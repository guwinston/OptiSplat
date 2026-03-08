#include "ops.h"
#include <iostream>

#ifndef CUDA_VERSION
#define CUDA_VERSION 8000
#endif

#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include <vector>
#include <iostream>
#include <fstream>


namespace flashgs {
namespace {

constexpr float log2e = 1.4426950216293334961f;
constexpr float ln2 = 0.69314718055f;

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float fast_max_f32(float a, float b)
{
	float d;
	asm volatile("max.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
	return d;
}

__forceinline__ __device__ float fast_sqrt_f32(float x)
{
	float y;
	asm volatile("sqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
	return y;
}

__forceinline__ __device__ float fast_rsqrt_f32(float x)
{
	float y;
	asm volatile("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
	return y;
}

__forceinline__ __device__ float fast_lg2_f32(float x)
{
	float y;
	asm volatile("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
	return y;
}

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ float3 transformPoint4x3(const glm::vec3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const glm::vec3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ void getRect(const float2 p, int width, int height, int2& rect_min, int2& rect_max, dim3 grid, int block_x, int block_y)
{
	rect_min = {
		min((int)grid.x, max((int)0, (int)((p.x - width) / (float)block_x))),
		min((int)grid.y, max((int)0, (int)((p.y - height) / (float)block_y)))
	};
	rect_max = {
		min((int)grid.x, max((int)0, (int)((p.x + width) / (float)block_x) + 1)),
		min((int)grid.y, max((int)0, (int)((p.y + height) / (float)block_y) + 1))
	};
}

// Forward version of 2D covariance matrix computation
__forceinline__ __device__ float3 computeCov2D(const glm::vec3& position, float focal_x, float focal_y, float tan_fovx, float tan_fovy,
	cov3d_t cov3D, glm::mat4 viewmatrix, bool is_ortho, bool is_fisheye, float k1, float k2, float k3, float k4)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002).
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(position, (float*)&viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J;
	if (is_ortho) {
		// orhographic projection
		// u = fx * x + cx, v = fy * y + cy
		J = glm::mat3(
			focal_x, 0.0f, 0,
			0.0f, focal_y, 0,
			0, 0, 0);
	}
	else if (is_fisheye) {
		// Fisheye projection
		float eps = 0.0000001f;
		float x2 = t.x * t.x + eps;
		float y2 = t.y * t.y;
		float xy = t.x * t.y;
		float x2y2 = x2 + y2 ;
		float len_xy = length(glm::vec2({t.x, t.y})) + eps;
		float x2y2z2_inv = 1.f / (x2y2 + t.z * t.z);

		// Kannala-Brandt distortion
		float theta = glm::atan(len_xy, t.z);
		float theta2 = theta * theta;
		float theta4 = theta2 * theta2;
		float theta_d = theta * ( 1 + k1 * theta2 + k2 * theta4 + k3 * theta2 * theta4 + k4 * theta4 * theta4);
		float D = 1 + 3 * k1 * theta2 + 5 * k2 * theta4 + 7 * k3 * theta2 * theta4 + 9 * k4 * theta4 * theta4;

		float b =  theta / len_xy / x2y2;
		float a = t.z * x2y2z2_inv / (x2y2);
		J = glm::mat3(
			focal_x * (x2 * a * D + y2 * b), focal_x * xy * (a * D - b),    - focal_x * t.x * x2y2z2_inv * D,
			focal_y * xy  * (a * D - b),    focal_y * (y2 * a * D + x2 * b), - focal_y * t.y * x2y2z2_inv * D,
			0, 0, 0
		);
	}
	else {
		// persperctive projection
		J = glm::mat3(
			focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
			0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
			0, 0, 0);
	}

	glm::mat3 W = glm::mat3(
		((float*)&viewmatrix)[0], ((float*)&viewmatrix)[4], ((float*)&viewmatrix)[8],
		((float*)&viewmatrix)[1], ((float*)&viewmatrix)[5], ((float*)&viewmatrix)[9],
		((float*)&viewmatrix)[2], ((float*)&viewmatrix)[6], ((float*)&viewmatrix)[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D.s[0], cov3D.s[1], cov3D.s[2],
		cov3D.s[1], cov3D.s[3], cov3D.s[4],
		cov3D.s[2], cov3D.s[4], cov3D.s[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__forceinline__ __device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, glm::vec3 p_orig, glm::vec3 campos, const glm::vec3* shs)
{
	// The implementation is loosely based on code for
	// "Differentiable Point-Based Radiance Fields for
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 dir = p_orig - campos;
	float l2 = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
	float rsqrt_l2 = fast_rsqrt_f32(l2);
	dir *= rsqrt_l2;

	// const auto& sh = ((const shs_deg3_t*)shs)[idx];
	const glm::vec3* sh = shs + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0] += 0.5f;

	if (deg > 0) {
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1) {
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2) {
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}

	result.x = fast_max_f32(result.x, 0.0f);
	result.y = fast_max_f32(result.y, 0.0f);
	result.z = fast_max_f32(result.z, 0.0f);
	return result;
}

__forceinline__ __device__ bool segment_intersect_ellipse(float a, float b, float c, float d, float l, float r)
{
	float delta = b * b - 4.0f * a * c;
	// return delta >= 0.0f && t1 <= sqrt(delta) && t2 >= -sqrt(delta)
	float t1 = (l - d) * (2.0f * a) + b;
	float t2 = (r - d) * (2.0f * a) + b;
	return delta >= 0.0f && (t1 <= 0.0f || t1 * t1 <= delta) && (t2 >= 0.0f || t2 * t2 <= delta);
}

__forceinline__ __device__ bool block_intersect_ellipse(int2 pix_min, int2 pix_max, float2 center, float3 conic, float power)
{
	float a, b, c, dx, dy;
	float w = 2.0f * power;

	if (center.x * 2.0f < pix_min.x + pix_max.x)
	{
		dx = center.x - pix_min.x;
	}
	else
	{
		dx = center.x - pix_max.x;
	}
	a = conic.z;
	b = -2.0f * conic.y * dx;
	c = conic.x * dx * dx - w;

	if (segment_intersect_ellipse(a, b, c, center.y, pix_min.y, pix_max.y))
	{
		return true;
	}

	if (center.y * 2.0f < pix_min.y + pix_max.y)
	{
		dy = center.y - pix_min.y;
	}
	else
	{
		dy = center.y - pix_max.y;
	}
	a = conic.x;
	b = -2.0f * conic.y * dy;
	c = conic.z * dy * dy - w;

	if (segment_intersect_ellipse(a, b, c, center.x, pix_min.x, pix_max.x))
	{
		return true;
	}

	return false;
}

__forceinline__ __device__ bool block_contains_center(int2 pix_min, int2 pix_max, float2 center)
{
	return center.x >= pix_min.x && center.x <= pix_max.x && center.y >= pix_min.y && center.y <= pix_max.y;
}

__global__ void preprocessCUDA(
	int P, int D, int M,
	const glm::vec3* __restrict__ positions,
	const float* __restrict__ opacities,
	const glm::vec3* __restrict__ shs,
	const glm::mat4 viewmatrix,
	const glm::mat4 projmatrix,
	const glm::vec3 cam_position,
	const int W, int H,
	int block_x, int block_y,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	bool is_ortho, bool is_fisheye, float k1, float k2, float k3, float k4,
	float2* __restrict__ points_xy,
	cov3d_t* __restrict__ cov3Ds,
	float4* __restrict__ rgb_depth,
	float4* __restrict__ conic_opacity,
	int* __restrict__ curr_offset,
	uint64_t* __restrict__ gaussian_keys_unsorted,
	uint32_t* __restrict__ gaussian_values_unsorted,
	const dim3 grid)
{
	int lane = threadIdx.y * blockDim.x + threadIdx.x;
	int warp_id = blockIdx.x * blockDim.z + threadIdx.z;
	int idx_vec = warp_id * FLASHGS_WARP_SIZE + lane;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	bool point_valid = false;
	glm::vec3 p_orig;
	int width = 0;
	int height = 0;
	float3 p_view;
	float2 point_xy;
	float3 conic;
	float opacity;
	float power;
	float log2_opacity;
	int2 rect_min;
	int2 rect_max;
	if (idx_vec < P)
	{
		do {
			// Perform near culling, quit if outside.
			p_orig = positions[idx_vec];
			p_view = transformPoint4x3(p_orig, (const float*)&viewmatrix);
			if (p_view.z <= 0.2f)
				break;
			opacity = opacities[idx_vec];
			if (255.0f * opacity < 1.0f)
				break;

			// Transform point by projecting
			float3 p_proj;
			if (is_fisheye) {
				// Fisheye Kannala-Brandt projection
				float xy_len = glm::length(glm::vec2({p_view.x, p_view.y})) + 0.000001f;
				float r = xy_len / (p_view.z + 0.000001f);

				float theta = atan2(r, 1.0f);
				if (abs(theta) > 3.14 * 0.44)
					break; 

				// Kannala-Brandt distortion
				float theta2 = theta * theta;
				float theta4 = theta2 * theta2;
				float theta_d = theta * ( 1 + k1 * theta2 + k2 * theta4 + k3 * theta2 * theta4 + k4 * theta4 * theta4);

				p_proj.x = 2 * p_view.x * focal_x * theta_d / (xy_len * W); // clip space
				p_proj.y = 2 * p_view.y * focal_y * theta_d / (xy_len * H);
				p_proj.z = 0;
			}
			else {
				float4 p_hom = transformPoint4x4(p_orig, (const float*)&projmatrix);
				if (is_ortho) {
					p_proj = { p_hom.x, p_hom.y, 0}; // 正交投影无需透视除法
				}
				else {
					float p_w = 1.0f / (p_hom.w + 0.0000001f);
					p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
				}
			}

			// Compute 2D screen-space covariance matrix
			float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3Ds[idx_vec], viewmatrix, is_ortho, is_fisheye, k1, k2, k3, k4);

			// Invert covariance (EWA algorithm)
			float det = (cov.x * cov.z - cov.y * cov.y);
			float det_inv = 1.f / det;
			conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

			log2_opacity = fast_lg2_f32(opacity);
			power = ln2 * 8.0f + ln2 * log2_opacity;
			width = (int)(1.414214f * fast_sqrt_f32(cov.x * power) + 1.0f);
			height = (int)(1.414214f * fast_sqrt_f32(cov.z * power) + 1.0f);

			point_xy = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
			getRect(point_xy, width, height, rect_min, rect_max, grid, block_x, block_y);
			point_valid = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) > 0;
		} while (false);
	}

	bool single_tile = point_valid && (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 1;
	if (single_tile)
	{
		int2 pix_min = { rect_min.x * block_x, rect_min.y * block_y };
		int2 pix_max = { pix_min.x + block_x - 1, pix_min.y + block_y - 1 };
		bool valid = block_contains_center(pix_min, pix_max, point_xy) ||
			block_intersect_ellipse(pix_min, pix_max, point_xy, conic, power);
		if (valid)
		{
			uint64_t key = rect_min.y * grid.x + rect_min.x;
			key <<= 32;
			key |= __float_as_uint(p_view.z);
			int offset = atomicAdd(curr_offset, 1);
			gaussian_keys_unsorted[offset] = key;
			gaussian_values_unsorted[offset] = idx_vec;
		}
		point_valid = false;
	}

	// Generate no key/value pair for invisible Gaussians
	int multi_tiles = __ballot_sync(~0, point_valid);
	bool vertex_valid = single_tile;
	while (multi_tiles)
	{
		int i = __ffs(multi_tiles) - 1;
		multi_tiles &= multi_tiles - 1;
		// Find this Gaussian's offset in buffer for writing keys/values.
		float2 my_point_xy = {
			__shfl_sync(~0, point_xy.x, i),
			__shfl_sync(~0, point_xy.y, i)
		};
		float3 my_conic = {
			__shfl_sync(~0, conic.x, i),
			__shfl_sync(~0, conic.y, i),
			__shfl_sync(~0, conic.z, i),
		};
		int2 my_rect_min = {
			__shfl_sync(~0, rect_min.x, i),
			__shfl_sync(~0, rect_min.y, i)
		};
		int2 my_rect_max = {
			__shfl_sync(~0, rect_max.x, i),
			__shfl_sync(~0, rect_max.y, i)
		};
		float my_depth = __shfl_sync(~0, p_view.z, i);
		float my_power = __shfl_sync(~0, power, i);
		int idx = warp_id * FLASHGS_WARP_SIZE + i;

		// For each tile that the bounding rect overlaps, emit a
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth.
		for (int y0 = my_rect_min.y; y0 < my_rect_max.y; y0 += blockDim.y)   //循环迭代tile范围，为每个tile生成键值对
		{
			int y = y0 + threadIdx.y;
			for (int x0 = my_rect_min.x; x0 < my_rect_max.x; x0 += blockDim.x)
			{
				int x = x0 + threadIdx.x;
				bool valid = y < my_rect_max.y && x < my_rect_max.x;

				if (valid)
				{
					int2 pix_min = { x * block_x, y * block_y };
					int2 pix_max = { pix_min.x + block_x - 1, pix_min.y + block_y - 1 };
					valid = block_contains_center(pix_min, pix_max, my_point_xy) ||
						block_intersect_ellipse(pix_min, pix_max, my_point_xy, my_conic, my_power);
				}

				int mask = __ballot_sync(~0, valid);
				if (mask == 0)
				{
					continue;
				}
				int my_offset;
				if (lane == 0)
				{
					my_offset = atomicAdd(curr_offset, __popc(mask));
				}
				vertex_valid = vertex_valid || i == lane;
				int count = __popc(mask & ((1 << lane) - 1));
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= __float_as_uint(my_depth);
				my_offset = __shfl_sync(~0, my_offset, 0);
				if (valid)
				{
					gaussian_keys_unsorted[my_offset + count] = key;
					gaussian_values_unsorted[my_offset + count] = idx;
				}
			}
		}
	}

	if (vertex_valid)
	{
		points_xy[idx_vec] = point_xy;
		conic_opacity[idx_vec] = { (-0.5f * log2e) * conic.x, -log2e * conic.y, (-0.5f * log2e) * conic.z, log2_opacity };
		auto color = computeColorFromSH(idx_vec, D, M, p_orig, cam_position, shs);
		rgb_depth[idx_vec] = {color.r, color.g, color.b, p_view.z};
	}
}


} // namespace

void preprocess(int P, int D, int M,
	glm::vec3* positions, glm::vec3* shs, const float* opacities, cov3d_t* cov3Ds,
	int width, int height, int block_x, int block_y,
	const glm::vec3 cam_position, const glm::mat3 cam_rotation, const glm::mat4 view_matrix, const glm::mat4 proj_matrix,
	float focal_x, float focal_y, float zFar, float zNear, float tan_fovx, float tan_fovy, bool is_ortho, bool is_fisheye, float k1, float k2, float k3, float k4,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	int* curr_offset, cudaStream_t stream
)
{
	dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y, 1);

#ifdef DEBUG
	std::cout << std::endl;
	std::cout << "Preprocessing with grid " << grid.x << "x" << grid.y << "x" << grid.z << " and block " << block_x << "x" << block_y << std::endl;
	std::cout << "Camera position: " << cam_position.x << ", " << cam_position.y << ", " << cam_position.z << std::endl;
	std::cout << "Camera rotation: " << cam_rotation[0][0] << ", " << cam_rotation[1][0] << ", " << cam_rotation[2][0] << std::endl;
	std::cout << "				   " << cam_rotation[0][1] << ", " << cam_rotation[1][1] << ", " << cam_rotation[2][1] << std::endl;
	std::cout << "				   " << cam_rotation[0][2] << ", " << cam_rotation[1][2] << ", " << cam_rotation[2][2] << std::endl;
	std::cout << "Focal length: " << focal_x << ", " << focal_y << std::endl;
	std::cout << "Z range: " << zNear << " - " << zFar << std::endl;
	std::cout << "Block size: " << block_x << "x" << block_y << std::endl;
	std::cout << "view matrix: " << view_matrix[0][0] << ", " << view_matrix[1][0] << ", " << view_matrix[2][0] << ", " << view_matrix[3][0] << std::endl;
	std::cout << "			   " << view_matrix[0][1] << ", " << view_matrix[1][1] << ", " << view_matrix[2][1] << ", " << view_matrix[3][1] << std::endl;
	std::cout << "			   " << view_matrix[0][2] << ", " << view_matrix[1][2] << ", " << view_matrix[2][2] << ", " << view_matrix[3][2] << std::endl;
	std::cout << "			   " << view_matrix[0][3] << ", " << view_matrix[1][3] << ", " << view_matrix[2][3] << ", " << view_matrix[3][3] << std::endl;
	std::cout << "proj matrix: " << proj_matrix[0][0] << ", " << proj_matrix[1][0] << ", " << proj_matrix[2][0] << ", " << proj_matrix[3][0] << std::endl;
	std::cout << "			   " << proj_matrix[0][1] << ", " << proj_matrix[1][1] << ", " << proj_matrix[2][1] << ", " << proj_matrix[3][1] << std::endl;
	std::cout << "			   " << proj_matrix[0][2] << ", " << proj_matrix[1][2] << ", " << proj_matrix[2][2] << ", " << proj_matrix[3][2] << std::endl;
	std::cout << "			   " << proj_matrix[0][3] << ", " << proj_matrix[1][3] << ", " << proj_matrix[2][3] << ", " << proj_matrix[3][3] << std::endl;
	std::cout << "tan fov: " << tan_fovx << ", " << tan_fovy << std::endl;
#endif

	preprocessCUDA<<<(P + 127) / 128, dim3(8, 4, 4), 0, stream>>>(
		P, D, M,
		positions,
		opacities,
		shs,
		view_matrix,
		proj_matrix,
		cam_position,
		width, height,
		block_x, block_y,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		is_ortho, is_fisheye, k1, k2, k3, k4,
		points_xy,
		cov3Ds,
		rgb_depth,
		conic_opacity,
		curr_offset,
		gaussian_keys_unsorted,
		gaussian_values_unsorted,
		grid);

}

} // namepace flashgs