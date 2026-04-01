/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

constexpr float log2e = 1.4426950216293334961f;

__device__ glm::vec3 loadSHCoeffHalf(const uint16_t* shsHalf, int idx, int max_coeffs, int coeffIdx)
{
	const half* sh = reinterpret_cast<const half*>(shsHalf) + (idx * max_coeffs + coeffIdx) * 3;
	return glm::vec3(__half2float(sh[0]), __half2float(sh[1]), __half2float(sh[2]));
}

__device__ float loadHalfAsFloat(const uint16_t* data, int idx)
{
	const half* values = reinterpret_cast<const half*>(data);
	return __half2float(values[idx]);
}

__device__ void loadCov3DHalf(const uint16_t* cov3DHalf, int idx, float* cov3D)
{
	const int base = idx * 6;
	#pragma unroll
	for (int i = 0; i < 6; ++i)
	{
		cov3D[i] = loadHalfAsFloat(cov3DHalf, base + i);
	}
}

__device__ uint32_t resolveSourceIndex(uint32_t global_idx, const uint32_t* source_indices)
{
	return source_indices != nullptr ? source_indices[global_idx] : global_idx;
}

__device__ int resolveInstanceIndex(uint32_t global_idx, const uint32_t* instance_indices)
{
	return instance_indices != nullptr ? static_cast<int>(instance_indices[global_idx]) : -1;
}

__device__ const float* resolveMatrixPointer(
	int instance_idx,
	const float* instance_matrices,
	const glm::mat4& fallback)
{
	return (instance_idx >= 0 && instance_matrices != nullptr) ?
		(instance_matrices + instance_idx * 16) :
		(reinterpret_cast<const float*>(&fallback));
}

__device__ glm::vec3 resolveCameraPosition(
	int instance_idx,
	const float* instance_cam_positions,
	const glm::vec3& fallback)
{
	if (instance_idx >= 0 && instance_cam_positions != nullptr) {
		const float* values = instance_cam_positions + instance_idx * 3;
		return glm::vec3(values[0], values[1], values[2]);
	}
	return fallback;
}

__device__ float resolveDepthScale(
	int instance_idx,
	const float* instance_depth_scales)
{
	return (instance_idx >= 0 && instance_depth_scales != nullptr) ? instance_depth_scales[instance_idx] : 1.0f;
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int sh_idx, int clamp_idx, int deg, int max_coeffs, const glm::vec3& pos, glm::vec3 campos, const float* shs, const uint16_t* shsHalf, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3 result;
	if (shs != nullptr) {
		glm::vec3* sh = ((glm::vec3*)shs) + sh_idx * max_coeffs;
		result = SH_C0 * sh[0];

		if (deg > 0)
		{
			float x = dir.x;
			float y = dir.y;
			float z = dir.z;
			result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

			if (deg > 1)
			{
				float xx = x * x, yy = y * y, zz = z * z;
				float xy = x * y, yz = y * z, xz = x * z;
				result = result +
					SH_C2[0] * xy * sh[4] +
					SH_C2[1] * yz * sh[5] +
					SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
					SH_C2[3] * xz * sh[7] +
					SH_C2[4] * (xx - yy) * sh[8];

				if (deg > 2)
				{
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
	} else {
		glm::vec3 sh0 = loadSHCoeffHalf(shsHalf, sh_idx, max_coeffs, 0);
		result = SH_C0 * sh0;

		if (deg > 0)
		{
			float x = dir.x;
			float y = dir.y;
			float z = dir.z;
			const glm::vec3 sh1 = loadSHCoeffHalf(shsHalf, sh_idx, max_coeffs, 1);
			const glm::vec3 sh2 = loadSHCoeffHalf(shsHalf, sh_idx, max_coeffs, 2);
			const glm::vec3 sh3 = loadSHCoeffHalf(shsHalf, sh_idx, max_coeffs, 3);
			result = result - SH_C1 * y * sh1 + SH_C1 * z * sh2 - SH_C1 * x * sh3;

			if (deg > 1)
			{
				float xx = x * x, yy = y * y, zz = z * z;
				float xy = x * y, yz = y * z, xz = x * z;
				const glm::vec3 sh4 = loadSHCoeffHalf(shsHalf, sh_idx, max_coeffs, 4);
				const glm::vec3 sh5 = loadSHCoeffHalf(shsHalf, sh_idx, max_coeffs, 5);
				const glm::vec3 sh6 = loadSHCoeffHalf(shsHalf, sh_idx, max_coeffs, 6);
				const glm::vec3 sh7 = loadSHCoeffHalf(shsHalf, sh_idx, max_coeffs, 7);
				const glm::vec3 sh8 = loadSHCoeffHalf(shsHalf, sh_idx, max_coeffs, 8);
				result = result +
					SH_C2[0] * xy * sh4 +
					SH_C2[1] * yz * sh5 +
					SH_C2[2] * (2.0f * zz - xx - yy) * sh6 +
					SH_C2[3] * xz * sh7 +
					SH_C2[4] * (xx - yy) * sh8;

				if (deg > 2)
				{
					const glm::vec3 sh9 = loadSHCoeffHalf(shsHalf, sh_idx, max_coeffs, 9);
					const glm::vec3 sh10 = loadSHCoeffHalf(shsHalf, sh_idx, max_coeffs, 10);
					const glm::vec3 sh11 = loadSHCoeffHalf(shsHalf, sh_idx, max_coeffs, 11);
					const glm::vec3 sh12 = loadSHCoeffHalf(shsHalf, sh_idx, max_coeffs, 12);
					const glm::vec3 sh13 = loadSHCoeffHalf(shsHalf, sh_idx, max_coeffs, 13);
					const glm::vec3 sh14 = loadSHCoeffHalf(shsHalf, sh_idx, max_coeffs, 14);
					const glm::vec3 sh15 = loadSHCoeffHalf(shsHalf, sh_idx, max_coeffs, 15);
					result = result +
						SH_C3[0] * y * (3.0f * xx - yy) * sh9 +
						SH_C3[1] * xy * z * sh10 +
						SH_C3[2] * y * (4.0f * zz - xx - yy) * sh11 +
						SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh12 +
						SH_C3[4] * x * (4.0f * zz - xx - yy) * sh13 +
						SH_C3[5] * z * (xx - yy) * sh14 +
						SH_C3[6] * x * (xx - 3.0f * yy) * sh15;
				}
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * clamp_idx + 0] = (result.x < 0);
	clamped[3 * clamp_idx + 1] = (result.y < 0);
	clamped[3 * clamp_idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix,
	const bool is_ortho, const bool is_fisheye, const float k1, const float k2, const float k3, const float k4)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J;
	if (is_ortho) {
		J = glm::mat3(
			focal_x, 0.0f, 0,
			0.0f, focal_y, 0,
			0, 0, 0);
	}
	else if (is_fisheye) // fisheye
	{
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

		float b = theta_d / len_xy / x2y2;
		float a = t.z * x2y2z2_inv / (x2y2);
		J = glm::mat3(
			focal_x * (x2 * a * D + y2 * b), focal_x * xy * (a * D - b),    - focal_x * t.x * x2y2z2_inv * D,
			focal_y * xy  * (a * D - b),    focal_y * (y2 * a * D + x2 * b), - focal_y * t.y * x2y2z2_inv * D,
			0, 0, 0
		);
	}
	else
	{
		J = glm::mat3(
			focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
			0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
			0, 0, 0);
	}

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const uint16_t* opacitiesHalf,
	const float* shs,
	const uint16_t* shsHalf,
	bool* clamped,
	const float* cov3D_precomp,
	const uint16_t* cov3DHalf,
	const float* colors_precomp,
	const glm::mat4 viewmatrix,
	const glm::mat4 projmatrix,
	const glm::vec3 cam_pos,
	const uint32_t* source_indices,
	const uint32_t* instance_indices,
	const float* instance_cam_positions,
	const float* instance_view_matrices,
	const float* instance_proj_matrices,
	const float* instance_depth_scales,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	bool is_ortho, bool is_fisheye, float k1, float k2, float k3, float k4, 
	int* radii,
	float2* points_xy_image,
	float* cov3Ds,
	float4* rgb_depth,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	const uint32_t src_idx = resolveSourceIndex(static_cast<uint32_t>(idx), source_indices);
	const int instance_idx = resolveInstanceIndex(static_cast<uint32_t>(idx), instance_indices);
	const float* view_ptr = resolveMatrixPointer(instance_idx, instance_view_matrices, viewmatrix);
	const float* proj_ptr = resolveMatrixPointer(instance_idx, instance_proj_matrices, projmatrix);
	const glm::vec3 local_cam_pos = resolveCameraPosition(instance_idx, instance_cam_positions, cam_pos);
	const float depth_scale = resolveDepthScale(instance_idx, instance_depth_scales);

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	float3 p_orig = { orig_points[3 * src_idx], orig_points[3 * src_idx + 1], orig_points[3 * src_idx + 2] };
	float3 p_view = transformPoint4x3(p_orig, view_ptr);
	if (p_view.z <= 0.2f)
		return;

	// Transform point by projecting
	float3 p_proj;
	if (is_fisheye) // panorama
	{
		float xy_len = glm::length(glm::vec2({p_view.x, p_view.y})) + 0.000001f;
		float theta = glm::atan(xy_len, p_view.z + 0.0000001f);
		float theta2 = theta * theta;
		float theta4 = theta2 * theta2;
		float theta_d = theta * ( 1 + k1 * theta2 + k2 * theta4 + k3 * theta2 * theta4 + k4 * theta4 * theta4);
		if (abs(theta) > 3.14 * 0.403)
			return; 
		p_proj.x = 2 * p_view.x * focal_x * theta_d / (xy_len * W);  // clip space
		p_proj.y = 2 * p_view.y * focal_y * theta_d / (xy_len * H);
		p_proj.z = 0;
	}
	else
	{
		float4 p_hom = transformPoint4x4(p_orig, proj_ptr);
		if (is_ortho) {
			p_proj = { p_hom.x, p_hom.y, 0}; // 正交投影无需透视除法
		}
		else {
			float p_w = 1.0f / (p_hom.w + 0.0000001f);
			p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
		}
	}

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	float cov3DDecoded[6];
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + src_idx * 6;
	}
	else if (cov3DHalf != nullptr)
	{
		loadCov3DHalf(cov3DHalf, src_idx, cov3DDecoded);
		cov3D = cov3DDecoded;
	}
	else
	{
		computeCov3D(scales[src_idx], scale_modifier, rotations[src_idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, view_ptr, is_ortho, is_fisheye, k1, k2, k3, k4);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov;

	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(src_idx, idx, D, M, glm::vec3(p_orig.x, p_orig.y, p_orig.z), local_cam_pos, shs, shsHalf, clamped);
		rgb_depth[idx].x = result.x;
		rgb_depth[idx].y = result.y;
		rgb_depth[idx].z = result.z;
		rgb_depth[idx].w = p_view.z * depth_scale;
	}

	// Store some useful helper data for the next steps.
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = opacities != nullptr ? opacities[src_idx] : loadHalfAsFloat(opacitiesHalf, src_idx);

	float log2_opacity; 
	asm volatile("lg2.approx.f32 %0, %1;" : "=f"(log2_opacity) : "f"(opacity * h_convolution_scaling));
	conic_opacity[idx] = { (-0.5f * log2e) * conic.x, -log2e * conic.y, (-0.5f * log2e) * conic.z, log2_opacity };


	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ features_depths,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float3  bg_color,
	float* __restrict__ out_color,
	float* __restrict__ invdepth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			// float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			float power = con_o.w + con_o.x * d.x * d.x + con_o.z * d.y * d.y + con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// float alpha = min(0.99f, con_o.w * exp(power));
			float alpha;
			asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(alpha) : "f"(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			// for (int ch = 0; ch < CHANNELS; ch++)
			// 	C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			C[0] += features_depths[collected_id[j]].x * alpha * T;
			C[1] += features_depths[collected_id[j]].y * alpha * T;
			C[2] += features_depths[collected_id[j]].z * alpha * T;

			if(invdepth)
			// expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;
				expected_invdepth += (1.0f / features_depths[collected_id[j]].w) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		out_color[pix_id * 4 + 0] = C[0] + T * bg_color.x; // HWC
		out_color[pix_id * 4 + 1] = C[1] + T * bg_color.y;
		out_color[pix_id * 4 + 2] = C[2] + T * bg_color.z;
		out_color[pix_id * 4 + 3] = 1.0f - T; // Store alpha in 4th channel

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float4* colors_depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float3 bg_color,
	float* out_color,
	float* depth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors_depths,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depth);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const uint16_t* opacitiesHalf,
	const float* shs,
	const uint16_t* shsHalf,
	bool* clamped,
	const float* cov3D_precomp,
	const uint16_t* cov3DHalf,
	const float* colors_precomp,
	const glm::mat4 viewmatrix,
	const glm::mat4 projmatrix,
	const glm::vec3 cam_pos,
	const uint32_t* source_indices,
	const uint32_t* instance_indices,
	const float* instance_cam_positions,
	const float* instance_view_matrices,
	const float* instance_proj_matrices,
	const float* instance_depth_scales,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	bool is_ortho, bool is_fisheye, float k1, float k2, float k3, float k4, 
	int* radii,
	float2* means2D,
	float* cov3Ds,
	float4* rgb_depth,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		opacitiesHalf,
		shs,
		shsHalf,
		clamped,
		cov3D_precomp,
		cov3DHalf,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		source_indices,
		instance_indices,
		instance_cam_positions,
		instance_view_matrices,
		instance_proj_matrices,
		instance_depth_scales,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		is_ortho, is_fisheye, k1, k2, k3, k4,
		radii,
		means2D,
		cov3Ds,
		rgb_depth,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing
		);
}
