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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <climits>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
// #include "backward.h"
#include "ops.h"
#include "tcgs.h"

// #define ENABLE_CUDA_TIMING
// #define DEBUG
#ifdef DEBUG
template <typename T>
void saveCudaArrayToFile(const T* d_array, size_t size, const std::string& filename) {
    T* h_array = new T[size];

    cudaMemcpy(h_array, d_array, size * sizeof(T), cudaMemcpyDeviceToHost);

    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        delete[] h_array;
        return;
    }

	// std::cout << "First element of array: " << h_array[0] << std::endl;

    outFile.write(reinterpret_cast<const char*>(h_array), size * sizeof(T));
    outFile.close();

    delete[] h_array;
}
#endif

/**
 * @brief Construct the world-to-camera (view) matrix.
 *
 * GLM uses column-major storage. The returned matrix transforms a point
 * from world coordinates into camera coordinates.
 *
 * @param[in] position  Camera center in world coordinates (t).
 * @param[in] rotation  Camera-to-world rotation matrix (R = C2W.R).
 *
 * @return 4x4 world-to-camera (view) matrix:
 *
 *         W2C = [ R^T   -R^T * t ]
 *               [ 0	      1   ]
 *
 * where t is the camera position in world coordinates.
 */
glm::mat4 getViewMatrix(const glm::vec3& position, const glm::mat3& rotation)
{
	return glm::mat4(
		glm::vec4(rotation[0][0], rotation[1][0], rotation[2][0], 0.0f),
		glm::vec4(rotation[0][1], rotation[1][1], rotation[2][1], 0.0f),
		glm::vec4(rotation[0][2], rotation[1][2], rotation[2][2], 0.0f),
		glm::vec4(glm::transpose(rotation) * -position, 1.0f));
}


glm::mat4 getProjectionMatrix(int width, int height, const glm::vec3& position, const glm::mat3& rotation, float focal_x, float focal_y, float zFar, float zNear, bool isOrtho=false)
{
	
	glm::mat4 P;
	memset(&P, 0, sizeof P);
	if (isOrtho) {
		// orthographic
        // 在正交模式下，focal_x 的含义是“物理世界 1 个单位长度对应屏幕上的多少个像素”
        float right = width / (2.0f * focal_x);
        float left = -right;
        float top = height / (2.0f * focal_y);
        float bottom = -top;

        P[0][0] = 2.0f / (right - left);
        P[1][1] = 2.0f / (top - bottom);
        P[2][2] = 1.0f / (zFar - zNear);
        P[0][3] = -(right + left) / (right - left);
        P[1][3] = -(top + bottom) / (top - bottom);
        P[2][3] = -zNear / (zFar - zNear);
        P[3][3] = 1.0f;
	}
	else {
		// perspective
		float z_sign = 1.0f;
		float top = height / (2.0f * focal_y) * zNear;
		float bottom = -top;
		float right = width / (2.0f * focal_x) * zNear;
		float left = -right;

		P[0][0] = 2.0f * zNear / (right - left);
		P[1][1] = 2.0f * zNear / (top - bottom);
		P[0][2] = (right + left) / (right - left);
		P[1][2] = (top + bottom) / (top - bottom);
		P[3][2] = z_sign;
		P[2][2] = z_sign * zFar / (zFar - zNear);
		P[2][3] = -(zFar * zNear) / (zFar - zNear);
	}

	return glm::transpose(P) * getViewMatrix(position, rotation);
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float4* rgb_depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&rgb_depths[idx].w);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

__global__ void scatterActiveIndices(
	int P,
	const uint32_t* active_flags,
	const uint32_t* active_offsets,
	uint32_t* active_indices)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	if (active_flags[idx] == 0)
		return;
	active_indices[active_offsets[idx] - 1] = idx;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(
	char*& chunk, size_t P, bool needsCov3DScratch)
{
	GeometryState geom;
	// obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	if (needsCov3DScratch) {
		obtain(chunk, geom.cov3D, P * 6, 128);
	} else {
		geom.cov3D = nullptr;
	}
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb_depth, P, 128);
	// obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::ActiveState CudaRasterizer::ActiveState::fromChunk(char*& chunk, size_t P)
{
	ActiveState active;
	obtain(chunk, active.active_flags, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, active.scan_size, active.active_flags, active.active_flags, P);
	obtain(chunk, active.scanning_space, active.scan_size, 128);
	obtain(chunk, active.active_offsets, P, 128);
	obtain(chunk, active.active_indices, P, 128);
	return active;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

namespace {

int growExactIntersectionCapacity(int currentCapacity, int requiredCount)
{
	const int minCapacity = 1 << 20;
	const int growthFromCurrent = currentCapacity > 0 ? currentCapacity + currentCapacity / 2 : 0;
	const int growthFromRequired = requiredCount + std::max(requiredCount / 4, minCapacity);
	return std::max(std::max(requiredCount, minCapacity), std::max(growthFromCurrent, growthFromRequired));
}

}


// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> activeBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M, int frameCapacity, int capacityLimit,
	bool useExactIntersection, int exactActiveSetMode, bool usePrefetchingPipeline, bool useTensorCore,
	const std::vector<float>& cpuCamPos, const std::vector<float>& cpuCamRot,
	const std::vector<float>& cpuViewMatrix, const std::vector<float>& cpuProjMatrix,
	float znear, float zfar, int* currOffset, int* overflowFlag,
	bool isOrtho, bool isFisheye, float k1, float k2, float k3, float k4, 
	const std::vector<float>& cpuBackground,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const uint16_t* shsHalf,
	const float* colors_precomp,
	const float* opacities,
	const uint16_t* opacitiesHalf,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const uint16_t* cov3DHalf,
	const uint32_t* sourceIndices,
	const uint32_t* instanceIndices,
	const float* instanceCamPositions,
	const float* instanceViewMatrices,
	const float* instanceProjMatrices,
	const float* instanceDepthScales,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* depth,
	bool antialiasing,
	int* resolvedCapacity,
	int* activeGaussians,
	int* radii,
	bool debug)
{
#ifdef ENABLE_CUDA_TIMING
	cudaEvent_t frameStartEvent;
	cudaEvent_t preprocessEndEvent;
	cudaEvent_t sortEndEvent;
	cudaEvent_t renderEndEvent;
	cudaEventCreate(&frameStartEvent);
	cudaEventCreate(&preprocessEndEvent);
	cudaEventCreate(&sortEndEvent);
	cudaEventCreate(&renderEndEvent);
	float preprocessTimeMs = 0.f;
	float sortTimeMs       = 0.f;
	float renderTimeMs     = 0.f;
	float totalFrameTimeMs = 0.f;

	cudaEventRecord(frameStartEvent);
#endif

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const glm::vec3 cpuCamPosGlm = glm::make_vec3(cpuCamPos.data());
	const glm::mat3 cpuCamRotGlm = glm::make_mat3(cpuCamRot.data());
	const glm::mat4 viewmatrixGlm = cpuViewMatrix.size() == 16 ?
		glm::make_mat4(cpuViewMatrix.data()) :
		getViewMatrix(cpuCamPosGlm, cpuCamRotGlm);
	const glm::mat4 projmatrixGlm = cpuProjMatrix.size() == 16 ?
		glm::make_mat4(cpuProjMatrix.data()) :
		getProjectionMatrix(width, height, cpuCamPosGlm, cpuCamRotGlm, focal_x, focal_y, zfar, znear, isOrtho);

	const bool needsCov3DScratch = (cov3D_precomp == nullptr && cov3DHalf == nullptr);
	GeometryState geomState{};
	size_t geometryPointCount = static_cast<size_t>(P);

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	BinningState binningState;
	if (!useExactIntersection && frameCapacity > 0) {
		size_t binning_chunk_size = required<BinningState>(frameCapacity);
		char* binning_chunkptr = binningBuffer(binning_chunk_size);
		binningState = BinningState::fromChunk(binning_chunkptr, frameCapacity);
	}

#ifdef DEBUG
	saveCudaArrayToFile(means3D, P * 3, "./output/means3d.bin");
	if (shs != nullptr) saveCudaArrayToFile(shs, P * D * M, "./output/shs.bin");
	if (opacities != nullptr) saveCudaArrayToFile(opacities, P, "./output/opacities.bin");
	if (scales) saveCudaArrayToFile(scales, P * 3, "./output/scales.bin");
	if (rotations) saveCudaArrayToFile(rotations, P * 4, "./output/rotations.bin");
	if (cov3D_precomp) saveCudaArrayToFile(cov3D_precomp, P * 6, "./output/cov3d_precomp.bin");
	saveCudaArrayToFile(viewmatrix, 16, "./output/viewmatrix.bin");
	saveCudaArrayToFile(projmatrix, 16, "./output/projmatrix.bin");
	saveCudaArrayToFile(cam_pos, 3, "./output/cam_pos.bin");
#endif

	int num_rendered = 0;
	if (useExactIntersection) {
		const bool useExactActiveSet = exactActiveSetMode != 0;
		const bool useCenterOnlyActiveSet = exactActiveSetMode == 2;
		int activeCount = P;
		const uint32_t* activeIndices = nullptr;
		if (useExactActiveSet && P > 0) {
			char* active_size = nullptr;
			ActiveState::fromChunk(active_size, P);
			size_t active_chunk_size = reinterpret_cast<size_t>(active_size) + 128;
			char* active_chunkptr = activeBuffer(active_chunk_size);
			ActiveState activeState = ActiveState::fromChunk(active_chunkptr, P);

				CHECK_CUDA(flashgs::markActive(
					P,
					(glm::vec3*)means3D,
					opacities,
					opacitiesHalf,
					(flashgs::cov3d_t*)cov3D_precomp,
					cov3DHalf,
					width, height, BLOCK_X, BLOCK_Y,
				cpuCamPosGlm, cpuCamRotGlm, viewmatrixGlm, projmatrixGlm,
				focal_x, focal_y, zfar, znear, tan_fovx, tan_fovy, isOrtho, isFisheye, k1, k2, k3, k4,
				sourceIndices, instanceIndices, instanceCamPositions, instanceViewMatrices, instanceProjMatrices, instanceDepthScales,
				useCenterOnlyActiveSet,
				activeState.active_flags
			), debug);
			CHECK_CUDA(cub::DeviceScan::InclusiveSum(
				activeState.scanning_space,
				activeState.scan_size,
				activeState.active_flags,
				activeState.active_offsets,
				P), debug);
			CHECK_CUDA(cudaMemcpy(&activeCount, activeState.active_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
			if (activeCount > 0) {
				scatterActiveIndices<<<(P + 255) / 256, 256>>>(
					P, activeState.active_flags, activeState.active_offsets, activeState.active_indices);
				activeIndices = activeState.active_indices;
			}
		}
		if (activeCount <= 0) {
			if (debug) {
				std::cerr << "[Exact] activeCount <= 0 (P=" << P
						  << ", exactActiveSetMode=" << exactActiveSetMode << ")" << std::endl;
			}
			if (activeGaussians != nullptr) {
				*activeGaussians = 0;
			}
			if (resolvedCapacity != nullptr) {
				*resolvedCapacity = std::max(0, frameCapacity);
			}
			return 0;
		}
		if (activeGaussians != nullptr) {
			*activeGaussians = activeCount;
		}

		geometryPointCount = static_cast<size_t>(std::max(activeCount, 0));
		char* geom_size = nullptr;
		GeometryState::fromChunk(geom_size, geometryPointCount, needsCov3DScratch);
		size_t chunk_size = reinterpret_cast<size_t>(geom_size) + 128;
		char* chunkptr = geometryBuffer(chunk_size);
		geomState = GeometryState::fromChunk(chunkptr, geometryPointCount, needsCov3DScratch);

		if (radii == nullptr)
		{
			radii = geomState.internal_radii;
		}

		int capacity = std::max(0, frameCapacity);
		const int effectiveCapacityLimit = capacityLimit > 0 ? capacityLimit : INT_MAX;
		bool needsRetry = false;
		int retryCount = 0;
		do {
			needsRetry = false;
			if (capacity > 0) {
				size_t binning_chunk_size = required<BinningState>(capacity);
				char* binning_chunkptr = binningBuffer(binning_chunk_size);
				binningState = BinningState::fromChunk(binning_chunkptr, capacity);
			}

			CHECK_CUDA(cudaMemset(currOffset, 0, sizeof(int)), debug);
			if (overflowFlag != nullptr) {
				CHECK_CUDA(cudaMemset(overflowFlag, 0, sizeof(int)), debug);
			}

			CHECK_CUDA(flashgs::preprocess(
				activeCount, D, M,
				(glm::vec3*)means3D, (glm::vec3*)shs, shsHalf, opacities, opacitiesHalf, (flashgs::cov3d_t*)cov3D_precomp, cov3DHalf,
				width, height, BLOCK_X, BLOCK_Y,
				cpuCamPosGlm, cpuCamRotGlm, viewmatrixGlm, projmatrixGlm,
				focal_x, focal_y, zfar, znear, tan_fovx, tan_fovy, isOrtho, isFisheye, k1, k2, k3, k4,
				sourceIndices, instanceIndices, instanceCamPositions, instanceViewMatrices, instanceProjMatrices, instanceDepthScales,
				(float2*)geomState.means2D, (float4*)geomState.rgb_depth, (float4*)geomState.conic_opacity,
				capacity > 0 ? (uint64_t*)binningState.point_list_keys_unsorted : nullptr,
				capacity > 0 ? (uint32_t*)binningState.point_list_unsorted : nullptr,
				currOffset,
				capacity,
				overflowFlag,
				activeIndices
			), debug);

			int overflowed = 0;
			CHECK_CUDA(cudaMemcpy(&num_rendered, currOffset, sizeof(int), cudaMemcpyDeviceToHost), debug);
			if (overflowFlag != nullptr) {
				CHECK_CUDA(cudaMemcpy(&overflowed, overflowFlag, sizeof(int), cudaMemcpyDeviceToHost), debug);
			}
			cudaDeviceSynchronize();
			if (debug && num_rendered <= 0) {
				std::cerr << "[Exact] preprocess produced num_rendered=" << num_rendered
						  << " (activeCount=" << activeCount
						  << ", capacity=" << capacity
						  << ", overflowed=" << overflowed << ")" << std::endl;
			}

			if (num_rendered <= 0) {
				break;
			}

			if (capacity == 0 || overflowed != 0 || num_rendered > capacity) {
				const int nextCapacity = std::min(growExactIntersectionCapacity(capacity, num_rendered), effectiveCapacityLimit);
				if (nextCapacity <= capacity) {
					std::cerr << "Error: exact-intersection buffer reached capacity limit "
							  << effectiveCapacityLimit << " with num_rendered=" << num_rendered
							  << ". Increase maxNumRenderedGaussians if you want a higher ceiling." << std::endl;
					return -1;
				}
				capacity = nextCapacity;
				needsRetry = true;
				++retryCount;
			}
		} while (needsRetry && retryCount < 8);

			if (needsRetry) {
				std::cerr << "Error: exact-intersection buffer did not converge after retries. Last num_rendered="
						  << num_rendered << " capacity=" << capacity << std::endl;
				return -1;
			}
			if (resolvedCapacity != nullptr) {
				*resolvedCapacity = capacity;
			}
		}
	else {
		if (activeGaussians != nullptr) {
			*activeGaussians = -1;
		}
		char* geom_size = nullptr;
		GeometryState::fromChunk(geom_size, P, needsCov3DScratch);
		size_t chunk_size = reinterpret_cast<size_t>(geom_size) + 128;
		char* chunkptr = geometryBuffer(chunk_size);
		geomState = GeometryState::fromChunk(chunkptr, P, needsCov3DScratch);

		if (radii == nullptr)
		{
			radii = geomState.internal_radii;
		}
		// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
		CHECK_CUDA(FORWARD::preprocess(
			P, D, M,
			means3D,
			(glm::vec3*)scales,
			scale_modifier,
			(glm::vec4*)rotations,
			opacities,
			opacitiesHalf,
			shs,
			shsHalf,
			geomState.clamped,
			cov3D_precomp,
			cov3DHalf,
			colors_precomp,
			viewmatrixGlm, projmatrixGlm,
			cpuCamPosGlm,
			sourceIndices,
			instanceIndices,
			instanceCamPositions,
			instanceViewMatrices,
			instanceProjMatrices,
			instanceDepthScales,
			width, height,
			focal_x, focal_y,
			tan_fovx, tan_fovy,
			isOrtho, isFisheye, k1, k2, k3, k4,
			radii,
			geomState.means2D,
			geomState.cov3D,
			geomState.rgb_depth,
			geomState.conic_opacity,
			tile_grid,
			geomState.tiles_touched,
			prefiltered,
			antialiasing
		), debug)

		// Compute prefix sum over full list of touched tile counts by Gaussians
		// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
		CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

		// Retrieve total number of Gaussian instances to launch and resize aux buffers
		CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
		if ((frameCapacity > 0 && num_rendered > frameCapacity) || num_rendered < 0) {
			std::cerr << "Error: num_rendered (" << num_rendered << ") exceeds frameCapacity (" << frameCapacity << "). Please increase maxNumRenderedGaussians." << std::endl;
			return -1;
		}

		if (frameCapacity < 0) {
			// std::cout << "Warning: frameCapacity is negative, using num_rendered for allocation." << std::endl;
			size_t binning_chunk_size = required<BinningState>(num_rendered);
			char* binning_chunkptr = binningBuffer(binning_chunk_size);
			binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);
		}
		if (resolvedCapacity != nullptr) {
			*resolvedCapacity = frameCapacity > 0 ? frameCapacity : num_rendered;
		}


		// For each instance to be rendered, produce adequate [ tile | depth ] key 
		// and corresponding dublicated Gaussian indices to be sorted
		duplicateWithKeys << <(P + 255) / 256, 256 >> > (
			P,
			geomState.means2D,
			geomState.rgb_depth,
			geomState.point_offsets,
			binningState.point_list_keys_unsorted,
			binningState.point_list_unsorted,
			radii,
			tile_grid)
		CHECK_CUDA(, debug)
	}

#ifdef ENABLE_CUDA_TIMING
	cudaEventRecord(preprocessEndEvent);
#endif

#ifdef DEBUG
	saveCudaArrayToFile((float*)geomState.means2D, P * 2, "./output/means2d.bin");
	saveCudaArrayToFile((float*)geomState.rgb_depth, P * 4, "./output/rgb_depths.bin");
	saveCudaArrayToFile((float*)geomState.conic_opacity, P * 4, "./output/conic_opacity.bin");
	saveCudaArrayToFile((uint64_t*)binningState.point_list_keys_unsorted, num_rendered, "./output/point_list_keys_unsorted.bin");
	saveCudaArrayToFile((uint32_t*)binningState.point_list_unsorted, num_rendered, "./output/point_list_unsorted.bin");
#endif

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

#ifdef ENABLE_CUDA_TIMING
	cudaEventRecord(sortEndEvent);
#endif

#ifdef DEBUG
	saveCudaArrayToFile((uint64_t**)binningState.point_list_keys, num_rendered, "./output/point_list_keys.bin");
	saveCudaArrayToFile((uint32_t*)binningState.point_list, num_rendered, "./output/point_list.bin");
	saveCudaArrayToFile((uint*)imgState.ranges, tile_grid.x * tile_grid.y * 2, "./output/ranges.bin");
#endif

	// Let each tile blend its range of Gaussians independently in parallel
	const float4* feature_ptr = colors_precomp != nullptr ? (const float4*)colors_precomp : geomState.rgb_depth;
	float3 bg_color = make_float3(cpuBackground[0], cpuBackground[1], cpuBackground[2]);
	if (usePrefetchingPipeline) {
		CHECK_CUDA(flashgs::render_16x16(num_rendered, width, height,
			(float2*)geomState.means2D, (float4*)feature_ptr, (float4*)geomState.conic_opacity,
			(uint64_t*)binningState.point_list_keys, (uint32_t*)binningState.point_list,
			(int2*)imgState.ranges, bg_color, (float4*)out_color, depth), debug)
	}
	else if (useTensorCore) {
		CHECK_CUDA(TCGS::renderCUDA_Forward(
			tile_grid, block,
			imgState.ranges,
			binningState.point_list,
			width, height, P,
			geomState.means2D,
			feature_ptr,
			geomState.conic_opacity,
			imgState.accum_alpha,
			imgState.n_contrib,
			bg_color,
			out_color,
			depth), debug)

	}
	else {
		CHECK_CUDA(FORWARD::render(
			tile_grid, block,
			imgState.ranges,
			binningState.point_list,
			width, height,
			geomState.means2D,
			feature_ptr,
			geomState.conic_opacity,
			imgState.accum_alpha,
			imgState.n_contrib,
			bg_color,
			out_color,
			depth), debug)
	}

#ifdef ENABLE_CUDA_TIMING
	cudaEventRecord(renderEndEvent);
	cudaEventSynchronize(renderEndEvent);

	cudaEventElapsedTime(&preprocessTimeMs, frameStartEvent, preprocessEndEvent);
	cudaEventElapsedTime(&sortTimeMs, preprocessEndEvent, sortEndEvent);
	cudaEventElapsedTime(&renderTimeMs, sortEndEvent, renderEndEvent);
	cudaEventElapsedTime(&totalFrameTimeMs, frameStartEvent, renderEndEvent);
	printf("Preprocess: %.3f ms\n", preprocessTimeMs);
	printf("Sort:       %.3f ms\n", sortTimeMs);
	printf("Render:     %.3f ms\n", renderTimeMs);
	printf("Total:      %.3f ms\n", totalFrameTimeMs);
	cudaEventDestroy(frameStartEvent);
	cudaEventDestroy(preprocessEndEvent);
	cudaEventDestroy(sortEndEvent);
	cudaEventDestroy(renderEndEvent);
#endif

	return num_rendered;
}
