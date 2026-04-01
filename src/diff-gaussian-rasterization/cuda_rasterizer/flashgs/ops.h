#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "glm/glm.hpp"

constexpr int FLASHGS_WARP_SIZE = 32;

#define FLASHGS_CHECK_CUDA(x)                                                                   \
	{                                                                                   \
		cudaError_t status = x;                                                         \
		if (status != cudaSuccess) {                                                    \
			fprintf(stderr, "%s\nline = %d\n", cudaGetErrorString(status), __LINE__);   \
			exit(1);                                                                    \
		}                                                                               \
	}

namespace flashgs {

union cov3d_t
{
    float2 f2[3];
    float s[6];
};

union shs_deg3_t
{
    float4 f4[12];
    glm::vec3 v3[16];
};

void preprocess(int P, int D, int M,
	glm::vec3* positions, glm::vec3* shs, const uint16_t* shsHalf, const float* opacities, const uint16_t* opacitiesHalf, cov3d_t* cov3Ds, const uint16_t* cov3DsHalf,
	int width, int height, int block_x, int block_y,
	const glm::vec3 cam_position, const glm::mat3 cam_rotation, const glm::mat4 view_matrix, const glm::mat4 proj_matrix,
	float focal_x, float focal_y, float zFar, float zNear, float tan_fovx, float tan_fovy, bool is_ortho, bool is_fisheye, float k1, float k2, float k3, float k4,
	const uint32_t* source_indices, const uint32_t* instance_indices,
	const float* instance_cam_positions, const float* instance_view_matrices, const float* instance_proj_matrices, const float* instance_depth_scales,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	int* curr_offset, int max_num_rendered, int* overflowed, const uint32_t* active_indices = nullptr, cudaStream_t stream = 0
);

void markActive(int P,
	glm::vec3* positions, const float* opacities, const uint16_t* opacitiesHalf, cov3d_t* cov3Ds, const uint16_t* cov3DsHalf,
	int width, int height, int block_x, int block_y,
	const glm::vec3 cam_position, const glm::mat3 cam_rotation, const glm::mat4 view_matrix, const glm::mat4 proj_matrix,
	float focal_x, float focal_y, float zFar, float zNear, float tan_fovx, float tan_fovy, bool is_ortho, bool is_fisheye, float k1, float k2, float k3, float k4,
	const uint32_t* source_indices, const uint32_t* instance_indices,
	const float* instance_cam_positions, const float* instance_view_matrices, const float* instance_proj_matrices, const float* instance_depth_scales,
	bool centerOnly,
	uint32_t* active_flags, cudaStream_t stream = 0);

void sort_gaussian(int num_rendered,
	int width, int height, int block_x, int block_y,
	char* list_sorting_space, size_t sorting_size,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted, cudaStream_t stream = 0);

size_t get_sort_buffer_size(int num_rendered, cudaStream_t stream = 0);

void render_16x16(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* ranges, float3 bg_color, float4* out_color, float* inv_depth, cudaStream_t stream = 0);

void render_32x16(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* ranges, float3 bg_color, float4* out_color, float* inv_depth, cudaStream_t stream = 0);

void render_32x32(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* ranges, float3 bg_color, float4* out_color, float* inv_depth, cudaStream_t stream = 0);

} // namespace flashgs
