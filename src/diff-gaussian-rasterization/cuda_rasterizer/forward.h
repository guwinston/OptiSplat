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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
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
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		bool is_ortho, bool is_fisheye, float k1, float k2, float k3, float k4, 
		int* radii,
		float2* points_xy_image,
		float* cov3Ds,
		float4* colors_depths,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool antialiasing);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float4* features_depths,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float3 bg_color,
		float* out_color,
		float* depth);
}


#endif
