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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
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
			int* resolvedCapacity = nullptr,
			int* activeGaussians = nullptr,
			int* radii = nullptr,
			bool debug = false);

	};
};

#endif
