#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


cmake_minimum_required(VERSION 3.20)


add_library(gaussian_splatting OBJECT
	${CMAKE_CURRENT_LIST_DIR}/cuda_rasterizer/backward.h
	${CMAKE_CURRENT_LIST_DIR}/cuda_rasterizer/backward.cu
	${CMAKE_CURRENT_LIST_DIR}/cuda_rasterizer/forward.h
	${CMAKE_CURRENT_LIST_DIR}/cuda_rasterizer/forward.cu
	${CMAKE_CURRENT_LIST_DIR}/cuda_rasterizer/auxiliary.h
	${CMAKE_CURRENT_LIST_DIR}/cuda_rasterizer/rasterizer_impl.cu
	${CMAKE_CURRENT_LIST_DIR}/cuda_rasterizer/rasterizer_impl.h
	${CMAKE_CURRENT_LIST_DIR}/cuda_rasterizer/rasterizer.h
)

target_include_directories(gaussian_splatting PUBLIC
	${CMAKE_SOURCE_DIR}/include
	${CMAKE_SOURCE_DIR}/third_party/glm
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set_target_properties(gaussian_splatting PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
	CUDA_ARCHITECTURES "75;86;89"
)