#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#


cmake_minimum_required(VERSION 3.20)

project(DiffRast LANGUAGES C CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_DEPENDS_USE_COMPILER TRUE)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

if(WIN32)
    set(CMAKE_GENERATOR_TOOLSET "v143,version=14.36")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

set(USE_CUDA_GDB OFF)
if (USE_CUDA_GDB)
	if(NOT CMAKE_BUILD_TYPE)
		set(CMAKE_BUILD_TYPE Debug)
	endif()
	set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
	set(CMAKE_CXX_FLAGS_DEBUG "-g")
endif()


# --- 1. 引入 GLFW ---
# 告诉 GLFW 不要编译它的测试程序
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/glfw-3.4)


# --- 2. 引入 ImGui ---
# 定义 ImGui 的源码路径
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui-1.92.6)
# 显式指定后端目录
set(IMGUI_BACKEND_DIR ${IMGUI_DIR}/backends)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    # 注意这里路径的变化：增加了 /backends/
    ${IMGUI_BACKEND_DIR}/imgui_impl_glfw.cpp
    ${IMGUI_BACKEND_DIR}/imgui_impl_opengl3.cpp
)

# 创建一个专门用于 ImGui 的静态库目标
add_library(imgui STATIC ${IMGUI_SOURCES})
# 指定 ImGui 编译时需要的头文件
target_include_directories(imgui PUBLIC 
    ${IMGUI_DIR} 
    ${IMGUI_BACKEND_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw-3.4/include
)

# --- 3. 引入 GLAD ---
set(GLAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glad)
# 创建 GLAD 静态库目标
add_library(glad STATIC ${GLAD_DIR}/src/glad.c)
target_include_directories(glad PUBLIC ${GLAD_DIR}/include)



set(GS_OBJECTS "")
set(GS_INCLUDES "")
add_subdirectory(src/diff-gaussian-rasterization)
list(APPEND GS_OBJECTS $<TARGET_OBJECTS:gaussian_splatting>)
list(APPEND GS_INCLUDES $<TARGET_PROPERTY:gaussian_splatting,INTERFACE_INCLUDE_DIRECTORIES>)

add_library(optisplat STATIC
    src/camera.cc
	src/render.cc
	src/utils.cc
	${GS_OBJECTS}
)

target_include_directories(optisplat PRIVATE 
	include
	src
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
	${CMAKE_CURRENT_LIST_DIR}/third_party
	${CMAKE_CURRENT_LIST_DIR}/third_party/glm
	${GS_INCLUDES}
)

target_link_libraries(optisplat PRIVATE 
    OpenMP::OpenMP_CXX 
    CUDA::cudart
)

set_target_properties(optisplat PROPERTIES 
    CUDA_ARCHITECTURES "75;86;89"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(optisplat_test
	src/main.cc
	src/viewer.cc
)

target_include_directories(optisplat_test PRIVATE 
	include
	src
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	${CMAKE_CURRENT_LIST_DIR}/third_party
	${CMAKE_CURRENT_LIST_DIR}/third_party/glm
	${GLAD_DIR}/include
)

target_link_libraries(optisplat_test
	PRIVATE 
	optisplat 
	imgui 
    glfw    # 这里直接写 glfw 名字，CMake 会自动寻找 add_subdirectory 定义的目标
	glad
    GL      # Linux 下需要显式链接 OpenGL 库
    dl      # Linux 下需要
    pthread # Linux 下需要
)





#### cmake ..
#### cmake --build . --config RelWithDebInfo
#### cmake --install . --config RelWithDebInfo
#### cmake --build . --config RelWithDebInfo --clean-first
#### cmake -G "Visual Studio 17 2022" -A x64 ..