#pragma once

#include "camera.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <glm/glm.hpp>

#include <math.h>
#include <cstdio>
#include <cfloat>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>
#include <array>


#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#define GS_INFO(fmt, ...)    printf("\033[32m[INFO] " fmt "\033[0m\n", ##__VA_ARGS__)
#define GS_WARNING(fmt, ...) printf("\033[33m[WARNING] " fmt "\033[0m\n", ##__VA_ARGS__)
#define GS_ERROR(fmt, ...)   printf("\033[31m[ERROR] %s:%d:%s(): " fmt "\033[0m\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__)


namespace optisplat {

struct GsConfig {
    std::string modelPath = "";
    std::string cameraPath = "";
    bool bUseTensorCore = true;
    bool bRebuildBinaryCache = false;
    GsConfig() = default;
};

template<int D>
class GaussianRender;

template<int D>
class SceneData;

template<int D>
struct RichPoint;

using Pos = glm::vec3;
using Rot = glm::vec4;
using Scale = glm::vec3;
template <int D>
using SHs = std::array<float, (D + 1) * (D + 1) * 3>;

inline uint64_t expandBits(uint32_t v) {
    // 将21位整数的每个位插入到64位整数的低、中、高三位，实现三维莫顿编码
    uint64_t x = v & 0x1fffff; // 只取低21位
    x = (x | (x << 32)) & 0x1f00000000ffff;
    x = (x | (x << 16)) & 0x1f0000ff0000ff;
    x = (x | (x << 8))  & 0x100f00f00f00f00f;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3;
    x = (x | (x << 2))  & 0x1249249249249249;
    return x;
}

inline uint64_t mortonEncode64(float x, float y, float z) {
    // 莫顿编码：假设输入归一化到[0,1)，并映射到[0, 2^21-1]
    uint32_t xx = static_cast<uint32_t>(x * 2097152.0f); // 2^21
    uint32_t yy = static_cast<uint32_t>(y * 2097152.0f);
    uint32_t zz = static_cast<uint32_t>(z * 2097152.0f);
    return (expandBits(xx) << 2) | (expandBits(yy) << 1) | (expandBits(zz));
}

template<typename T>
void safeCudaFree(T*& ptr, cudaStream_t stream = 0) {
    if (ptr != nullptr) {
        cudaFreeAsync(ptr, stream);
        ptr = nullptr;
    }
}

template<typename T>
void safeDelete(T*& ptr) {
    if (ptr != nullptr) {
        delete[] ptr;
        ptr = nullptr;
    }
}

template<typename T>
T* cudaMallocAndMemcpy(const T* hPtr, size_t nElem, cudaStream_t stream = 0) {
    T* dPtr = nullptr;
    if (hPtr != nullptr && nElem > 0) {
        cudaMallocAsync(&dPtr, nElem * sizeof(T), stream);
        cudaMemcpyAsync(dPtr, hPtr, nElem * sizeof(T), cudaMemcpyHostToDevice, stream);
    }
    return dPtr;
}

inline float sigmoid(const float m1)
{
	return 1.0f / (1.0f + exp(-m1));
}

inline float inverse_sigmoid(const float m1)
{
	return log(m1 / (1.0f - m1));
}

void readPlyHeader(std::string filename, int& numVertex, int& shsDegree);


class IGaussianRender {
public:

    virtual float render(GsCamera& inCamera, float*& outImage, float*& outAllMap, bool debug = false) { GS_ERROR("To be implement!"); return -FLT_MAX; }

    virtual void setDefaultCamera(int fov, int width, int height) {}

    virtual ~IGaussianRender() = default;
    
    static std::shared_ptr<IGaussianRender> CreateRenderer(GsConfig config);

public:
    GsCamera defaultCamera;
};



template <int D>
class GaussianRender : public IGaussianRender {
public:
    GsConfig config;
    float   sceneMin[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    float   sceneMax[3] = {FLT_MAX, FLT_MAX, FLT_MAX};

    float*  cudaImage = nullptr;
    float*  cudaAllMap = nullptr;
    void*   cudaGeometryState = nullptr;
    void*   cudaBinningState = nullptr;
    void*   cudaImageState = nullptr;
    size_t  allocatedCudaImage = 0;
    size_t  allocatedCudaAllMap = 0;
    size_t  allocatedGeometryState = 0;
    size_t  allocatedBinningState = 0;
    size_t  allocatedImageState = 0;

    float*  cudaBackground = nullptr;
    float*  cudaView = nullptr;
    float*  cudaProj = nullptr;
    float*  cudaCamPos = nullptr;

    int* cudaRadii = nullptr;

    SceneData<D> sceneData;

    cudaStream_t stream = 0;

    GaussianRender(GsConfig config);
    
    ~GaussianRender();


    float render(GsCamera& inCamera, float*& outImage, float*& outAllMap, bool debug = false) override;

    void initCuda(int device = 0);

    void setDefaultCamera(int fov, int width, int height) override;

    void setCudaImageParams(const GsCamera& camera, bool debug = false);

    void setCudaCameraParams(const GsCamera& cameras, bool debug = false);

    void setCudaAuxiliary();

};

template <int D>
class SceneData {
public:

    void initResource(std::string modelPath,  std::string cacheSavePath, bool rebuildBinaryCache);


    int loadPly(
        std::string filename, 
        std::vector<Pos>& pos, 
        std::vector<SHs<D>>& shs, 
        std::vector<float>& opacity,
        std::vector<Scale>& scale,
        std::vector<Rot>& rot,
        glm::vec3& minn,
    	glm::vec3& maxx,
	    glm::vec3& mean);
    
    int loadPlyFlexible(
        std::string filename, 
        std::vector<Pos>& pos, 
        std::vector<SHs<D>>& shs, 
        std::vector<float>& opacity,
        std::vector<Scale>& scale,
        std::vector<Rot>& rot,
        glm::vec3& minn,
    	glm::vec3& maxx,
	    glm::vec3& mean);

    void loadCache(std::string cacheDir);

    void saveCache(std::string cacheDir);

    bool isCacheExist(std::string cacheDir);

    void uploadDataToGPU();

    ~SceneData() {
        safeCudaFree(cudaGaussianPoints);
        safeCudaFree(cudaGaussianSHs);
        safeCudaFree(cudaGaussianOpacity);
        safeCudaFree(cudaGaussianScale);
        safeCudaFree(cudaGaussianRot);
    }

public:
    std::vector<Pos> gaussianPoints;
    std::vector<SHs<D>> gaussianSHs;
    std::vector<float> gaussianOpacity;
    std::vector<Scale> gaussianScale;
    std::vector<Rot> gaussianRot;

    glm::vec3 sceneMin = {FLT_MAX, FLT_MAX, FLT_MAX};
    glm::vec3 sceneMax = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    std::string cacheName = "gaussian_splatting.cache";

    int numPoints = 0;
    float*  cudaGaussianPoints = nullptr;
    float*  cudaGaussianSHs = nullptr;
    float*  cudaGaussianOpacity = nullptr;
    float*  cudaGaussianScale = nullptr;
    float*  cudaGaussianRot = nullptr;
};

template<int D>
struct RichPoint
{
    Pos pos;
    float n[3];
    SHs<D> shs;
    float opacity;
    Scale scale;
    Rot rot;
};


}