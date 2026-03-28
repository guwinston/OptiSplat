#pragma once

#include "camera.h"
#include "common.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <Eigen/Geometry>

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

struct RenderRuntimeStats {
    int numPoints = 0;
    int allocatedRenderedGaussians = 0;
    int allocatedRenderedGaussiansLimit = -1;
    bool useExactIntersection = false;
};

struct GsConfig {
    std::string modelPath = "";
    std::string cameraPath = "";
    bool bRebuildBinaryCache = false;
    int maxNumRenderedGaussians = 200000000;
    bool bUseFlashGSExactIntersection = false;
    bool bUseFlashGSPrefetchingPipeline = false;
    bool bUseTensorCore = false;
    GsConfig() = default;
};

template<int D>
class GaussianRender;

template<int D>
class SceneData;

template<int D>
struct RichPoint;

// Pos, Rot, Scale, SHs, sigmoid, mortonEncode64 -- see common.h

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

// sigmoid / inverse_sigmoid -- see common.h

void readPlyHeader(std::string filename, int& numVertex, int& shsDegree);


class IGaussianRender {
public:

    virtual float render(GsCamera& inCamera, float*& outImage, float*& outAllMap, bool debug = false) { GS_ERROR("To be implement!"); return -FLT_MAX; }

    virtual void setDefaultCamera(int fov, int width, int height) {}

    virtual RenderRuntimeStats getRuntimeStats() const { return {}; }

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
    int* cudaCurrOffset = nullptr;
    int* cudaExactOverflow = nullptr;
    int allocatedRenderedCapacity = 0;
    int capacityLimit = -1;

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

    RenderRuntimeStats getRuntimeStats() const override;

};

template <int D>
class SceneData {
public:

    void initResource(std::string modelPath, std::string cacheSavePath, bool rebuildBinaryCache);

    void uploadDataToGPU();

    ~SceneData() {
        safeCudaFree(cudaGaussianPoints);
        safeCudaFree(cudaGaussianSHs);
        safeCudaFree(cudaGaussianOpacity);
        safeCudaFree(cudaGaussianScale);
        safeCudaFree(cudaGaussianRot);
        safeCudaFree(cudaGaussianCov3D);
    }

public:
    std::vector<Pos> gaussianPoints;
    std::vector<SHs<D>> gaussianSHs;
    std::vector<float> gaussianOpacity;
    std::vector<Scale> gaussianScale;
    std::vector<Rot> gaussianRot;
    std::vector<float> gaussianCov3D;

    Eigen::Vector3f sceneMin = {FLT_MAX, FLT_MAX, FLT_MAX};
    Eigen::Vector3f sceneMax = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    std::string cacheName = "gaussian_splatting.sog";

    int numPoints = 0;
    float*  cudaGaussianPoints = nullptr;
    float*  cudaGaussianSHs = nullptr;
    float*  cudaGaussianOpacity = nullptr;
    float*  cudaGaussianScale = nullptr;
    float*  cudaGaussianRot = nullptr;
    float*  cudaGaussianCov3D = nullptr;
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
