#pragma once


#include "camera.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <chrono>

#include "cuda.h"
#include "cuda_runtime.h"

namespace optisplat {

struct ResourceUsageStats {
    float processCpuPercent = 0.0f;
    float systemCpuPercent = 0.0f;
    float processMemoryMB = 0.0f;
    float gpuUsedMB = 0.0f;
    float gpuTotalMB = 0.0f;
    float gpuUsedPercent = 0.0f;
    bool cpuAvailable = false;
    bool processMemoryAvailable = false;
    bool gpuAvailable = false;
};

struct MemoryFootprint {
    float processMemoryMB = 0.0f;
    float gpuUsedMB = 0.0f;
    float gpuTotalMB = 0.0f;
    bool processMemoryAvailable = false;
    bool gpuAvailable = false;
};

class ResourceUsageSampler {
public:
    ResourceUsageSampler();

    const ResourceUsageStats& update();

private:
    void updateCpuStats();
    void updateGpuStats();

    ResourceUsageStats stats_{};
    bool cpuAvailable_ = false;
    uint64_t lastSystemTotalTicks_ = 0;
    uint64_t lastSystemIdleTicks_ = 0;
    uint64_t lastProcessTicks_ = 0;
    uint32_t lastNumProcessors_ = 1;
    std::chrono::steady_clock::time_point lastWallTime_{};
    std::chrono::steady_clock::time_point lastUpdateTime_{};
};

class Utils {
public:
    static int64_t nowUs() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);
        return microseconds.count();
    }

    static float nowGPUMB() {
        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        float usedMem = (float)(totalMem - freeMem) / (1024.0 * 1024.0);
        return usedMem;
    }

    static bool readGpuMemoryMB(float& gpuUsedMB, float& gpuTotalMB);

    static bool readProcessMemoryMB(float& processMemoryMB);

    static MemoryFootprint sampleMemoryFootprint();

    static void logMemoryFootprint(const std::string& label, const MemoryFootprint& footprint);

    static void logMemoryFootprintDelta(const std::string& label, const MemoryFootprint& start, const MemoryFootprint& end);

    template<typename T>
    static bool saveCpuArrayToBin(const std::string& fileName, const T* data, size_t count) {
        std::ofstream out(fileName, std::ios::binary);
        if (!out) return false;

        out.write(reinterpret_cast<const char*>(data), count * sizeof(T));
        std::cout << "Cpu raw data has been saved to " << fileName << std::endl;

        return out.good();
    }

    template<typename T>
    static bool saveCudaArrayToBin(const std::string& fileName, const T* deviceData, size_t count) {
        std::vector<T> hostBuffer(count);

        cudaError_t err = cudaMemcpy(hostBuffer.data(), deviceData, count * sizeof(T), cudaMemcpyDeviceToHost);

        if (err != cudaSuccess)
            return false;

        std::ofstream out(fileName, std::ios::binary);
        if (!out) return false;

        out.write(reinterpret_cast<const char*>(hostBuffer.data()), count * sizeof(T));
        std::cout << "Cuda raw data has been saved to " << fileName << std::endl;

        return out.good();

    }

    static std::vector<GsCamera> readCamerasFromJson(std::string filePath);

    static GaussianSceneDesc readGaussianSceneFromJson(const std::string& filePath);
    
    static void saveCamerasToJson(const std::vector<GsCamera>& cameras, const std::string& filePath);

    static void saveImages(const float* cudaData, int N, int H, int W, std::vector<std::string> filenames);

    static void saveAllMaps(const float* cudaData, int N, int H, int W, std::vector<std::string> filenames, bool saveRawData = false);

    static std::string absolutePath(std::string path);

    static std::vector<std::string> absolutePaths(std::vector<std::string> paths);
};


#if ENABLE_SCOPE_TIMER
class ScopeTimer {
public:
    ScopeTimer(const std::string& name = "")
        : name_(name) {
        cudaDeviceSynchronize();
        start_ = std::chrono::high_resolution_clock::now();
    }
    ~ScopeTimer() {
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        std::cout << name_ << " elapsed: " << ms << " ms" << std::endl;
    }
private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};
#else
class ScopeTimer {
public:
    ScopeTimer(const std::string& name = "") {}
};
#endif


class ProgressBar {
public:
    ProgressBar(int total, const std::string& desc = "")
        : total_(total), desc_(desc), current_(0) {}

    void update(int step = 1, const std::string& frameDesc = "") {
        current_ += step;
        display(frameDesc);
    }

    void setDesc(const std::string& desc) {
        desc_ = desc;
    }

    void display(const std::string& frameDesc) const {
        int width = 50;  // Width of the progress bar
        int pos = width * current_ / total_;
        std::cout << "\r" << desc_ << " [";
        for (int i = 0; i < width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(current_ * 100.0 / total_) << "% " << frameDesc;
        std::cout.flush();
    }

    void close() const {
        std::cout << std::endl;
    }

    void show_progress(int done, float time, int num_rendered) {
        int barWidth = 50;
        std::cout << "\r" << desc_ << " [";
        float progress = (float)done / (float)total_;
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " % (" << done << "/" << total_ << ", num_rendered = " << num_rendered << ", Time = " << time << " ms)\r";
        std::cout.flush();
    }

private:
    int total_;
    std::string desc_;
    int current_;
};


}
