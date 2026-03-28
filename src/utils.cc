#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stbi_image_write.h" // Single header file image library in the third-party library of original 3DGS 

#include "utils.h"
#include "camera.h"
#include "render.h"
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <map>
#include <unordered_map>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <sstream>

#if defined(__linux__)
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif


namespace optisplat {
namespace {

bool readSystemCpuTicks(uint64_t& totalTicks, uint64_t& idleTicks) {
#if defined(__linux__)
    std::ifstream statFile("/proc/stat");
    if (!statFile.is_open()) return false;

    std::string cpuLabel;
    uint64_t user = 0, nice = 0, system = 0, idle = 0, iowait = 0;
    uint64_t irq = 0, softirq = 0, steal = 0, guest = 0, guestNice = 0;
    statFile >> cpuLabel >> user >> nice >> system >> idle >> iowait
             >> irq >> softirq >> steal >> guest >> guestNice;
    if (cpuLabel != "cpu") return false;

    idleTicks = idle + iowait;
    totalTicks = user + nice + system + idle + iowait + irq + softirq + steal + guest + guestNice;
    return true;
#elif defined(_WIN32)
    FILETIME idleTime;
    FILETIME kernelTime;
    FILETIME userTime;
    if (!GetSystemTimes(&idleTime, &kernelTime, &userTime)) return false;

    ULARGE_INTEGER idleValue, kernelValue, userValue;
    idleValue.LowPart = idleTime.dwLowDateTime;
    idleValue.HighPart = idleTime.dwHighDateTime;
    kernelValue.LowPart = kernelTime.dwLowDateTime;
    kernelValue.HighPart = kernelTime.dwHighDateTime;
    userValue.LowPart = userTime.dwLowDateTime;
    userValue.HighPart = userTime.dwHighDateTime;

    idleTicks = idleValue.QuadPart;
    totalTicks = kernelValue.QuadPart + userValue.QuadPart;
    return true;
#else
    (void)totalTicks;
    (void)idleTicks;
    return false;
#endif
}

bool readProcessCpuTicks(uint64_t& processTicks) {
#if defined(__linux__)
    std::ifstream statFile("/proc/self/stat");
    if (!statFile.is_open()) return false;

    std::string line;
    std::getline(statFile, line);
    if (line.empty()) return false;

    const size_t rparen = line.rfind(')');
    if (rparen == std::string::npos || rparen + 2 >= line.size()) return false;

    std::istringstream iss(line.substr(rparen + 2));
    std::string token;
    uint64_t utime = 0;
    uint64_t stime = 0;
    for (int field = 3; iss >> token; ++field) {
        if (field == 14) utime = std::stoull(token);
        else if (field == 15) {
            stime = std::stoull(token);
            break;
        }
    }

    processTicks = utime + stime;
    return true;
#elif defined(_WIN32)
    FILETIME creationTime;
    FILETIME exitTime;
    FILETIME kernelTime;
    FILETIME userTime;
    if (!GetProcessTimes(GetCurrentProcess(), &creationTime, &exitTime, &kernelTime, &userTime)) return false;

    ULARGE_INTEGER kernelValue, userValue;
    kernelValue.LowPart = kernelTime.dwLowDateTime;
    kernelValue.HighPart = kernelTime.dwHighDateTime;
    userValue.LowPart = userTime.dwLowDateTime;
    userValue.HighPart = userTime.dwHighDateTime;

    processTicks = kernelValue.QuadPart + userValue.QuadPart;
    return true;
#else
    (void)processTicks;
    return false;
#endif
}

uint32_t getProcessorCount() {
#if defined(_WIN32)
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return std::max<DWORD>(1, sysInfo.dwNumberOfProcessors);
#else
    return 1;
#endif
}

} // namespace

bool Utils::readGpuMemoryMB(float& gpuUsedMB, float& gpuTotalMB) {
    size_t freeMem = 0;
    size_t totalMem = 0;
    const cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess || totalMem == 0) {
        cudaGetLastError();
        return false;
    }

    const size_t usedMem = totalMem - freeMem;
    gpuUsedMB = static_cast<float>(usedMem) / (1024.0f * 1024.0f);
    gpuTotalMB = static_cast<float>(totalMem) / (1024.0f * 1024.0f);
    return true;
}

bool Utils::readProcessMemoryMB(float& processMemoryMB) {
#if defined(__linux__)
    std::ifstream statmFile("/proc/self/statm");
    if (!statmFile.is_open()) return false;

    long totalPages = 0;
    long residentPages = 0;
    statmFile >> totalPages >> residentPages;
    if (residentPages <= 0) return false;

    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0) return false;

    processMemoryMB = static_cast<float>(residentPages) * static_cast<float>(pageSize) /
                      (1024.0f * 1024.0f);
    return true;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS memCounters;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &memCounters, sizeof(memCounters))) return false;

    processMemoryMB = static_cast<float>(memCounters.WorkingSetSize) / (1024.0f * 1024.0f);
    return true;
#else
    (void)processMemoryMB;
    return false;
#endif
}

MemoryFootprint Utils::sampleMemoryFootprint() {
    MemoryFootprint footprint;
    footprint.processMemoryAvailable = readProcessMemoryMB(footprint.processMemoryMB);
    footprint.gpuAvailable = readGpuMemoryMB(footprint.gpuUsedMB, footprint.gpuTotalMB);
    return footprint;
}

void Utils::logMemoryFootprint(const std::string& label, const MemoryFootprint& footprint) {
    if (footprint.processMemoryAvailable) {
        GS_INFO("%s process memory: %.1f MB", label.c_str(), footprint.processMemoryMB);
    } else {
        GS_INFO("%s process memory: unavailable", label.c_str());
    }

    if (footprint.gpuAvailable) {
        GS_INFO("%s GPU memory: %.1f / %.1f MB", label.c_str(), footprint.gpuUsedMB, footprint.gpuTotalMB);
    } else {
        GS_INFO("%s GPU memory: unavailable", label.c_str());
    }
}

void Utils::logMemoryFootprintDelta(const std::string& label, const MemoryFootprint& start, const MemoryFootprint& end) {
    if (start.processMemoryAvailable && end.processMemoryAvailable) {
        GS_INFO("%s process memory delta: start=%.1f MB end=%.1f MB delta=%+.1f MB",
                label.c_str(), start.processMemoryMB, end.processMemoryMB,
                end.processMemoryMB - start.processMemoryMB);
    } else {
        GS_INFO("%s process memory delta: unavailable", label.c_str());
    }

    if (start.gpuAvailable && end.gpuAvailable) {
        GS_INFO("%s GPU memory delta: start=%.1f MB end=%.1f MB delta=%+.1f MB (total %.1f MB)",
                label.c_str(), start.gpuUsedMB, end.gpuUsedMB,
                end.gpuUsedMB - start.gpuUsedMB, end.gpuTotalMB);
    } else {
        GS_INFO("%s GPU memory delta: unavailable", label.c_str());
    }
}

ResourceUsageSampler::ResourceUsageSampler() {
    lastNumProcessors_ = getProcessorCount();
    const bool haveSystem = readSystemCpuTicks(lastSystemTotalTicks_, lastSystemIdleTicks_);
    const bool haveProcess = readProcessCpuTicks(lastProcessTicks_);
    lastWallTime_ = std::chrono::steady_clock::now();
    cpuAvailable_ = haveSystem && haveProcess;
    updateGpuStats();
}

const ResourceUsageStats& ResourceUsageSampler::update() {
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<float>(now - lastUpdateTime_).count();
    if (elapsed < 0.5f) return stats_;

    lastUpdateTime_ = now;
    updateCpuStats();
    updateGpuStats();
    return stats_;
}

void ResourceUsageSampler::updateCpuStats() {
    if (!cpuAvailable_) {
        stats_.cpuAvailable = false;
        return;
    }

    uint64_t currentTotalTicks = 0;
    uint64_t currentIdleTicks = 0;
    uint64_t currentProcessTicks = 0;
    if (!readSystemCpuTicks(currentTotalTicks, currentIdleTicks) ||
        !readProcessCpuTicks(currentProcessTicks)) {
        stats_.cpuAvailable = false;
        cpuAvailable_ = false;
        return;
    }

    const uint64_t totalDelta = currentTotalTicks - lastSystemTotalTicks_;
    const uint64_t idleDelta = currentIdleTicks - lastSystemIdleTicks_;
    const uint64_t processDelta = currentProcessTicks - lastProcessTicks_;

    if (totalDelta > 0) {
        stats_.systemCpuPercent = 100.0f * (1.0f - static_cast<float>(idleDelta) /
            static_cast<float>(totalDelta));

#if defined(__linux__)
        const auto now = std::chrono::steady_clock::now();
        const float wallSeconds = std::chrono::duration<float>(now - lastWallTime_).count();
        const long ticksPerSecond = sysconf(_SC_CLK_TCK);
        if (wallSeconds > 0.0f && ticksPerSecond > 0) {
            stats_.processCpuPercent = 100.0f *
                (static_cast<float>(processDelta) / static_cast<float>(ticksPerSecond)) / wallSeconds;
        }
        lastWallTime_ = now;
#elif defined(_WIN32)
        stats_.processCpuPercent = 100.0f *
            (static_cast<float>(processDelta) / static_cast<float>(totalDelta)) *
            static_cast<float>(lastNumProcessors_);
#else
        stats_.processCpuPercent = 0.0f;
#endif
    }

    stats_.processMemoryAvailable = Utils::readProcessMemoryMB(stats_.processMemoryMB);
    stats_.cpuAvailable = true;
    lastSystemTotalTicks_ = currentTotalTicks;
    lastSystemIdleTicks_ = currentIdleTicks;
    lastProcessTicks_ = currentProcessTicks;
}

void ResourceUsageSampler::updateGpuStats() {
    if (!Utils::readGpuMemoryMB(stats_.gpuUsedMB, stats_.gpuTotalMB)) {
        stats_.gpuAvailable = false;
        return;
    }

    stats_.gpuUsedPercent = 100.0f * stats_.gpuUsedMB / stats_.gpuTotalMB;
    stats_.gpuAvailable = true;
}

std::vector<GsCamera> Utils::readCamerasFromJson(std::string filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        GS_ERROR("Failed to open file %s", filePath.c_str());
        return {};
    }

    nlohmann::json data;
    try {
        file >> data;
    } catch (nlohmann::json::parse_error& e) {
        GS_ERROR("Parse error: %s", e.what());
        return {};
    }

    std::vector<GsCamera> cameras;
    for (auto& item : data) {
        GsCamera cam;
        cam.model = stringToCameraModel("PINHOLE");
        cam.coordSystem = CameraCoordSystem::COLMAP;
        
        if (item.contains("model")) {
            cam.model = stringToCameraModel(item["model"]);
        }
        if (item.contains("coord_system")) {
            cam.coordSystem = stringToCameraCoordSystem(item["coord_system"]);
        }
        cam.cameraId = item["id"];
        cam.imageName = item["img_name"];
        cam.height = item["height"];
        cam.width  = item["width"];
        cam.fx = item["fx"];
        cam.fy = item["fy"];
        
        cam.cx = item["width"].get<double>() / 2.0;
        cam.cy = item["height"].get<double>() / 2.0;
        if (item.contains("cx") && item.contains("cy")) {
            cam.cx = item["cx"];
            cam.cy = item["cy"];
        }

        cam.position.x() = item["position"][0];
        cam.position.y() = item["position"][1];
        cam.position.z() = item["position"][2];

        if (item.contains("k1") && item.contains("k2") && item.contains("k3") && item.contains("k4")) {
            cam.k1 = item["k1"];
            cam.k2 = item["k2"];
            cam.k3 = item["k3"];
            cam.k4 = item["k4"];
        }
        
        // 旋转处理
        Eigen::Matrix3d rotationMatrix;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                rotationMatrix(i, j) = item["rotation"][i][j];

        cam.quaternion = Eigen::Quaterniond(rotationMatrix).cast<float>();

        cameras.push_back(cam);
    }
    GS_INFO("Read %d cameras from %s", (int)cameras.size(), filePath.c_str());
    return cameras;
}

void Utils::saveCamerasToJson(const std::vector<GsCamera>& cameras, const std::string& filePath) {
    nlohmann::json data = nlohmann::json::array();

    for (const auto& cam : cameras) {
        nlohmann::json item;
        item["model"] = cameraModelToString(cam.model);
        item["coord_system"] = cameraCoordSystemToString(cam.coordSystem);
        item["id"] = cam.cameraId;
        item["img_name"] = cam.imageName;
        item["height"] = cam.height;
        item["width"] = cam.width;
        item["fx"] = cam.fx;
        item["fy"] = cam.fy;
        item["cx"] = cam.cx;
        item["cy"] = cam.cy;
        item["position"] = { cam.position.x(), cam.position.y(), cam.position.z() };
        
        item["k1"] = cam.k1;
        item["k2"] = cam.k2;
        item["k3"] = cam.k3;
        item["k4"] = cam.k4;

        Eigen::Matrix3f rotationMatrix = cam.quaternion.toRotationMatrix();

        item["rotation"] = {
            { rotationMatrix(0,0), rotationMatrix(0,1), rotationMatrix(0,2) }, // Row 0
            { rotationMatrix(1,0), rotationMatrix(1,1), rotationMatrix(1,2) }, // Row 1
            { rotationMatrix(2,0), rotationMatrix(2,1), rotationMatrix(2,2) }  // Row 2
        };

        data.push_back(item);
    }

    std::ofstream file(filePath);
    if (!file.is_open()) {
        GS_ERROR("Failed to open file %s for writing", filePath.c_str());
        return;
    }
    file << data.dump(4);
    GS_INFO("Saved %d cameras to %s", (int)cameras.size(), filePath.c_str());
}


void Utils::saveImages(const float* cudaData, int N, int H, int W, std::vector<std::string> filenames) {
    // cudaData must be in NHWC format
    const int NUM_IMAGE_CHANNELS = 4; // RGBA
    size_t imageSize = N * NUM_IMAGE_CHANNELS * H * W;
    std::vector<float> hostData(imageSize);
    cudaMemcpy(hostData.data(), cudaData, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Iterate through each image and save as PNG
    for (int n = 0; n < N; ++n) {
		std::string imageFilename = filenames[n];
		std::filesystem::path filePath(filenames[n]);
		std::string extension = filePath.extension().string();
        if (!std::filesystem::exists(filePath.parent_path().string())) {
            std::filesystem::create_directories(filePath.parent_path().string());
        }

        const float* imageDataStart = hostData.data() + n * NUM_IMAGE_CHANNELS * H * W;

        // Create an array to hold the image data in unsigned char format for stb_image_write
        std::vector<unsigned char> imageUC(H * W * NUM_IMAGE_CHANNELS);
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < NUM_IMAGE_CHANNELS; ++c) {
                    int srcIdx = h * W * NUM_IMAGE_CHANNELS + w * NUM_IMAGE_CHANNELS + c;
                    int dstIdx = h * W * NUM_IMAGE_CHANNELS + w * NUM_IMAGE_CHANNELS + c;
                    imageUC[dstIdx] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, imageDataStart[srcIdx] * 255.0f)));
                }
            }
        }

		// Write the image in the specified format
        if (std::string(extension) == ".png") {
            if (!stbi_write_png(imageFilename.c_str(), W, H, NUM_IMAGE_CHANNELS, static_cast<const void*>(imageUC.data()), W * NUM_IMAGE_CHANNELS)) {
                GS_ERROR("Failed to write PNG image to %s", imageFilename.c_str());
            }
        } else if (std::string(extension) == ".jpg" || std::string(extension) == ".jpeg" || std::string(extension) == ".JPG") {
            if (!stbi_write_jpg(imageFilename.c_str(), W, H, NUM_IMAGE_CHANNELS, static_cast<const void*>(imageUC.data()), 75)) { 
                GS_ERROR("Failed to write JPEG image to %s", imageFilename.c_str());
            }
        } else {
            GS_ERROR("Unsupported format: %s", extension.c_str());
        }
    }
}


void Utils::saveAllMaps(const float* cudaData, int N, int H, int W, std::vector<std::string> filenames, bool saveRawData) {
    const int NUM_ALLMAP_CHANNELS = 1; // Only Depth (1)
    size_t imageSize = static_cast<size_t>(N) * NUM_ALLMAP_CHANNELS * H * W;
    std::vector<float> hostData(imageSize);
    
    cudaMemcpy(hostData.data(), cudaData, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

    if (saveRawData && !filenames.empty()) {
        std::filesystem::path filePath(filenames[0]);
        std::string filename = filePath.parent_path().string() + "/" + filePath.stem().string() + ".bin";
        std::ofstream file(filename, std::ios::binary);
        file.write(reinterpret_cast<const char*>(hostData.data()), hostData.size() * sizeof(float));
        file.close();
    }

    for (int n = 0; n < N; ++n) {   
        std::filesystem::path filePath(filenames[n]);
        std::string folderPath = filePath.parent_path().string();
        std::string extension = filePath.extension().string();
        std::string name = filePath.stem().string();
        std::string depthFilename = folderPath + "/" + name + "_depth" + extension;

        if (!std::filesystem::exists(folderPath)) {
            std::filesystem::create_directories(folderPath);
        }

        const float* imageDataStart = hostData.data() + n * H * W;

        float depthMin = std::numeric_limits<float>::max();
        float depthMax = std::numeric_limits<float>::lowest();
        for (int i = 0; i < H * W; ++i) {
            float depthValue = imageDataStart[i];
            if (std::isfinite(depthValue)) {
                depthMin = std::min(depthMin, depthValue);
                depthMax = std::max(depthMax, depthValue);
            }
        }

        if (depthMax <= depthMin) depthMax = depthMin + 1.0f;

        std::vector<unsigned char> depthUC(H * W);
        for (int i = 0; i < H * W; ++i) {
            float val = imageDataStart[i];
            float normalized = (val - depthMin) / (depthMax - depthMin);
            depthUC[i] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, normalized * 255.0f)));
        }

        if (extension == ".png") {
            if (!stbi_write_png(depthFilename.c_str(), W, H, 1, depthUC.data(), W)) {
                GS_ERROR("Failed to write PNG depth image to %s", depthFilename.c_str());
            }
        } else if (extension == ".jpg" || extension == ".jpeg" || extension == ".JPG") {
            if (!stbi_write_jpg(depthFilename.c_str(), W, H, 1, depthUC.data(), 100)) {
                GS_ERROR("Failed to write JPEG depth image to %s", depthFilename.c_str());
            }
        } else {
            GS_ERROR("Unsupported format: %s", extension.c_str());
        }
    }
}

std::string Utils::absolutePath(std::string path) {
    std::string outPath = path;
    std::filesystem::path inputPath = path;
    if (inputPath.is_relative()) {
        std::filesystem::path absPath = std::filesystem::absolute(inputPath);
        outPath = absPath.string();
    }
    return outPath;
}

std::vector<std::string> Utils::absolutePaths(std::vector<std::string> paths) {
    std::vector<std::string> outPaths;
    for (int i = 0; i< paths.size(); i++) {
        std::string inputPath = paths[i];
        outPaths.push_back(absolutePath(inputPath));
    }
    return outPaths;
}

}
