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


namespace optisplat {
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