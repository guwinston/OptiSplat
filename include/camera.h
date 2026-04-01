#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <unordered_map>


namespace optisplat {

enum class CameraCoordSystem { 
    COLMAP, SIBR, UNREAL
};

enum class CameraModel {
    PINHOLE = 0, FISHEYE = 1, ORTHOGRAPHIC = 2
};
inline const char* CameraModelNames[] = { "Pinhole", "Fisheye", "Orthographic" };

inline CameraModel stringToCameraModel(const std::string& str) {
    if (str == "PINHOLE") return CameraModel::PINHOLE;
    if (str == "FISHEYE") return CameraModel::FISHEYE;
    if (str == "ORTHOGRAPHIC") return CameraModel::ORTHOGRAPHIC;
    throw std::invalid_argument("Unknown CameraModel: " + str);
}

inline CameraCoordSystem stringToCameraCoordSystem(const std::string& str) {
    if (str == "COLMAP") return CameraCoordSystem::COLMAP;
    if (str == "SIBR") return CameraCoordSystem::SIBR;
    if (str == "UNREAL") return CameraCoordSystem::UNREAL;
    throw std::invalid_argument("Unknown CameraCoordSystem: " + str);
}

inline std::string cameraModelToString(CameraModel model) {
    switch (model) {
        case CameraModel::PINHOLE:       return "PINHOLE";
        case CameraModel::FISHEYE:       return "FISHEYE";
        case CameraModel::ORTHOGRAPHIC:  return "ORTHOGRAPHIC";
        default: throw std::invalid_argument("Unknown CameraModel enum value");
    }
}

inline std::string cameraCoordSystemToString(CameraCoordSystem coord) {
    switch (coord) {
        case CameraCoordSystem::COLMAP:   return "COLMAP";
        case CameraCoordSystem::SIBR:     return "SIBR";
        case CameraCoordSystem::UNREAL:   return "UNREAL";
        default: throw std::invalid_argument("Unknown CameraCoordSystem enum value");
    }
}


/**
 * @brief Camera information used for constructing extrinsic/intrinsic and rendering images
 */
struct GsCamera {
    CameraModel model = CameraModel::PINHOLE;
    CameraCoordSystem coordSystem = CameraCoordSystem::COLMAP;
    
    // --- Camera Extrinsics
    Eigen::Vector3f position{0.0f, 0.0f, 0.0f};
    Eigen::Quaternionf quaternion{1.0f, 0.0f, 0.0f, 0.0f}; // (w, x, y, z)

    // --- Camera Intrinsics ---
    int width = 1920;
    int height = 1080;
    float fx = -1.0f;
    float fy = -1.0f;
    float cx = 0.5f;
    float cy = 0.5f;
    float znear = 0.01f;
    float zfar  = 1000.0f;
    
    // Distortion coefficients for opencv fisheye model
    float k1 = 0.0f, k2 = 0.0f, k3 = 0.0f, k4 = 0.0f;

    // --- Rendering Information ---
    Eigen::Vector3f bgColor{0.0f, 0.0f, 0.0f}; // background color
    float scale = 1.0f; // scale factor of gaussian ellipsoid
    int cameraId = 0;
    std::string imageName = "";

    Eigen::Matrix4f getPerspectiveMatrix() const;
    Eigen::Matrix4f getOrthographicMatrix() const;
    Eigen::Matrix4f getProjectionMatrix() const;
    Eigen::Matrix4f getWorld2CameraMatrix() const;
    Eigen::Matrix3f getIntrinsicMatrix() const;
    void setResolution(int targetW, int targetH);
    void rescaleResolution(float scaleFactor);
    bool isOrthographic() const { return model == CameraModel::ORTHOGRAPHIC; }
    bool isFisheye() const      { return model == CameraModel::FISHEYE; }
    bool isPerspective() const  { return model == CameraModel::PINHOLE; }
};

struct GaussianInstanceDesc {
    Eigen::Vector3f translation{0.0f, 0.0f, 0.0f};
    Eigen::Quaternionf rotation{1.0f, 0.0f, 0.0f, 0.0f};
    float scale = 1.0f;
    std::string name = "";
};

struct GaussianObjectDesc {
    std::string name = "";
    std::string modelPath = "";
    std::vector<GaussianInstanceDesc> instances;
};

struct GaussianSceneDesc {
    std::vector<GaussianObjectDesc> objects;
};

GsCamera createCamera(Eigen::Vector3f position, Eigen::Quaternionf quaternion, int width, int height, float fov);

float fov2focal(float fovRadian, float pixels);

float focal2fov(float focal, float pixels);

float fovx2fovy(float fovx, float width, float height);

float fovy2fovx(float fovy, float width, float height);

float rad2Deg(float radians);

float deg2Rad(float degrees);

}
