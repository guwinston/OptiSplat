#include "camera.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace optisplat {

void GsCamera::setResolution(int targetW, int targetH) {
    if (targetW <= 0 || targetH <= 0)
        return;

    float scaleX = static_cast<float>(targetW) / static_cast<float>(width);
    float scaleY = static_cast<float>(targetH) / static_cast<float>(height);

    if (fx > 0.0f) fx *= scaleX;
    if (fy > 0.0f) fy *= scaleY;

    cx *= scaleX;
    cy *= scaleY;

    width  = targetW;
    height = targetH;
}

void GsCamera::rescaleResolution(float scaleFactor) {
    if (scaleFactor <= 0.0f)
        return;

    int targetW = static_cast<int>(std::round(width  / scaleFactor));
    int targetH = static_cast<int>(std::round(height / scaleFactor));

    setResolution(targetW, targetH);
}

Eigen::Matrix4f GsCamera::getPerspectiveMatrix() const {
    const float xScale = fx / (width / 2.0f);
    const float yScale = fy / (height / 2.0f);
    const float dx = 2.0f * (cx / width) - 1.0f;
    const float dy = 2.0f * (cy / height) - 1.0f;
    
    float zsign = 1.0f;
    float zf = zfar;
    float zn = znear;

    Eigen::Matrix4f m;
    m << xScale, 0.0f,   dx,                     0.0f,                      // Row 0
         0.0f,   yScale, dy,                     0.0f,                      // Row 1
         0.0f,   0.0f,   zsign * zf / (zf - zn), -(zn * zf) / (zf - zn),    // Row 2
         0.0f,   0.0f,   zsign,                  0.0f;                      // Row 3
    return m;
}

Eigen::Matrix4f GsCamera::getOrthographicMatrix() const {
    // 正交投影不用进行齐次除法，正交投影矩阵直接转换相机坐标转换到ndc空间，而透视投影还需要进行齐次除法才能转到ndc空间
    // 推导：u = sx * x + cx, ==> ndc_x = (u / width) * 2 - 1 = (2 * sx / width) * x + (2 * cx / width) - 1; 
    // 因此 xscale = 2 * sx / width, dx = 2 * cx / width - 1, yscale和dy同理
    float xScale = fx / (width / 2.0f);
    float yScale = fy / (height / 2.0f);
    float dx = 2.0f * (cx / (float)width) - 1.0f;
    float dy = 2.0f * (cy / (float)height) - 1.0f;
    float zn = znear;
    float zf = zfar;

    Eigen::Matrix4f m;
    m << xScale, 0.0f,   0.0f,          dx,           // Row 0
         0.0f,   yScale, 0.0f,          dy,           // Row 1
         0.0f,   0.0f,   1.0f/(zf-zn), -zn/(zf-zn),   // Row 2
         0.0f,   0.0f,   0.0f,          1.0f;         // Row 3

    return m;
}

Eigen::Matrix4f GsCamera::getProjectionMatrix() const {
    // Proj = Perspective * View
    return getPerspectiveMatrix() * getWorld2CameraMatrix();
}


Eigen::Matrix4f GsCamera::getWorld2CameraMatrix() const {
    Eigen::Quaternionf q = quaternion.normalized();
    Eigen::Matrix3f R = q.toRotationMatrix();

    Eigen::Matrix3f Rt = R.transpose();
    Eigen::Vector3f t_inv = -Rt * position;

    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m.block<3, 3>(0, 0) = Rt;
    m.block<3, 1>(0, 3) = t_inv;
    
    return m;
}

Eigen::Matrix3f GsCamera::getIntrinsicMatrix() const {
    Eigen::Matrix3f K;
    K << fx,   0.0f, cx,
         0.0f, fy,   cy,
         0.0f, 0.0f, 1.0f;

    return K;
}

GsCamera createCamera(Eigen::Vector3f position, Eigen::Quaternionf quaternion, int width, int height, float fov) {
    GsCamera camera;
    camera.position = position;
    camera.quaternion = quaternion;
    camera.width = width;
    camera.height = height;
    camera.fx = fov2focal(fov, width);
    camera.fy = camera.fx;
    camera.cx = width / 2.0f;
    camera.cy = height / 2.0f;
    return camera;
}

float fov2focal(float fovRadian, float pixels) {
    return (pixels / 2.f) / std::tan(fovRadian / 2.f);
}

float focal2fov(float focal, float pixels) {
    return 2 * std::atan(pixels / (2 * focal));
}

float fovx2fovy(float fovx, float width, float height) {
    return 2 * std::atan((height / width) * std::tan(fovx / 2));
}

float fovy2fovx(float fovy, float width, float height) {
    return 2 * std::atan((width / height) * std::tan(fovy / 2));
}

float rad2Deg(float radians) {
    return radians * (180.0 / M_PI);
}

float deg2Rad(float degrees) {
    return degrees * (M_PI / 180.0);
}

}