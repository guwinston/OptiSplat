#include "camera.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace optisplat {

glm::mat4 GsCamera::getPerspectiveMatrix() const {
    const float xScale = fx / (width / 2.0f);
    const float yScale = fy / (height / 2.0f);
    const float dx = 2.0f * (cx / width) - 1.0f;
    const float dy = 2.0f * (cy / height) - 1.0f;
    
    float zsign	 = 1.0;
    float zf = zfar;
    float zn = znear;

    // GLM 矩阵构造是列主序: mat4(col0, col1, col2, col3)
    return glm::mat4(
        xScale, 0.0f,   0.0f,                   0.0f,   // col 0
        0.0f,   yScale, 0.0f,                   0.0f,   // col 1
        dx,     dy,     zsign * zf / (zf - zn), zsign,  // col 2
        0.0f,   0.0f,   -(zn * zf) / (zf - zn), 0.0f    // col 3
    );
}

glm::mat4 GsCamera::getOrthographicMatrix() const {
    // 正交投影不用进行齐次除法，正交投影矩阵直接转换相机坐标转换到ndc空间，而透视投影还需要进行齐次除法才能转到ndc空间
    // 推导：u = sx * x + cx, ==> ndc_x = (u / width) * 2 - 1 = (2 * sx / width) * x + (2 * cx / width) - 1; 
    // 因此 xscale = 2 * sx / width, dx = 2 * cx / width - 1, yscale和dy同理
    float xScale = fx / (width / 2.0f);
    float yScale = fy / (height / 2.0f);
    float dx = 2.0f * (cx / width) - 1.0f;
    float dy = 2.0f * (cy / height) - 1.0f;
    float zn = znear;
    float zf = zfar;

    return glm::mat4(
        xScale, 0.0f,   0.0f,          0.0f, // col 0
        0.0f,   yScale, 0.0f,          0.0f, // col 1
        0.0f,   0.0f,   1.0f/(zf-zn),  0.0f, // col 2
        dx,     dy,     -zn/(zf-zn),   1.0f  // col 3
    );
}

glm::mat4 GsCamera::getProjectionMatrix() const {
    // Proj = Perspective * View
    return getPerspectiveMatrix() * getWorld2CameraMatrix();
}


glm::mat4 GsCamera::getWorld2CameraMatrix() const {
    // 1. 归一化并构造旋转矩阵 (注意 GLM quat 构造顺序: w, x, y, z)
    glm::quat q = glm::normalize(glm::quat(quaternion.w, quaternion.x, quaternion.y, quaternion.z));
    glm::mat3 R = glm::mat3_cast(q); 
    
    // 2. 优化：不直接使用 inverse()。
    // W2C 矩阵的旋转部分是 C2W 的转置，位移部分是 -R^T * t
    glm::mat3 Rt = glm::transpose(R);
    glm::vec3 t_inv = -Rt * position;

    // 3. 组合成 4x4 矩阵 (列主序填充)
    return glm::mat4(
        Rt[0][0], Rt[0][1], Rt[0][2], 0.0f, // col 0
        Rt[1][0], Rt[1][1], Rt[1][2], 0.0f, // col 1
        Rt[2][0], Rt[2][1], Rt[2][2], 0.0f, // col 2
        t_inv.x,  t_inv.y,  t_inv.z,  1.0f  // col 3
    );
}

glm::mat3 GsCamera::getIntrinsicMatrix() const {
    // 列主序构造
    return glm::mat3(
        fx,   0.0f, 0.0f, // col 0
        0.0f, fy,   0.0f, // col 1
        cx,   cy,   1.0f  // col 2
    );
}

GsCamera createCamera(glm::vec3 position, glm::quat quaternion, int width, int height, float fov) {
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