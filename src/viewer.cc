#include "viewer.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace optisplat {

// 全局变量用于捕获滚轮位移
static float gScrollOffset = 0.0f;
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    // 只有当鼠标不在 ImGui 面板上时才响应 3D 缩放
    if (!ImGui::GetIO().WantCaptureMouse) {
        gScrollOffset = static_cast<float>(yoffset);
    }
}

/**
 * @brief 处理相机姿态更新（UE 风格交互逻辑）
 * @param cam 当前要修改的相机对象
 * @param window GLFW窗口指针
 * @param deltaTime 帧间隔时间（用于平滑移动）
 * @param moveSpeed 移动系数
 * @param mouseSensitivity 鼠标灵敏度
 * @param lastX/lastY 上一帧鼠标位置
 * @param scrollDelta 滚轮位移
 */
void updateCameraPose(GsCamera& cam, GLFWwindow* window, float deltaTime, 
                      float moveSpeed, float mouseSensitivity, 
                      double& lastX, double& lastY, float scrollDelta) {
    
    bool isCameraMoved = false;

    // 从四元数推导当前局部坐标系的基向量
    Eigen::Matrix3f rotMat = cam.quaternion.toRotationMatrix();
    Eigen::Vector3f right   = rotMat.col(0); // 第一列：右向量
    Eigen::Vector3f up      = rotMat.col(1); // 第二列：上向量
    Eigen::Vector3f forward = rotMat.col(2); // 第三列：前向量

    // 1. 键盘 WASD 平移逻辑
    float velocity = moveSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {cam.position += forward * velocity; isCameraMoved = true;}
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {cam.position -= forward * velocity; isCameraMoved = true;}
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {cam.position -= right * velocity; isCameraMoved = true;}
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {cam.position += right * velocity; isCameraMoved = true;}
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {cam.position += up * velocity; isCameraMoved = true;}
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {cam.position -= up * velocity; isCameraMoved = true;}

    // 2. 鼠标滚轮物理推进逻辑
    if (std::abs(scrollDelta) > 0.01f) {
        cam.position -= forward * (scrollDelta * moveSpeed * 0.5f);
        isCameraMoved = true;
    }

    // 3. 鼠标交互（旋转与平移）
    double curX, curY;
    glfwGetCursorPos(window, &curX, &curY);
    float deltaX = static_cast<float>(curX - lastX);
    float deltaY = static_cast<float>(curY - lastY);
    lastX = curX; lastY = curY;

    if (!ImGui::GetIO().WantCaptureMouse) {
        // 左键修改旋转 (Yaw & Pitch)
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            // 绕相机局部坐标系 up 轴进行水平旋转 (Yaw)
            Eigen::AngleAxisf yaw(deltaX * mouseSensitivity, up);
            // 绕相机局部坐标系 Right 轴进行垂直旋转 (Pitch)
            Eigen::AngleAxisf pitch(-deltaY * mouseSensitivity, right);
            
            cam.quaternion = (Eigen::Quaternionf(yaw) * Eigen::Quaternionf(pitch) * cam.quaternion).normalized();
            isCameraMoved = true;
        }
        
        // 右键进行平移 (Pan)
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            cam.position -= right * (deltaX * mouseSensitivity * 2.0f);
            cam.position -= up * (deltaY * mouseSensitivity * 2.0f);
            isCameraMoved = true;
        }
    }

    if (isCameraMoved) {
        GS_INFO("Camera position = {%f, %f, %f}, quaternion = {%f, %f, %f, %f}, width = %d, height = %d, fov = %f",
            cam.position.x(), cam.position.y(), cam.position.z(), cam.quaternion.w(), 
            cam.quaternion.x(), cam.quaternion.y(), cam.quaternion.z(), cam.width, cam.height, rad2Deg(focal2fov(cam.fx, cam.width)));
    }
}

int runViewer(std::shared_ptr<IGaussianRender> renderer, std::vector<GsCamera> cameras, int maxWidth, int maxHeight, int lockFPS, bool debug) {
    // --- 1. 初始化工作相机列表 (深拷贝以保护原始数据) ---
    GS_INFO("Launching Gaussian Splatting Viewer...");
    std::vector<GsCamera> workCameras;
    if (cameras.empty()) {
        workCameras.push_back(renderer->defaultCamera);
    } else {
        workCameras = cameras;
    }

    // --- 2. 初始化窗口与 OpenGL 环境 ---
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    int width = maxWidth <= 0 ? workCameras[0].width : std::min(workCameras[0].width, maxWidth);
    int height = maxHeight <= 0 ? workCameras[0].height : std::min(workCameras[0].height, maxHeight);
    GS_INFO("Gaussian Splatting Viewer window resolotion: {%d}x{%d}", width, height);
    // int width = workCameras[0].width;
    // int height = workCameras[0].height;
    GLFWwindow* window = glfwCreateWindow(width, height, "Gaussian Splatting Viewer", NULL, NULL);
    if (!window) return -1;
    
    glfwMakeContextCurrent(window);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSwapInterval(1); // 开启垂直同步
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    // --- 3. 初始化 ImGui ---
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // --- 4. 准备 GPU 渲染缓冲区与纹理 ---
    GLuint imageTexture;
    glGenTextures(1, &imageTexture);
    glBindTexture(GL_TEXTURE_2D, imageTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // 锁页内存 (Pinned Memory) 加速显存到内存的 DMA 拷贝
    float* hPinnedPtr = nullptr;
    size_t imgSize = width * height * 4 * sizeof(float);
    cudaHostAlloc((void**)&hPinnedPtr, imgSize, cudaHostAllocDefault);
    
    // 设备侧图像缓冲区
    float* dOutImage = nullptr;
    cudaMalloc(&dOutImage, imgSize);

    // --- 5. 初始化状态与计时变量 ---
    int currentCamIdx = 0;
    float moveSpeed = 2.0f;
    float mouseSensitivity = 0.002f;
    double lastMouseX, lastMouseY;
    glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
    
    auto lastFrameTime = std::chrono::high_resolution_clock::now();
    float avgRenderMs = 0.0f; // 纯渲染耗时（平滑值）
    float avgTotalMs = 0.0f;  // 完整帧耗时（平滑值）
    float numRendered = 0.0f; // 渲染的高斯点数量
    bool camSwitchLocked = false;
    GS_INFO("Launching Gaussian Splatting Viewer successful.");

    // --- 6. 主循环 ---
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // 快捷键退出逻辑
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

        // 重置视角逻辑 (R)
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            workCameras[currentCamIdx] = cameras.empty() ? renderer->defaultCamera : cameras[currentCamIdx];
            workCameras[currentCamIdx].setResolution(width, height);
        }

        // 计算 DeltaTime (秒)
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastFrameTime).count();
        lastFrameTime = currentTime;

        // 相机索引切换逻辑 (防止长按导致连续跳过)
        if (!camSwitchLocked) {
            if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
                currentCamIdx = (currentCamIdx + 1) % static_cast<int>(workCameras.size());
                camSwitchLocked = true;
            } else if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
                currentCamIdx = (currentCamIdx - 1 + static_cast<int>(workCameras.size())) % static_cast<int>(workCameras.size());
                camSwitchLocked = true;
            }
        }
        if (camSwitchLocked && glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_RELEASE && glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_RELEASE && 
            glfwGetKey(window, GLFW_KEY_UP) == GLFW_RELEASE && glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_RELEASE) {
            camSwitchLocked = false;
        }

        // 更新相机姿态
        updateCameraPose(workCameras[currentCamIdx], window, deltaTime, moveSpeed, mouseSensitivity, lastMouseX, lastMouseY, gScrollOffset);
        gScrollOffset = 0.0f;

        // --- A. 纯渲染过程 (测量开始) ---
        auto renderStart = std::chrono::high_resolution_clock::now();
        float* dOutAllMap = nullptr; 
        workCameras[currentCamIdx].setResolution(width, height);
        numRendered = renderer->render(workCameras[currentCamIdx], dOutImage, dOutAllMap, debug);
        auto renderEnd = std::chrono::high_resolution_clock::now();
        
        float currentRenderMs = std::chrono::duration<float, std::milli>(renderEnd - renderStart).count();
        avgRenderMs = 0.95f * avgRenderMs + 0.05f * currentRenderMs; // 低通滤波平滑显示

        // --- B. 后处理过程 (包含 DMA 拷贝与纹理更新) ---
        cudaMemcpy(hPinnedPtr, dOutImage, imgSize, cudaMemcpyDeviceToHost);
        glBindTexture(GL_TEXTURE_2D, imageTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, hPinnedPtr);

        // --- C. UI 与性能看板绘制 ---
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        ImGui::SetNextWindowPos(ImVec2(width - 260.0f, 10.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Renderer Statistics", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Rendered Gaussians: %.0f", numRendered);
        ImGui::Separator();
        
        // 显示纯渲染耗时与对应 FPS
        ImGui::Text("GPU Render: %.2f ms (%.1f FPS)", avgRenderMs, 1000.0f / (avgRenderMs + 1e-6f));
        
        // 显示包含拷贝在内的总帧耗时
        float currentTotalMs = deltaTime * 1000.0f;
        avgTotalMs = 0.95f * avgTotalMs + 0.05f * currentTotalMs;
        ImGui::Text("Total Frame: %.2f ms (%.1f FPS)", avgTotalMs, 1000.0f / (avgTotalMs + 1e-6f));
        ImGui::End();

        ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Camera Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Camera Settings");
        int currentMode = (int)workCameras[currentCamIdx].model; 
        if (ImGui::Combo("Projection", &currentMode, CameraModelNames, IM_ARRAYSIZE(CameraModelNames))) {
            workCameras[currentCamIdx].model = (CameraModel)currentMode;
        }
        // 畸变系数 k1-k4 进度条 (仅在鱼眼模式下显示)
        if (currentMode == (int)CameraModel::FISHEYE) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.2f, 1.0f)); // 鱼眼模式高亮提示
            ImGui::Text("Kannala-Brandt Distortion (k1-k4)");
            ImGui::PopStyleColor();

            // 使用 DragFloat 方便微调且不占空间
            ImGui::DragFloat("k1", &workCameras[currentCamIdx].k1, 0.001f, -1.0f, 1.0f, "%.4f");
            ImGui::DragFloat("k2", &workCameras[currentCamIdx].k2, 0.001f, -1.0f, 1.0f, "%.4f");
            ImGui::DragFloat("k3", &workCameras[currentCamIdx].k3, 0.001f, -1.0f, 1.0f, "%.4f");
            ImGui::DragFloat("k4", &workCameras[currentCamIdx].k4, 0.001f, -1.0f, 1.0f, "%.4f");
            if (ImGui::Button("Reset Distortion")) {
                workCameras[currentCamIdx].k1 = cameras[currentCamIdx].k1;
                workCameras[currentCamIdx].k1 = cameras[currentCamIdx].k2;
                workCameras[currentCamIdx].k1 = cameras[currentCamIdx].k3;
                workCameras[currentCamIdx].k1 = cameras[currentCamIdx].k4;
            }
        }
        // 正交缩放 (仅在正交模式下显示)
        else if (currentMode == (int)CameraModel::ORTHOGRAPHIC) {
            float focalScale = workCameras[currentCamIdx].fy / workCameras[currentCamIdx].fx;
            if (ImGui::SliderFloat("Ortho Scale", &workCameras[currentCamIdx].fx, 0.1f, 2000.0f)) {
                workCameras[currentCamIdx].fy = workCameras[currentCamIdx].fx * focalScale;
            }
            if (ImGui::Button("Reset Ortho Scale")) {
                workCameras[currentCamIdx].width = cameras[currentCamIdx].width;
                workCameras[currentCamIdx].height = cameras[currentCamIdx].height;
                workCameras[currentCamIdx].fx = cameras[currentCamIdx].fx;
                workCameras[currentCamIdx].fy = cameras[currentCamIdx].fy;
                workCameras[currentCamIdx].setResolution(width, height);
            }
        }

        ImGui::Separator();
        ImGui::Text("View Frustum Control");
        if (currentMode != (int)CameraModel::ORTHOGRAPHIC) { // 正交模式下，FOV 没有意义
            float fovDegreesX = rad2Deg(focal2fov(workCameras[currentCamIdx].fx, workCameras[currentCamIdx].width));
            float fovDegreesY = rad2Deg(focal2fov(workCameras[currentCamIdx].fy, workCameras[currentCamIdx].height));
            float focalScale = workCameras[currentCamIdx].fy / workCameras[currentCamIdx].fx; // 记录当前的 fx/fy 比例（通常为 1.0，但为了兼容非正方形像素，我们动态计算）
            if (ImGui::SliderFloat("Field of View", &fovDegreesX, 1.0f, 179.0f, "%.2f deg")) {
                workCameras[currentCamIdx].fx = fov2focal(deg2Rad(fovDegreesX), workCameras[currentCamIdx].width);
                workCameras[currentCamIdx].fy = workCameras[currentCamIdx].fx * focalScale;
            }
            float fovx = rad2Deg(focal2fov(workCameras[currentCamIdx].fx, workCameras[currentCamIdx].width));
            float fovy = rad2Deg(focal2fov(workCameras[currentCamIdx].fy, workCameras[currentCamIdx].height));
            ImGui::Text("FovX = %.2f, FovY = %.2f", fovx, fovy);
            if (ImGui::Button("Reset FOV")) {
                workCameras[currentCamIdx].width = cameras[currentCamIdx].width;
                workCameras[currentCamIdx].height = cameras[currentCamIdx].height;
                workCameras[currentCamIdx].fx = cameras[currentCamIdx].fx;
                workCameras[currentCamIdx].fy = cameras[currentCamIdx].fy;
                workCameras[currentCamIdx].setResolution(width, height);
            }
        }
        
        ImGui::Separator();
        ImGui::Text("Camera: %d / %zu", currentCamIdx, workCameras.size());
        ImGui::SliderFloat("Move Speed", &moveSpeed, 0.1f, 10.0f);
        if (ImGui::Button("Reset View (R)")) {
             workCameras[currentCamIdx] = cameras.empty() ? renderer->defaultCamera : cameras[currentCamIdx];
        }
        ImGui::End();

        // --- D. 最终 OpenGL 画面呈现 ---
        int dw, dh;
        glfwGetFramebufferSize(window, &dw, &dh);
        glViewport(0, 0, dw, dh);
        glClear(GL_COLOR_BUFFER_BIT);
        
        // 将 CUDA 渲染出的纹理全屏覆盖
        ImGui::GetBackgroundDrawList()->AddImage((ImTextureID)(intptr_t)imageTexture, ImVec2(0, 0), ImVec2((float)dw, (float)dh));
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }
    GS_INFO("Shutting down...");

    // --- 7. 资源清理与释放 ---
    cudaFreeHost(hPinnedPtr);
    cudaFree(dOutImage);
    glDeleteTextures(1, &imageTexture);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    GS_INFO("Shutdown complete.");

    return 0;
}

} // namespace optisplat