#include "viewer.h"
#include <iostream>
#include <vector>
#include <chrono>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

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
    
    // 从四元数推导当前局部坐标系的基向量
    glm::mat3 rotMat = glm::mat3_cast(cam.quaternion);
    glm::vec3 right   = rotMat[0]; // 第一列：右向量
    glm::vec3 up      = rotMat[1]; // 第二列：上向量
    glm::vec3 forward = rotMat[2]; // 第三列：前向量

    // 1. 键盘 WASD 平移逻辑
    float velocity = moveSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) cam.position -= forward * velocity;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) cam.position += forward * velocity;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) cam.position -= right * velocity;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) cam.position += right * velocity;

    // 2. 鼠标滚轮物理推进逻辑
    if (std::abs(scrollDelta) > 0.01f) {
        cam.position -= forward * (scrollDelta * moveSpeed * 0.5f);
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
            // 绕世界坐标系 Y 轴进行水平旋转 (Yaw)
            glm::quat yaw = glm::angleAxis(-deltaX * mouseSensitivity, glm::vec3(0, 1, 0));
            // 绕相机局部坐标系 Right 轴进行垂直旋转 (Pitch)
            glm::quat pitch = glm::angleAxis(-deltaY * mouseSensitivity, right);
            
            cam.quaternion = glm::normalize(yaw * pitch * cam.quaternion);
        }
        
        // 右键进行平移 (Pan)
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            cam.position += right * (deltaX * mouseSensitivity * 2.0f);
            cam.position -= up * (deltaY * mouseSensitivity * 2.0f);
        }
    }
}

int runViewer(std::shared_ptr<IGaussianRender> renderer, std::vector<GsCamera> cameras, int lockFPS, bool debug) {
    // --- 1. 初始化工作相机列表 (深拷贝以保护原始数据) ---
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

    int width = workCameras[0].width;
    int height = workCameras[0].height;
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

    // --- 6. 主循环 ---
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // 快捷键退出逻辑
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

        // 重置视角逻辑 (R)
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            workCameras[currentCamIdx] = cameras.empty() ? renderer->defaultCamera : cameras[currentCamIdx];
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
        
        ImGui::Begin("Renderer Statistics");
        ImGui::Text("Rendered Gaussians: %.0f", numRendered);
        ImGui::Separator();
        
        // 显示纯渲染耗时与对应 FPS
        ImGui::Text("GPU Render: %.2f ms (%.1f FPS)", avgRenderMs, 1000.0f / (avgRenderMs + 1e-6f));
        
        // 显示包含拷贝在内的总帧耗时
        float currentTotalMs = deltaTime * 1000.0f;
        avgTotalMs = 0.95f * avgTotalMs + 0.05f * currentTotalMs;
        ImGui::Text("Total Frame: %.2f ms (%.1f FPS)", avgTotalMs, 1000.0f / (avgTotalMs + 1e-6f));
        
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

    // --- 7. 资源清理与释放 ---
    cudaFreeHost(hPinnedPtr);
    cudaFree(dOutImage);
    glDeleteTextures(1, &imageTexture);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

} // namespace optisplat