#pragma once

#include "render.h"
#include "camera.h"
#include <vector>
#include <string>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>





namespace optisplat {
    // 启动可视化器的入口函数
    int runViewer(std::shared_ptr<IGaussianRender> renderer,  std::vector<GsCamera> cameras, int lockFPS = 60, bool debug = false);
}