#pragma once

#include "render.h"
#include "camera.h"
#include <vector>
#include <string>
#include <memory>


namespace optisplat {
    int runViewer(std::shared_ptr<IGaussianRender> renderer,  std::vector<GsCamera> cameras, int maxWidth, int maxHeight, int lockFPS = 60, bool debug = false);
}