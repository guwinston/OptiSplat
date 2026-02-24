

#include "render.h"
#include "utils.h"
#include "camera.h"
#include "viewer.h"

#include <cuda.h>
#include <cuda_runtime.h>

using namespace optisplat;



int main(int argc, char** argv) {
    float m0 = Utils::nowGPUMB();

    bool debug = true;
    bool bRunViewer = true;
	GsConfig config;
    config.modelPath = "/mnt/e/Dataset/GaussianSplattingModels/bicycle/point_cloud/iteration_30000/point_cloud.ply";
    config.cameraPath = "/mnt/e/Dataset/GaussianSplattingModels/bicycle/cameras.json";
    config.bRebuildBinaryCache = false;
    config.bUseTensorCore = false;

    std::vector<GsCamera> cameras = Utils::readCamerasFromJson(config.cameraPath);
    std::shared_ptr<IGaussianRender> renderer = IGaussianRender::CreateRenderer(config);
    if (bRunViewer) {
        return runViewer(renderer, cameras, debug);
    }

    ProgressBar progress(cameras.size(), "Rendering");
    float time = 0;
    float* outImage = nullptr;
    float* outAllmap = nullptr;
    for (int i = 0; i < cameras.size(); i++) {
        auto t0 = Utils::nowUs();
        auto camera = cameras[i];
        float numRendered = renderer->render(camera, outImage, outAllmap, debug);
        auto t1 = Utils::nowUs();
        time += (t1-t0)/ 1000.0f; // Convert to milliseconds
        progress.show_progress(i+1, (t1-t0) / 1000, numRendered);
        break;
    }

    std::vector<std::string> filenames = { "output/output.jpg" };
    Utils::saveImages(outImage, 1, cameras.back().height, cameras.back().width, filenames);
    Utils::saveAllMaps(outAllmap, 1, cameras.back().height, cameras.back().width, filenames);
    progress.close();
    std::cout << "Average time: " << time / cameras.size() << " ms" << std::endl;
    std::cout << "Memory usage: " << Utils::nowGPUMB() - m0 << " MB" << std::endl;
    
    return 0;
}



