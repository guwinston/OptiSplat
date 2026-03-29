#include <filesystem>

#include "render.h"
#include "utils.h"
#include "camera.h"
#include "viewer.h"

#include <cuda.h>
#include <cuda_runtime.h>

using namespace optisplat;



int main(int argc, char** argv) {
    const MemoryFootprint startFootprint = Utils::sampleMemoryFootprint();
    Utils::logMemoryFootprint("Main start", startFootprint);

    bool debug = false;
    bool testPerformance = true;
    bool bRunViewer = true;
	GsConfig config;
    config.modelPath = "/mnt/e/Dataset/GaussianSplattingModels/bicycle/point_cloud/iteration_30000/point_cloud.ply";
    config.cameraPath = "/mnt/e/Dataset/GaussianSplattingModels/bicycle/cameras.json";
    config.bRebuildBinaryCache = false;
    config.bKeepCpuSceneData = false;
    config.bUseHalfPrecisionSH = true;
    config.bUseHalfPrecisionCov3DOpacity = true;
    config.exactActiveSetMode = ExactActiveSetMode::CenterOnly;
    config.bUseFlashGSExactIntersection = true;
    config.bUseFlashGSPrefetchingPipeline = false;
    config.bUseTensorCore = true;
    config.maxNumRenderedGaussians = -1; // 预分配中间显存，设为-1表示动态分配

    std::vector<GsCamera> cameras = Utils::readCamerasFromJson(config.cameraPath);
    std::shared_ptr<IGaussianRender> renderer = IGaussianRender::CreateRenderer(config);
    int exitCode = 0;
    if (bRunViewer) {
        int maxWindowWidth = 1920;
        int maxWindowHeight = 1080;
        exitCode = runViewer(renderer, cameras, maxWindowWidth, maxWindowHeight, -1, debug);
        const MemoryFootprint endFootprint = Utils::sampleMemoryFootprint();
        Utils::logMemoryFootprint("Main end", endFootprint);
        Utils::logMemoryFootprintDelta("Main runtime", startFootprint, endFootprint);
        return exitCode;
    }

    ProgressBar progress(cameras.size(), "Rendering");
    float time = 0;
    float* outImage = nullptr;
    float* outAllmap = nullptr;
    GsCamera camera = cameras[0];
    if (testPerformance)
        for (int i = 0; i < 10; i++)
            renderer->render(camera, outImage, outAllmap, debug); // warm-up
    for (int i = 0; i < cameras.size(); i++) {
        auto t0 = Utils::nowUs();
        camera = cameras[i];
        camera.rescaleResolution(4);
        // camera.setResolution(1920, 1080);
        float numRendered = renderer->render(camera, outImage, outAllmap, debug);
        auto t1 = Utils::nowUs();
        time += (t1-t0)/ 1000.0f; // Convert to milliseconds
        progress.show_progress(i+1, (t1-t0) / 1000, numRendered);
        // break;
    }
    progress.close();

    std::filesystem::path mainFilePath(__FILE__);
    std::filesystem::path projectDir = mainFilePath.parent_path().parent_path();
    std::vector<std::string> filenames = { (projectDir / "output/output.jpg").string() };
    Utils::saveImages(outImage, 1, camera.height, camera.width, filenames);
    Utils::saveAllMaps(outAllmap, 1, camera.height, camera.width, filenames);
    Utils::saveCudaArrayToBin((projectDir / "output/output.bin").string() , outImage, camera.height * camera.width * 4);
    std::cout << "Average time: " << time / cameras.size() << " ms" << std::endl;
    std::cout << "Average FPS:  " << 1000 / (time / cameras.size()) << std::endl;
    const MemoryFootprint endFootprint = Utils::sampleMemoryFootprint();
    Utils::logMemoryFootprint("Main end", endFootprint);
    Utils::logMemoryFootprintDelta("Main runtime", startFootprint, endFootprint);
    
    return exitCode;
}



