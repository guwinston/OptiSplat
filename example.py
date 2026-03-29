import sys
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    import optisplat
    print("Import success!")
except ImportError as e:
    print(f"Still failed: {e}")


if __name__ == "__main__":
    debug = False
    testPerformance = True
    bRunViewer = True
    config = optisplat.GsConfig()
    config.modelPath = "/mnt/e/Dataset/GaussianSplattingModels/bicycle/point_cloud/iteration_30000/point_cloud.ply"
    config.cameraPath = "/mnt/e/Dataset/GaussianSplattingModels/bicycle/cameras.json"
    config.bRebuildBinaryCache = False
    config.bKeepCpuSceneData = False
    config.bUseHalfPrecisionSH = True
    config.bUseHalfPrecisionCov3DOpacity = True
    config.exactActiveSetMode = optisplat.ExactActiveSetMode.CENTER_ONLY
    config.bUseFlashGSExactIntersection = True
    config.bUseFlashGSPrefetchingPipeline = False
    config.bUseTensorCore = True
    config.maxNumRenderedGaussians = -1; # 预分配的中间显存，预分配显存可以提高渲染效率，值可以根据实际情况调整

    renderer = optisplat.IGaussianRender.CreateRenderer(config)
    cameras = optisplat.readCamerasFromJson(config.cameraPath)

    if bRunViewer:
        maxWinWidth = 1920
        maxWinHeight = 1080
        optisplat.runViewer(renderer, cameras, maxWinWidth, maxWinHeight, -1, debug)
        exit(0)

    if testPerformance:
        for _ in range(10):
            num, img_ptr, map_ptr = renderer.render(cameras[0], debug)

    pbar = tqdm(cameras, desc='Rendering [FPS: 0.00, Delay: 0.00 ms]')
    total_time = 0
    for cam in pbar:
        t0 = time.time()
        num, img_ptr, map_ptr = renderer.render(cam, debug)
        t1 = time.time()
        total_time += t1 - t0
        pbar.set_description(f"Rendering [FPS: {1/(t1-t0):.2f}, Delay: {(t1-t0)*1000:.2f} ms]")
    avg_time = total_time / len(cameras)
    print(f"Rendering shape: {cam.width}x{cam.height}")
    print(f"AVG FPS: {1/avg_time:.2f}, AVG Delay: {avg_time*1000:.2f} ms")

    image = optisplat.copyToHost(img_ptr, cam.height, cam.width, 4) # 4 通道 RGBA
    depth = optisplat.copyToHost(map_ptr, cam.height, cam.width, 1) # 深度图是 1 通道

    if image is not None:
        plt.imshow(image[..., :3]) # 只显示 RGB
        plt.show()