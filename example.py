import torch
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

def ptr_to_tensor(ptr, height, width, channels, device_id=0):
    """将 C++ 传回的 CUDA 指针包装为 torch.Tensor"""
    if ptr == 0:
        return None
    
    # 构造 CUDA Array Interface 字典
    ctx = {
        "shape": (height, width, channels),
        "typestr": "<f4",     # <f4 代表 float32
        "data": (ptr, False), # 指针地址，False 表示非只读
        "version": 3,
    }
    
    # 创建一个哑对象来承载接口
    class Holder:
        def __init__(self, interface):
            self.__cuda_array_interface__ = interface

    return torch.as_tensor(Holder(ctx), device=f"cuda:{device_id}")

if __name__ == "__main__":
    debug = False
    testPerformance = True
    bRunViewer = True
    config = optisplat.GsConfig()
    config.modelPath = "/mnt/e/Dataset/GaussianSplattingModels/bicycle/point_cloud/iteration_30000/point_cloud.ply"
    config.cameraPath = "/mnt/e/Dataset/GaussianSplattingModels/bicycle/cameras.json"
    config.bRebuildBinaryCache = False
    config.bUseFlashGSExactIntersection = True
    config.bUseFlashGSPrefetchingPipeline = False
    config.bUseTensorCore = True

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

    image_tensor = ptr_to_tensor(img_ptr, cam.height, cam.width, 4) # 4 通道 RGBA
    depth_tensor = ptr_to_tensor(map_ptr, cam.height, cam.width, 1) # 深度图是 1 通道

    if image_tensor is not None:
        img_np = image_tensor.detach().cpu().numpy()
        plt.imshow(img_np[..., :3]) # 只显示 RGB
        plt.show()