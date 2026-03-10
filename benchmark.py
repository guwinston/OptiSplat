import os
import sys
import time
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    import optisplat
except ImportError as e:
    print(f"无法导入 optisplat，请先完成构建或通过 `pip install -e .` 安装: {e}")
    sys.exit(1)

# 不同渲染配置组合（与 README 中的示例类似）
CONFIGS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "ExactIntersection + TensorCore",
        {
            "bUseFlashGSExactIntersection": True,
            "bUseFlashGSPrefetchingPipeline": False,
            "bUseTensorCore": True,
        },
    ),
    (
        "ExactIntersection + Prefetching",
        {
            "bUseFlashGSExactIntersection": True,
            "bUseFlashGSPrefetchingPipeline": True,
            "bUseTensorCore": False,
        },
    ),
    (
        "ExactIntersection Only",
        {
            "bUseFlashGSExactIntersection": True,
            "bUseFlashGSPrefetchingPipeline": False,
            "bUseTensorCore": False,
        },
    ),
    (
        "Baseline",
        {
            "bUseFlashGSExactIntersection": False,
            "bUseFlashGSPrefetchingPipeline": False,
            "bUseTensorCore": False,
        },
    ),
]

# 分辨率测试组合：原始、2 倍下采样、4 倍下采样、1080p
# value 含义：
# - None: 原始分辨率
# - 2 / 4: 调用 rescaleResolution(下采样倍数)
# - (w, h): 调用 setResolution(w, h)
RESOLUTIONS: List[Tuple[str, Any]] = [
    ("Original", None),
    ("2x Downsample", 2),
    ("4x Downsample", 4),
    ("1080p", (1920, 1080)),
]


def _prepare_cameras_for_resolution(camera_path: str, res_cfg):
    """读取相机并根据分辨率配置，使用 rescaleResolution / setResolution 调整。"""
    cameras = optisplat.readCamerasFromJson(camera_path)
    if not cameras:
        return []

    if res_cfg is None:
        # 原始分辨率
        return cameras

    if isinstance(res_cfg, (int, float)):
        # 下采样倍数：2 表示长宽各除以 2，4 表示各除以 4
        for cam in cameras:
            cam.rescaleResolution(float(res_cfg))
        return cameras

    if isinstance(res_cfg, tuple) and len(res_cfg) == 2:
        w, h = res_cfg
        for cam in cameras:
            cam.setResolution(int(w), int(h))
        return cameras

    raise ValueError(f"未知的分辨率配置: {res_cfg}")


def run_benchmark(
    model_path: str,
    camera_path: str,
    warmup_iters: int = 10,
    debug: bool = False,
) -> None:
    """对多种配置和多种分辨率进行性能测试，输出 FPS 和单帧延迟。"""

    if not os.path.isfile(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    if not os.path.isfile(camera_path):
        print(f"相机文件不存在: {camera_path}")
        return

    print(f"使用模型: {model_path}")
    print(f"使用相机: {camera_path}")

    results = []

    # 外层按渲染配置循环，确保每种配置只创建 / 加载一次 renderer（模型只 load 一次）
    for cfg_name, cfg in CONFIGS:
        print("\n" + "#" * 80)
        print(f"开始配置组: {cfg_name}")
        print(
            f"  bUseFlashGSExactIntersection={cfg['bUseFlashGSExactIntersection']}, "
            f"bUseFlashGSPrefetchingPipeline={cfg['bUseFlashGSPrefetchingPipeline']}, "
            f"bUseTensorCore={cfg['bUseTensorCore']}"
        )

        config = optisplat.GsConfig()
        config.modelPath = model_path
        config.cameraPath = camera_path
        config.bRebuildBinaryCache = False
        config.bUseFlashGSExactIntersection = cfg["bUseFlashGSExactIntersection"]
        config.bUseFlashGSPrefetchingPipeline = cfg["bUseFlashGSPrefetchingPipeline"]
        config.bUseTensorCore = cfg["bUseTensorCore"]

        renderer = optisplat.IGaussianRender.CreateRenderer(config)

        for res_name, res_cfg in RESOLUTIONS:
            cameras = _prepare_cameras_for_resolution(camera_path, res_cfg)
            if not cameras:
                print(f"分辨率 `{res_name}` 下相机列表为空，跳过。")
                continue

            cam0 = cameras[0]
            full_name = f"{cfg_name} @ {res_name}"

            print("\n" + "=" * 80)
            print(
                f"开始测试: {full_name} "
                f"(分辨率: {cam0.width}x{cam0.height})"
            )

            # 预热，避免首次调用偏慢影响统计
            for _ in range(warmup_iters):
                renderer.render(cameras[0], debug)

            total_time = 0.0
            pbar = tqdm(
                cameras, desc=f"[{full_name}] FPS: 0.00, Delay: 0.00 ms"
            )

            for cam in pbar:
                t0 = time.time()
                _, _, _ = renderer.render(cam, debug)
                t1 = time.time()

                dt = t1 - t0
                total_time += dt

                fps_inst = 1.0 / dt if dt > 0 else 0.0
                delay_ms_inst = dt * 1000.0
                pbar.set_description(
                    f"[{full_name}] FPS: {fps_inst:.2f}, Delay: {delay_ms_inst:.2f} ms"
                )

            avg_time = total_time / len(cameras)
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0.0
            avg_delay_ms = avg_time * 1000.0

            print(
                f"配置 `{full_name}` 测试完成，分辨率: {cam0.width}x{cam0.height}, "
                f"AVG FPS: {avg_fps:.2f}, AVG Delay: {avg_delay_ms:.2f} ms"
            )

            results.append(
                {
                    "name": cfg_name,
                    "resolution_group": res_name,
                    "fps": avg_fps,
                    "delay_ms": avg_delay_ms,
                    "width": cam0.width,
                    "height": cam0.height,
                }
            )

    # 汇总表格：先按分辨率从高到低排序，同一分辨率按配置固定顺序输出
    print("\n" + "#" * 80)
    print("各配置在不同分辨率下的平均性能汇总：")
    config_order = {"Baseline": 0,"ExactIntersection Only": 1, "ExactIntersection + Prefetching": 2,"ExactIntersection + TensorCore": 3,}
    def sort_key(r: Dict[str, Any]):
        pixels = r["width"] * r["height"]
        order = config_order.get(r["name"], 99)
        return (-pixels, order)
    results_sorted = sorted(results, key=sort_key)
    print(f"{'Resolution Group':18s} | {'Resolution':12s} | {'Config':35s} | {'FPS':>10s} | {'Delay (ms)':>12s}")
    print("-" * 110)
    for r in results_sorted:
        reso = f"{r['width']}x{r['height']}"
        print(f"{r['resolution_group']:18s} | {reso:12s} | {r['name']:35s} | {r['fps']:10.2f} | {r['delay_ms']:12.2f}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="OptiSplat 多配置延时 / FPS 性能测试脚本")

    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/e/Dataset/GaussianSplattingModels/bicycle/point_cloud/iteration_30000/point_cloud.ply",
        help="高斯点云模型路径 (.ply)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="/mnt/e/Dataset/GaussianSplattingModels/bicycle/cameras.json",
        help="相机参数 JSON 路径",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="每个配置的预热渲染次数",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="是否开启调试渲染",
    )

    args = parser.parse_args()
    run_benchmark(
        model_path=args.model,
        camera_path=args.cameras,
        warmup_iters=args.warmup,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()

