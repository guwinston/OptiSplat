import argparse
import ctypes
import ctypes.util
import glob
import importlib.machinery
import importlib.util
import json
import math
import os
import sys
import types
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

try:
    from PIL import Image
except ImportError:
    Image = None


sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


class _FallbackExactActiveSetMode:
    DISABLED = 0
    PRECISE = 1
    CENTER_ONLY = 2


def load_optisplat_module() -> Any:
    root_dir = os.path.dirname(__file__)
    search_dirs = [
        os.path.join(root_dir, "build"),
        os.path.join(root_dir, "src", "optisplat"),
    ]
    preferred_suffixes = importlib.machinery.EXTENSION_SUFFIXES
    suffixes = preferred_suffixes + [".so"]
    candidates: List[str] = []
    for search_dir in search_dirs:
        for suffix in suffixes:
            candidates.extend(glob.glob(os.path.join(search_dir, f"_C*{suffix}")))

    seen = set()
    unique_candidates = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique_candidates.append(path)

    last_error: Optional[BaseException] = None
    for stale_name in ["optisplat", "optisplat._C", "_C"]:
        sys.modules.pop(stale_name, None)
    for path in unique_candidates:
        try:
            spec = importlib.util.spec_from_file_location("_C", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            required = ["GsConfig", "IGaussianRender", "readCamerasFromJson"]
            missing = [name for name in required if not hasattr(module, name)]
            if missing:
                raise ImportError(f"Missing symbols in {path}: {', '.join(missing)}")
            namespace = types.SimpleNamespace(
                GsConfig=module.GsConfig,
                IGaussianRender=module.IGaussianRender,
                readCamerasFromJson=module.readCamerasFromJson,
            )
            if hasattr(module, "ExactActiveSetMode"):
                namespace.ExactActiveSetMode = module.ExactActiveSetMode
            else:
                namespace.ExactActiveSetMode = _FallbackExactActiveSetMode
            if hasattr(module, "copyToHost"):
                namespace.copyToHost = module.copyToHost
            return namespace
        except BaseException as inner_exc:
            last_error = inner_exc

    try:
        import optisplat as module
        return module
    except ImportError as exc:
        print(f"无法导入 optisplat，请先完成构建或通过 `pip install -e .` 安装: {exc}")
        if last_error is not None:
            print(f"尝试直接加载底层扩展也失败了: {last_error}")
        sys.exit(1)


def build_cuda_copy_to_host() -> Any:
    libcudart_path = ctypes.util.find_library("cudart")
    if libcudart_path is None:
        candidates = [
            "libcudart.so",
            "libcudart.so.12",
            "libcudart.so.11.0",
        ]
        for candidate in candidates:
            try:
                ctypes.CDLL(candidate)
                libcudart_path = candidate
                break
            except OSError:
                continue
    if libcudart_path is None:
        return None

    libcudart = ctypes.CDLL(libcudart_path)
    cuda_memcpy = libcudart.cudaMemcpy
    cuda_memcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    cuda_memcpy.restype = ctypes.c_int

    cuda_get_error_string = libcudart.cudaGetErrorString
    cuda_get_error_string.argtypes = [ctypes.c_int]
    cuda_get_error_string.restype = ctypes.c_char_p

    cuda_memcpy_device_to_host = 2

    def copy_to_host(ptr: int, h: int, w: int, c: int) -> np.ndarray:
        array = np.empty((h, w, c), dtype=np.float32)
        byte_count = int(array.nbytes)
        err = cuda_memcpy(
            ctypes.c_void_p(array.ctypes.data),
            ctypes.c_void_p(int(ptr)),
            ctypes.c_size_t(byte_count),
            cuda_memcpy_device_to_host,
        )
        if err != 0:
            err_text = cuda_get_error_string(err)
            message = err_text.decode("utf-8") if err_text else f"cuda error {err}"
            raise RuntimeError(f"cudaMemcpy failed in metrics.py: {message}")
        return array

    return copy_to_host


optisplat = load_optisplat_module()
cuda_copy_to_host = build_cuda_copy_to_host()


@dataclass
class CameraMetric:
    index: int
    camera_id: int
    width: int
    height: int
    psnr: float
    mse: float
    invalid_pixel_ratio: float
    max_abs_diff: float
    baseline_mean_luma: float
    optimized_mean_luma: float
    baseline_max_value: float
    optimized_max_value: float


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_for_save(img: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    return (sanitize_for_save(img) * 255.0 + 0.5).astype(np.uint8)


def write_ppm(path: str, rgb_u8: np.ndarray) -> None:
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("write_ppm expects HxWx3 uint8 image.")
    ensure_dir(os.path.dirname(path) or ".")
    h, w, _ = rgb_u8.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(np.ascontiguousarray(rgb_u8).tobytes())


def write_pgm(path: str, gray_u8: np.ndarray) -> None:
    if gray_u8.ndim != 2:
        raise ValueError("write_pgm expects HxW uint8 image.")
    ensure_dir(os.path.dirname(path) or ".")
    h, w = gray_u8.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(np.ascontiguousarray(gray_u8).tobytes())


def write_rgb_image(path: str, rgb_u8: np.ndarray, image_format: str) -> str:
    ext = "png" if image_format == "png" else "ppm"
    final_path = os.path.splitext(path)[0] + f".{ext}"
    if image_format == "png" and Image is not None:
        ensure_dir(os.path.dirname(final_path) or ".")
        Image.fromarray(rgb_u8, mode="RGB").save(final_path)
        return final_path
    write_ppm(final_path, rgb_u8)
    return final_path


def write_gray_image(path: str, gray_u8: np.ndarray, image_format: str) -> str:
    ext = "png" if image_format == "png" else "pgm"
    final_path = os.path.splitext(path)[0] + f".{ext}"
    if image_format == "png" and Image is not None:
        ensure_dir(os.path.dirname(final_path) or ".")
        Image.fromarray(gray_u8, mode="L").save(final_path)
        return final_path
    write_pgm(final_path, gray_u8)
    return final_path


def build_diff_heatmap(ref: np.ndarray, test: np.ndarray) -> np.ndarray:
    valid_mask = np.isfinite(ref) & np.isfinite(test)
    diff = np.zeros_like(ref, dtype=np.float32)
    diff[valid_mask] = np.abs(ref[valid_mask] - test[valid_mask])
    magnitude = np.max(diff, axis=2)
    if np.any(np.isfinite(magnitude)):
        scale = float(np.nanmax(magnitude))
        if scale > 1e-12:
            magnitude = magnitude / scale
    heat = np.zeros_like(ref, dtype=np.float32)
    heat[..., 0] = magnitude
    heat[..., 1] = np.clip(1.0 - np.abs(magnitude - 0.5) * 2.0, 0.0, 1.0) * 0.6
    heat[..., 2] = 1.0 - magnitude
    return np.clip(heat, 0.0, 1.0)


def build_invalid_overlay(img: np.ndarray, invalid_mask: np.ndarray) -> np.ndarray:
    overlay = sanitize_for_save(img).copy()
    overlay[invalid_mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return overlay


def save_visualizations(
    save_dir: str,
    metric: CameraMetric,
    baseline_rgb: np.ndarray,
    optimized_rgb: np.ndarray,
    image_format: str,
) -> None:
    frame_dir = os.path.join(save_dir, f"frame_{metric.index:04d}_cam_{metric.camera_id:04d}")
    ensure_dir(frame_dir)

    baseline_invalid = ~np.all(np.isfinite(baseline_rgb), axis=2)
    optimized_invalid = ~np.all(np.isfinite(optimized_rgb), axis=2)

    baseline_clean = sanitize_for_save(baseline_rgb)
    optimized_clean = sanitize_for_save(optimized_rgb)
    diff_heat = build_diff_heatmap(baseline_rgb, optimized_rgb)
    side_by_side = np.concatenate([baseline_clean, optimized_clean], axis=1)
    baseline_overlay = build_invalid_overlay(baseline_rgb, baseline_invalid)
    optimized_overlay = build_invalid_overlay(optimized_rgb, optimized_invalid)

    write_rgb_image(os.path.join(frame_dir, "baseline"), to_uint8_rgb(baseline_clean), image_format)
    write_rgb_image(os.path.join(frame_dir, "optimized"), to_uint8_rgb(optimized_clean), image_format)
    write_rgb_image(os.path.join(frame_dir, "side_by_side"), to_uint8_rgb(side_by_side), image_format)
    write_rgb_image(os.path.join(frame_dir, "diff_heatmap"), to_uint8_rgb(diff_heat), image_format)
    write_rgb_image(os.path.join(frame_dir, "baseline_invalid_overlay"), to_uint8_rgb(baseline_overlay), image_format)
    write_rgb_image(os.path.join(frame_dir, "optimized_invalid_overlay"), to_uint8_rgb(optimized_overlay), image_format)
    write_gray_image(os.path.join(frame_dir, "baseline_invalid_mask"), baseline_invalid.astype(np.uint8) * 255, image_format)
    write_gray_image(os.path.join(frame_dir, "optimized_invalid_mask"), optimized_invalid.astype(np.uint8) * 255, image_format)


def copy_rgb_image(img_ptr: int, cam: Any) -> np.ndarray:
    if hasattr(optisplat, "copyToHost"):
        rgba = optisplat.copyToHost(img_ptr, cam.height, cam.width, 4)
    else:
        if cuda_copy_to_host is None:
            raise RuntimeError("Neither optisplat.copyToHost nor libcudart-based fallback is available.")
        rgba = cuda_copy_to_host(img_ptr, cam.height, cam.width, 4)
    if rgba is None:
        raise RuntimeError("copyToHost returned None for rendered image.")
    rgb = np.asarray(rgba[..., :3], dtype=np.float32)
    return rgb


def psnr_from_images(ref: np.ndarray, test: np.ndarray) -> (float, float, float, float):
    valid_mask = np.isfinite(ref) & np.isfinite(test)
    invalid_pixel_ratio = 1.0 - float(np.mean(valid_mask))
    if not np.any(valid_mask):
        return float("nan"), float("nan"), 1.0, float("nan")

    ref_valid = ref[valid_mask].astype(np.float32)
    test_valid = test[valid_mask].astype(np.float32)
    diff = ref_valid - test_valid
    max_abs_diff = float(np.max(np.abs(diff))) if diff.size > 0 else 0.0
    mse = float(np.mean(diff * diff))
    if not math.isfinite(mse):
        return float("nan"), float("nan"), invalid_pixel_ratio, max_abs_diff
    if mse <= 1e-12:
        return float("inf"), 0.0, invalid_pixel_ratio, max_abs_diff
    psnr = 10.0 * math.log10(1.0 / mse)
    return psnr, mse, invalid_pixel_ratio, max_abs_diff


def prepare_cameras(camera_path: str, max_cameras: int, stride: int, resolution: Optional[List[int]]) -> List[Any]:
    cameras = optisplat.readCamerasFromJson(camera_path)
    if not cameras:
        raise RuntimeError(f"Failed to load cameras from {camera_path}")

    if resolution is not None:
        width, height = resolution
        for cam in cameras:
            cam.setResolution(int(width), int(height))

    if stride > 1:
        cameras = cameras[::stride]
    if max_cameras > 0:
        cameras = cameras[:max_cameras]
    return cameras


def apply_config_flags(config: Any, args: argparse.Namespace, prefix: str) -> None:
    active_set_mode = optisplat.ExactActiveSetMode.DISABLED
    if bool(getattr(args, f"{prefix}_use_center_only_active_set")):
        active_set_mode = optisplat.ExactActiveSetMode.CENTER_ONLY
    elif bool(getattr(args, f"{prefix}_use_exact_active_set")):
        active_set_mode = optisplat.ExactActiveSetMode.PRECISE

    config.bKeepCpuSceneData = bool(getattr(args, f"{prefix}_keep_cpu_scene_data"))
    config.bUseHalfPrecisionSH = bool(getattr(args, f"{prefix}_use_half_sh"))
    config.bUseHalfPrecisionCov3DOpacity = bool(getattr(args, f"{prefix}_use_half_cov_opacity"))
    config.bUseFlashGSExactIntersection = bool(getattr(args, f"{prefix}_use_exact"))
    config.exactActiveSetMode = active_set_mode
    config.bUseFlashGSPrefetchingPipeline = bool(getattr(args, f"{prefix}_use_prefetch"))
    config.bUseTensorCore = bool(getattr(args, f"{prefix}_use_tensor_core"))
    config.maxNumRenderedGaussians = int(getattr(args, f"{prefix}_max_num_rendered"))


def build_config(args: argparse.Namespace, optimized: bool) -> Any:
    config = optisplat.GsConfig()
    config.modelPath = args.model
    config.cameraPath = args.cameras
    config.bRebuildBinaryCache = args.rebuild_cache
    apply_config_flags(config, args, "optimized" if optimized else "baseline")

    return config


def warmup_renderer(renderer: Any, cameras: List[Any], warmup: int, debug: bool) -> None:
    if warmup <= 0 or not cameras:
        return
    warmup_cam = cameras[0]
    for _ in range(warmup):
        renderer.render(warmup_cam, debug)


def summarize_metrics(metrics: List[CameraMetric]) -> Dict[str, Any]:
    psnrs = np.array([m.psnr for m in metrics], dtype=np.float64)
    finite_mask = np.isfinite(psnrs)
    inf_mask = np.isinf(psnrs)
    nan_mask = np.isnan(psnrs)
    finite_psnrs = psnrs[finite_mask & ~inf_mask]
    if finite_psnrs.size == 0:
        avg_psnr = float("nan")
        min_psnr = float("nan")
        max_psnr = float("nan")
    else:
        avg_psnr = float(np.mean(finite_psnrs))
        min_psnr = float(np.min(finite_psnrs))
        max_psnr = float(np.max(finite_psnrs))

    finite_mses = [m.mse for m in metrics if math.isfinite(m.mse)]
    mean_mse = float(np.mean(finite_mses)) if finite_mses else float("nan")
    mean_invalid_pixel_ratio = float(np.mean([m.invalid_pixel_ratio for m in metrics])) if metrics else 0.0
    finite_max_abs_diffs = [m.max_abs_diff for m in metrics if math.isfinite(m.max_abs_diff)]
    mean_max_abs_diff = float(np.mean(finite_max_abs_diffs)) if finite_max_abs_diffs else float("nan")
    max_max_abs_diff = float(np.max(finite_max_abs_diffs)) if finite_max_abs_diffs else float("nan")
    near_black_identical_count = int(np.sum([
        math.isinf(m.psnr)
        and m.baseline_max_value <= 1e-6
        and m.optimized_max_value <= 1e-6
        for m in metrics
    ]))
    return {
        "num_cameras": len(metrics),
        "avg_psnr": avg_psnr,
        "min_psnr": min_psnr,
        "max_psnr": max_psnr,
        "infinite_psnr_count": int(np.sum(inf_mask)),
        "nan_psnr_count": int(np.sum(nan_mask)),
        "mean_mse": mean_mse,
        "mean_invalid_pixel_ratio": mean_invalid_pixel_ratio,
        "mean_max_abs_diff": mean_max_abs_diff,
        "max_max_abs_diff": max_max_abs_diff,
        "near_black_identical_count": near_black_identical_count,
    }


def active_set_mode_name(use_exact_active_set: bool, use_center_only_active_set: bool) -> str:
    if use_center_only_active_set:
        return "center_only"
    if use_exact_active_set:
        return "precise"
    return "disabled"


def render_and_compare(
    baseline_renderer: Any,
    optimized_renderer: Any,
    cameras: List[Any],
    debug: bool,
    save_visual_dir: str,
    save_visuals_max: int,
    save_only_invalid_frames: bool,
    image_format: str,
) -> List[CameraMetric]:
    metrics: List[CameraMetric] = []
    reported_invalid_input = False
    saved_visuals = 0
    pbar = tqdm(enumerate(cameras), total=len(cameras), desc="Comparing PSNR")
    for idx, cam in pbar:
        _, base_img_ptr, _ = baseline_renderer.render(cam, debug)
        _, opt_img_ptr, _ = optimized_renderer.render(cam, debug)

        base_rgb = copy_rgb_image(base_img_ptr, cam)
        opt_rgb = copy_rgb_image(opt_img_ptr, cam)
        if not reported_invalid_input:
            base_invalid_ratio = 1.0 - float(np.mean(np.isfinite(base_rgb)))
            opt_invalid_ratio = 1.0 - float(np.mean(np.isfinite(opt_rgb)))
            if base_invalid_ratio > 0.0 or opt_invalid_ratio > 0.0:
                print(
                    f"[WARN] First compared frame contains non-finite pixels: "
                    f"baseline={base_invalid_ratio:.6f}, optimized={opt_invalid_ratio:.6f}"
                )
            reported_invalid_input = True
        psnr, mse, invalid_pixel_ratio, max_abs_diff = psnr_from_images(base_rgb, opt_rgb)
        baseline_mean_luma = float(np.mean(np.clip(np.nan_to_num(base_rgb, nan=0.0), 0.0, 1.0)))
        optimized_mean_luma = float(np.mean(np.clip(np.nan_to_num(opt_rgb, nan=0.0), 0.0, 1.0)))
        baseline_max_value = float(np.max(np.nan_to_num(base_rgb, nan=0.0))) if base_rgb.size else 0.0
        optimized_max_value = float(np.max(np.nan_to_num(opt_rgb, nan=0.0))) if opt_rgb.size else 0.0

        metric = CameraMetric(
            index=idx,
            camera_id=int(getattr(cam, "cameraId", idx)),
            width=int(cam.width),
            height=int(cam.height),
            psnr=psnr,
            mse=mse,
            invalid_pixel_ratio=invalid_pixel_ratio,
            max_abs_diff=max_abs_diff,
            baseline_mean_luma=baseline_mean_luma,
            optimized_mean_luma=optimized_mean_luma,
            baseline_max_value=baseline_max_value,
            optimized_max_value=optimized_max_value,
        )
        metrics.append(metric)

        if save_visual_dir:
            should_save = save_visuals_max <= 0 or saved_visuals < save_visuals_max
            if should_save and (not save_only_invalid_frames or invalid_pixel_ratio > 0.0):
                save_visualizations(save_visual_dir, metric, base_rgb, opt_rgb, image_format)
                saved_visuals += 1

        if math.isnan(psnr):
            psnr_text = "nan"
        elif math.isinf(psnr):
            psnr_text = "inf"
        else:
            psnr_text = f"{psnr:.3f}"
        pbar.set_description(f"Comparing PSNR [last={psnr_text} dB]")
    if save_visual_dir:
        print(f"[INFO] Saved {saved_visuals} visualization frame(s) to {save_visual_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline OptiSplat rendering against an optimized configuration and report PSNR."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/e/Dataset/GaussianSplattingModels/bicycle/point_cloud/iteration_30000/point_cloud.ply",
        help="模型路径 (.ply 或 .sog)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="/mnt/e/Dataset/GaussianSplattingModels/bicycle/cameras.json",
        help="相机 JSON 路径",
    )
    parser.add_argument("--warmup", type=int, default=1, help="每个 renderer 的预热帧数")
    parser.add_argument("--max-cameras", type=int, default=0, help="最多比较多少个相机，0 表示全部")
    parser.add_argument("--stride", type=int, default=1, help="相机采样步长，例如 2 表示每隔一个相机取一帧")
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="可选，统一把所有相机渲染到指定分辨率",
    )
    parser.add_argument("--debug", action="store_true", help="是否以 debug 模式调用 render")
    parser.add_argument("--rebuild-cache", action="store_true", help="是否重建 SOG cache")
    parser.add_argument("--output-json", type=str, default="", help="可选，把结果保存成 JSON")
    parser.add_argument("--save-visual-dir", type=str, default="", help="可选，保存每帧可视化对比到该目录（PPM/PGM）")
    parser.add_argument("--save-visuals-max", type=int, default=0, help="最多保存多少帧可视化，0 表示全部")
    parser.add_argument("--save-only-invalid-frames", action="store_true", help="只保存包含 NaN/Inf 像素的帧")
    parser.add_argument("--save-image-format", choices=["auto", "png", "ppm"], default="auto", help="保存图片格式，auto 会优先用 png，Pillow 不可用时回退到 ppm/pgm")

    parser.add_argument("--baseline-max-num-rendered", type=int, default=200000000, help="baseline 配置下的实例容量上限")
    parser.add_argument("--baseline-keep-cpu-scene-data", action="store_true", help="baseline 配置下保留 CPU 场景数据")
    parser.add_argument("--baseline-use-half-sh", action="store_true", help="baseline 配置：开启 half SH")
    parser.add_argument("--baseline-use-half-cov-opacity", action="store_true", help="baseline 配置：开启 half cov3D/opacity")
    parser.add_argument("--baseline-use-exact", action="store_true", help="baseline 配置：开启 exact intersection")
    parser.add_argument("--baseline-use-exact-active-set", action="store_true", help="baseline 配置：开启 exact active set")
    parser.add_argument("--baseline-use-center-only-active-set", action="store_true", help="baseline 配置：开启 center-only active set")
    parser.add_argument("--baseline-use-prefetch", action="store_true", help="baseline 配置：开启 prefetching pipeline")
    parser.add_argument("--baseline-use-tensor-core", action="store_true", help="baseline 配置：开启 tensor core path")

    parser.add_argument("--optimized-max-num-rendered", type=int, default=200000000, help="优化配置下的实例容量上限")
    parser.add_argument("--optimized-keep-cpu-scene-data", action="store_true", help="优化配置下保留 CPU 场景数据")
    parser.add_argument("--optimized-use-half-sh", action="store_true", help="优化配置：开启 half SH")
    parser.add_argument("--optimized-use-half-cov-opacity", action="store_true", help="优化配置：开启 half cov3D/opacity")
    parser.add_argument("--optimized-use-exact", action="store_true", help="优化配置：开启 exact intersection")
    parser.add_argument("--optimized-use-exact-active-set", action="store_true", help="优化配置：开启 exact active set")
    parser.add_argument("--optimized-use-center-only-active-set", action="store_true", help="优化配置：开启 center-only active set")
    parser.add_argument("--optimized-use-prefetch", action="store_true", help="优化配置：开启 prefetching pipeline")
    parser.add_argument("--optimized-use-tensor-core", action="store_true", help="优化配置：开启 tensor core path")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model file does not exist: {args.model}")
    if not os.path.isfile(args.cameras):
        raise FileNotFoundError(f"Camera file does not exist: {args.cameras}")
    if args.stride <= 0:
        raise ValueError("--stride must be >= 1")
    if args.max_cameras < 0:
        raise ValueError("--max-cameras must be >= 0")

    cameras = prepare_cameras(args.cameras, args.max_cameras, args.stride, args.resolution)
    if not cameras:
        raise RuntimeError("No cameras selected for comparison.")

    baseline_config = build_config(args, optimized=False)
    optimized_config = build_config(args, optimized=True)

    print("Creating baseline renderer...")
    baseline_renderer = optisplat.IGaussianRender.CreateRenderer(baseline_config)
    print("Creating optimized renderer...")
    optimized_renderer = optisplat.IGaussianRender.CreateRenderer(optimized_config)

    warmup_renderer(baseline_renderer, cameras, args.warmup, args.debug)
    warmup_renderer(optimized_renderer, cameras, args.warmup, args.debug)

    metrics = render_and_compare(
        baseline_renderer,
        optimized_renderer,
        cameras,
        args.debug,
        args.save_visual_dir,
        args.save_visuals_max,
        args.save_only_invalid_frames,
        "png" if args.save_image_format == "png" or (args.save_image_format == "auto" and Image is not None) else "ppm",
    )
    summary = summarize_metrics(metrics)

    print("\nBaseline config:")
    print(
        "  exact={exact}, active_set_mode={mode}, prefetch={prefetch}, tensor_core={tc}, half_sh={half_sh}, half_cov_opacity={half_cov}".format(
            exact=args.baseline_use_exact,
            mode=active_set_mode_name(args.baseline_use_exact_active_set, args.baseline_use_center_only_active_set),
            prefetch=args.baseline_use_prefetch,
            tc=args.baseline_use_tensor_core,
            half_sh=args.baseline_use_half_sh,
            half_cov=args.baseline_use_half_cov_opacity,
        )
    )
    print("Optimized config:")
    print(
        "  exact={exact}, active_set_mode={mode}, prefetch={prefetch}, tensor_core={tc}, half_sh={half_sh}, half_cov_opacity={half_cov}".format(
            exact=args.optimized_use_exact,
            mode=active_set_mode_name(args.optimized_use_exact_active_set, args.optimized_use_center_only_active_set),
            prefetch=args.optimized_use_prefetch,
            tc=args.optimized_use_tensor_core,
            half_sh=args.optimized_use_half_sh,
            half_cov=args.optimized_use_half_cov_opacity,
        )
    )

    print("\nPSNR summary:")
    print(f"  Cameras compared: {summary['num_cameras']}")
    print(f"  Average PSNR: {summary['avg_psnr']:.4f} dB" if math.isfinite(summary["avg_psnr"]) else "  Average PSNR: nan")
    print(f"  Min PSNR:     {summary['min_psnr']:.4f} dB" if math.isfinite(summary["min_psnr"]) else "  Min PSNR: nan")
    print(f"  Max PSNR:     {summary['max_psnr']:.4f} dB" if math.isfinite(summary["max_psnr"]) else "  Max PSNR: nan")
    print(f"  Mean MSE:     {summary['mean_mse']:.8e}" if math.isfinite(summary["mean_mse"]) else "  Mean MSE:     nan")
    print(f"  Identical frames: {summary['infinite_psnr_count']}")
    print(f"  Invalid frames:   {summary['nan_psnr_count']}")
    print(f"  Mean invalid pixel ratio: {summary['mean_invalid_pixel_ratio']:.6f}")
    print(f"  Mean max abs diff: {summary['mean_max_abs_diff']:.8e}" if math.isfinite(summary["mean_max_abs_diff"]) else "  Mean max abs diff: nan")
    print(f"  Max max abs diff:  {summary['max_max_abs_diff']:.8e}" if math.isfinite(summary["max_max_abs_diff"]) else "  Max max abs diff:  nan")
    print(f"  Near-black identical frames: {summary['near_black_identical_count']}")

    finite_metrics = [m for m in metrics if math.isfinite(m.psnr) and not math.isinf(m.psnr)]
    if finite_metrics:
        worst = min(finite_metrics, key=lambda m: m.psnr)
        print(
            f"  Worst camera: idx={worst.index}, camera_id={worst.camera_id}, "
            f"resolution={worst.width}x{worst.height}, psnr={worst.psnr:.4f} dB"
        )

    if args.output_json:
        payload = {
            "model": args.model,
            "cameras": args.cameras,
            "baseline": {
                "bUseFlashGSExactIntersection": args.baseline_use_exact,
                "exactActiveSetMode": active_set_mode_name(args.baseline_use_exact_active_set, args.baseline_use_center_only_active_set),
                "bUseFlashGSPrefetchingPipeline": args.baseline_use_prefetch,
                "bUseTensorCore": args.baseline_use_tensor_core,
                "bUseHalfPrecisionSH": args.baseline_use_half_sh,
                "bUseHalfPrecisionCov3DOpacity": args.baseline_use_half_cov_opacity,
                "maxNumRenderedGaussians": args.baseline_max_num_rendered,
            },
            "optimized": {
                "bUseFlashGSExactIntersection": args.optimized_use_exact,
                "exactActiveSetMode": active_set_mode_name(args.optimized_use_exact_active_set, args.optimized_use_center_only_active_set),
                "bUseFlashGSPrefetchingPipeline": args.optimized_use_prefetch,
                "bUseTensorCore": args.optimized_use_tensor_core,
                "bUseHalfPrecisionSH": args.optimized_use_half_sh,
                "bUseHalfPrecisionCov3DOpacity": args.optimized_use_half_cov_opacity,
                "maxNumRenderedGaussians": args.optimized_max_num_rendered,
            },
            "summary": summary,
            "per_camera": [asdict(m) for m in metrics],
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved JSON report to {args.output_json}")


if __name__ == "__main__":
    main()

'''
python3 metrics.py \
  --model /mnt/e/Dataset/GaussianSplattingModels/bicycle/point_cloud/iteration_30000/point_cloud.ply \
  --cameras /mnt/e/Dataset/GaussianSplattingModels/bicycle/cameras.json \
  --optimized-use-prefetch \
  --optimized-use-exact \
  --optimized-use-exact-active-set \
  --optimized-use-center-only-active-set \
  --optimized-use-half-sh \
  --optimized-use-half-cov-opacity \
  --output-json output/metrics_result.json \
  --save-visual-dir output/metrics_result \
  --debug --max-cameras 12
'''
