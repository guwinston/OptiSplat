# optisplat/__init__.py
import os
import sys
import platform

if sys.platform == "win32":
    cuda_path = os.environ.get("CUDA_PATH")
    found_cuda = False
    if cuda_path:
        bin_path = os.path.join(cuda_path, "bin")
        if os.path.exists(bin_path):
            os.add_dll_directory(bin_path)
            found_cuda = True
        else:
            print(f"[OptiSplat] WARNING: CUDA_PATH is set to '{cuda_path}', but '{bin_path}' does not exist.")
    if not found_cuda:
        default_cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.exists(default_cuda_root):
            versions = os.listdir(default_cuda_root)
            if versions:
                latest_version = sorted(versions)[-1]
                bin_path = os.path.join(default_cuda_root, latest_version, "bin")
                if os.path.exists(bin_path):
                    os.add_dll_directory(bin_path)
                    print(f"[OptiSplat] Found CUDA at default location: {bin_path}")
                    found_cuda = True

    if not found_cuda:
        print("[OptiSplat] ERROR: CUDA Runtime not found. Please ensure CUDA Toolkit is installed and CUDA_PATH environment variable is set.")

try:
    from . import _C
    from ._C import (
        GsCamera, 
        CameraModel, 
        CameraCoordSystem, 
        IGaussianRender, 
        GsConfig, 
        ExactActiveSetMode,
        readCamerasFromJson, 
        runViewer, 
        copyToHost
    )
except ImportError as e:
    print("\n" + "="*50)
    print(f"[OptiSplat] CRITICAL ERROR: Failed to load native module '_C'")
    print(f"Error Detail: {e}")
    print("-"*50)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Current Directory: {os.path.dirname(__file__)}")
    print(f"Directory Contents: {os.listdir(os.path.dirname(__file__))}")
    
    if sys.platform == "win32" and "DLL load failed" in str(e):
        print("\n[Diagnosis Tip]:")
        print("This usually means a dependency (.dll) is missing. Possible causes:")
        print("1. CUDA Runtime DLLs (cudart64_XX.dll) are not in your PATH or added via os.add_dll_directory.")
        print("2. Your OptiSplat C++ core was built as a DLL but the .dll file is not next to the .pyd file.")
        print("3. Visual C++ Redistributable is missing.")
        print("\nTry running: 'dumpbin /dependents _C.cp39-win_amd64.pyd' or use 'Dependencies' GUI tool.")
    
    print("="*50 + "\n")
    raise e

__all__ = [
    "GsCamera", 
    "CameraModel", 
    "CameraCoordSystem", 
    "IGaussianRender", 
    "GsConfig", 
    "ExactActiveSetMode",
    "readCamerasFromJson", 
    "runViewer", 
    "copyToHost"
]