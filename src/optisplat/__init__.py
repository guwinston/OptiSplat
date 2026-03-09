# optisplat/__init__.py
import os
import sys

try:
    from . import _C
    from ._C import GsCamera, CameraModel, CameraCoordSystem, IGaussianRender, GsConfig, readCamerasFromJson, runViewer
except ImportError as e:
    import traceback
    print(f"Failed to load native module _C. Error: {e}")
    traceback.print_exc()
    print(f"Current directory content: {os.listdir(os.path.dirname(__file__))}")

__all__ = ["GsCamera", "CameraModel", "CameraCoordSystem", "IGaussianRender", "GsConfig", "readCamerasFromJson", "runViewer"]