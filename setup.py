import os
import sys
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # 这里的参数要和你在命令行 cmake .. 时传的一致
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release", # 默认 Release 提高渲染性能
        ]

        build_args = []
        if sys.platform.startswith("win"):
            cmake_args += ["-A", "x64"]
            build_args += ["--config", "Release"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--target", "_C"] + build_args, cwd=self.build_temp)

setup(
    name="optisplat",
    version="0.0.1",
    author="guwencong",
    package_dir={"": "src/optisplat"}, 
    packages=find_packages(where="src/optisplat"),
    ext_modules=[CMakeExtension("._C")], # 对应编译出的 _C.so 或 _C.pyd
    cmdclass={"build_ext": CMakeBuild},
    install_requires=["numpy", 'pybind11'],
    zip_safe=False,
)
