#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h> // 必须包含，处理 Eigen 类型
#include "camera.h"
#include "render.h"
#include "utils.h"
#include "viewer.h"

namespace py = pybind11;
using namespace optisplat;

py::array_t<float> copyToHost(uintptr_t ptr, int h, int w, int c) {
    auto result = py::array_t<float>({h, w, c});
    cudaMemcpy(result.mutable_data(), (void*)ptr, h * w * c * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

PYBIND11_MODULE(_C, m) {
    // 导出 GsConfig 结构体
    py::class_<GsConfig>(m, "GsConfig")
        .def(py::init<>()) // 导出默认构造函数
        .def_readwrite("modelPath", &GsConfig::modelPath)
        .def_readwrite("cameraPath", &GsConfig::cameraPath)
        .def_readwrite("bRebuildBinaryCache", &GsConfig::bRebuildBinaryCache)
        .def_readwrite("maxNumRenderedGaussians", &GsConfig::maxNumRenderedGaussians)
        .def_readwrite("bUseFlashGSExactIntersection", &GsConfig::bUseFlashGSExactIntersection)
        .def_readwrite("bUseFlashGSPrefetchingPipeline", &GsConfig::bUseFlashGSPrefetchingPipeline)
        .def_readwrite("bUseTensorCore", &GsConfig::bUseTensorCore);
    // 绑定枚举
    py::enum_<CameraModel>(m, "CameraModel")
        .value("PINHOLE", CameraModel::PINHOLE)
        .value("FISHEYE", CameraModel::FISHEYE)
        .value("ORTHOGRAPHIC", CameraModel::ORTHOGRAPHIC)
        .export_values();

    py::enum_<CameraCoordSystem>(m, "CameraCoordSystem")
        .value("COLMAP", CameraCoordSystem::COLMAP)
        .value("SIBR", CameraCoordSystem::SIBR)
        .value("UNREAL", CameraCoordSystem::UNREAL)
        .export_values();

    // 绑定 GsCamera 结构体
    py::class_<GsCamera>(m, "GsCamera")
        .def(py::init<>())
        .def_readwrite("model", &GsCamera::model)
        .def_readwrite("coordSystem", &GsCamera::coordSystem)
        .def_readwrite("position", &GsCamera::position)
        .def_property("quaternion",
            // Getter: 将四元数转为 Python 列表 [w, x, y, z]
            [](const GsCamera &self) {
                return std::vector<float>{self.quaternion.w(), self.quaternion.x(), self.quaternion.y(), self.quaternion.z()};
            },
            // Setter: 接受 NumPy 数组或列表并构造四元数
            [](GsCamera &self, py::array_t<float> q) {
                auto r = q.unchecked<1>();
                if (r.size() != 4) 
                    throw std::runtime_error("Quaternion must have 4 elements (w, x, y, z)");
                self.quaternion = Eigen::Quaternionf(r(0), r(1), r(2), r(3));
            }
        )
        .def_readwrite("width", &GsCamera::width)
        .def_readwrite("height", &GsCamera::height)
        .def_readwrite("fx", &GsCamera::fx)
        .def_readwrite("fy", &GsCamera::fy)
        .def_readwrite("cx", &GsCamera::cx)
        .def_readwrite("cy", &GsCamera::cy)
        .def_readwrite("znear", &GsCamera::znear)
        .def_readwrite("zfar", &GsCamera::zfar)
        .def_readwrite("k1", &GsCamera::k1)
        .def_readwrite("k2", &GsCamera::k2)
        .def_readwrite("k3", &GsCamera::k3)
        .def_readwrite("k4", &GsCamera::k4)
        .def("setResolution", &GsCamera::setResolution)
        .def("rescaleResolution", &GsCamera::rescaleResolution)
        .def("getProjectionMatrix", &GsCamera::getProjectionMatrix);

    // 绑定渲染器基类
    py::class_<IGaussianRender, std::shared_ptr<IGaussianRender>>(m, "IGaussianRender")
        .def_static("CreateRenderer", &IGaussianRender::CreateRenderer)
        .def("render", [](IGaussianRender& self, GsCamera& cam, bool debug) {
            float* outImage = nullptr;
            float* outAllMap = nullptr;
            float num = self.render(cam, outImage, outAllMap, debug);
            // 关键：将 CUDA 指针转为 uintptr_t 返回给 Python
            return py::make_tuple(num, reinterpret_cast<uintptr_t>(outImage), reinterpret_cast<uintptr_t>(outAllMap));
        }, py::arg("camera"), py::arg("debug") = false);

    // 绑定辅助函数
    m.def("fov2focal", &fov2focal);
    m.def("focal2fov", &focal2fov);
    m.def("readCamerasFromJson", &Utils::readCamerasFromJson, "Read cameras from a JSON file");
    m.def("runViewer", &runViewer, "run Guassian Splatting Viewer");
    m.def("copyToHost", &copyToHost);
}