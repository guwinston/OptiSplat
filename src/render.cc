#include "rasterizer.h"
#include "render.h"
#include "load.h"
#include "utils.h"
#include "camera.h"
#include "common.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <climits>
#include <limits>
#include <unordered_map>
#include <cuda_fp16.h>


namespace optisplat {

template class GaussianModelRender<0>;
template class GaussianModelRender<1>;
template class GaussianModelRender<2>;
template class GaussianModelRender<3>;
template class GaussianSceneRender<0>;
template class GaussianSceneRender<1>;
template class GaussianSceneRender<2>;
template class GaussianSceneRender<3>;

template class SceneData<0>;
template class SceneData<1>;
template class SceneData<2>;
template class SceneData<3>;

std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S, bool debug = false) {
	auto lambda = [ptr, &S, debug](size_t N) {
		if (N > S)
		{	
			if (*ptr) CHECK_CUDA(cudaFree(*ptr), debug);
			const size_t kMaxSlackBytes = 256ull * 1024ull * 1024ull;
			size_t newSize = N;
			if (S > 0) {
				const size_t extra = std::min(S, kMaxSlackBytes);
				if (S <= SIZE_MAX - extra) {
					newSize = std::max(N, S + extra);
				}
			}
			CHECK_CUDA(cudaMalloc(ptr, newSize), debug);
			CHECK_CUDA(cudaMemset(reinterpret_cast<char*>(*ptr), 0, newSize), debug);
			S = newSize;
		}
		return reinterpret_cast<char*>(*ptr);
	};
	return lambda;
}

int clampRenderedCapacity(int64_t requested, int limit) {
	if (requested <= 0) return 0;
	int64_t capped = requested;
	if (limit > 0) capped = std::min<int64_t>(capped, limit);
	capped = std::min<int64_t>(capped, INT_MAX);
	return static_cast<int>(capped);
}

int chooseInitialExactCapacity(int numPoints, int limit) {
	const int64_t baseline = std::max<int64_t>(1 << 20, std::min<int64_t>(static_cast<int64_t>(numPoints) * 2, 16LL << 20));
	return clampRenderedCapacity(baseline, limit);
}

bool isGaussianScenePath(const std::string& path) {
	return std::filesystem::path(path).extension() == ".json";
}

std::string buildDefaultCachePath(const std::string& modelPath, std::string& cacheName) {
	std::filesystem::path path(modelPath);
	if (std::filesystem::is_regular_file(path) && path.extension().string() == ".sog") {
		cacheName = path.filename().string();
		return path.string();
	}
	if (std::filesystem::is_regular_file(path)) {
		cacheName = path.stem().string() + ".sog";
		return (path.parent_path() / ".cache" / cacheName).string();
	}
	cacheName = "gaussian_splatting.sog";
	return (path / ".cache" / cacheName).string();
}

void setCudaImageBuffers(float*& cudaImage, size_t& allocatedCudaImage,
	float*& cudaAllMap, size_t& allocatedCudaAllMap,
	const GsCamera& camera, cudaStream_t stream, bool debug)
{
	const int NUM_IMAGE_CHANNELS = 4;
	const int NUM_ALLMAP_CHANNELS = 1;
	resizeFunctional((void**)&cudaImage, allocatedCudaImage)(camera.width * camera.height * NUM_IMAGE_CHANNELS * sizeof(float));
	resizeFunctional((void**)&cudaAllMap, allocatedCudaAllMap)(camera.width * camera.height * NUM_ALLMAP_CHANNELS * sizeof(float));
	CHECK_CUDA(cudaMemsetAsync(cudaImage, 0, camera.width * camera.height * NUM_IMAGE_CHANNELS * sizeof(float), stream), debug);
	CHECK_CUDA(cudaMemsetAsync(cudaAllMap, 0, camera.width * camera.height * NUM_ALLMAP_CHANNELS * sizeof(float), stream), debug);
	cudaStreamSynchronize(stream);
}

GsCamera transformCameraToInstanceLocal(const GsCamera& worldCamera, const GaussianInstanceDesc& transform) {
	GsCamera localCamera = worldCamera;
	const Eigen::Quaternionf invRotation = transform.rotation.normalized().conjugate();
	localCamera.position = (invRotation * (worldCamera.position - transform.translation)) / transform.scale;
	localCamera.quaternion = (invRotation * worldCamera.quaternion).normalized();
	localCamera.znear = worldCamera.znear / transform.scale;
	localCamera.zfar = worldCamera.zfar / transform.scale;
	localCamera.bgColor = Eigen::Vector3f::Zero();
	return localCamera;
}

std::vector<float> matrix4fToColumnMajorVector(const Eigen::Matrix4f& matrix) {
	return std::vector<float>(matrix.data(), matrix.data() + 16);
}

template <int D>
std::vector<uint16_t> packSHsToHalf(const std::vector<SHs<D>>& shs);

void expandWorldBounds(const Eigen::Vector3f& localMin, const Eigen::Vector3f& localMax,
	const GaussianInstanceDesc& transform,
	Eigen::Vector3f& worldMin, Eigen::Vector3f& worldMax)
{
	for (int mask = 0; mask < 8; ++mask) {
		Eigen::Vector3f corner(
			(mask & 1) ? localMax.x() : localMin.x(),
			(mask & 2) ? localMax.y() : localMin.y(),
			(mask & 4) ? localMax.z() : localMin.z());
		const Eigen::Vector3f worldCorner = transform.translation + transform.rotation * (corner * transform.scale);
		worldMin = worldMin.cwiseMin(worldCorner);
		worldMax = worldMax.cwiseMax(worldCorner);
	}
}

template <int D>
void appendLoadResult(SceneData<D>& dst, LoadResult<D>&& src)
{
	const int baseCount = dst.numPoints;
	const int appendCount = src.numPoints;
	if (appendCount <= 0) {
		return;
	}

	dst.gaussianPoints.insert(dst.gaussianPoints.end(),
		std::make_move_iterator(src.points.begin()),
		std::make_move_iterator(src.points.end()));

	const bool dstUsesHalfSH = !dst.gaussianSHsHalfHost.empty();
	const bool srcUsesHalfSH = !src.shsHalf.empty();
	if (dstUsesHalfSH || srcUsesHalfSH) {
		if (!dstUsesHalfSH && !dst.gaussianSHs.empty()) {
			dst.gaussianSHsHalfHost = packSHsToHalf<D>(dst.gaussianSHs);
			std::vector<SHs<D>>().swap(dst.gaussianSHs);
		}
		if (!srcUsesHalfSH && !src.shs.empty()) {
			src.shsHalf = packSHsToHalf<D>(src.shs);
		}
		dst.gaussianSHsHalfHost.insert(dst.gaussianSHsHalfHost.end(),
			std::make_move_iterator(src.shsHalf.begin()),
			std::make_move_iterator(src.shsHalf.end()));
	} else {
		dst.gaussianSHs.insert(dst.gaussianSHs.end(),
			std::make_move_iterator(src.shs.begin()),
			std::make_move_iterator(src.shs.end()));
	}
	dst.gaussianOpacity.insert(dst.gaussianOpacity.end(),
		std::make_move_iterator(src.opacity.begin()),
		std::make_move_iterator(src.opacity.end()));
	dst.gaussianScale.insert(dst.gaussianScale.end(),
		std::make_move_iterator(src.scale.begin()),
		std::make_move_iterator(src.scale.end()));
	dst.gaussianRot.insert(dst.gaussianRot.end(),
		std::make_move_iterator(src.rot.begin()),
		std::make_move_iterator(src.rot.end()));

	dst.numPoints = baseCount + appendCount;
	dst.sceneMin = dst.sceneMin.cwiseMin(src.sceneMin);
	dst.sceneMax = dst.sceneMax.cwiseMax(src.sceneMax);
}

template <int D>
void moveLoadResult(SceneData<D>& dst, LoadResult<D>&& src)
{
	dst.gaussianPoints = std::move(src.points);
	dst.gaussianSHs = std::move(src.shs);
	dst.gaussianSHsHalfHost = std::move(src.shsHalf);
	dst.gaussianOpacity = std::move(src.opacity);
	dst.gaussianScale = std::move(src.scale);
	dst.gaussianRot = std::move(src.rot);
	dst.numPoints = src.numPoints;
	dst.sceneMin = src.sceneMin;
	dst.sceneMax = src.sceneMax;
}

template <int D>
int renderSceneDataInternal(
	const GsConfig& config,
	const int renderPointCount,
	SceneData<D>& sceneData,
	const uint32_t* sourceIndices,
	const uint32_t* instanceIndices,
	const float* instanceCamPositions,
	const float* instanceViewMatrices,
	const float* instanceProjMatrices,
	const float* instanceDepthScales,
	void*& cudaGeometryState, size_t& allocatedGeometryState,
	void*& cudaActiveState, size_t& allocatedActiveState,
	void*& cudaBinningState, size_t& allocatedBinningState,
	void*& cudaImageState, size_t& allocatedImageState,
	int& allocatedRenderedCapacity, int capacityLimit,
	int* cudaCurrOffset, int* cudaExactOverflow,
	int* cudaRadii,
	const GsCamera& inCamera,
	float* outImage, float* outAllMap,
	int* activeGaussiansOut,
	bool debug)
{
	const int P = renderPointCount;
	const int M = (D + 1) * (D + 1);
	const int frameCapacity = config.bUseFlashGSExactIntersection ? allocatedRenderedCapacity : config.maxNumRenderedGaussians;
	int resolvedCapacity = frameCapacity;
	int activeGaussians = -1;
	const float* gaussianOpacity = sceneData.cudaGaussianOpacityHalf ? nullptr : sceneData.cudaGaussianOpacity;
	const uint16_t* gaussianOpacityHalf = sceneData.cudaGaussianOpacityHalf;
	const bool hasPrecomputedCov3D = sceneData.cudaGaussianCov3D != nullptr || sceneData.cudaGaussianCov3DHalf != nullptr;
	const float* gaussianScale = hasPrecomputedCov3D ? nullptr : sceneData.cudaGaussianScale;
	const float* gaussianRot = hasPrecomputedCov3D ? nullptr : sceneData.cudaGaussianRot;
	const float* gaussianCov3D = sceneData.cudaGaussianCov3DHalf ? nullptr : sceneData.cudaGaussianCov3D;
	const uint16_t* gaussianCov3DHalf = sceneData.cudaGaussianCov3DHalf;
	const float tanHalfFovx = inCamera.width / (2.0f * inCamera.fx);
	const float tanHalfFovy = inCamera.height / (2.0f * inCamera.fy);
	const bool isOrtho = inCamera.model == CameraModel::ORTHOGRAPHIC;
	const bool isFisheye = inCamera.model == CameraModel::FISHEYE;
	const float k1 = inCamera.k1, k2 = inCamera.k2, k3 = inCamera.k3, k4 = inCamera.k4;

	const Eigen::Vector3f cpuCamPosEigen = inCamera.position;
	const Eigen::Matrix3f cpuCamRotEigen = inCamera.quaternion.toRotationMatrix();
	const Eigen::Matrix4f cpuViewMatrixEigen = inCamera.getWorld2CameraMatrix();
	const Eigen::Matrix4f cpuProjMatrixEigen = inCamera.getProjectionMatrix();
	std::vector<float> cpuCamPos(cpuCamPosEigen.data(), cpuCamPosEigen.data() + 3);
	std::vector<float> cpuCamRot(cpuCamRotEigen.data(), cpuCamRotEigen.data() + 9);
	std::vector<float> cpuViewMatrix = matrix4fToColumnMajorVector(cpuViewMatrixEigen);
	std::vector<float> cpuProjMatrix = matrix4fToColumnMajorVector(cpuProjMatrixEigen);
	std::vector<float> cpuBackground = {inCamera.bgColor.x(), inCamera.bgColor.y(), inCamera.bgColor.z()};

	int numRendered = 0;
	CHECK_CUDA(
		numRendered = CudaRasterizer::Rasterizer::forward(
			resizeFunctional(&cudaGeometryState, allocatedGeometryState),
			resizeFunctional(&cudaActiveState, allocatedActiveState),
			resizeFunctional(&cudaBinningState, allocatedBinningState),
			resizeFunctional(&cudaImageState, allocatedImageState),
			P, D, M, frameCapacity, capacityLimit,
			config.bUseFlashGSExactIntersection, static_cast<int>(config.exactActiveSetMode), config.bUseFlashGSPrefetchingPipeline, config.bUseTensorCore,
			cpuCamPos, cpuCamRot, cpuViewMatrix, cpuProjMatrix, inCamera.znear, inCamera.zfar, cudaCurrOffset, cudaExactOverflow,
			isOrtho, isFisheye, k1, k2, k3, k4,
			cpuBackground,
			inCamera.width, inCamera.height,
			sceneData.cudaGaussianPoints,
			sceneData.cudaGaussianSHs,
			sceneData.cudaGaussianSHsHalf,
			nullptr,
			gaussianOpacity,
			gaussianOpacityHalf,
			gaussianScale,
			inCamera.scale,
			gaussianRot,
			gaussianCov3D,
			gaussianCov3DHalf,
			sourceIndices,
			instanceIndices,
			instanceCamPositions,
			instanceViewMatrices,
			instanceProjMatrices,
			instanceDepthScales,
			tanHalfFovx, tanHalfFovy,
			false,
			outImage,
			outAllMap,
			false,
			&resolvedCapacity,
			&activeGaussians,
			cudaRadii,
			debug), debug);
	if (activeGaussiansOut != nullptr) {
		*activeGaussiansOut = activeGaussians;
	}
	if (config.bUseFlashGSExactIntersection && resolvedCapacity > allocatedRenderedCapacity) {
		GS_INFO("Exact-intersection capacity grow: %d -> %d (num_rendered=%d)",
			allocatedRenderedCapacity, resolvedCapacity, numRendered);
		allocatedRenderedCapacity = resolvedCapacity;
	}
	return numRendered;
}

template <int D>
std::vector<uint16_t> packSHsToHalf(const std::vector<SHs<D>>& shs) {
	std::vector<uint16_t> packed;
	if (shs.empty()) return packed;

	const size_t numValues = shs.size() * shs[0].size();
	packed.resize(numValues);
	size_t dst = 0;
	for (const auto& coeffs : shs) {
		for (float value : coeffs) {
			const __half halfValue = __float2half(value);
			std::memcpy(&packed[dst], &halfValue, sizeof(uint16_t));
			++dst;
		}
	}
	return packed;
}

std::vector<uint16_t> packFloatsToHalf(const std::vector<float>& values) {
	std::vector<uint16_t> packed(values.size());
	for (size_t i = 0; i < values.size(); ++i) {
		const __half halfValue = __float2half(values[i]);
		std::memcpy(&packed[i], &halfValue, sizeof(uint16_t));
	}
	return packed;
}

void computeCov3D(const Eigen::Vector3f& scale, float mod, const Eigen::Vector4f& rot, float* cov3D)
{
    // Create scaling matrix
    Eigen::Matrix3f S = Eigen::Matrix3f::Identity();
    S(0,0) = mod * scale.x();
    S(1,1) = mod * scale.y();
    S(2,2) = mod * scale.z();

    // Normalize quaternion to get valid rotation
    Eigen::Vector4f q = rot; //
    float r = q.x();
    float x = q.y();
    float y = q.z();
    float z = q.w();

    // Compute rotation matrix from quaternion
    Eigen::Matrix3f R;
    R << 1.f - 2.f * (y*y + z*z), 2.f * (x*y - r*z), 2.f * (x*z + r*y),
         2.f * (x*y + r*z), 1.f - 2.f * (x*x + z*z), 2.f * (y*z - r*x),
         2.f * (x*z - r*y), 2.f * (y*z + r*x), 1.f - 2.f * (x*x + y*y);
	R.transposeInPlace();

    Eigen::Matrix3f M = S * R;

    // Compute 3D world covariance matrix Sigma
    Eigen::Matrix3f Sigma = M.transpose() * M;

    // Covariance is symmetric, only store upper right
    cov3D[0] = Sigma(0,0);
    cov3D[1] = Sigma(0,1);
    cov3D[2] = Sigma(0,2);
    cov3D[3] = Sigma(1,1);
    cov3D[4] = Sigma(1,2);
    cov3D[5] = Sigma(2,2);
}

void computeCov3Ds(const Eigen::Vector3f* scales, float mod, const Eigen::Vector4f* rots, float* cov3Ds, int num) {
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < num; i++) {
		computeCov3D(scales[i], mod, rots[i], cov3Ds + i * 6);
	}
}



std::shared_ptr<IGaussianRender> IGaussianRender::CreateRenderer(GsConfig config) {
	if (isGaussianScenePath(config.modelPath)) {
		const GaussianSceneDesc scene = Utils::readGaussianSceneFromJson(config.modelPath);
		int sceneDegree = -1;
		int totalPoints = 0;
		for (const auto& object : scene.objects) {
			int numVertex = 0;
			int shsDegree = 0;
			if (!LoaderFactory<3>::peekInfo(object.modelPath, &shsDegree, &numVertex)) {
				GS_ERROR("Cannot read model file info: %s", object.modelPath.c_str());
				return nullptr;
			}
			if (sceneDegree < 0) sceneDegree = shsDegree;
			if (sceneDegree != shsDegree) {
				throw std::runtime_error("All Gaussian scene objects must use the same SH degree.");
			}
			totalPoints += numVertex * static_cast<int>(object.instances.size());
		}
		GS_INFO("Gaussian scene sh degree = %d, total instantiated points = %d", sceneDegree, totalPoints);
		switch (sceneDegree)
		{
			case 0: return std::make_shared<GaussianSceneRender<0>>(config, scene);
			case 1: return std::make_shared<GaussianSceneRender<1>>(config, scene);
			case 2: return std::make_shared<GaussianSceneRender<2>>(config, scene);
			case 3: return std::make_shared<GaussianSceneRender<3>>(config, scene);
			default:
				throw std::runtime_error("Unsupported SH Degree: " + std::to_string(sceneDegree));
		}
	}

	int numVertex = 0;
	int shsDegree = 0;
	if (!LoaderFactory<3>::peekInfo(config.modelPath, &shsDegree, &numVertex)) {
		GS_ERROR("Cannot read model file info: %s", config.modelPath.c_str());
		return nullptr;
	}
	if (numVertex <= 0 || shsDegree < 0 || shsDegree > 3) {
		GS_ERROR("Invalid model file: %s", config.modelPath.c_str());
		return nullptr;
	}
	GS_INFO("Model sh degree = %d", shsDegree);

	std::shared_ptr<IGaussianRender> renderer = nullptr;
	switch (shsDegree)
	{
		case 0:
			renderer = std::make_shared<GaussianModelRender<0>>(config);
			break;
		case 1:
			renderer = std::make_shared<GaussianModelRender<1>>(config);
			break;
		case 2:
			renderer = std::make_shared<GaussianModelRender<2>>(config);
			break;
		case 3:
			renderer = std::make_shared<GaussianModelRender<3>>(config);
			break;
		default:
			GS_ERROR("Unsupported SH Degree: %d", shsDegree);
			throw std::runtime_error("Unsupported SH Degree: " + std::to_string(shsDegree));
	}
	return renderer;
}

template <int D>
GaussianModelRender<D>::GaussianModelRender(GsConfig config) {
	initCuda(0);

	std::filesystem::path modelPath = config.modelPath;
	this->config = config;

	std::filesystem::path sogPath;
	const bool inputIsSog = std::filesystem::is_regular_file(modelPath) &&
	                        modelPath.extension().string() == ".sog";
	if (inputIsSog) {
		this->sceneData.cacheName = modelPath.filename().string();
		sogPath = modelPath;
	} else if (std::filesystem::is_regular_file(modelPath)) {
		this->sceneData.cacheName = modelPath.stem().string() + ".sog";
		const std::filesystem::path cacheDir = modelPath.parent_path() / ".cache";
		sogPath = cacheDir / this->sceneData.cacheName;
	} else {
		this->sceneData.cacheName = "gaussian_splatting.sog";
		sogPath = modelPath / ".cache" / this->sceneData.cacheName;
	}

	sceneData.initResource(modelPath.string(), sogPath.string(), config.bRebuildBinaryCache, config.bKeepCpuSceneData, config.bUseHalfPrecisionSH, config.bUseHalfPrecisionCov3DOpacity);
	capacityLimit = config.maxNumRenderedGaussians > 0 ? config.maxNumRenderedGaussians : -1;
	allocatedRenderedCapacity = config.bUseFlashGSExactIntersection ?
		chooseInitialExactCapacity(sceneData.numPoints, capacityLimit) :
		config.maxNumRenderedGaussians;
	
	setDefaultCamera(90, 1920, 1080);
	setCudaAuxiliary();
}

template <int D>
GaussianModelRender<D>::~GaussianModelRender() {
	safeCudaFree(cudaImage);
	safeCudaFree(cudaAllMap);
	safeCudaFree(cudaGeometryState);
	safeCudaFree(cudaActiveState);
	safeCudaFree(cudaBinningState);
	safeCudaFree(cudaImageState);
	
	safeCudaFree(cudaBackground);
	safeCudaFree(cudaView);
	safeCudaFree(cudaProj);
	safeCudaFree(cudaCamPos);
	safeCudaFree(cudaRadii);
	safeCudaFree(cudaCurrOffset);
	safeCudaFree(cudaExactOverflow);
}

template <int D>
float GaussianModelRender<D>::render(GsCamera& inCamera, float*& outImage, float*& outAllMap, bool debug) {
	ScopeTimer timer("Render camera " + std::to_string(inCamera.cameraId));

	{
		ScopeTimer timer("set camera and image cuda");
		setCudaImageParams(inCamera, debug);
	}

	int activeGaussians = -1;
	ScopeTimer forwardTimer("forward");
	const int numRendered = renderSceneDataInternal(
		config,
		sceneData.numPoints,
		sceneData,
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		cudaGeometryState, allocatedGeometryState,
		cudaActiveState, allocatedActiveState,
		cudaBinningState, allocatedBinningState,
		cudaImageState, allocatedImageState,
		allocatedRenderedCapacity, capacityLimit,
		cudaCurrOffset, cudaExactOverflow,
		cudaRadii,
		inCamera,
		cudaImage,
		cudaAllMap,
		&activeGaussians,
		debug);
	lastActiveGaussians = activeGaussians;
	CHECK_CUDA(cudaDeviceSynchronize(), debug);
	outImage = cudaImage;
	outAllMap = cudaAllMap;

	return numRendered;
}

template <int D>
RenderRuntimeStats GaussianModelRender<D>::getRuntimeStats() const {
	RenderRuntimeStats stats;
	stats.numPoints = sceneData.numPoints;
	stats.activeGaussians = lastActiveGaussians;
	stats.allocatedRenderedInstances = config.bUseFlashGSExactIntersection ? allocatedRenderedCapacity : config.maxNumRenderedGaussians;
	stats.allocatedRenderedInstancesLimit = capacityLimit;
	stats.useExactIntersection = config.bUseFlashGSExactIntersection;
	stats.exactActiveSetMode = config.exactActiveSetMode;
	return stats;
}


template <int D>
void GaussianModelRender<D>::initCuda(int device) {
	CHECK_CUDA(cudaSetDevice(device), true);
	cudaDeviceProp prop;
	CHECK_CUDA(cudaGetDeviceProperties(&prop, device), true);
	if (prop.major < 7) {
		throw std::runtime_error("Sorry, need at least compute capability 7.0+!");
	}
}


template <int D>
void GaussianModelRender<D>::setDefaultCamera(int fov, int width, int height) {
    defaultCamera.coordSystem = CameraCoordSystem::COLMAP;
    defaultCamera.width = width;
    defaultCamera.height = height;
    defaultCamera.fx = (float)width / 2.0f / std::tan(deg2Rad((float)fov / 2.0f));
    defaultCamera.fy = defaultCamera.fx;
    defaultCamera.cx = (float)width / 2.0f;
    defaultCamera.cy = (float)height / 2.0f;
    defaultCamera.bgColor = Eigen::Vector3f::Zero();

    Eigen::Vector3f sceneCenter = (sceneData.sceneMax + sceneData.sceneMin) * 0.5f;
    Eigen::Vector3f sceneSize = sceneData.sceneMax - sceneData.sceneMin;

    Eigen::Vector3f cameraPos(
        sceneCenter.x(), 
        sceneCenter.y() + sceneSize.y() * 0.5f, 
        sceneCenter.z() + sceneSize.z() * 0.5f
    );
    Eigen::Vector3f cameraTarget = sceneCenter;
    
    Eigen::Vector3f up(0.0f, 1.0f, 0.0f);
    Eigen::Vector3f forward = (cameraTarget - cameraPos).normalized();
    Eigen::Vector3f right = up.cross(forward).normalized(); 
    Eigen::Vector3f newUp = forward.cross(right);

    Eigen::Matrix3f R;
    R.col(0) = right;
    R.col(1) = newUp;
    R.col(2) = forward;

    defaultCamera.quaternion = Eigen::Quaternionf(R);
    defaultCamera.position = cameraPos;
}

template <int D>
void GaussianModelRender<D>::setCudaImageParams(const GsCamera& camera, bool debug) {
	setCudaImageBuffers(cudaImage, allocatedCudaImage, cudaAllMap, allocatedCudaAllMap, camera, stream, debug);
}


template <int D>
void GaussianModelRender<D>::setCudaCameraParams(const GsCamera& camera, bool debug) 
{
	Eigen::Vector3f background = camera.bgColor;
    Eigen::Matrix4f viewMats = camera.getWorld2CameraMatrix();
    Eigen::Matrix4f projMats = camera.getProjectionMatrix();
	Eigen::Vector3f camPos = camera.position;

	if (cudaView == nullptr || cudaProj == nullptr || cudaCamPos == nullptr || cudaBackground == nullptr) {
		safeCudaFree(cudaView);
		safeCudaFree(cudaProj);
		safeCudaFree(cudaCamPos);
		safeCudaFree(cudaBackground);
		CHECK_CUDA(cudaMallocAsync((void**)&cudaView,    	sizeof(Eigen::Matrix4f), stream), debug);
		CHECK_CUDA(cudaMallocAsync((void**)&cudaProj,    	sizeof(Eigen::Matrix4f), stream), debug);
		CHECK_CUDA(cudaMallocAsync((void**)&cudaCamPos,  	sizeof(Eigen::Vector3f), stream), debug);
		CHECK_CUDA(cudaMallocAsync((void**)&cudaBackground, sizeof(Eigen::Vector3f), stream), debug);
	}
	CHECK_CUDA(cudaMemcpyAsync(cudaBackground, 	&background, 	sizeof(background), cudaMemcpyHostToDevice, stream), debug);
	CHECK_CUDA(cudaMemcpyAsync(cudaView, 		&viewMats, 		sizeof(viewMats), 	cudaMemcpyHostToDevice, stream), debug);
	CHECK_CUDA(cudaMemcpyAsync(cudaProj, 		&projMats, 		sizeof(projMats), 	cudaMemcpyHostToDevice, stream), debug);
	CHECK_CUDA(cudaMemcpyAsync(cudaCamPos, 		&camPos, 		sizeof(camPos), 	cudaMemcpyHostToDevice, stream), debug);
	cudaStreamSynchronize(stream);
}

template <int D>
void GaussianModelRender<D>::setCudaAuxiliary() {
	int numPoints = sceneData.numPoints;
	cudaMallocAsync((void**)&cudaRadii, sizeof(int) * numPoints, stream);
	cudaMallocAsync((void**)&cudaCurrOffset, sizeof(int), stream);
	cudaMallocAsync((void**)&cudaExactOverflow, sizeof(int), stream);
	cudaStreamSynchronize(stream);
}

template <int D>
GaussianSceneRender<D>::GaussianSceneRender(GsConfig config)
	: GaussianSceneRender(std::move(config), Utils::readGaussianSceneFromJson(config.modelPath)) {}

template <int D>
GaussianSceneRender<D>::GaussianSceneRender(GsConfig config, GaussianSceneDesc scene) {
	const auto initStart = Utils::nowUs();
	const auto startMem = Utils::nowGPUMB();
	initCuda(0);
	this->config = config;

	std::unordered_map<std::string, int> assetIndexByPath;
	Eigen::Vector3f worldMin(FLT_MAX, FLT_MAX, FLT_MAX);
	Eigen::Vector3f worldMax(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	for (const auto& object : scene.objects) {
		if (object.instances.empty()) {
			continue;
		}

		int assetIndex = -1;
		auto found = assetIndexByPath.find(object.modelPath);
		if (found == assetIndexByPath.end()) {
			assetIndex = static_cast<int>(assets.size());
			assetIndexByPath[object.modelPath] = assetIndex;
			LoadResult<D> loaded;
			std::string cacheName = "gaussian_splatting.sog";
			const std::string cachePath = buildDefaultCachePath(object.modelPath, cacheName);
			const auto loader = LoaderFactory<D>::create(
				object.modelPath, cachePath, config.bRebuildBinaryCache, config.bUseHalfPrecisionSH);
			const std::string loadPath = LoaderFactory<D>::resolveLoadPath(object.modelPath, cachePath, config.bRebuildBinaryCache);
			loader->load(loadPath, loaded);

			assets.push_back({});
			auto& asset = assets.back();
			asset.name = object.name;
			asset.modelPath = object.modelPath;
			asset.basePoint = sourceSceneData.numPoints;
			asset.numPoints = loaded.numPoints;
			asset.sceneMin = loaded.sceneMin;
			asset.sceneMax = loaded.sceneMax;
			if (sourceSceneData.numPoints == 0) {
				moveLoadResult(sourceSceneData, std::move(loaded));
			} else {
				appendLoadResult(sourceSceneData, std::move(loaded));
			}
		} else {
			assetIndex = found->second;
		}

		for (const auto& instanceTransform : object.instances) {
			instances.push_back({assetIndex, instanceTransform});
			const auto& asset = assets[assetIndex];
			totalScenePoints += asset.numPoints;
			expandWorldBounds(asset.sceneMin, asset.sceneMax, instanceTransform, worldMin, worldMax);
		}
	}

	if (instances.empty()) {
		throw std::runtime_error("Gaussian scene contains no instances.");
	}
	if (sourceSceneData.numPoints <= 0) {
		throw std::runtime_error("Gaussian scene contains no source Gaussian points.");
	}
	const auto afterLoad = Utils::nowUs();

	totalSourcePoints = sourceSceneData.numPoints;
	sourceIndicesHost.resize(totalScenePoints);
	instanceIndicesHost.resize(totalScenePoints);
	size_t writeOffset = 0;
	for (size_t instanceIdx = 0; instanceIdx < instances.size(); ++instanceIdx) {
		const auto& instance = instances[instanceIdx];
		const auto& asset = assets[instance.assetIndex];
		for (int localIdx = 0; localIdx < asset.numPoints; ++localIdx) {
			sourceIndicesHost[writeOffset] = static_cast<uint32_t>(asset.basePoint + localIdx);
			instanceIndicesHost[writeOffset] = static_cast<uint32_t>(instanceIdx);
			++writeOffset;
		}
	}
	const auto afterIndexBuild = Utils::nowUs();

	sourceSceneData.uploadDataToGPU(config.bKeepCpuSceneData, config.bUseHalfPrecisionSH, config.bUseHalfPrecisionCov3DOpacity);
	const auto afterSourceUpload = Utils::nowUs();
	uploadInstanceIndexData(true);
	const auto afterIndexUpload = Utils::nowUs();

	sceneMin[0] = worldMin.x(); sceneMin[1] = worldMin.y(); sceneMin[2] = worldMin.z();
	sceneMax[0] = worldMax.x(); sceneMax[1] = worldMax.y(); sceneMax[2] = worldMax.z();
	capacityLimit = config.maxNumRenderedGaussians > 0 ? config.maxNumRenderedGaussians : -1;
	allocatedRenderedCapacity = config.bUseFlashGSExactIntersection ?
		chooseInitialExactCapacity(totalScenePoints, capacityLimit) :
		config.maxNumRenderedGaussians;

	setDefaultCamera(90, 1920, 1080);
	setCudaAuxiliary();
	const auto afterAux = Utils::nowUs();

	GS_INFO("Total sourcePoints = %d, instantiatedPoints = %d, instances = %d",
		totalSourcePoints, totalScenePoints, static_cast<int>(instances.size()));
	GS_INFO("Multi-scene init breakdown ms: load=%.2f, indexBuild=%.2f, sourceUpload(+gpuCov)=%.2f, indexUpload=%.2f, aux=%.2f",
		static_cast<float>(afterLoad - initStart) / 1000.0f,
		static_cast<float>(afterIndexBuild - afterLoad) / 1000.0f,
		static_cast<float>(afterSourceUpload - afterIndexBuild) / 1000.0f,
		static_cast<float>(afterIndexUpload - afterSourceUpload) / 1000.0f,
		static_cast<float>(afterAux - afterIndexUpload) / 1000.0f);
	GS_INFO("Multi-scene init elapsed %.2f ms", static_cast<float>(afterAux - initStart) / 1000.0f);
	GS_INFO("Multi-scene init used %.2f MB", static_cast<float>(Utils::nowGPUMB() - startMem));
}

template <int D>
GaussianSceneRender<D>::~GaussianSceneRender() {
	safeCudaFree(cudaImage);
	safeCudaFree(cudaAllMap);
	safeCudaFree(cudaGeometryState);
	safeCudaFree(cudaActiveState);
	safeCudaFree(cudaBinningState);
	safeCudaFree(cudaImageState);
	safeCudaFree(cudaRadii);
	safeCudaFree(cudaCurrOffset);
	safeCudaFree(cudaExactOverflow);
	safeCudaFree(cudaSourceIndices);
	safeCudaFree(cudaInstanceIndices);
	safeCudaFree(cudaInstanceCamPositions);
	safeCudaFree(cudaInstanceViewMatrices);
	safeCudaFree(cudaInstanceProjMatrices);
	safeCudaFree(cudaInstanceDepthScales);
}

template <int D>
float GaussianSceneRender<D>::render(GsCamera& inCamera, float*& outImage, float*& outAllMap, bool debug) {
	ScopeTimer timer("Render multi camera " + std::to_string(inCamera.cameraId));
	setCudaImageParams(inCamera, debug);
	updateInstanceCameraData(inCamera, debug);

	int activeGaussians = -1;
	const int numRendered = renderSceneDataInternal(
		config,
		totalScenePoints,
		sourceSceneData,
		cudaSourceIndices,
		cudaInstanceIndices,
		cudaInstanceCamPositions,
		cudaInstanceViewMatrices,
		cudaInstanceProjMatrices,
		cudaInstanceDepthScales,
		cudaGeometryState, allocatedGeometryState,
		cudaActiveState, allocatedActiveState,
		cudaBinningState, allocatedBinningState,
		cudaImageState, allocatedImageState,
		allocatedRenderedCapacity, capacityLimit,
		cudaCurrOffset, cudaExactOverflow,
		cudaRadii,
		inCamera,
		cudaImage,
		cudaAllMap,
		&activeGaussians,
		debug);
	lastActiveGaussians = activeGaussians;
	CHECK_CUDA(cudaDeviceSynchronize(), debug);
	outImage = cudaImage;
	outAllMap = cudaAllMap;
	return static_cast<float>(numRendered);
}

template <int D>
void GaussianSceneRender<D>::initCuda(int device) {
	CHECK_CUDA(cudaSetDevice(device), true);
	cudaDeviceProp prop;
	CHECK_CUDA(cudaGetDeviceProperties(&prop, device), true);
	if (prop.major < 7) {
		throw std::runtime_error("Sorry, need at least compute capability 7.0+!");
	}
}

template <int D>
void GaussianSceneRender<D>::setDefaultCamera(int fov, int width, int height) {
	defaultCamera.coordSystem = CameraCoordSystem::COLMAP;
	defaultCamera.width = width;
	defaultCamera.height = height;
	defaultCamera.fx = static_cast<float>(width) / 2.0f / std::tan(deg2Rad(static_cast<float>(fov) / 2.0f));
	defaultCamera.fy = defaultCamera.fx;
	defaultCamera.cx = static_cast<float>(width) / 2.0f;
	defaultCamera.cy = static_cast<float>(height) / 2.0f;
	defaultCamera.bgColor = Eigen::Vector3f::Zero();

	const Eigen::Vector3f sceneCenter(
		(sceneMin[0] + sceneMax[0]) * 0.5f,
		(sceneMin[1] + sceneMax[1]) * 0.5f,
		(sceneMin[2] + sceneMax[2]) * 0.5f);
	const Eigen::Vector3f sceneSize(
		sceneMax[0] - sceneMin[0],
		sceneMax[1] - sceneMin[1],
		sceneMax[2] - sceneMin[2]);
	const Eigen::Vector3f cameraPos(
		sceneCenter.x(),
		sceneCenter.y() + sceneSize.y() * 0.5f,
		sceneCenter.z() + sceneSize.z() * 0.5f);
	const Eigen::Vector3f cameraTarget = sceneCenter;
	const Eigen::Vector3f up(0.0f, 1.0f, 0.0f);
	const Eigen::Vector3f forward = (cameraTarget - cameraPos).normalized();
	const Eigen::Vector3f right = up.cross(forward).normalized();
	const Eigen::Vector3f newUp = forward.cross(right);
	Eigen::Matrix3f R;
	R.col(0) = right;
	R.col(1) = newUp;
	R.col(2) = forward;
	defaultCamera.quaternion = Eigen::Quaternionf(R);
	defaultCamera.position = cameraPos;
}

template <int D>
void GaussianSceneRender<D>::setCudaImageParams(const GsCamera& camera, bool debug) {
	setCudaImageBuffers(cudaImage, allocatedCudaImage, cudaAllMap, allocatedCudaAllMap, camera, stream, debug);
}

template <int D>
void GaussianSceneRender<D>::setCudaAuxiliary() {
	cudaMallocAsync((void**)&cudaRadii, sizeof(int) * totalScenePoints, stream);
	cudaMallocAsync((void**)&cudaCurrOffset, sizeof(int), stream);
	cudaMallocAsync((void**)&cudaExactOverflow, sizeof(int), stream);
	cudaStreamSynchronize(stream);
}

template <int D>
void GaussianSceneRender<D>::uploadInstanceIndexData(bool debug) {
	cudaSourceIndices = cudaMallocAndMemcpy(sourceIndicesHost.data(), sourceIndicesHost.size(), stream);
	cudaInstanceIndices = cudaMallocAndMemcpy(instanceIndicesHost.data(), instanceIndicesHost.size(), stream);
	if (!instances.empty()) {
		CHECK_CUDA(cudaMallocAsync((void**)&cudaInstanceCamPositions, sizeof(float) * instances.size() * 3, stream), debug);
		CHECK_CUDA(cudaMallocAsync((void**)&cudaInstanceViewMatrices, sizeof(float) * instances.size() * 16, stream), debug);
		CHECK_CUDA(cudaMallocAsync((void**)&cudaInstanceProjMatrices, sizeof(float) * instances.size() * 16, stream), debug);
		CHECK_CUDA(cudaMallocAsync((void**)&cudaInstanceDepthScales, sizeof(float) * instances.size(), stream), debug);
	}
	cudaStreamSynchronize(stream);
}

template <int D>
void GaussianSceneRender<D>::updateInstanceCameraData(const GsCamera& camera, bool debug) {
	instanceCamPositionsHost.resize(instances.size() * 3);
	instanceViewMatricesHost.resize(instances.size() * 16);
	instanceProjMatricesHost.resize(instances.size() * 16);
	instanceDepthScalesHost.resize(instances.size());

	for (size_t instanceIdx = 0; instanceIdx < instances.size(); ++instanceIdx) {
		const auto& instance = instances[instanceIdx];
		GsCamera localCamera = transformCameraToInstanceLocal(camera, instance.transform);
		const Eigen::Matrix4f view = localCamera.getWorld2CameraMatrix();
		const Eigen::Matrix4f proj = localCamera.getProjectionMatrix();

		std::memcpy(instanceCamPositionsHost.data() + instanceIdx * 3, localCamera.position.data(), sizeof(float) * 3);
		std::memcpy(instanceViewMatricesHost.data() + instanceIdx * 16, view.data(), sizeof(float) * 16);
		std::memcpy(instanceProjMatricesHost.data() + instanceIdx * 16, proj.data(), sizeof(float) * 16);
		instanceDepthScalesHost[instanceIdx] = instance.transform.scale;
	}

	CHECK_CUDA(cudaMemcpyAsync(cudaInstanceCamPositions, instanceCamPositionsHost.data(),
		sizeof(float) * instanceCamPositionsHost.size(), cudaMemcpyHostToDevice, stream), debug);
	CHECK_CUDA(cudaMemcpyAsync(cudaInstanceViewMatrices, instanceViewMatricesHost.data(),
		sizeof(float) * instanceViewMatricesHost.size(), cudaMemcpyHostToDevice, stream), debug);
	CHECK_CUDA(cudaMemcpyAsync(cudaInstanceProjMatrices, instanceProjMatricesHost.data(),
		sizeof(float) * instanceProjMatricesHost.size(), cudaMemcpyHostToDevice, stream), debug);
	CHECK_CUDA(cudaMemcpyAsync(cudaInstanceDepthScales, instanceDepthScalesHost.data(),
		sizeof(float) * instanceDepthScalesHost.size(), cudaMemcpyHostToDevice, stream), debug);
	cudaStreamSynchronize(stream);
}

template <int D>
RenderRuntimeStats GaussianSceneRender<D>::getRuntimeStats() const {
	RenderRuntimeStats stats;
	stats.numPoints = totalScenePoints;
	stats.activeGaussians = lastActiveGaussians;
	stats.allocatedRenderedInstances = config.bUseFlashGSExactIntersection ? allocatedRenderedCapacity : config.maxNumRenderedGaussians;
	stats.allocatedRenderedInstancesLimit = capacityLimit;
	stats.useExactIntersection = config.bUseFlashGSExactIntersection;
	stats.exactActiveSetMode = config.exactActiveSetMode;
	return stats;
}


template <int D>
void SceneData<D>::initResource(std::string modelPath, std::string cacheSavePath, bool rebuildBinaryCache, bool keepCpuSceneData, bool useHalfPrecisionSH, bool useHalfPrecisionCov3DOpacity) {

	auto startM = Utils::nowGPUMB();
	auto startT = Utils::nowUs();

	// LoaderFactory decides the right loader (PLY / SOG) and the source path.
	// For PLY: it may use the SOG cache (or rebuild it).
	// For SOG: it loads directly.
	LoadResult<D> loaded;
	const auto loader = LoaderFactory<D>::create(modelPath, cacheSavePath, rebuildBinaryCache, useHalfPrecisionSH);
	const std::string loadPath =
		LoaderFactory<D>::resolveLoadPath(modelPath, cacheSavePath, rebuildBinaryCache);
	loader->load(loadPath, loaded);

	numPoints = loaded.numPoints;
	gaussianPoints = std::move(loaded.points);
	gaussianSHs    = std::move(loaded.shs);
	gaussianSHsHalfHost = std::move(loaded.shsHalf);
	gaussianOpacity= std::move(loaded.opacity);
	gaussianScale  = std::move(loaded.scale);
	gaussianRot    = std::move(loaded.rot);
	sceneMin = loaded.sceneMin;
	sceneMax = loaded.sceneMax;

	auto tAfterLoad = Utils::nowUs();
	// CPU cov3D precomputation -- kept here for reference, migrated to GPU below.
	// gaussianCov3D.resize(numPoints * 6);
	// computeCov3Ds(gaussianScale.data(), 1.0f, gaussianRot.data(), gaussianCov3D.data(), numPoints);

	auto tUploadStart = Utils::nowUs();
	uploadDataToGPU(keepCpuSceneData, useHalfPrecisionSH, useHalfPrecisionCov3DOpacity);
	auto tUploadDone = Utils::nowUs();

	GS_INFO("Total numPoints = %d", numPoints);
	GS_INFO("Init breakdown ms: load=%.2f, upload(+gpuCov)=%.2f",
		(float)(tAfterLoad - startT) / 1000.0f,
		(float)(tUploadDone - tUploadStart) / 1000.0f);
	GS_INFO("Initializing scene data elapsed %.2f ms", (float)(Utils::nowUs() - startT) / 1000);
	GS_INFO("Initializing scene data used %.2f MB", (float)(Utils::nowGPUMB() - startM));

}


template <int D>
void SceneData<D>::uploadDataToGPU(bool keepCpuSceneData, bool useHalfPrecisionSH, bool useHalfPrecisionCov3DOpacity) {
	cudaStream_t stream = 0;
	cudaGaussianPoints = (float*)cudaMallocAndMemcpy((Pos*)gaussianPoints.data(), numPoints, stream);
	if (useHalfPrecisionSH) {
		if (!gaussianSHsHalfHost.empty()) {
			cudaGaussianSHsHalf = cudaMallocAndMemcpy(gaussianSHsHalfHost.data(), gaussianSHsHalfHost.size(), stream);
		} else {
			std::vector<uint16_t> packedSHs = packSHsToHalf<D>(gaussianSHs);
			cudaGaussianSHsHalf = cudaMallocAndMemcpy(packedSHs.data(), packedSHs.size(), stream);
		}
	} else {
		cudaGaussianSHs = (float*)cudaMallocAndMemcpy((SHs<D>*)gaussianSHs.data(), numPoints, stream);
	}
	if (useHalfPrecisionCov3DOpacity) {
		std::vector<uint16_t> packedOpacity = packFloatsToHalf(gaussianOpacity);
		cudaGaussianOpacityHalf = cudaMallocAndMemcpy(packedOpacity.data(), packedOpacity.size(), stream);
	} else {
		cudaGaussianOpacity = (float*)cudaMallocAndMemcpy((float*)gaussianOpacity.data(), numPoints, stream);
	}
	cudaGaussianScale = (float*)cudaMallocAndMemcpy((Scale*)gaussianScale.data(), numPoints, stream);
	cudaGaussianRot = (float*)cudaMallocAndMemcpy((Rot*)gaussianRot.data(), numPoints, stream);

	// Precompute cov3D on the GPU (runs once at init, never again per frame).
	// Allocate output first, then launch the kernel in-place. After that we can
	// release scale/rot on the GPU because the renderer only needs cov3D.
	cudaMallocAsync(&cudaGaussianCov3D, (size_t)numPoints * 6 * sizeof(float), stream);
	// Synchronize to ensure Scale/Rot transfers are complete before the kernel starts.
	cudaStreamSynchronize(stream);
	launchComputeCov3D(cudaGaussianScale, cudaGaussianRot, cudaGaussianCov3D, 1.0f, numPoints, stream);
	cudaStreamSynchronize(stream);

	if (useHalfPrecisionCov3DOpacity) {
		cudaMallocAsync(&cudaGaussianCov3DHalf, static_cast<size_t>(numPoints) * 6 * sizeof(uint16_t), stream);
		launchPackFloatsToHalf(cudaGaussianCov3D, cudaGaussianCov3DHalf, numPoints * 6, stream);
		cudaStreamSynchronize(stream);
		safeCudaFree(cudaGaussianCov3D, stream);
	}

	safeCudaFree(cudaGaussianScale, stream);
	safeCudaFree(cudaGaussianRot, stream);

	if (!keepCpuSceneData) {
		std::vector<Pos>().swap(gaussianPoints);
		std::vector<SHs<D>>().swap(gaussianSHs);
		std::vector<uint16_t>().swap(gaussianSHsHalfHost);
		std::vector<float>().swap(gaussianOpacity);
		std::vector<Scale>().swap(gaussianScale);
		std::vector<Rot>().swap(gaussianRot);
		std::vector<float>().swap(gaussianCov3D);
		GS_INFO("Released CPU scene data after GPU upload.");
	}
}



}
