#include "rasterizer.h"
#include "render.h"
#include "utils.h"
#include "camera.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
// #include <glm/glm.hpp>
// #include <glm/gtc/quaternion.hpp>

#include <numeric>
#include <filesystem>



namespace optisplat {

template class GaussianRender<0>;
template class GaussianRender<1>;
template class GaussianRender<2>;
template class GaussianRender<3>;

template class SceneData<0>;
template class SceneData<1>;
template class SceneData<2>;
template class SceneData<3>;

std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S, bool debug = false) {
	auto lambda = [ptr, &S, debug](size_t N) {
		if (N > S)
		{	
			if (*ptr) CHECK_CUDA(cudaFree(*ptr), debug);
			size_t scale = (S == 0) ? 1 : 2; // s = 0 means the first allocation
			// std::cout << "Resize: " << N << " -> " << scale * N << std::endl;
			CHECK_CUDA(cudaMalloc(ptr, scale * N), debug);
			CHECK_CUDA(cudaMemset(reinterpret_cast<char*>(*ptr), 0, scale * N), debug);
			S = scale * N;
		}
		return reinterpret_cast<char*>(*ptr);
	};
	return lambda;
}

// void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
// {
//     // Create scaling matrix
//     glm::mat3 S = glm::mat3(1.0f);
//     S[0][0] = mod * scale.x;
//     S[1][1] = mod * scale.y;
//     S[2][2] = mod * scale.z;

//     // Normalize quaternion to get valid rotation
//     glm::vec4 q = rot;// / glm::length(rot);
//     float r = q.x;
//     float x = q.y;
//     float y = q.z;
//     float z = q.w;

//     // Compute rotation matrix from quaternion
//     glm::mat3 R = glm::mat3(
//         1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
//         2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
//         2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
//     );

//     glm::mat3 M = S * R;

//     // Compute 3D world covariance matrix Sigma
//     glm::mat3 Sigma = glm::transpose(M) * M;

//     // Covariance is symmetric, only store upper right
//     cov3D[0] = Sigma[0][0];
//     cov3D[1] = Sigma[0][1];
//     cov3D[2] = Sigma[0][2];
//     cov3D[3] = Sigma[1][1];
//     cov3D[4] = Sigma[1][2];
//     cov3D[5] = Sigma[2][2];
// }

// void computeCov3Ds(const glm::vec3* scales, float mod, const glm::vec4* rots, float* cov3Ds, int num) {
// 	for (int i = 0; i < num; i++) {
// 		computeCov3D(scales[i], mod, rots[i], cov3Ds + i * 6);
// 	}
// }

// void computeCov3D(const Eigen::Vector3f& scale, float mod, const Eigen::Vector4f& rot, float* cov3D)
// {
//     // 1. 创建缩放矩阵 (利用 diagonal 特性更高效)
//     // S = diag(mod * sx, mod * sy, mod * sz)
//     Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
//     S.diagonal() = scale * mod;

//     // 2. 从四元数构造旋转矩阵
//     // 注意：输入 rot 是 (w, x, y, z) 还是 (x, y, z, w)? 
//     // 根据你原代码 float r = q.x (即 w), x = q.y, y = q.z, z = q.w
//     // 我们直接构造 Eigen 四元数。Eigen 构造函数顺序是 (w, x, y, z)
//     Eigen::Quaternionf q(rot[0], rot[1], rot[2], rot[3]);
    
//     // 必须归一化以防止 NaN 和精度漂移，这是解决你之前问题的关键
//     q.normalize(); 
//     Eigen::Matrix3f R = q.toRotationMatrix();

//     // 3. 计算中间矩阵 M
//     // 原代码逻辑: M = S * R
//     Eigen::Matrix3f M = S * R;

//     // 4. 计算 3D 协方差矩阵 Sigma = M^T * M
//     // Eigen 的 transpose() * M 在内部有高度优化
//     Eigen::Matrix3f Sigma = M.transpose() * M;

//     // 5. 存储上三角元素 (Symmetric Matrix)
//     // 索引说明: Sigma(row, col)
//     cov3D[0] = Sigma(0, 0);
//     cov3D[1] = Sigma(0, 1);
//     cov3D[2] = Sigma(0, 2);
//     cov3D[3] = Sigma(1, 1);
//     cov3D[4] = Sigma(1, 2);
//     cov3D[5] = Sigma(2, 2);
// }


// 单个物体 3D 协方差
void computeCov3D(const Eigen::Vector3f& scale, float mod, const Eigen::Vector4f& rot, float* cov3D)
{
    // Create scaling matrix
    Eigen::Matrix3f S = Eigen::Matrix3f::Identity();
    S(0,0) = mod * scale.x();
    S(1,1) = mod * scale.y();
    S(2,2) = mod * scale.z();

    // Normalize quaternion to get valid rotation
    Eigen::Vector4f q = rot; // / rot.norm();  // 如果需要归一化可以取消注释
    float r = q.x();  // 对应原 glm: r = q.x
    float x = q.y();  // 对应原 glm: x = q.y
    float y = q.z();  // 对应原 glm: y = q.z
    float z = q.w();  // 对应原 glm: z = q.w

    // Compute rotation matrix from quaternion
    Eigen::Matrix3f R;
    R << 1.f - 2.f * (y*y + z*z), 2.f * (x*y - r*z), 2.f * (x*z + r*y),
         2.f * (x*y + r*z), 1.f - 2.f * (x*x + z*z), 2.f * (y*z - r*x),
         2.f * (x*z - r*y), 2.f * (y*z + r*x), 1.f - 2.f * (x*x + y*y);
	R.transposeInPlace(); // 转置以匹配原 glm 代码的行主序访问

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
    #pragma omp parallel for
	for (int i = 0; i < num; i++) {
        computeCov3D(scales[i], mod, rots[i], cov3Ds + i * 6);
    }
}


std::shared_ptr<IGaussianRender> IGaussianRender::CreateRenderer(GsConfig config) {
	int numVertex, shsDegree;
	readPlyHeader(config.modelPath, numVertex, shsDegree);
	if (numVertex == 0 || shsDegree == 0) {
		GS_ERROR("Invalid PLY file: %s", config.modelPath.c_str());
		return nullptr;
	}

	std::shared_ptr<IGaussianRender> renderer = nullptr;
	switch (shsDegree)
	{
		case 0:
			renderer = std::make_shared<GaussianRender<0>>(config);
			break;
		case 1:
			renderer = std::make_shared<GaussianRender<1>>(config);
			break;
		case 2:
			renderer = std::make_shared<GaussianRender<2>>(config);
			break;
		case 3:
			renderer = std::make_shared<GaussianRender<3>>(config);
			break;
		default:
			GS_ERROR("Unsupported SH Degree: %d", shsDegree);
			throw std::runtime_error("Unsupported SH Degree: " + std::to_string(shsDegree));
	}
	return renderer;
}

template <int D>
GaussianRender<D>::GaussianRender(GsConfig config) {
	initCuda(0);

	std::filesystem::path modelPath = config.modelPath;
	this->config = config;
	this->sceneData.cacheName = modelPath.filename().string() + ".cache";

	std::string cacheSavePath;
	if (std::filesystem::is_regular_file(modelPath)) {
		std::filesystem::path modelDir = modelPath.parent_path();
		cacheSavePath = (modelDir / ".cache").string();
	} else {
		cacheSavePath = (modelPath / ".cache").string();
	}
	 
	sceneData.initResource(modelPath.string(), cacheSavePath, config.bRebuildBinaryCache);
	
	setDefaultCamera(90, 1920, 1080);
	setCudaAuxiliary();
}

template <int D>
GaussianRender<D>::~GaussianRender() {
	safeCudaFree(cudaImage);
	safeCudaFree(cudaAllMap);
	safeCudaFree(cudaGeometryState);
	safeCudaFree(cudaBinningState);
	safeCudaFree(cudaImageState);
	
	safeCudaFree(cudaBackground);
	safeCudaFree(cudaView);
	safeCudaFree(cudaProj);
	safeCudaFree(cudaCamPos);
	safeCudaFree(cudaRadii);
	safeCudaFree(cudaCurrOffset);
}

template <int D>
float GaussianRender<D>::render(GsCamera& inCamera, float*& outImage, float*& outAllMap, bool debug) {
	ScopeTimer timer("Render camera " + std::to_string(inCamera.cameraId));

	{
		ScopeTimer timer("set camera and image cuda");
		setCudaImageParams(inCamera, debug);
		setCudaCameraParams(inCamera, debug);
	}


	auto numRendered = 0;
	{
		int P = sceneData.numPoints;
		int M = (D + 1) * (D + 1);
		float tanHalfFovx = inCamera.width / (2.0f * inCamera.fx);
		float tanHalfFovy = inCamera.height / (2.0f * inCamera.fy);

		Eigen::Vector3f cpuCamPosEigen = inCamera.position;
		Eigen::Matrix3f cpuCamRotEigen = inCamera.quaternion.toRotationMatrix();
		std::vector<float> cpuCamPos(cpuCamPosEigen.data(), cpuCamPosEigen.data() + 3);
		std::vector<float> cpuCamRot(cpuCamRotEigen.data(), cpuCamRotEigen.data() + 9);
		ScopeTimer timer("forward");
		CHECK_CUDA(
			numRendered = CudaRasterizer::Rasterizer::forward2(
			resizeFunctional(&cudaGeometryState, allocatedGeometryState),
			resizeFunctional(&cudaBinningState, allocatedBinningState),
			resizeFunctional(&cudaImageState, allocatedImageState),
			P, D, M, config.maxNumRenderedGaussians,
			config.bUseFlashGSExactIntersection, config.bUseFlashGSPrefetchingPipeline, config.bUseTensorCore,
			cpuCamPos, cpuCamRot, inCamera.znear, inCamera.zfar, cudaCurrOffset,
			cudaBackground,
			inCamera.width, inCamera.height,
			sceneData.cudaGaussianPoints,
			sceneData.cudaGaussianSHs,
			nullptr,
			sceneData.cudaGaussianOpacity,
			sceneData.cudaGaussianScale,
			inCamera.scale,
			sceneData.cudaGaussianRot,
			// nullptr,
			sceneData.cudaGaussianCov3D,
			cudaView,
			cudaProj,
			cudaCamPos,
			tanHalfFovx, tanHalfFovy,
			false,
			cudaImage,
			cudaAllMap,
			false,
			cudaRadii,
			debug), debug
		);
		// CHECK_CUDA(
		// 	numRendered = CudaRasterizer::Rasterizer::forward(
		// 	resizeFunctional(&cudaGeometryState, allocatedGeometryState),
		// 	resizeFunctional(&cudaBinningState, allocatedBinningState),
		// 	resizeFunctional(&cudaImageState, allocatedImageState),
		// 	P, D, M, 
		// 	cudaBackground,
		// 	inCamera.width, inCamera.height,
		// 	sceneData.cudaGaussianPoints,
		// 	sceneData.cudaGaussianSHs,
		// 	nullptr,
		// 	sceneData.cudaGaussianOpacity,
		// 	sceneData.cudaGaussianScale,
		// 	inCamera.scale,
		// 	sceneData.cudaGaussianRot,
		// 	nullptr,
		// 	// sceneData.cudaGaussianCov3D,
		// 	cudaView,
		// 	cudaProj,
		// 	cudaCamPos,
		// 	tanHalfFovx, tanHalfFovy,
		// 	false,
		// 	cudaImage,
		// 	cudaAllMap,
		// 	false,
		// 	cudaRadii,
		// 	debug), debug
		// );
		// cudaStreamSynchronize(stream);
	}

	outImage = cudaImage;
	outAllMap = cudaAllMap;

	return numRendered;
}


template <int D>
void GaussianRender<D>::initCuda(int device) {
	CHECK_CUDA(cudaSetDevice(device), true);
	cudaDeviceProp prop;
	CHECK_CUDA(cudaGetDeviceProperties(&prop, device), true);
	if (prop.major < 7) {
		throw std::runtime_error("Sorry, need at least compute capability 7.0+!");
	}
}


// template <int D>
// void GaussianRender<D>::setDefaultCamera(int fov, int width, int height) {
//     defaultCamera.coordSystem = CameraCoordSystem::COLMAP;
//     defaultCamera.width = width;
//     defaultCamera.height = height;
//     defaultCamera.fx = width / 2.f / std::tan(deg2Rad(fov / 2));
//     defaultCamera.fy = defaultCamera.fx;
//     defaultCamera.cx = width / 2.0f;
//     defaultCamera.cy = height / 2.0f;
//     defaultCamera.bgColor = glm::vec3(0.0f, 0.0f, 0.0f);

//     glm::vec3 sceneCenter = (sceneData.sceneMax + sceneData.sceneMin) * 0.5f;
//     glm::vec3 sceneSize = sceneData.sceneMax - sceneData.sceneMin;

//     // 相机位置与朝向
//     glm::vec3 cameraPos(sceneCenter.x, sceneCenter.y + sceneSize.y * 0.5f, sceneCenter.z + sceneSize.z * 0.5f);
//     glm::vec3 cameraTarget = sceneCenter;
    
//     glm::vec3 up(0.0f, 1.0f, 0.0f);
//     glm::vec3 forward = glm::normalize(cameraTarget - cameraPos);
//     glm::vec3 right = glm::normalize(glm::cross(up, forward)); // GLM cross(x, y)
//     glm::vec3 newUp = glm::cross(forward, right);

//     // 构建旋转矩阵
//     // 注意：GLM 矩阵构造是列主序，glm::mat3(col0, col1, col2)
//     glm::mat3 R(right, newUp, forward);

//     glm::quat quat = glm::quat_cast(R); 
//     defaultCamera.quaternion = glm::quat_cast(R);
//     defaultCamera.position = cameraPos;
// }

template <int D>
void GaussianRender<D>::setDefaultCamera(int fov, int width, int height) {
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
void GaussianRender<D>::setCudaImageParams(const GsCamera& camera, bool debug) {
	const int NUM_IMAGE_CHANNELS = 4; // RGBA
	const int NUM_ALLMAP_CHANNELS = 1; // depth
	resizeFunctional((void **)&cudaImage, allocatedCudaImage)(camera.width * camera.height * NUM_IMAGE_CHANNELS * sizeof(float));
	resizeFunctional((void **)&cudaAllMap, allocatedCudaAllMap)(camera.width * camera.height * NUM_ALLMAP_CHANNELS * sizeof(float));
	CHECK_CUDA(cudaMemsetAsync(cudaImage, 0, camera.width * camera.height * NUM_IMAGE_CHANNELS * sizeof(float), stream), debug);
	CHECK_CUDA(cudaMemsetAsync(cudaAllMap, 0, camera.width * camera.height * NUM_ALLMAP_CHANNELS * sizeof(float), stream), debug);
	cudaStreamSynchronize(stream);
}


template <int D>
void GaussianRender<D>::setCudaCameraParams(const GsCamera& camera, bool debug) 
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
void GaussianRender<D>::setCudaAuxiliary() {
	int numPoints = sceneData.numPoints;
	cudaMallocAsync((void**)&cudaRadii, sizeof(int) * numPoints, stream);
	cudaMallocAsync((void**)&cudaCurrOffset, sizeof(int), stream);
	cudaStreamSynchronize(stream);
}


template <int D>
void SceneData<D>::initResource(std::string modelPath,  std::string cacheSavePath, bool rebuildBinaryCache) {

	auto startM = Utils::nowGPUMB();
	auto startT = Utils::nowUs();
	if (rebuildBinaryCache || !isCacheExist(cacheSavePath)) {
		numPoints = 0;
		auto t0 = Utils::nowUs();
		auto& path = modelPath;

		std::vector<Pos> posVec;
		std::vector<SHs<D>> shsVec; 
		std::vector<float> opacityVec;
		std::vector<Scale> scaleVec;
		std::vector<Rot> rotVec;
		Eigen::Vector3f minn, maxx, mean;
		int numElem = loadPlyFlexible(path, posVec, shsVec, opacityVec, scaleVec, rotVec, minn, maxx, mean);
		sceneMin = sceneMin.cwiseMin(minn);
		sceneMax = sceneMax.cwiseMax(maxx);

		gaussianPoints.insert(gaussianPoints.end(), posVec.begin(), posVec.end());
		gaussianSHs.insert(gaussianSHs.end(), shsVec.begin(), shsVec.end());
		gaussianOpacity.insert(gaussianOpacity.end(), opacityVec.begin(), opacityVec.end());
		gaussianScale.insert(gaussianScale.end(), scaleVec.begin(), scaleVec.end());
		gaussianRot.insert(gaussianRot.end(), rotVec.begin(), rotVec.end());
		// gaussianCov3D.resize(numElem * 6);
		// computeCov3Ds(scaleVec.data(), 1.0f, rotVec.data(), gaussianCov3D.data(), numElem);

		numPoints += numElem;
		GS_INFO("Loaded %d gaussians from %s elapsed %.2f ms", numPoints, path.c_str(), (float)(Utils::nowUs() - t0) / 1000);
		saveCache(cacheSavePath);
	}
	else {
		loadCache(cacheSavePath);
	}

	gaussianCov3D.resize(numPoints * 6);
	computeCov3Ds(gaussianScale.data(), 1.0f, gaussianRot.data(), gaussianCov3D.data(), numPoints);
	
	uploadDataToGPU();

	GS_INFO("Total numPoints = %d", numPoints);
	GS_INFO("Initializing scene data elapsed %.2f ms", (float)(Utils::nowUs() - startT) / 1000);
	GS_INFO("Initializing scene data used %.2f MB", (float)(Utils::nowGPUMB() - startM));

}


template <int D>
int SceneData<D>::loadPlyFlexible(
	std::string filename,
	std::vector<Pos>& posVec,
	std::vector<SHs<D>>& shsVec,
	std::vector<float>& opacityVec,
	std::vector<Scale>& scaleVec,
	std::vector<Rot>& rotVec,
	Eigen::Vector3f& minn,
	Eigen::Vector3f& maxx,
	Eigen::Vector3f& mean,
	bool mortonOrder
) {
	std::ifstream infile(filename, std::ios_base::binary);
	if (!infile.good()){
		GS_ERROR("Cannot open file: %s, skip", filename.c_str());
		return -1;
	}

	// 1) Parse header
	int numPoints = -1;
	std::string line;
	std::vector<std::string> props;
	bool binaryLittle = false;
	int totalFloats = 0;
	while (std::getline(infile, line)) {
		if (!line.empty() && line.back()=='\r') line.pop_back();
		std::istringstream iss(line);
		std::string token;
		iss >> token;
		if (token == "format") {
			std::string fmt;
			iss >> fmt;
			if (fmt == "binary_little_endian") binaryLittle = true;
		}
		else if (token == "element") {
			std::string type;
			iss >> type;
			if (type == "vertex") iss >> numPoints;
		}
		else if (token == "property") {
			std::string ty, name;
			iss >> ty >> name;
			props.push_back(name);
			totalFloats++;
		}
		else if (token == "end_header") {
			break;
		}
	}
	if (totalFloats < 0 || numPoints < 0) {
		GS_ERROR("Parse PLY file failed: no vertex count found.: %s, skip", filename.c_str());
		return -1;
    }

	// allocate
	posVec.resize(numPoints);
	shsVec.resize(numPoints);
	opacityVec.resize(numPoints);
	scaleVec.resize(numPoints);
	rotVec.resize(numPoints);

	// 2) Read binary data per-vertex
	// Track sum for mean
	struct PropInfo {
        enum Type { POS, F_DC, F_REST, OPACITY, SCALE, ROT, UNKNOWN };
        Type type = UNKNOWN;
        int channel = -1;
    };
    std::vector<PropInfo> propInfos(props.size());
    const int SH_N = (D + 1) * (D + 1);
    for (size_t j = 0; j < props.size(); ++j) {
        const std::string& n = props[j];
        if 		(n == "x") propInfos[j] = {PropInfo::POS, 0};
        else if (n == "y") propInfos[j] = {PropInfo::POS, 1};
        else if (n == "z") propInfos[j] = {PropInfo::POS, 2};
        else if (n.rfind("f_dc_", 0) == 0) {
            propInfos[j] = {PropInfo::F_DC, std::stoi(n.substr(5))};
        }
        else if (n.rfind("f_rest_", 0) == 0) {
            int num = std::stoi(n.substr(7));
			int ch = num / (SH_N - 1);
			int idx = num % (SH_N - 1) + 1;
			propInfos[j] = {PropInfo::F_REST, idx * 3 + ch};
        }
        else if (n == "opacity") propInfos[j] = {PropInfo::OPACITY, 0};
        else if (n.rfind("scale_", 0) == 0) {
            propInfos[j] = {PropInfo::SCALE, std::stoi(n.substr(6))};
        }
        else if (n.rfind("rot_", 0) == 0) {
            propInfos[j] = {PropInfo::ROT, std::stoi(n.substr(4))};
        }
    }

	std::vector<float> buffer(totalFloats * numPoints);
	infile.read(reinterpret_cast<char*>(buffer.data()), totalFloats * numPoints * sizeof(float));
	Eigen::Vector3d sumPos(0.0, 0.0, 0.0);
	Eigen::Vector3f minPos(FLT_MAX, FLT_MAX, FLT_MAX);
	Eigen::Vector3f maxPos(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	for (int i = 0; i < numPoints; ++i) {
		for (size_t j = 0; j < props.size(); ++j) {
			size_t base = i * totalFloats;
			float value = buffer[base + j];
            const PropInfo& info = propInfos[j];
            switch (info.type) {
                case PropInfo::POS: posVec[i][info.channel] = value; break;
                case PropInfo::F_DC: shsVec[i][info.channel] = value; break;
                case PropInfo::F_REST: shsVec[i][info.channel] = value; break;
                case PropInfo::OPACITY: opacityVec[i] = sigmoid(value); break;
                case PropInfo::SCALE: scaleVec[i][info.channel] = std::exp(value); break;
                case PropInfo::ROT: rotVec[i][info.channel] = value; break;
                default: break;
            }
		}
		// must normalize quat of rot, otherwise the cov3d will be wrong!
		// float length2 = 0;
		// for (int j = 0; j < 4; j++) length2 += rotVec[i][j] * rotVec[i][j];
		// float length = sqrt(length2);
		// for (int j = 0; j < 4; j++) rotVec[i][j] = rotVec[i][j] / length;
		// rotVec[i] = glm::normalize(rotVec[i]);
		rotVec[i] = rotVec[i].normalized();
		// glm::vec4 tmp(rotVec[i].x(), rotVec[i].y(), rotVec[i].z(), rotVec[i].w());
		// tmp = glm::normalize(tmp);
		// rotVec[i] = Eigen::Vector4f(tmp.x, tmp.y, tmp.z, tmp.w);

		Eigen::Vector3f curPos = posVec[i]; 
		sumPos += curPos.cast<double>();
		maxPos = maxPos.cwiseMax(curPos);
		minPos = minPos.cwiseMin(curPos);
	}

	// 3) Compute mean, bounding box, Morton order
	Eigen::Vector3d avgPos = sumPos / double(numPoints);
	mean = avgPos.cast<float>();
	minn = minPos;
	maxx = maxPos;

	if (!mortonOrder) {
		return numPoints;
	}
	std::vector<uint64_t> mortonCodes(numPoints);
    std::vector<int> indices(numPoints);
	for (int i = 0; i < numPoints; i++) {
		mortonCodes[i] = mortonEncode64(posVec[i][0], posVec[i][1], posVec[i][2]);
    	indices[i] = i;
	}
	std::sort(indices.begin(), indices.end(), [&](int a, int b){ return mortonCodes[a] < mortonCodes[b]; });

	auto t0 = Utils::nowUs();
	// 4) Permute into sorted output
	auto permute = [&](auto& vec) {
		using T = typename std::decay<decltype(vec)>::type::value_type;
		std::vector<T> tmp(numPoints);
		for (int i = 0; i < numPoints; ++i) tmp[i] = vec[indices[i]];
		vec.swap(tmp);
	};
	permute(posVec);
	permute(shsVec);
	permute(opacityVec);
	permute(scaleVec);
	permute(rotVec);
	GS_INFO("permute time cost %.2f ms", float(Utils::nowUs() - t0) / 1000);
	return numPoints;
}


template <int D>
void SceneData<D>::saveCache(std::string cacheDir) {
	if (!std::filesystem::exists(cacheDir)) {
		std::filesystem::create_directory(cacheDir);
		GS_INFO("Cache directory %s created!", cacheDir.c_str());
	}
	else {
		GS_INFO("Cache directory %s already exists!", cacheDir.c_str());
	}

	std::string cachePath = cacheDir + "/" + cacheName;
	std::ofstream outfile(cachePath, std::ios::binary);
	if (!outfile.is_open()) {
		GS_ERROR("File %s can not be open!", cachePath.c_str());
        throw std::runtime_error("File can not be open for serialization");
    }

	auto t0 = Utils::nowUs();
	outfile.write(reinterpret_cast<const char*>(&sceneMin), sizeof(Eigen::Vector3f));
	outfile.write(reinterpret_cast<const char*>(&sceneMax), sizeof(Eigen::Vector3f));
	outfile.write(reinterpret_cast<const char*>(&numPoints), sizeof(numPoints));
	outfile.write(reinterpret_cast<const char*>(gaussianPoints.data()), sizeof(Pos) * numPoints);
	outfile.write(reinterpret_cast<const char*>(gaussianSHs.data()), sizeof(SHs<D>) * numPoints);
	outfile.write(reinterpret_cast<const char*>(gaussianOpacity.data()), sizeof(float) * numPoints);
	outfile.write(reinterpret_cast<const char*>(gaussianScale.data()), sizeof(Scale) * numPoints);
	outfile.write(reinterpret_cast<const char*>(gaussianRot.data()), sizeof(Rot) * numPoints);
	GS_INFO("Cache has been written to %s, elapsed %.2f ms", cachePath.c_str(), (float)(Utils::nowUs() - t0) / 1000);
}

template <int D>
void SceneData<D>::loadCache(std::string cacheDir) {
	if (!std::filesystem::exists(cacheDir)) {
        throw std::runtime_error("Cache folder does not exist: " + cacheDir);
	}
	if (!std::filesystem::exists(cacheDir + "/" + cacheName)) {
		throw std::runtime_error("Cache file does not exist: " + cacheDir + "/" + cacheName);
	}
	
	std::string cachePath = cacheDir + "/" + cacheName;
	std::ifstream infile(cachePath, std::ios::binary);
	if (!infile.is_open()) {
		GS_ERROR("File %s can not be open!", cachePath.c_str());
        throw std::runtime_error("File can not be open for deserialization");
    }

	GS_INFO("Loading cache from %s", cachePath.c_str());
	auto t0 = Utils::nowUs();
	infile.read(reinterpret_cast<char*>(&sceneMin), sizeof(Eigen::Vector3f));
	infile.read(reinterpret_cast<char*>(&sceneMax), sizeof(Eigen::Vector3f));
	infile.read(reinterpret_cast<char*>(&numPoints), sizeof(numPoints));
	gaussianPoints.resize(numPoints);
	gaussianSHs.resize(numPoints);
	gaussianOpacity.resize(numPoints);
	gaussianScale.resize(numPoints);
	gaussianRot.resize(numPoints);
	infile.read(reinterpret_cast<char*>(gaussianPoints.data()), sizeof(Pos) * numPoints);
	infile.read(reinterpret_cast<char*>(gaussianSHs.data()), sizeof(SHs<D>) * numPoints);
	infile.read(reinterpret_cast<char*>(gaussianOpacity.data()), sizeof(float) * numPoints);
	infile.read(reinterpret_cast<char*>(gaussianScale.data()), sizeof(Scale) * numPoints);
	infile.read(reinterpret_cast<char*>(gaussianRot.data()), sizeof(Rot) * numPoints);
	GS_INFO("Loaded %d gaussians from %s, elapsed %.2f ms", numPoints, cachePath.c_str(), (float)(Utils::nowUs() - t0) / 1000);
}

template <int D>
bool SceneData<D>::isCacheExist(std::string cacheDir) {
	std::string cachePath = cacheDir + "/" + cacheName;
	return (std::filesystem::exists(cacheDir) && std::filesystem::exists(cachePath));
}

template <int D>
void SceneData<D>::uploadDataToGPU() {
	cudaStream_t stream = 0;
	cudaGaussianPoints = (float*)cudaMallocAndMemcpy((Pos*)gaussianPoints.data(), numPoints, stream);
	cudaGaussianSHs = (float*)cudaMallocAndMemcpy((SHs<D>*)gaussianSHs.data(), numPoints, stream);
	cudaGaussianOpacity = (float*)cudaMallocAndMemcpy((float*)gaussianOpacity.data(), numPoints, stream);
	cudaGaussianScale = (float*)cudaMallocAndMemcpy((Scale*)gaussianScale.data(), numPoints, stream);
	cudaGaussianRot = (float*)cudaMallocAndMemcpy((Rot*)gaussianRot.data(), numPoints, stream);
	cudaGaussianCov3D = (float*)cudaMallocAndMemcpy((float*)gaussianCov3D.data(), numPoints * 6, stream);
}


void readPlyHeader(std::string filename, int& numVertex, int& shsDegree) {
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		GS_ERROR("Error: Unable to open file");
		throw std::runtime_error("Unable to open file");
	}

	std::string line;
	bool headerEnd = false;
	int numCoefficient = 0;

	while (std::getline(file, line)) {
		if (line == "end_header") {
			headerEnd = true;
			break;
		}

		std::istringstream iss(line);
		std::string keyword;
		iss >> keyword;

		if (keyword == "property") {
			std::string type, name;
			iss >> type >> name;
			if (name.find("f_dc") == 0 || name.find("f_rest") == 0) {
				numCoefficient++;
			}
		}

		if (keyword == "element") {
			std::string name;
			iss >> name >> numVertex;
		}
	}

	if (!headerEnd) {
		GS_ERROR("Error: Invalid PLY file: Missing end_header");
		throw std::runtime_error("Invalid PLY file: Missing end_header");
	}

	if (numCoefficient % 3 != 0) {
		GS_ERROR("Error: Invalid PLY file: Invalid number of SH coefficients");
		throw std::runtime_error("Invalid PLY file: Invalid number of SH coefficients");
	}

	shsDegree = std::sqrt(numCoefficient / 3)-1;
}



}