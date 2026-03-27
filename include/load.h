#pragma once

#include "common.h"

#include <cfloat>
#include <memory>
#include <string>
#include <vector>

namespace optisplat {

// -----------------------------------------------------------------------
// LoadResult: raw Gaussian data filled by a loader, consumed by SceneData.
// -----------------------------------------------------------------------
template <int D>
struct LoadResult {
    std::vector<Pos>     points;
    std::vector<SHs<D>>  shs;
    std::vector<float>   opacity;
    std::vector<Scale>   scale;
    std::vector<Rot>     rot;
    Eigen::Vector3f      sceneMin = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    Eigen::Vector3f      sceneMax = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    int                  numPoints = 0;
};

// -----------------------------------------------------------------------
// IGaussianLoader: common interface for all scene loaders.
// Add a new format by creating a subclass and registering it in
// LoaderFactory::create().
// -----------------------------------------------------------------------
template <int D>
class IGaussianLoader {
public:
    virtual ~IGaussianLoader() = default;

    // Load Gaussian data from the given path into result.
    // Throws std::runtime_error on failure.
    virtual void load(const std::string& path, LoadResult<D>& result) = 0;
};

// -----------------------------------------------------------------------
// PlyLoader: loads a standard 3DGS PLY file and optionally saves an SOG
// cache so the next run can use SogLoader instead.
// -----------------------------------------------------------------------
template <int D>
class PlyLoader : public IGaussianLoader<D> {
public:
    // If sogCachePath is non-empty the loader writes an SOG cache after
    // reading the PLY.
    explicit PlyLoader(std::string sogCachePath = "")
        : cachePath_(std::move(sogCachePath)) {}

    void load(const std::string& path, LoadResult<D>& result) override;

    // Read just the header to obtain vertex count and SH degree.
    static bool peekInfo(const std::string& path, int* outShDegree, int* outNumPoints);

private:
    std::string cachePath_;
};

// -----------------------------------------------------------------------
// SogLoader: loads the compressed Supersplat SOG format (.sog / .zip).
// -----------------------------------------------------------------------
template <int D>
class SogLoader : public IGaussianLoader<D> {
public:
    void load(const std::string& path, LoadResult<D>& result) override;

    // Read meta.json inside the SOG archive to obtain count and SH degree.
    static bool peekInfo(const std::string& path, int* outShDegree, int* outNumPoints);
};

// -----------------------------------------------------------------------
// LoaderFactory: selects the right loader from a file path and handles
// the PLY -> SOG caching logic transparently.
//
// Extension   -> Primary loader  -> Cache behaviour
// ----------  -----------------  ----------------
// .ply        PlyLoader          writes .sog cache next to the file
// .sog        SogLoader          no cache (already optimised)
// (future)    ...
// -----------------------------------------------------------------------
template <int D>
class LoaderFactory {
public:
    // Create a loader appropriate for the given model path.
    // sogCachePath is only used when the source is not already an SOG.
    static std::unique_ptr<IGaussianLoader<D>> create(
        const std::string& modelPath,
        const std::string& sogCachePath,
        bool               rebuildCache);

    // Resolve which file should actually be loaded. When a cached SOG
    // exists but libwebp decode support is unavailable, this falls back
    // to the original source file instead of the cache.
    static std::string resolveLoadPath(
        const std::string& modelPath,
        const std::string& sogCachePath,
        bool               rebuildCache);

    // Peek model metadata without creating a full loader.
    static bool peekInfo(const std::string& modelPath, int* outShDegree, int* outNumPoints);

    // Derive the default SOG cache path for a PLY model:
    //   <model_dir>/.cache/<model_stem>.sog
    static std::string defaultSogCachePath(const std::string& modelPath);
};

} // namespace optisplat
