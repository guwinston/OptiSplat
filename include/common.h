#pragma once

#include <Eigen/Dense>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace optisplat {

// ---------------------------------------------------------------------------
// Core per-Gaussian attribute types
// ---------------------------------------------------------------------------
using Pos   = Eigen::Vector3f;
using Rot   = Eigen::Vector4f;
using Scale = Eigen::Vector3f;

template <int D>
using SHs = std::array<float, (D + 1) * (D + 1) * 3>;

// ---------------------------------------------------------------------------
// Lightweight math helpers
// ---------------------------------------------------------------------------
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float inverse_sigmoid(float x) {
    return std::log(x / (1.0f - x));
}

inline uint64_t expandBits(uint32_t v) {
    uint64_t x = v & 0x1fffffu;
    x = (x | (x << 32u)) & 0x1f00000000ffffu;
    x = (x | (x << 16u)) & 0x1f0000ff0000ffu;
    x = (x | (x <<  8u)) & 0x100f00f00f00f00fu;
    x = (x | (x <<  4u)) & 0x10c30c30c30c30c3u;
    x = (x | (x <<  2u)) & 0x1249249249249249u;
    return x;
}

inline uint64_t mortonEncode64(float x, float y, float z) {
    const uint32_t xx = static_cast<uint32_t>(x * 2097152.0f);
    const uint32_t yy = static_cast<uint32_t>(y * 2097152.0f);
    const uint32_t zz = static_cast<uint32_t>(z * 2097152.0f);
    return (expandBits(xx) << 2u) | (expandBits(yy) << 1u) | expandBits(zz);
}

/**
 * Launch a CUDA kernel that computes the 3D covariance matrix (upper-triangle)
 * for every Gaussian point in parallel.
 *
 * Formula:  Sigma = M^T * M,   M = S * R^T
 *   S = diag(mod*sx, mod*sy, mod*sz)
 *   R is built from the unit quaternion (r, x, y, z)
 *
 * Parameters:
 *   cudaScale  -- GPU pointer to Scale data, N * 3 floats  (sx, sy, sz)
 *   cudaRot    -- GPU pointer to Rot   data, N * 4 floats  (r, x, y, z)
 *   cudaCov3D  -- GPU output,              N * 6 floats  (c00,c01,c02,c11,c12,c22)
 *   mod        -- uniform scale modifier (typically 1.0f)
 *   N          -- number of Gaussian points
 *   stream     -- CUDA stream (default: 0)
 *
 * Preconditions:
 *   - cudaScale and cudaRot must be fully uploaded before this call.
 *   - cudaCov3D must be pre-allocated to N * 6 * sizeof(float) bytes.
 */
void launchComputeCov3D(
    const float* cudaScale,
    const float* cudaRot,
    float*       cudaCov3D,
    float        mod,
    int          N,
    cudaStream_t stream = 0);

void launchPackFloatsToHalf(
    const float* src,
    uint16_t*    dst,
    int          count,
    cudaStream_t stream = 0);

} // namespace optisplat
