/**
 * common.cu
 *
 * GPU precomputation utilities for Gaussian Splatting.
 * Functions here are invoked once at initialization (uploadDataToGPU),
 * so per-frame rendering does not repeat any of this work.
 */

#include "common.h"
#include <cuda_runtime.h>

namespace optisplat {

// -----------------------------------------------------------------------
// Kernel: compute 3D covariance matrices for all Gaussian points in parallel.
//
// Each thread processes one Gaussian point.
//
// Formula: Sigma = M^T * M,  where M = S * R^T
//   S   = diag(mod*sx, mod*sy, mod*sz)          -- scaling matrix
//   R^T = transposed rotation matrix built from unit quaternion (r, x, y, z)
//
// Only the upper-triangle of the symmetric result is stored:
//   out[6*i + 0] = Sigma(0,0)
//   out[6*i + 1] = Sigma(0,1)
//   out[6*i + 2] = Sigma(0,2)
//   out[6*i + 3] = Sigma(1,1)
//   out[6*i + 4] = Sigma(1,2)
//   out[6*i + 5] = Sigma(2,2)
//
// Input layout (matches CPU-side Eigen types):
//   scales [N,3]: (sx, sy, sz)
//   rots   [N,4]: (r, x, y, z)  -- normalized quaternion
// -----------------------------------------------------------------------
__global__ void computeCov3DKernel(
    const float* __restrict__ scales,
    const float* __restrict__ rots,
    float*       __restrict__ cov3Ds,
    float mod,
    int   N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // --- scale ---
    float sx = mod * scales[idx * 3 + 0];
    float sy = mod * scales[idx * 3 + 1];
    float sz = mod * scales[idx * 3 + 2];

    // --- quaternion (r, x, y, z) ---
    float r = rots[idx * 4 + 0];
    float x = rots[idx * 4 + 1];
    float y = rots[idx * 4 + 2];
    float z = rots[idx * 4 + 3];

    // Build R^T directly (equivalent to building R then transposing,
    // which is what the CPU reference code does via R.transposeInPlace()).
    //
    // Standard quaternion -> rotation matrix, then transpose:
    //   Rt[row][col] = R[col][row]
    float Rt00 = 1.f - 2.f*(y*y + z*z);  float Rt01 = 2.f*(x*y + r*z);          float Rt02 = 2.f*(x*z - r*y);
    float Rt10 = 2.f*(x*y - r*z);         float Rt11 = 1.f - 2.f*(x*x + z*z);   float Rt12 = 2.f*(y*z + r*x);
    float Rt20 = 2.f*(x*z + r*y);         float Rt21 = 2.f*(y*z - r*x);          float Rt22 = 1.f - 2.f*(x*x + y*y);

    // M = S * Rt  (S is diagonal, so row i of M = scale_i * row i of Rt)
    float M00 = sx*Rt00,  M01 = sx*Rt01,  M02 = sx*Rt02;
    float M10 = sy*Rt10,  M11 = sy*Rt11,  M12 = sy*Rt12;
    float M20 = sz*Rt20,  M21 = sz*Rt21,  M22 = sz*Rt22;

    // Sigma = M^T * M  (symmetric; store upper triangle only)
    //   Sigma(i,j) = sum_k  M[k][i] * M[k][j]
    int base = idx * 6;
    cov3Ds[base + 0] = M00*M00 + M10*M10 + M20*M20;  // (0,0)
    cov3Ds[base + 1] = M00*M01 + M10*M11 + M20*M21;  // (0,1)
    cov3Ds[base + 2] = M00*M02 + M10*M12 + M20*M22;  // (0,2)
    cov3Ds[base + 3] = M01*M01 + M11*M11 + M21*M21;  // (1,1)
    cov3Ds[base + 4] = M01*M02 + M11*M12 + M21*M22;  // (1,2)
    cov3Ds[base + 5] = M02*M02 + M12*M12 + M22*M22;  // (2,2)
}

// -----------------------------------------------------------------------
// Launch wrapper for computeCov3DKernel.
//
// cudaScale and cudaRot must already reside in GPU memory (uploaded before
// this call).  cudaCov3D must be pre-allocated to N*6*sizeof(float) bytes.
// -----------------------------------------------------------------------
void launchComputeCov3D(
    const float* cudaScale,
    const float* cudaRot,
    float*       cudaCov3D,
    float        mod,
    int          N,
    cudaStream_t stream)
{
    if (N <= 0) return;
    constexpr int kBlockSize = 256;
    const int gridSize = (N + kBlockSize - 1) / kBlockSize;
    computeCov3DKernel<<<gridSize, kBlockSize, 0, stream>>>(
        cudaScale, cudaRot, cudaCov3D, mod, N);
}

} // namespace optisplat
