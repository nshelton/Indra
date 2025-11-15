#include "PointCloudKernel.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// CUDA kernel to animate points with audio-reactive effects using FFT data
__global__ void animatePointsKernel(float* vertices, int numPoints, float time,
                                     const float* fftData, int numFFTBins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numPoints)
    {
        // Each vertex has 7 floats: x, y, z, r, g, b, a
        int offset = idx * 7;

        float x = vertices[offset + 0];
        float z = vertices[offset + 2];

        // Base wave effect
        float wave = 0;

        // Add audio reactivity if FFT data is available
        float audioFactor = 0.0f;
        if (fftData != nullptr && numFFTBins > 0)
        {
            // Map point position to FFT bin (use x position)
             int binIndex = static_cast<int>(x / 1024) % numFFTBins;
            binIndex = max(0, min(binIndex, numFFTBins - 1));

            // Get FFT magnitude for this bin (normalized)
            audioFactor = fftData[binIndex] * 0.01f; // Scale down FFT values

            // Add audio-driven displacement
            wave = audioFactor;
        }
        vertices[offset + 1] += wave;

    }
}

// Host function to launch the kernel
cudaError_t launchAnimatePointsKernel(float* d_vertices, int numPoints, float time,
                                       const float* d_fftData, int numFFTBins)
{
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    animatePointsKernel<<<numBlocks, blockSize>>>(d_vertices, numPoints, time, d_fftData, numFFTBins);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return err;
    }

    // Wait for kernel to complete
    return cudaDeviceSynchronize();
}
