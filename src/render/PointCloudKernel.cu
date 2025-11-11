#include "PointCloudKernel.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// CUDA kernel to animate points (example: wave effect)
__global__ void animatePointsKernel(float* vertices, int numPoints, float time)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numPoints)
    {
        // Each vertex has 7 floats: x, y, z, r, g, b, a
        int offset = idx * 7;

        float x = vertices[offset + 0];
        float z = vertices[offset + 2];

        // Example: add a wave effect to the Y coordinate
        float wave = sinf(x * 0.5f + time) * cosf(z * 0.5f + time) * 0.2f;
        vertices[offset + 1] += wave;

        // Example: animate color based on height
        float heightFactor = (vertices[offset + 1] + 1.0f) * 0.5f; // Normalize to [0,1]
        vertices[offset + 3] = heightFactor;        // R
        vertices[offset + 4] = 1.0f - heightFactor; // G
        vertices[offset + 5] = 0.5f;                // B
    }
}

// Host function to launch the kernel
cudaError_t launchAnimatePointsKernel(float* d_vertices, int numPoints, float time)
{
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    animatePointsKernel<<<numBlocks, blockSize>>>(d_vertices, numPoints, time);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return err;
    }

    // Wait for kernel to complete
    return cudaDeviceSynchronize();
}
