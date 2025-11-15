#include "AudioAnalyzer.h"
#include <cuda_runtime.h>
#include <cufft.h>

// CUDA kernel to compute magnitudes from complex FFT output
__global__ void computeMagnitudesKernel(const cufftComplex* complexData, float* magnitudes, unsigned int numBins)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numBins) {
        float real = complexData[idx].x;
        float imag = complexData[idx].y;
        magnitudes[idx] = sqrtf(real * real + imag * imag);
    }
}

// Host function to launch the magnitude computation kernel
void launchComputeMagnitudes(const cufftComplex* d_complexData, float* d_magnitudes, unsigned int numBins)
{
    const int threadsPerBlock = 256;
    const int numBlocks = (numBins + threadsPerBlock - 1) / threadsPerBlock;

    computeMagnitudesKernel<<<numBlocks, threadsPerBlock>>>(d_complexData, d_magnitudes, numBins);
}