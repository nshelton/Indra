#pragma once

#include <cuda_runtime.h>

// Host function to launch the animation kernel on device vertices with FFT data
cudaError_t launchAnimatePointsKernel(float* d_vertices, int numPoints, float time,
                                       const float* d_fftData, int numFFTBins);
