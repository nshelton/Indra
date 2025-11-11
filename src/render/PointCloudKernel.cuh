#pragma once

#include <cuda_runtime.h>

// Host function to launch the animation kernel on device vertices
cudaError_t launchAnimatePointsKernel(float* d_vertices, int numPoints, float time);
