#pragma once

#include <cuda_runtime.h>

// Particle data structure for physics simulation
struct ParticleData
{
    float vx, vy, vz;     // Velocity (vec3)
    float age;            // Current age in seconds
    float maxAge;         // Maximum age before respawn
    float pressure;       // Pressure value
    float u, v;           // UV coordinates
};

// Host function to launch the animation kernel on device vertices with FFT data
cudaError_t launchAnimatePointsKernel(float* d_vertices, ParticleData* d_particleData,
                                       int numPoints, float deltaTime,
                                       const float* d_fftData, int numFFTBins);
