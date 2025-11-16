// Particle data structure for physics simulation
struct ParticleData
{
    float vx, vy, vz;     // Velocity (vec3)
    float age;            // Current age in seconds
    float maxAge;         // Maximum age before respawn
    float pressure;       // Pressure value
    float u, v;           // UV coordinates
};


// Simple hash-based random number generator for CUDA
__device__ float randomFloat(unsigned int seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return float(seed) / 4294967296.0f; // Normalize to [0, 1]
}

// Generate random float in range [min, max]
__device__ float randomRange(unsigned int seed, float minVal, float maxVal)
{
    return minVal + randomFloat(seed) * (maxVal - minVal);
}

__device__ float3 TurboColormap(float x) {
  const float4 kRedVec4 = make_float4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  const float4 kGreenVec4 = make_float4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  const float4 kBlueVec4 = make_float4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  const float2 kRedVec2 = make_float2(-152.94239396, 59.28637943);
  const float2 kGreenVec2 = make_float2(4.27729857, 2.82956604);
  const float2 kBlueVec2 = make_float2(-89.90310912, 27.34824973);

  // Saturate: clamp x to [0, 1]
  x = fminf(fmaxf(x, 0.0f), 1.0f);

  float4 v4 = make_float4(1.0, x, x * x, x * x * x);
  float2 v2 = make_float2(v4.z * v4.z, v4.w * v4.z);

  // Manual dot products
  float r = v4.x * kRedVec4.x + v4.y * kRedVec4.y + v4.z * kRedVec4.z + v4.w * kRedVec4.w + v2.x * kRedVec2.x + v2.y * kRedVec2.y;
  float g = v4.x * kGreenVec4.x + v4.y * kGreenVec4.y + v4.z * kGreenVec4.z + v4.w * kGreenVec4.w + v2.x * kGreenVec2.x + v2.y * kGreenVec2.y;
  float b = v4.x * kBlueVec4.x + v4.y * kBlueVec4.y + v4.z * kBlueVec4.z + v4.w * kBlueVec4.w + v2.x * kBlueVec2.x + v2.y * kBlueVec2.y;

  return make_float3(r, g, b);
}

// CUDA kernel to animate points with audio-reactive effects using FFT data
extern "C" __global__ void animatePointsKernel(float *vertices, ParticleData *particleData,
                                    int numPoints, float deltaTime,
                                    const float *fftData, int numFFTBins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numPoints)
    {
        // Each vertex has 7 floats: x, y, z, r, g, b, a
        int offset = idx * 7;

        // Load particle data
        ParticleData particle = particleData[idx];

        // Update age
        particle.age -= randomRange(idx * 12345, 0.8, 1.2);

        // Check if particle needs to respawn
        if (particle.age <= 0.0f)
        {
            // Generate new random seed based on particle index and current time
            unsigned int seed = idx * 73856093 ^ static_cast<unsigned int>(clock64());
            // Respawn at random position in  x-z plane
            vertices[offset + 0] = 100.0f * randomRange(seed * 10001, -1.0f, 1.0f); // x
            vertices[offset + 1] = idx;
            vertices[offset + 2] = -50.0f;

            // Reset random velocity
            particle.vx = 0.0f;
            particle.vy = 0.0f;
            particle.vz = 0.001f + 0.001f  * randomRange(seed * 20002, 0.0f, 1.0f);

            // Reset age to a random max age between 8 and 10 seconds
            particle.maxAge = 100.0f * randomRange(seed, 0.9f, 1.1f);
            particle.age = particle.maxAge;

            // Set random pressure
            particle.pressure = 0.f;

            if (fftData != nullptr)
            {
                // Add audio reactivity if FFT data is available
                // Map particle x position to FFT bin
                float parameter = (vertices[offset + 0] + 100.0f) / 200.0f;
                 parameter = pow(parameter, 2.0f); // Adjust distribution
                parameter = logf(parameter + 1.0f); // / logf(10.0f) ; // Emphasize lower frequencies
                int binIndex = static_cast<int>(numFFTBins * parameter);
                binIndex = max(0, min(binIndex, numFFTBins - 1));

                // Get FFT magnitude for this bin
                float audioFactorRaw = fftData[binIndex]/ 100.0f;

                // Enhance
                audioFactorRaw = logf(audioFactorRaw + 1.0f) * 10.0f * (1 + 10.0f *parameter);
                // Apply audio-driven displacement (set y position)
                vertices[offset + 1] = audioFactorRaw;

                // Modulate color based on audio

                float3 color  = TurboColormap(min(audioFactorRaw , 1.0f));
                vertices[offset + 3] = color.x * audioFactorRaw; // r
                vertices[offset + 4] = color.y * audioFactorRaw; // g
                vertices[offset + 5] = color.z * audioFactorRaw; // b


                particle.vy = 0.1f * audioFactorRaw * 0.001f  * randomRange(seed * 20002, 0.0f, 1.0f); 

            }
        }
        unsigned int seed = idx * 73856093 ^ static_cast<unsigned int>(clock64());
        // Update position using velocity
        vertices[offset + 0] += particle.vx * deltaTime; // x
        vertices[offset + 1] += particle.vy * deltaTime; // y
        vertices[offset + 2] += particle.vz * deltaTime; // z

        // // Update color alpha based on age (fade out as particle ages)
        // float ageRatio = particle.age / particle.maxAge;
        // vertices[offset + 6] = ageRatio; // a
        // Write particle data back
        particleData[idx] = particle;
    }
}

// Host function to launch the kernel
