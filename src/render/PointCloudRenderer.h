#pragma once

#include <glad/glad.h>
#include <vector>
#include <memory>
#include "core/core.h"
#include "Camera.h"
#include "ShaderProgram.h"

// Forward declare CUDA types to avoid requiring CUDA headers in this header
struct cudaGraphicsResource;

class PointCloudRenderer
{
public:
    bool init();
    void shutdown();

    void clear();
    void setPointSize(float w) { m_pointRadius = w; }
    float pointSize() const { return m_pointRadius; }

    // Add a colored 3D line segment
    void addLine(vec3 a, vec3 b, color c);
    // Add a 3D point to be rendered as a filled circle (GL_POINTS sprite)
    void addPoint(vec3 p, color c);

    // Add a grid on the XZ plane (horizontal ground plane)
    // center: center point of the grid
    // size: total size of the grid in world units
    // divisions: number of grid cells in each direction (will draw divisions+1 lines per axis)
    // gridColor: color for regular grid lines
    // centerColor: color for the center lines (X and Z axes)
    void addGrid(vec3 center, float size, int divisions, color gridColor, color centerColor);

    int totalVertices() const { return static_cast<int>(m_points.size()); }

    void uploadToGPU();  // Upload CPU data to VBO
    void draw(const Camera& camera);

    // CUDA Interop methods
    bool initCudaInterop();
    void shutdownCudaInterop();
    bool runCudaKernel(float deltaTime);  // Example: animate points with CUDA

    // Audio reactivity
    void uploadFFTData(const std::vector<float>& fftMagnitudes);
    // Use FFT data directly from GPU (avoids CPU round-trip)
    void setFFTDataGPU(const float* d_fftData, int numBins);

    // Hot-reload shaders
    bool reloadShaders();

private:
    struct GLVertex { float x, y, z, r, g, b, a; };

    // Shader management
    std::unique_ptr<ShaderProgram> m_shaderProgram;
    GLuint m_vao{0};
    GLuint m_vbo{0};

    float m_pointRadius{1.0f};

    std::vector<GLVertex> m_points;   // point vertices (single positions)

    // CUDA Interop resources
    cudaGraphicsResource* m_cudaVboResource{nullptr};
    bool m_cudaInteropInitialized{false};
    bool m_cudaInteropAttempted{false};  // Track if we've tried to init CUDA

    // FFT data for audio reactivity
    float* m_d_fftData{nullptr};  // Device pointer for FFT data
    int m_numFFTBins{0};
    bool m_ownsFFTData{false};  // Track if we allocated the FFT buffer (and should free it)
};

