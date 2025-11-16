#pragma once

#include <glad/glad.h>
#include <vector>
#include <memory>
#include "core/core.h"
#include "Camera.h"
#include "ShaderProgram.h"
#include "Scene.h"

class Raymarcher
{
public:
    bool init();
    void shutdown();

    void draw(const Camera& camera, const Scene& scene);

    bool reloadShaders();

private:
    struct GLVertex { float x, y, z, r, g, b, a; };

    // Shader management
    std::unique_ptr<ShaderProgram> m_shaderProgram;


    float m_pointRadius{1.0f};

    std::vector<GLVertex> m_points;   // point vertices (single positions)

    // FFT data for audio reactivity
    // float* m_d_fftData{nullptr};  // Device pointer for FFT data
    // int m_numFFTBins{0};
    // bool m_ownsFFTData{false};  // Track if we allocated the FFT buffer (and should free it)

};

