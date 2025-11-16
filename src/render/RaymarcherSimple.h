#pragma once

#include <glad/glad.h>
#include <memory>
#include "core/core.h"
#include "Camera.h"
#include "ComputeShader.h"
#include "../ShaderState.h"

/// @brief Simple per-pixel raymarcher (no hierarchical optimization)
class RaymarcherSimple
{
public:
    bool init();
    void shutdown();

    void draw(const Camera& camera, const ShaderState& shaderState);

    bool reloadShaders();

    void setViewportSize(int width, int height);

    /// @brief Get the output texture that the raymarcher writes to
    GLuint getOutputTexture() const { return m_outputTexture; }

    /// @brief Get GPU execution time of the last raymarch pass
    /// @return Execution time in milliseconds
    float getExecutionTimeMs() const
    {
        return m_computeShader ? m_computeShader->getLastExecutionTimeMs() : 0.0f;
    }

    /// @brief Get work group size used by the compute shader
    void getWorkGroupSize(GLint& sizeX, GLint& sizeY, GLint& sizeZ) const
    {
        if (m_computeShader)
        {
            m_computeShader->getWorkGroupSize(sizeX, sizeY, sizeZ);
        }
        else
        {
            sizeX = sizeY = sizeZ = 0;
        }
    }

private:
    void createOutputTexture();

    // Compute shader for raymarching
    std::unique_ptr<ComputeShader> m_computeShader;

    // Output texture that the compute shader writes to
    GLuint m_outputTexture{0};

    // Viewport dimensions
    int m_viewportWidth{1920};
    int m_viewportHeight{1080};
};
