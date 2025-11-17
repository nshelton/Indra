#pragma once

#include <glad/glad.h>
#include <memory>
#include "core/core.h"
#include "Camera.h"
#include "ComputeShader.h"
#include "../ShaderState.h"

/// @brief Hierarchical depth pyramid raymarcher
/// Uses mipmapped depth texture for progressive refinement
class RaymarcherSimple
{
public:
    bool init();
    void shutdown();

    void draw(const Camera& camera, const ShaderState& shaderState);

    bool reloadShaders();
    void uploadCameraParameters(const Camera& camera, ComputeShader* shader);

    void setViewportSize(int width, int height);

    /// @brief Get the output texture that the raymarcher writes to
    GLuint getOutputTexture() const { return m_outputTexture; }

    /// @brief Get GPU execution time of the last raymarch pass
    /// @return Execution time in milliseconds
    float getExecutionTimeMs() const
    {
        return m_baseDepthShader ? m_baseDepthShader->getLastExecutionTimeMs() : 0.0f;
    }

    /// @brief Get work group size used by the compute shader
    void getWorkGroupSize(GLint& sizeX, GLint& sizeY, GLint& sizeZ) const
    {
        if (m_baseDepthShader)
        {
            m_baseDepthShader->getWorkGroupSize(sizeX, sizeY, sizeZ);
        }
        else
        {
            sizeX = sizeY = sizeZ = 0;
        }
    }

private:
    void createOutputTexture();
    void createDepthPyramid();
    void raymarchDepthPyramid(const Camera& camera, const ShaderState& shaderState);
    void shadeFromDepth(const Camera& camera, const ShaderState& shaderState);

    // Compute shaders for hierarchical raymarching
    std::unique_ptr<ComputeShader> m_baseDepthShader;    // 4x4 base level
    std::unique_ptr<ComputeShader> m_refineDepthShader;  // Refinement levels
    std::unique_ptr<ComputeShader> m_shadingShader;      // Final shading pass

    // Depth pyramid (mipmapped R32F texture)
    GLuint m_depthPyramid{0};
    int m_numLevels{0};
    int m_baseLevelIndex{0};  // Index of the ~4x4 starting level

    // Output texture (final RGBA16F shaded result)
    GLuint m_outputTexture{0};

    // Viewport dimensions
    int m_viewportWidth{1920};
    int m_viewportHeight{1080};
};
