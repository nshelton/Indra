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

    void drawGui();

private:
    void createOutputTextures();
    void createDepthPyramid();
    void raymarchDepthPyramid(const Camera& camera, const ShaderState& shaderState);
    void shadeFromDepth(const Camera& camera, const ShaderState& shaderState);
    void reconstruction(const Camera& camera, const ShaderState& shaderState);
    // Compute shaders for hierarchical raymarching
    std::unique_ptr<ComputeShader> m_baseDepthShader;    // 4x4 base level
    std::unique_ptr<ComputeShader> m_shadingShader;      // Final shading pass
    std::unique_ptr<ComputeShader> m_reconstructionShader;      // Reconstruction pass

    // Depth pyramid (mipmapped R32F texture)
    GLuint m_depthPyramid{0};
    int m_numLevels{0};
    int m_baseLevelIndex{0};  // Index of the ~4x4 starting level

    // Output texture (final RGBA16F shaded result)
    GLuint m_outputTexture{0};
    GLuint m_outputTextureSwap{0};
    GLuint m_currentShadedFrame{0};

    matrix4 m_lastCameraTransform;

    // Viewport dimensions
    int m_viewportWidth{1920};
    int m_viewportHeight{1080};

    // GPU timing
    std::unordered_map<std::string, float> m_lastExecutionTimes;
};
