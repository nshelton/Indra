#pragma once

#include <glad/glad.h>
#include <vector>
#include <memory>
#include "core/core.h"
#include "Camera.h"
#include "ComputeShader.h"
#include "../ShaderState.h"

class RaymarcherLottes
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

    /// @brief Reset accumulation buffer (call when camera moves)
    void resetAccumulation();

    /// @brief Set frame number for temporal variation
    void setFrameNumber(int frame) { m_frameNumber = frame; }

    /// @brief Set camera changed flag
    void setCameraChanged(bool changed) { m_cameraChanged = changed; }

    /// @brief Get accumulation texture
    GLuint getAccumulationTexture() const { return m_accumulationTexture; }

    /// @brief Notify that shader parameters changed (triggers reset)
    void notifyParametersChanged() { resetAccumulation(); }

private:
    // Helper functions
    void createOutputTexture();
    void createAccumulationTexture();
    void createWorkQueueBuffer();
    void createDepthCacheBuffer();

    // Compute shader for raymarching
    std::unique_ptr<ComputeShader> m_computeShader;

    // Output texture that the compute shader writes to
    GLuint m_outputTexture{0};

    // Accumulation texture for temporal refinement
    GLuint m_accumulationTexture{0};

    // Work queue SSBO (atomic counter + total rays)
    GLuint m_workQueueSSBO{0};

    // Depth cache SSBO (breadcrumbs from parent rays)
    GLuint m_depthCacheSSBO{0};

    // Viewport dimensions
    int m_viewportWidth{1920};
    int m_viewportHeight{1080};

    // Hierarchical raymarching parameters
    int m_frameNumber{0};
    bool m_cameraChanged{true};
    int m_iterationBudget{10000000};  // Total iterations per frame (10M for progressive refinement)
    int m_iterationsPerThread{512};   // Iterations per thread
};

