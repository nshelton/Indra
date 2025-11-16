#pragma once

#include "core/Core.h"
#include "render/LineRenderer.h"
#include "render/Raymarcher.h"
#include "render/Framebuffer.h"
#include "render/PostProcessor.h"
#include "render/TextureBlit.h"
#include "Interaction.h"
#include "Camera.h"
#include "ShaderState.h"

#include <glog/logging.h>
#include <memory>

class Renderer
{
public:
    Renderer();

    /// @brief Initialize OpenGL resources - must be called after OpenGL context is ready
    void init();

    void setSize(int width, int height)
    {
        LOG(INFO) << "GL size set to " << width << "x" << height;
        m_width = width;
        m_height = height;

        m_raymarcher.setViewportSize(width, height);

        if (m_hdrFramebuffer && m_hdrFramebuffer->getFBO() != 0) {
            m_hdrFramebuffer->resize(width, height);
        } else if (m_width > 0 && m_height > 0) {
            // First time initialization
            m_hdrFramebuffer->createHDR(m_width, m_height);
            m_postProcessor->init(m_width, m_height);
        }

        if (m_postProcessor && m_width > 0 && m_height > 0) {
            m_postProcessor->resize(width, height);
        }
    }

    void render(const Camera &camera, const ShaderState &shaderState, const InteractionState &uiState);
    void shutdown();
    void drawGui();

    // HDR postprocessing controls
    void setExposure(float exposure) { if (m_postProcessor) m_postProcessor->setExposure(exposure); }
    void setBloomStrength(float strength) { if (m_postProcessor) m_postProcessor->setBloomStrength(strength); }
    void setGrainAmount(float amount) { if (m_postProcessor) m_postProcessor->setGrainAmount(amount); }

    float getExposure() const { return m_postProcessor ? m_postProcessor->getExposure() : 1.0f; }
    float getBloomStrength() const { return m_postProcessor ? m_postProcessor->getBloomStrength() : 0.04f; }
    float getGrainAmount() const { return m_postProcessor ? m_postProcessor->getGrainAmount() : 0.02f; }

    // Audio reactivity

    // Hot-reload shaders and kernels
    bool reloadShaders() { return m_raymarcher.reloadShaders(); }

    // Performance metrics
    float getRaymarcherExecutionTimeMs() const { return m_raymarcher.getExecutionTimeMs(); }
    void getRaymarcherWorkGroupSize(GLint& sizeX, GLint& sizeY, GLint& sizeZ) const
    {
        m_raymarcher.getWorkGroupSize(sizeX, sizeY, sizeZ);
    }

private:
    bool m_initialized{false};

    LineRenderer m_lines{};
    Raymarcher m_raymarcher{};
    float m_time{0.0f};

    // HDR rendering
    std::unique_ptr<Framebuffer> m_hdrFramebuffer;
    std::unique_ptr<PostProcessor> m_postProcessor;
    int m_width{0};
    int m_height{0};
};
