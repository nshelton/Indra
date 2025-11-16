#pragma once

#include "core/Core.h"
#include "render/LineRenderer.h"
#include "render/RaymarcherSimple.h"
#include "render/Framebuffer.h"
#include "render/PostProcessor.h"
#include "render/TextureBlit.h"
#include "Interaction.h"
#include "Camera.h"
#include "ShaderState.h"

#include <memory>

class Renderer
{
public:
    Renderer();

    /// @brief Initialize OpenGL resources - must be called after OpenGL context is ready
    void init();

    void setSize(int width, int height);

    void render(const Camera &camera, const ShaderState &shaderState, const InteractionState &uiState);
    void shutdown();
    void drawGui();

    // HDR postprocessing controls
    void setExposure(float exposure) { if (m_postProcessor) m_postProcessor->setExposure(exposure); }
    void setBloomStrength(float strength) { if (m_postProcessor) m_postProcessor->setBloomStrength(strength); }

    float getExposure() const { return m_postProcessor ? m_postProcessor->getExposure() : 1.0f; }
    float getBloomStrength() const { return m_postProcessor ? m_postProcessor->getBloomStrength() : 0.04f; }

    void fromJson(const nlohmann::json &j)
    {
        m_lines.setLineWidth(j.value("lineWidth", 1.0f));
        m_postProcessor->setExposure(j.value("exposure", 1.0f));
        m_postProcessor->setBloomStrength(j.value("bloomStrength", 0.04f));
    }

    void toJson(nlohmann::json &j) const
    {
        j["lineWidth"] = m_lines.lineWidth();
        j["exposure"] = getExposure();
        j["bloomStrength"] = getBloomStrength();
    }

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
    RaymarcherSimple m_raymarcher{};
    float m_time{0.0f};

    // HDR rendering
    std::unique_ptr<Framebuffer> m_hdrFramebuffer;
    std::unique_ptr<PostProcessor> m_postProcessor;
    int m_width{0};
    int m_height{0};
};
