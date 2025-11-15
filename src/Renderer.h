#pragma once

#include "core/Core.h"
#include "render/LineRenderer.h"
#include "render/MeshRenderer.h"
#include "render/PointCloudRenderer.h"
#include "render/Framebuffer.h"
#include "render/PostProcessor.h"
#include "Interaction.h"
#include "Scene.h"
#include "Camera.h"

#include <glog/logging.h>
#include <memory>

class Renderer
{
public:
    Renderer();

    void setSize(int width, int height)
    {
        LOG(INFO) << "GL size set to " << width << "x" << height;
        m_width = width;
        m_height = height;

        // Only resize if already initialized (FBO != 0)
        // Otherwise, let render() create it on first frame
        if (m_hdrFramebuffer && m_hdrFramebuffer->getFBO() != 0) {
            m_hdrFramebuffer->resize(width, height);
        }
        if (m_postProcessor) {
            m_postProcessor->resize(width, height);
        }
    }

    void render(const Camera &camera, const SceneModel &scene, const InteractionState &uiState);
    void setPoints(const std::vector<vec3> &points, color col);
    void shutdown();
    void drawGui();

    int totalVertices() const { return static_cast<int>(m_lines.totalVertices()); }

    // HDR postprocessing controls
    void setExposure(float exposure) { if (m_postProcessor) m_postProcessor->setExposure(exposure); }
    void setBloomStrength(float strength) { if (m_postProcessor) m_postProcessor->setBloomStrength(strength); }
    void setGrainAmount(float amount) { if (m_postProcessor) m_postProcessor->setGrainAmount(amount); }

    float getExposure() const { return m_postProcessor ? m_postProcessor->getExposure() : 1.0f; }
    float getBloomStrength() const { return m_postProcessor ? m_postProcessor->getBloomStrength() : 0.04f; }
    float getGrainAmount() const { return m_postProcessor ? m_postProcessor->getGrainAmount() : 0.02f; }

    // Audio reactivity
    void uploadFFTData(const std::vector<float>& fftMagnitudes) { m_points.uploadFFTData(fftMagnitudes); }
    void setFFTDataGPU(const float* d_fftData, int numBins) { m_points.setFFTDataGPU(d_fftData, numBins); }

    // Hot-reload shaders
    bool reloadShaders() { return m_points.reloadShaders(); }

private:
    LineRenderer m_lines{};
    MeshRenderer m_meshes{};
    PointCloudRenderer m_points{};
    float m_time{0.0f};

    // HDR rendering
    std::unique_ptr<Framebuffer> m_hdrFramebuffer;
    std::unique_ptr<PostProcessor> m_postProcessor;
    int m_width{0};
    int m_height{0};
};
