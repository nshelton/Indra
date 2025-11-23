#pragma once

#include "Framebuffer.h"
#include <glad/glad.h>
#include "shader/GraphicsShader.h"
#include <memory>

class PostProcessor {
public:
    PostProcessor();
    ~PostProcessor();

    // Disable copy
    PostProcessor(const PostProcessor&) = delete;
    PostProcessor& operator=(const PostProcessor&) = delete;

    void init(int width, int height);
    void resize(int width, int height);

    // Process HDR input texture and render to screen
    void process(GLuint hdrTexture);

    // Individual effect controls
    void setExposure(float exposure) { m_exposure = exposure; }
    void setBloomStrength(float strength) { m_bloomStrength = strength; }

    float getExposure() const { return m_exposure; }
    float getBloomStrength() const { return m_bloomStrength; }

    bool reloadShaders()
    {
        bool success = true;
        if (m_brightPassShader)
            success &= m_brightPassShader->reload();
        if (m_blurShader)
            success &= m_blurShader->reload();
        if (m_finalShader)
            success &= m_finalShader->reload();
        return success;
    }

private:
    void setupQuad();
    void createShaders();
    void renderQuad();

    // Shaders
    std::unique_ptr<GraphicsShader> m_brightPassShader = nullptr;
    std::unique_ptr<GraphicsShader> m_blurShader = nullptr;
    std::unique_ptr<GraphicsShader> m_finalShader = nullptr;

    // Framebuffers for bloom
    std::unique_ptr<Framebuffer> m_brightnessBuffer;
    std::unique_ptr<Framebuffer> m_blurHBuffer;
    std::unique_ptr<Framebuffer> m_blurVBuffer;

    // Fullscreen quad
    GLuint m_quadVAO = 0;
    GLuint m_quadVBO = 0;

    // Effect parameters
    float m_exposure = 1.0f;
    float m_bloomStrength = 0.04f;
    float m_time = 0.0f;

    int m_width = 0;
    int m_height = 0;
};
