#pragma once

#include "Framebuffer.h"
#include <glad/glad.h>
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

private:
    void setupQuad();
    void createShaders();
    void renderQuad();

    // Shaders
    GLuint m_brightPassShader = 0;
    GLuint m_blurShader = 0;
    GLuint m_finalShader = 0;

    // Framebuffers for bloom
    std::unique_ptr<Framebuffer> m_brightPass;
    std::unique_ptr<Framebuffer> m_blurH;
    std::unique_ptr<Framebuffer> m_blurV;

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
