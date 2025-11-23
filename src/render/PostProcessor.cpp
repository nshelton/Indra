#include "PostProcessor.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

PostProcessor::PostProcessor()
{
}

PostProcessor::~PostProcessor()
{
    if (m_quadVAO)
        glDeleteVertexArrays(1, &m_quadVAO);
    if (m_quadVBO)
        glDeleteBuffers(1, &m_quadVBO);
}

void PostProcessor::init(int width, int height)
{
    m_width = width;
    m_height = height;

    setupQuad();
    createShaders();

    // Create bloom framebuffers at half resolution for performance
    int bloomWidth = width / 2;
    int bloomHeight = height / 2;

    m_brightnessBuffer = std::make_unique<Framebuffer>();
    m_brightnessBuffer->createColorOnly(bloomWidth, bloomHeight, GL_RGBA16F);

    m_blurHBuffer = std::make_unique<Framebuffer>();
    m_blurHBuffer->createColorOnly(bloomWidth, bloomHeight, GL_RGBA16F);
    m_blurVBuffer = std::make_unique<Framebuffer>();
    m_blurVBuffer->createColorOnly(bloomWidth, bloomHeight, GL_RGBA16F);
}

void PostProcessor::resize(int width, int height)
{
    if (width == m_width && height == m_height)
        return;

    m_width = width;
    m_height = height;

    int bloomWidth = width / 2;
    int bloomHeight = height / 2;

    if (m_brightnessBuffer)
        m_brightnessBuffer->resize(bloomWidth, bloomHeight);
    if (m_blurHBuffer)
        m_blurHBuffer->resize(bloomWidth, bloomHeight);
    if (m_blurVBuffer)
        m_blurVBuffer->resize(bloomWidth, bloomHeight);
}

void PostProcessor::setupQuad()
{
    // Fullscreen quad: position (xy) and texcoord (zw)
    float quadVertices[] = {
        -1.0f, 1.0f, 0.0f, 1.0f,  // Top-left
        -1.0f, -1.0f, 0.0f, 0.0f, // Bottom-left
        1.0f, -1.0f, 1.0f, 0.0f,  // Bottom-right
        -1.0f, 1.0f, 0.0f, 1.0f,  // Top-left
        1.0f, -1.0f, 1.0f, 0.0f,  // Bottom-right
        1.0f, 1.0f, 1.0f, 1.0f    // Top-right
    };

    glGenVertexArrays(1, &m_quadVAO);
    glGenBuffers(1, &m_quadVBO);

    glBindVertexArray(m_quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));

    glBindVertexArray(0);
}

void PostProcessor::createShaders()
{

    m_brightPassShader = std::make_unique<GraphicsShader>();
    m_brightPassShader->loadFromFiles("../../shaders/basicQuadVert.glsl", "../../shaders/postprocess_brightnessThreshold.frag");
    m_blurShader = std::make_unique<GraphicsShader>();
    m_blurShader->loadFromFiles("../../shaders/basicQuadVert.glsl", "../../shaders/postprocess_blur.frag");
    m_finalShader = std::make_unique<GraphicsShader>();
    m_finalShader->loadFromFiles("../../shaders/basicQuadVert.glsl", "../../shaders/postprocess_comp.frag");
}

void PostProcessor::renderQuad()
{
    glBindVertexArray(m_quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void PostProcessor::process(GLuint hdrTexture)
{
    m_time += 0.016f; // Approximate frame time for noise animation

    glDisable(GL_DEPTH_TEST);

    // 1. Bright pass: extract bright pixels
    m_brightnessBuffer->bind();
    glClear(GL_COLOR_BUFFER_BIT);
    m_brightPassShader->use();

    glUniform1i(glGetUniformLocation(m_brightPassShader->getProgram(), "uHdrTexture"), 0);
    glUniform1f(glGetUniformLocation(m_brightPassShader->getProgram(), "uThreshold"), 1.0f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, hdrTexture);
    renderQuad();

    // 2. Horizontal blur
    glClear(GL_COLOR_BUFFER_BIT);
    m_blurShader->use();
    glUniform1i(glGetUniformLocation(m_blurShader->getProgram(), "uTexture"), 0);
    glUniform2f(glGetUniformLocation(m_blurShader->getProgram(), "uDirection"), 1.0f, 0.0f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_brightnessBuffer->getColorTexture());
    renderQuad();

    // 3. Vertical blur
    m_blurVBuffer->bind();
    glClear(GL_COLOR_BUFFER_BIT);
    m_blurShader->use();
    glUniform1i(glGetUniformLocation(m_blurShader->getProgram(), "uTexture"), 0);
    glUniform2f(glGetUniformLocation(m_blurShader->getProgram(), "uDirection"), 0.0f, 1.0f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_blurHBuffer->getColorTexture());
    renderQuad();

    // 4. Final composite to screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, m_width, m_height);
    glClear(GL_COLOR_BUFFER_BIT);
    m_finalShader->use();
    glUniform1i(glGetUniformLocation(m_finalShader->getProgram(), "uHdrTexture"), 0);
    glUniform1i(glGetUniformLocation(m_finalShader->getProgram(), "uBloomTexture"), 1);
    glUniform1f(glGetUniformLocation(m_finalShader->getProgram(), "uExposure"), m_exposure);
    glUniform1f(glGetUniformLocation(m_finalShader->getProgram(), "uBloomStrength"), m_bloomStrength);
    glUniform1f(glGetUniformLocation(m_finalShader->getProgram(), "uTime"), m_time);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, hdrTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_blurVBuffer->getColorTexture());

    renderQuad();

    glEnable(GL_DEPTH_TEST);
}
