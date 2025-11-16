#include "Raymarcher.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <glog/logging.h>
#include <cmath>


void Raymarcher::createOutputTexture()
{
    if (m_outputTexture != 0)
    {
        glDeleteTextures(1, &m_outputTexture);
    }

    glGenTextures(1, &m_outputTexture);
    glBindTexture(GL_TEXTURE_2D, m_outputTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, m_viewportWidth, m_viewportHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    LOG(INFO) << "Created output texture: " << m_viewportWidth << "x" << m_viewportHeight;
}

bool Raymarcher::init()
{
    // Load compute shader from source directory for hot-reload support
    m_computeShader = std::make_unique<ComputeShader>();
    if (!m_computeShader->loadFromFile("../../shaders/raymarch.comp"))
    {
        LOG(ERROR) << "Failed to load compute shader";
        return false;
    }

    // Create output texture
    createOutputTexture();

    LOG(INFO) << "Raymarcher initialized successfully";
    return true;
}

void Raymarcher::shutdown()
{
    m_computeShader.reset();

    if (m_outputTexture != 0)
    {
        glDeleteTextures(1, &m_outputTexture);
        m_outputTexture = 0;
    }
}

void Raymarcher::setViewportSize(int width, int height)
{
    if (m_viewportWidth != width || m_viewportHeight != height)
    {
        m_viewportWidth = width;
        m_viewportHeight = height;
        createOutputTexture();
    }
}

/// @brief Raymarch the scene to a framebuffer using a compute shader
/// @param camera The camera to use for rendering
/// @param shaderState The shader parameters to use for rendering
void Raymarcher::draw(const Camera &camera, const ShaderState &shaderState)
{
    if (!m_computeShader || !m_computeShader->isValid() || m_outputTexture == 0)
        return;

    // Bind the compute shader
    m_computeShader->use();

    // Bind output texture for writing
    glBindImageTexture(0, m_outputTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

    // Upload camera uniforms
    vec3 cameraPos = camera.getPosition();
    vec3 cameraForward = camera.getForward();
    vec3 cameraRight = camera.getRight();
    vec3 cameraUp = camera.getUp();
    float fov = camera.getFov();
    float aspect = camera.getAspect();

    // Calculate tan(fov/2) for the shader
    float fovRadians = fov * 3.14159265359f / 180.0f;
    float tanHalfFov = std::tan(fovRadians / 2.0f);

    glUniform3f(m_computeShader->getUniformLocation("uCameraPos"), cameraPos.x, cameraPos.y, cameraPos.z);
    glUniform3f(m_computeShader->getUniformLocation("uCameraForward"), cameraForward.x, cameraForward.y, cameraForward.z);
    glUniform3f(m_computeShader->getUniformLocation("uCameraRight"), cameraRight.x, cameraRight.y, cameraRight.z);
    glUniform3f(m_computeShader->getUniformLocation("uCameraUp"), cameraUp.x, cameraUp.y, cameraUp.z);
    glUniform1f(m_computeShader->getUniformLocation("uTanHalfFov"), tanHalfFov);
    glUniform1f(m_computeShader->getUniformLocation("uAspect"), aspect);

    // Upload all shader parameters (scene + raymarching params) automatically
    const_cast<ShaderState&>(shaderState).uploadUniforms(m_computeShader.get());

    // Upload viewport parameters
    glUniform1i(m_computeShader->getUniformLocation("uViewportWidth"), m_viewportWidth);
    glUniform1i(m_computeShader->getUniformLocation("uViewportHeight"), m_viewportHeight);

    // Dispatch compute shader
    int workGroupsX = (m_viewportWidth + 15) / 16;
    int workGroupsY = (m_viewportHeight + 15) / 16;
    m_computeShader->dispatch(workGroupsX, workGroupsY, 1);

    // Wait for compute shader to finish
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

bool Raymarcher::reloadShaders()
{
    if (m_computeShader && !m_computeShader->getComputePath().empty())
    {
        bool success = m_computeShader->reload();
        if (success)
        {
            LOG(INFO) << "Compute shader reloaded successfully";
        }
        return success;
    }
    return false;
}