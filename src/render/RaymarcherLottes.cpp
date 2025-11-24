#include "RaymarcherLottes.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <glog/logging.h>
#include <cmath>


void RaymarcherLottes::createOutputTexture()
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

void RaymarcherLottes::createAccumulationTexture()
{
    if (m_accumulationTexture != 0)
    {
        glDeleteTextures(1, &m_accumulationTexture);
    }

    // Initialize texture with zeros (important for alpha channel check)
    std::vector<float> zeros(m_viewportWidth * m_viewportHeight * 4, 0.0f);

    glGenTextures(1, &m_accumulationTexture);
    glBindTexture(GL_TEXTURE_2D, m_accumulationTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, m_viewportWidth, m_viewportHeight, 0, GL_RGBA, GL_FLOAT, zeros.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    LOG(INFO) << "Created accumulation texture: " << m_viewportWidth << "x" << m_viewportHeight;
}

void RaymarcherLottes::createWorkQueueBuffer()
{
    if (m_workQueueSSBO != 0)
    {
        glDeleteBuffers(1, &m_workQueueSSBO);
    }

    // Work queue contains: [currentRayIndex, totalRays]
    // Total rays includes all pyramid levels:
    // Level 0: (w/4)*(h/4) coarse pixels
    // Level 1: (w/2)*(h/2) medium pixels
    // Level 2: w*h ALL pixels (full coverage)
    GLuint raysLevel0 = (m_viewportWidth / 4) * (m_viewportHeight / 4);
    GLuint raysLevel1 = (m_viewportWidth / 2) * (m_viewportHeight / 2);
    GLuint raysLevel2 = m_viewportWidth * m_viewportHeight;
    GLuint totalRays = raysLevel0 + raysLevel1 + raysLevel2;

    GLuint workQueueData[2] = {0, totalRays};

    glGenBuffers(1, &m_workQueueSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_workQueueSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(workQueueData), workQueueData, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    LOG(INFO) << "Created work queue SSBO (totalRays: " << totalRays
              << ", L0:" << raysLevel0 << " L1:" << raysLevel1 << " L2:" << raysLevel2 << ")";
}

void RaymarcherLottes::createDepthCacheBuffer()
{
    if (m_depthCacheSSBO != 0)
    {
        glDeleteBuffers(1, &m_depthCacheSSBO);
    }

    // Depth cache stores one float per RAY (not per pixel!)
    // Need to store depths for ALL pyramid levels
    GLuint raysLevel0 = (m_viewportWidth / 4) * (m_viewportHeight / 4);
    GLuint raysLevel1 = (m_viewportWidth / 2) * (m_viewportHeight / 2);
    GLuint raysLevel2 = m_viewportWidth * m_viewportHeight;
    GLuint totalRays = raysLevel0 + raysLevel1 + raysLevel2;
    size_t bufferSize = totalRays * sizeof(float);

    glGenBuffers(1, &m_depthCacheSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_depthCacheSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    LOG(INFO) << "Created depth cache SSBO (" << (bufferSize / 1024.0f / 1024.0f) << " MB)";
}

bool RaymarcherLottes::init()
{
    // Load compute shader from source directory for hot-reload support
    m_computeShader = std::make_unique<ComputeShader>();
    if (!m_computeShader->loadFromFile("../../shaders/raymarch_lottes.comp"))
    {
        LOG(ERROR) << "Failed to load compute shader";
        return false;
    }

    // Create textures and buffers
    createOutputTexture();
    createAccumulationTexture();
    createWorkQueueBuffer();
    createDepthCacheBuffer();

    LOG(INFO) << "Raymarcher initialized successfully";
    return true;
}

void RaymarcherLottes::shutdown()
{
    m_computeShader.reset();

    if (m_outputTexture != 0)
    {
        glDeleteTextures(1, &m_outputTexture);
        m_outputTexture = 0;
    }

    if (m_accumulationTexture != 0)
    {
        glDeleteTextures(1, &m_accumulationTexture);
        m_accumulationTexture = 0;
    }

    if (m_workQueueSSBO != 0)
    {
        glDeleteBuffers(1, &m_workQueueSSBO);
        m_workQueueSSBO = 0;
    }

    if (m_depthCacheSSBO != 0)
    {
        glDeleteBuffers(1, &m_depthCacheSSBO);
        m_depthCacheSSBO = 0;
    }
}

void RaymarcherLottes::setViewportSize(int width, int height)
{
    if (m_viewportWidth != width || m_viewportHeight != height)
    {
        m_viewportWidth = width;
        m_viewportHeight = height;
        createOutputTexture();
        createAccumulationTexture();
        createWorkQueueBuffer();
        createDepthCacheBuffer();
        resetAccumulation();  // Reset when viewport changes
    }
}

/// @brief Raymarch the scene to a framebuffer using a compute shader
/// @param camera The camera to use for rendering
void RaymarcherLottes::draw(const Camera &camera)
{
    if (!m_computeShader || !m_computeShader->isValid() || m_outputTexture == 0)
        return;

    // Only reset work queue counter if camera changed
    // Otherwise continue from where we left off (progressive refinement)
    if (m_cameraChanged)
    {
        GLuint zero = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_workQueueSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &zero);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    // Bind the compute shader
    m_computeShader->use();

    // Bind textures
    glBindImageTexture(0, m_outputTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
    glBindImageTexture(1, m_accumulationTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA16F);

    // Bind SSBOs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_workQueueSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_depthCacheSSBO);

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
    // const_cast<ShaderState&>(shaderState).uploadUniforms(m_computeShader.get());

    // Upload viewport parameters
    glUniform1i(m_computeShader->getUniformLocation("uViewportWidth"), m_viewportWidth);
    glUniform1i(m_computeShader->getUniformLocation("uViewportHeight"), m_viewportHeight);

    // Upload hierarchical raymarching parameters
    glUniform1i(m_computeShader->getUniformLocation("uIterationsPerThread"), m_iterationsPerThread);
    glUniform1i(m_computeShader->getUniformLocation("uFrameNumber"), m_frameNumber);
    glUniform1i(m_computeShader->getUniformLocation("uCameraChanged"), m_cameraChanged ? 1 : 0);
    glUniform1i(m_computeShader->getUniformLocation("uPyramidLevels"), 3);  // 3 levels: coarse, medium, fine

    // Dispatch compute shader with 1D workgroups
    // Calculate number of threads needed based on iteration budget
    int numThreads = m_iterationBudget / m_iterationsPerThread;
    int workGroupsX = (numThreads + 255) / 256;  // 256 threads per workgroup
    m_computeShader->dispatch(workGroupsX, 1, 1);

    // Wait for compute shader to finish
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    // Debug: Read back progress
    GLuint rayProgress;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_workQueueSSBO);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &rayProgress);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // Calculate actual total rays (all pyramid levels)
    GLuint raysLevel0 = (m_viewportWidth / 4) * (m_viewportHeight / 4);
    GLuint raysLevel1 = (m_viewportWidth / 2) * (m_viewportHeight / 2);
    GLuint raysLevel2 = m_viewportWidth * m_viewportHeight;
    GLuint totalRays = raysLevel0 + raysLevel1 + raysLevel2;
    float percentComplete = (float)rayProgress / (float)totalRays * 100.0f;

    // Log every frame for debugging, or reduce frequency later
    if (rayProgress < totalRays)
    {
        LOG(INFO) << "Frame " << m_frameNumber
                  << ": Progressive refinement - " << rayProgress << "/" << totalRays
                  << " rays (" << percentComplete << "%)";
    }
    else if (m_frameNumber == 0)
    {
        LOG(INFO) << "Frame completed in single pass! (" << rayProgress << " rays)";
    }

    // After first frame, camera is no longer considered "changed"
    if (m_cameraChanged)
    {
        m_cameraChanged = false;
    }
}

void RaymarcherLottes::resetAccumulation()
{
    // Reset work queue counter
    GLuint zero = 0;
    if (m_workQueueSSBO != 0)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_workQueueSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &zero);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    // Clear output texture to black - makes progressive refinement visible
    if (m_outputTexture != 0)
    {
        std::vector<float> zeros(m_viewportWidth * m_viewportHeight * 4, 0.0f);
        glBindTexture(GL_TEXTURE_2D, m_outputTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_viewportWidth, m_viewportHeight,
                        GL_RGBA, GL_FLOAT, zeros.data());
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // Clear accumulation texture
    if (m_accumulationTexture != 0)
    {
        std::vector<float> zeros(m_viewportWidth * m_viewportHeight * 4, 0.0f);
        glBindTexture(GL_TEXTURE_2D, m_accumulationTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_viewportWidth, m_viewportHeight,
                        GL_RGBA, GL_FLOAT, zeros.data());
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // Clear depth cache (for ALL rays, not just pixels!)
    if (m_depthCacheSSBO != 0)
    {
        GLuint raysLevel0 = (m_viewportWidth / 4) * (m_viewportHeight / 4);
        GLuint raysLevel1 = (m_viewportWidth / 2) * (m_viewportHeight / 2);
        GLuint raysLevel2 = m_viewportWidth * m_viewportHeight;
        GLuint totalRays = raysLevel0 + raysLevel1 + raysLevel2;
        size_t bufferSize = totalRays * sizeof(float);
        std::vector<float> zeros(totalRays, 0.0f);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_depthCacheSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bufferSize, zeros.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    // Reset frame counter
    m_frameNumber = 0;
    m_cameraChanged = true;

    LOG(INFO) << "Accumulation reset";
}

bool RaymarcherLottes::reloadShaders()
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