#include "RaymarcherSimple.h"
#include <iostream>
#include <glog/logging.h>
#include <cmath>

#define TOP_LEVEL 5

void createTexture(GLuint &texture, GLenum internalFormat, int width, int height, int levels, GLenum format, GLenum type)
{
    if (texture != 0)
    {
        glDeleteTextures(1, &texture);
    }

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, levels, internalFormat, width, height);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void RaymarcherSimple::createOutputTextures()
{
    createTexture(m_outputTexture, GL_RGBA16F, m_viewportWidth, m_viewportHeight, 1, GL_RGBA, GL_HALF_FLOAT);
    createTexture(m_outputTextureSwap, GL_RGBA16F, m_viewportWidth, m_viewportHeight, 1, GL_RGBA, GL_HALF_FLOAT);
    createTexture(m_currentShadedFrame, GL_RGBA16F, m_viewportWidth, m_viewportHeight, 1, GL_RGBA, GL_HALF_FLOAT);
    LOG(INFO) << "RaymarcherSimple: Created output textures: " << m_viewportWidth << "x" << m_viewportHeight;
}

void RaymarcherSimple::createDepthPyramid()
{
    if (m_depthPyramid != 0)
    {
        glDeleteTextures(1, &m_depthPyramid);
    }

    // Calculate number of mip levels needed
    int maxDim = std::max(m_viewportWidth, m_viewportHeight);
    m_numLevels = 1 + static_cast<int>(std::floor(std::log2(maxDim)));

    // Create mipmapped R32F texture
    glGenTextures(1, &m_depthPyramid);
    glBindTexture(GL_TEXTURE_2D, m_depthPyramid);
    glTexStorage2D(GL_TEXTURE_2D, m_numLevels, GL_R32F, m_viewportWidth, m_viewportHeight);
    glGenerateMipmap(GL_TEXTURE_2D);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    LOG(INFO) << "RaymarcherSimple: Created depth pyramid with " << m_numLevels
              << " levels, base level " << m_baseLevelIndex
              << " (" << (maxDim >> m_baseLevelIndex) << "x" << (maxDim >> m_baseLevelIndex) << ")";
}

bool RaymarcherSimple::init()
{
    // Load base depth shader (for coarse level)
    m_baseDepthShader = std::make_unique<ComputeShader>();
    if (!m_baseDepthShader->loadFromFile("../../shaders/raymarch_depth_base.comp"))
    {
        LOG(ERROR) << "RaymarcherSimple: Failed to load base depth shader";
        return false;
    }

    // Load shading shader
    m_shadingShader = std::make_unique<ComputeShader>();
    if (!m_shadingShader->loadFromFile("../../shaders/shade_from_depth.comp"))
    {
        LOG(ERROR) << "RaymarcherSimple: Failed to load shading shader";
        return false;
    }

    // Load reconstruction shader
    m_reconstructionShader = std::make_unique<ComputeShader>();
    if (!m_reconstructionShader->loadFromFile("../../shaders/reconstruction.comp"))
    {
        LOG(ERROR) << "RaymarcherSimple: Failed to load reconstruction shader";
        return false;
    }

    // Create depth pyramid and output texture
    createDepthPyramid();
    createOutputTextures();

    LOG(INFO) << "RaymarcherSimple: Initialized successfully with hierarchical depth pyramid";
    return true;
}

void RaymarcherSimple::shutdown()
{
    m_baseDepthShader.reset();
    m_shadingShader.reset();

    if (m_depthPyramid != 0)
    {
        glDeleteTextures(1, &m_depthPyramid);
        m_depthPyramid = 0;
    }

    if (m_outputTexture != 0)
    {
        glDeleteTextures(1, &m_outputTexture);
        m_outputTexture = 0;
    }

    if (m_outputTextureSwap != 0)
    {
        glDeleteTextures(1, &m_outputTextureSwap);
        m_outputTextureSwap = 0;
    }
    if (m_currentShadedFrame != 0)
    {
        glDeleteTextures(1, &m_currentShadedFrame);
        m_currentShadedFrame = 0;
    }
}

void RaymarcherSimple::setViewportSize(int width, int height)
{
    if (m_viewportWidth != width || m_viewportHeight != height)
    {
        m_viewportWidth = width;
        m_viewportHeight = height;
        createDepthPyramid();
        createOutputTextures();
    }
}

void RaymarcherSimple::uploadCameraParameters(const Camera &camera, ComputeShader *shader)
{
    vec3 cameraPos = camera.getPosition();
    vec3 cameraForward = camera.getForward();
    vec3 cameraRight = camera.getRight();
    vec3 cameraUp = camera.getUp();
    float fov = camera.getFov();
    float aspect = camera.getAspect();

    // Calculate tan(fov/2) for the shader
    float fovRadians = fov * 3.14159265359f / 180.0f;
    float tanHalfFov = std::tan(fovRadians / 2.0f);

    glUniform3f(shader->getUniformLocation("uCameraPos"), cameraPos.x, cameraPos.y, cameraPos.z);
    glUniform3f(shader->getUniformLocation("uCameraForward"), cameraForward.x, cameraForward.y, cameraForward.z);
    glUniform3f(shader->getUniformLocation("uCameraRight"), cameraRight.x, cameraRight.y, cameraRight.z);
    glUniform3f(shader->getUniformLocation("uCameraUp"), cameraUp.x, cameraUp.y, cameraUp.z);
    glUniform1f(shader->getUniformLocation("uTanHalfFov"), tanHalfFov);
    glUniform1f(shader->getUniformLocation("uAspect"), aspect);

    // Upload viewport parameters
    glUniform1i(shader->getUniformLocation("uViewportWidth"), m_viewportWidth);
    glUniform1i(shader->getUniformLocation("uViewportHeight"), m_viewportHeight);
}

void RaymarcherSimple::raymarchDepthPyramid(const Camera &camera, const ShaderState &shaderState)
{

    if (!m_baseDepthShader || m_depthPyramid == 0)
        return;

    // Pass 1: Build depth pyramid (coarse to fine)

    m_baseDepthShader->use();
    uploadCameraParameters(camera, m_baseDepthShader.get());
    const_cast<ShaderState &>(shaderState).uploadUniforms(m_baseDepthShader.get());

    // Bind current level for writing

    int location = m_baseDepthShader->getUniformLocationCached("uSeed4");
    glUniform4f(location, 
        static_cast<float>(std::rand()) / RAND_MAX,
        static_cast<float>(std::rand()) / RAND_MAX,
        static_cast<float>(std::rand()) / RAND_MAX,
        static_cast<float>(std::rand()) / RAND_MAX);

    for (int level = TOP_LEVEL; level >= 0; level--)
    {
        auto start = std::chrono::steady_clock::now();

        int location = m_baseDepthShader->getUniformLocationCached("uLevel");
        glUniform1i(location, level);

        int levelWidth = m_viewportWidth >> level;
        int levelHeight = m_viewportHeight >> level;

        // Bind previous level for reading
        if (level < TOP_LEVEL)
        {
            glBindImageTexture(0, m_depthPyramid, level + 1, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        }

        // Bind this levelfor writing
        glBindImageTexture(1, m_depthPyramid, level, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        // Dispatch
        int workGroupsX = (levelWidth + 15) / 16;
        int workGroupsY = (levelHeight + 15) / 16;
        m_baseDepthShader->dispatch(workGroupsX, workGroupsY, 1);
        // Memory barrier to ensure level is complete before next level reads it
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        m_lastExecutionTimes["raymarch l" + std::to_string(level)] = duration.count() / 1000.0f;
    }
}

void RaymarcherSimple::shadeFromDepth(const Camera &camera, const ShaderState &shaderState)
{
    if (!m_shadingShader || m_depthPyramid == 0 || m_outputTexture == 0)
        return;

    m_shadingShader->use();
    uploadCameraParameters(camera, m_shadingShader.get());
    const_cast<ShaderState &>(shaderState).uploadUniforms(m_shadingShader.get());

    glUniform1f(m_shadingShader->getUniformLocationCached("uSeed"), static_cast<float>(std::rand()) / RAND_MAX);

    // Bind depth pyramid level 0 for reading
    glBindImageTexture(0, m_depthPyramid, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);

    // Bind output texture for writing
    glBindImageTexture(1, m_currentShadedFrame, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

    // Dispatch
    int workGroupsX = (m_viewportWidth + 15) / 16;
    int workGroupsY = (m_viewportHeight + 15) / 16;
    m_shadingShader->dispatch(workGroupsX, workGroupsY, 1);

    // Memory barrier
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void RaymarcherSimple::reconstruction(const Camera &camera, const ShaderState &shaderState)
{
    if (!m_reconstructionShader || m_outputTexture == 0 || m_outputTextureSwap == 0)
        return;

    m_reconstructionShader->use();
    uploadCameraParameters(camera, m_reconstructionShader.get());
    const_cast<ShaderState &>(shaderState).uploadUniforms(m_reconstructionShader.get());

    // Bind input texture (shaded output) for reading
    glBindImageTexture(0, m_currentShadedFrame, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
    glBindImageTexture(1, m_outputTextureSwap, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);

    // Bind output texture for writing
    glBindImageTexture(2, m_outputTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

    matrix4 currentCameraTransform = camera.getViewMatrix();

    matrix4 thisFrameToLastFrame = m_lastCameraTransform * currentCameraTransform.inverse();
    int location = m_reconstructionShader->getUniformLocationCached("uThisFrameToLastFrame");
    glUniformMatrix4fv(location, 1, GL_TRUE, &thisFrameToLastFrame.m[0][0]);

    // Dispatch
    int workGroupsX = (m_viewportWidth + 15) / 16;
    int workGroupsY = (m_viewportHeight + 15) / 16;
    m_reconstructionShader->dispatch(workGroupsX, workGroupsY, 1);

    // Memory barrier
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    m_lastCameraTransform = camera.getViewMatrix();

    // Swap output textures
    std::swap(m_outputTexture, m_outputTextureSwap);
}

void RaymarcherSimple::draw(const Camera &camera, const ShaderState &shaderState)
{
    auto frameStart = std::chrono::steady_clock::now();

    if (!m_baseDepthShader || !m_shadingShader)
        return;

    if (m_depthPyramid == 0 || m_outputTexture == 0)
        return;

    // Pass 1: Build depth pyramid (coarse to fine)
    auto start = std::chrono::steady_clock::now();
    raymarchDepthPyramid(camera, shaderState);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    m_lastExecutionTimes["raymarchDepthPyramid"] = duration.count() / 1000.0f;

    // Pass 2: Shade from depth
    start = std::chrono::steady_clock::now();
    shadeFromDepth(camera, shaderState);
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    m_lastExecutionTimes["shadeFromDepth"] = duration.count() / 1000.0f;

    // Pass 3: Reconstruction
    start = std::chrono::steady_clock::now();
    reconstruction(camera, shaderState);
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    m_lastExecutionTimes["reconstruction"] = duration.count() / 1000.0f;

    auto frameEnd = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(frameEnd - frameStart);
    m_lastExecutionTimes["frame"] = duration.count() / 1000.0f;
}

bool RaymarcherSimple::reloadShaders()
{
    bool allSuccess = true;

    if (m_baseDepthShader && !m_baseDepthShader->getComputePath().empty())
    {
        bool success = m_baseDepthShader->reload();
        if (success)
        {
            LOG(INFO) << "RaymarcherSimple: Base depth shader reloaded successfully";
        }
        allSuccess &= success;
    }

    if (m_shadingShader && !m_shadingShader->getComputePath().empty())
    {
        bool success = m_shadingShader->reload();
        if (success)
        {
            LOG(INFO) << "RaymarcherSimple: Shading shader reloaded successfully";
        }
        allSuccess &= success;
    }

    if (m_reconstructionShader && !m_reconstructionShader->getComputePath().empty())
    {
        bool success = m_reconstructionShader->reload();
        if (success)
        {
            LOG(INFO) << "RaymarcherSimple: Reconstruction shader reloaded successfully";
        }
        allSuccess &= success;
    }

    return allSuccess;
}

void RaymarcherSimple::drawGui()
{
    ImGui::Separator();
    ImGui::Text("Raymarcher Simple GPU Timings (ms)");

    for (const auto &entry : m_lastExecutionTimes)
    {
        ImGui::Text("%s: %.3f ms", entry.first.c_str(), entry.second);
    }
}