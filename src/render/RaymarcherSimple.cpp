#include "RaymarcherSimple.h"
#include <iostream>
#include <glog/logging.h>
#include <cmath>

void RaymarcherSimple::createOutputTexture()
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

    LOG(INFO) << "RaymarcherSimple: Created output texture: " << m_viewportWidth << "x" << m_viewportHeight;
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

    // Load refinement depth shader
    m_refineDepthShader = std::make_unique<ComputeShader>();
    if (!m_refineDepthShader->loadFromFile("../../shaders/raymarch_depth_refine.comp"))
    {
        LOG(ERROR) << "RaymarcherSimple: Failed to load refinement depth shader";
        return false;
    }

    // Load shading shader
    m_shadingShader = std::make_unique<ComputeShader>();
    if (!m_shadingShader->loadFromFile("../../shaders/shade_from_depth.comp"))
    {
        LOG(ERROR) << "RaymarcherSimple: Failed to load shading shader";
        return false;
    }

    // Create depth pyramid and output texture
    createDepthPyramid();
    createOutputTexture();

    LOG(INFO) << "RaymarcherSimple: Initialized successfully with hierarchical depth pyramid";
    return true;
}

void RaymarcherSimple::shutdown()
{
    m_baseDepthShader.reset();
    m_refineDepthShader.reset();
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
}

void RaymarcherSimple::setViewportSize(int width, int height)
{
    if (m_viewportWidth != width || m_viewportHeight != height)
    {
        m_viewportWidth = width;
        m_viewportHeight = height;
        createDepthPyramid();
        createOutputTexture();
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
    if (!m_baseDepthShader || !m_refineDepthShader || m_depthPyramid == 0)
        return;

    m_baseDepthShader->use();
    uploadCameraParameters(camera, m_baseDepthShader.get());
    const_cast<ShaderState &>(shaderState).uploadUniforms(m_baseDepthShader.get());

    // Bind current level for writing
    int level = 0;
    int levelWidth = m_viewportWidth >> level;
    int levelHeight = m_viewportHeight >> level;
    glBindImageTexture(0, m_depthPyramid, level, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    // Dispatch
    int workGroupsX = (levelWidth + 15) / 16;
    int workGroupsY = (levelHeight + 15) / 16;
    m_baseDepthShader->dispatch(workGroupsX, workGroupsY, 1);

    // // Raymarch from coarse to fine (high mip level to low mip level)
    // for (int level = m_baseLevelIndex; level >= 0; level--)
    // {
    //     int levelWidth = m_viewportWidth >> level;
    //     int levelHeight = m_viewportHeight >> level;

    //     if (level == m_baseLevelIndex)
    //     {
    //         // Base level: raymarch from scratch
    //         m_baseDepthShader->use();
    //         uploadCameraParameters(camera, m_baseDepthShader.get());
    //         const_cast<ShaderState &>(shaderState).uploadUniforms(m_baseDepthShader.get());

    //         // Bind current level for writing
    //         glBindImageTexture(0, m_depthPyramid, level, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    //         // Dispatch
    //         int workGroupsX = (levelWidth + 15) / 16;
    //         int workGroupsY = (levelHeight + 15) / 16;
    //         m_baseDepthShader->dispatch(workGroupsX, workGroupsY, 1);
    //     }
    //     else
    //     {
    //         // Refinement level: continue from parent
    //         m_refineDepthShader->use();
    //         uploadCameraParameters(camera, m_refineDepthShader.get());
    //         const_cast<ShaderState&>(shaderState).uploadUniforms(m_refineDepthShader.get());

    //         // Bind parent level for reading (coarser level = higher index)
    //         glBindImageTexture(0, m_depthPyramid, level + 1, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);

    //         // Bind current level for writing
    //         glBindImageTexture(1, m_depthPyramid, level, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    //         // Dispatch
    //         int workGroupsX = (levelWidth + 15) / 16;
    //         int workGroupsY = (levelHeight + 15) / 16;
    //         m_refineDepthShader->dispatch(workGroupsX, workGroupsY, 1);
    //     }

    // Memory barrier to ensure level is complete before next level reads it
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void RaymarcherSimple::shadeFromDepth(const Camera &camera, const ShaderState &shaderState)
{
    if (!m_shadingShader || m_depthPyramid == 0 || m_outputTexture == 0)
        return;

    m_shadingShader->use();
    uploadCameraParameters(camera, m_shadingShader.get());
    const_cast<ShaderState &>(shaderState).uploadUniforms(m_shadingShader.get());

    // Bind depth pyramid level 0 for reading
    glBindImageTexture(0, m_depthPyramid, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);

    // Bind output texture for writing
    glBindImageTexture(1, m_outputTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

    // Dispatch
    int workGroupsX = (m_viewportWidth + 15) / 16;
    int workGroupsY = (m_viewportHeight + 15) / 16;
    m_shadingShader->dispatch(workGroupsX, workGroupsY, 1);

    // Memory barrier
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void RaymarcherSimple::draw(const Camera &camera, const ShaderState &shaderState)
{
    if (!m_baseDepthShader || !m_refineDepthShader || !m_shadingShader)
        return;

    if (m_depthPyramid == 0 || m_outputTexture == 0)
        return;

    // Pass 1: Build depth pyramid (coarse to fine)
    raymarchDepthPyramid(camera, shaderState);

    // Pass 2: Shade from depth
    shadeFromDepth(camera, shaderState);
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

    if (m_refineDepthShader && !m_refineDepthShader->getComputePath().empty())
    {
        bool success = m_refineDepthShader->reload();
        if (success)
        {
            LOG(INFO) << "RaymarcherSimple: Refinement depth shader reloaded successfully";
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

    return allSuccess;
}
