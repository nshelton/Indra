#include "RaymarcherSimple.h"
#include <iostream>
#include <glog/logging.h>
#include <cmath>

#define TOP_LEVEL 7

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
    createTexture(m_outputTexture, GL_RGBA16F, m_viewportSize.x, m_viewportSize.y, 1, GL_RGBA, GL_HALF_FLOAT);
    createTexture(m_outputTextureSwap, GL_RGBA16F, m_viewportSize.x, m_viewportSize.y, 1, GL_RGBA, GL_HALF_FLOAT);
    createTexture(m_currentShadedFrame, GL_RGBA16F, m_viewportSize.x, m_viewportSize.y, 1, GL_RGBA, GL_HALF_FLOAT);
    LOG(INFO) << "RaymarcherSimple: Created output textures: " << m_viewportSize.x << "x" << m_viewportSize.y;
}

void RaymarcherSimple::createDepthPyramid()
{
    if (m_depthPyramid != 0)
    {
        glDeleteTextures(1, &m_depthPyramid);
    }

    // Calculate number of mip levels needed
    int maxDim = std::max(m_viewportSize.x, m_viewportSize.y);
    m_numLevels = 1 + static_cast<int>(std::floor(std::log2(maxDim)));

    // Create mipmapped R32F texture
    glGenTextures(1, &m_depthPyramid);
    glBindTexture(GL_TEXTURE_2D, m_depthPyramid);
    glTexStorage2D(GL_TEXTURE_2D, m_numLevels, GL_R32F, m_viewportSize.x, m_viewportSize.y);
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
    if (!m_reconstructionShader->loadFromFile("../../shaders/reconstruction_test.comp"))
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
    if (m_viewportSize.x != width || m_viewportSize.y != height)
    {
        m_viewportSize = vec2i(width, height);
        createDepthPyramid();
        createOutputTextures();
    }
}

void RaymarcherSimple::setCameraParameters(const Camera &camera, ComputeShader *shader)
{
    shader->set("uCameraPos", camera.getPosition());
    shader->set("uCameraForward", camera.getForward());
    shader->set("uCameraRight", camera.getRight());
    shader->set("uCameraUp", camera.getUp());

    // Calculate tan(fov/2) for the shader
    float fov = camera.getFov();
    float aspect = camera.getAspect();
    float fovRadians = fov * 3.14159265359f / 180.0f;
    float tanHalfFov = std::tan(fovRadians / 2.0f);

    shader->set("uTanHalfFov", tanHalfFov);
    shader->set("uAspect", aspect);
    shader->set("uViewportSize", m_viewportSize);
}

// void RaymarcherSimple::raymarchDepthPyramid(const Camera &camera)
// {

//     if (!m_baseDepthShader || m_depthPyramid == 0)
//         return;

//     // Pass 1: Build depth pyramid (coarse to fine)
//     m_baseDepthShader->use();
//     setCameraParameters(camera, m_baseDepthShader.get());

//     // Bind current level for writing
//     int location = m_baseDepthShader->set(
//         "uSeed3",
//         vec3(
//             static_cast<float>(std::rand()) / RAND_MAX,
//             static_cast<float>(std::rand()) / RAND_MAX,
//             static_cast<float>(std::rand()) / RAND_MAX));

//     for (int level = TOP_LEVEL; level >= 0; level--)
//     {
//         auto start = std::chrono::steady_clock::now();

//         int location = m_baseDepthShader->set("uLevel", level);

//         int levelWidth = m_viewportSize.x >> level;
//         int levelHeight = m_viewportSize.y >> level;

//         // Bind previous level for reading
//         if (level < TOP_LEVEL)
//         {
//             glBindImageTexture(0, m_depthPyramid, level + 1, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
//         }

//         // Bind this levelfor writing
//         glBindImageTexture(1, m_depthPyramid, level, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

//         // Dispatch
//         // Note: Top levels (7-8) have poor occupancy due to small resolution + texture reads
//         // Consider batching or skipping intermediate levels for better performance
//         int workGroupsX = (levelWidth + 15) / 16;
//         int workGroupsY = (levelHeight + 15) / 16;
//         m_baseDepthShader->dispatch(workGroupsX, workGroupsY, 1);
//         // Memory barrier to ensure level is complete before next level reads it
//         glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

//         auto end = std::chrono::steady_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//         m_lastExecutionTimes["raymarch l" + std::to_string(level)] = duration.count() / 1000.0f;
//     }
// }

// void RaymarcherSimple::shadeFromDepth(const Camera &camera)
// {
//     if (!m_shadingShader || m_depthPyramid == 0 || m_outputTexture == 0)
//         return;

//     m_shadingShader->use();
//     setCameraParameters(camera, m_shadingShader.get());
//     // const_cast<ShaderState &>(shaderState).uploadUniforms(m_shadingShader.get());

//     m_shadingShader->set("uSeed", static_cast<float>(std::rand()) / RAND_MAX);

//     // Bind depth pyramid level 0 for reading
//     glBindImageTexture(0, m_depthPyramid, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);

//     // Bind output texture for writing
//     glBindImageTexture(1, m_currentShadedFrame, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

//     // Dispatch
//     int workGroupsX = (m_viewportSize.x + 15) / 16;
//     int workGroupsY = (m_viewportSize.y + 15) / 16;
//     m_shadingShader->dispatch(workGroupsX, workGroupsY, 1);

//     // Memory barrier
//     glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
// }

void RaymarcherSimple::reconstruction(const Camera &camera)
{
    if (!m_reconstructionShader || m_outputTexture == 0 || m_outputTextureSwap == 0)
        return;

    m_reconstructionShader->use();
    setCameraParameters(camera, m_reconstructionShader.get());
    // const_cast<ShaderState &>(shaderState).uploadUniforms(m_reconstructionShader.get());

    // Bind input texture (shaded output) for reading
    // glBindImageTexture(0, m_currentShadedFrame, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
    // glBindImageTexture(1, m_outputTextureSwap, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);

    // Bind output texture for writing
    glBindImageTexture(2, m_outputTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

    matrix4 currentCameraTransform = camera.getViewMatrix();
    matrix4 thisFrameToLastFrame = m_lastCameraTransform * currentCameraTransform.inverse();
    m_reconstructionShader->set("uCurrentCameraTransform", thisFrameToLastFrame);

    // Dispatch
    int workGroupsX = (m_viewportSize.x + 15) / 16;
    int workGroupsY = (m_viewportSize.y + 15) / 16;
    m_reconstructionShader->dispatch(workGroupsX, workGroupsY, 1);

    // Memory barrier
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    m_lastCameraTransform = camera.getViewMatrix();

    // Swap output textures
    std::swap(m_outputTexture, m_outputTextureSwap);
}

void RaymarcherSimple::draw(const Camera &camera)
{
    auto frameStart = std::chrono::steady_clock::now();

    // if (!m_baseDepthShader || !m_shadingShader)
    //     return;

    // if (m_depthPyramid == 0 || m_outputTexture == 0)
    //     return;

    // // Pass 1: Build depth pyramid (coarse to fine)
    // auto start = std::chrono::steady_clock::now();
    // raymarchDepthPyramid(camera);
    // auto end = std::chrono::steady_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // m_lastExecutionTimes["raymarchDepthPyramid"] = duration.count() / 1000.0f;

    // // Pass 2: Shade from depth
    // start = std::chrono::steady_clock::now();
    // shadeFromDepth(camera);
    // end = std::chrono::steady_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // m_lastExecutionTimes["shadeFromDepth"] = duration.count() / 1000.0f;

    // Pass 3: Reconstruction
    auto start = std::chrono::steady_clock::now();
    reconstruction(camera);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
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

void DrawShaderGui(ComputeShader *shader, const std::string &shaderName)
{
    if (!shader)
        return;

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.8f, 0.2f, 1.0f));
    ImGui::Text("%s Uniform Locations:", shaderName.c_str());
    ImGui::PopStyleColor();

    shader->uniforms().forEach([](auto &u)
                               {
        ImGui::Text("%s: %d", u.name.c_str(), u.location);
        ImGui::SameLine();
        if constexpr (std::same_as<decltype(u.value), float>)
        {
            ImGui::Text("Value: %.3f", u.value);
            if (u.hasMetadata)
                ImGui::SliderFloat(("##" + u.name).c_str(), &u.value, u.minValue, u.maxValue);
        }
        else if constexpr (std::same_as<decltype(u.value), int>)
        {
            ImGui::Text("Value: %d", u.value);
            if (u.hasMetadata)
                ImGui::SliderInt(("##" + u.name).c_str(), &u.value, u.minValue, u.maxValue);
        }
        else if constexpr (std::same_as<decltype(u.value), vec2>)
        {
            ImGui::Text("Value: (%.3f, %.3f)", u.value.x, u.value.y);
            if (u.hasMetadata)
                ImGui::SliderFloat2(("##" + u.name).c_str(), &u.value.x, u.minValue.x, u.maxValue.x);
        }
        else if constexpr (std::same_as<decltype(u.value), vec2i>)
        {
            ImGui::Text("Value: (%d, %d)", u.value.x, u.value.y);
            if (u.hasMetadata)
                ImGui::SliderInt2(("##" + u.name).c_str(), &u.value.x, u.minValue.x, u.maxValue.x);
        }
        else if constexpr (std::same_as<decltype(u.value), vec3>)
        {
            ImGui::Text("Value: (%.3f, %.3f, %.3f)", u.value.x, u.value.y, u.value.z);
            if (u.hasMetadata)
                ImGui::SliderFloat3(("##" + u.name).c_str(), &u.value.x, u.minValue.x, u.maxValue.x);
        }
        else if constexpr (std::same_as<decltype(u.value), color>)
        {
            ImGui::ColorEdit4(("##" + u.name).c_str(), &u.value.r);
        }
        else if constexpr (std::same_as<decltype(u.value), matrix4>)
        {
            ImGui::Text("Value: [matrix4]");
        } });
}

void RaymarcherSimple::drawGui()
{

    ImGui::Separator();
    ImGui::Text("Shaders:");
    ImGui::Separator();

    if (!m_baseDepthShader || !m_shadingShader || !m_reconstructionShader)
    {
        ImGui::Text("Shaders not initialized");
        return;
    }

    DrawShaderGui(m_baseDepthShader.get(), "m_baseDepthShader");
    DrawShaderGui(m_shadingShader.get(), "m_shadingShader");
    DrawShaderGui(m_reconstructionShader.get(), "m_reconstructionShader");

    ImGui::Separator();
    ImGui::Text("PERFORMANCE:");

    for (const auto &entry : m_lastExecutionTimes)
    {
        ImGui::Text("%s: %.3f ms", entry.first.c_str(), entry.second);
    }
}

nlohmann::json RaymarcherSimple::toJson() const
{
    nlohmann::json j;
    j["baseDepthShader"] = m_baseDepthShader ? m_baseDepthShader->toJson() : nlohmann::json();
    j["shadingShader"] = m_shadingShader ? m_shadingShader->toJson() : nlohmann::json();
    j["reconstructionShader"] = m_reconstructionShader ? m_reconstructionShader->toJson() : nlohmann::json();
    return j;
}

void RaymarcherSimple::fromJson(const nlohmann::json &j)
{
    if (m_baseDepthShader)
    {
        m_baseDepthShader->fromJson(j.value("baseDepthShader", nlohmann::json()));
    }
    if (m_shadingShader)
    {
        m_shadingShader->fromJson(j.value("shadingShader", nlohmann::json()));
    }
    if (m_reconstructionShader)
    {
        m_reconstructionShader->fromJson(j.value("reconstructionShader", nlohmann::json()));
    }
}