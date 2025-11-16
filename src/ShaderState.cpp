#include "ShaderState.h"
#include "render/ComputeShader.h"
#include <imgui.h>

ShaderState::ShaderState()
{
    addParameter<ColorParameter>("backgroundColor", "Background Color", "uBackgroundColor", color(0.0f, 0.0f, 0.0f, 1.0f));
    addParameter<Vec3Parameter>("u_paramA", "u_paramA", "u_paramA", -1.0f, 1.0f, vec3(0.0f, 0.0f, 0.0f));
    addParameter<Vec3Parameter>("u_paramB", "u_paramB", "u_paramB", -1.0f, 1.0f, vec3(0.0f, 0.0f, 0.0f));
    addParameter<Vec3Parameter>("u_paramC", "u_paramC", "u_paramC", -1.0f, 1.0f, vec3(0.0f, 0.0f, 0.0f));
    addParameter<FloatParameter>("_LEVELS", "Levels", "_LEVELS", 1.0f, 10.0f, 6.0f);

    addParameter<IntParameter>("maxSteps", "Max Steps", "uMaxSteps", 1, 500, 100);
    addParameter<FloatParameter>("maxDistance", "Max Distance", "uMaxDistance", 1.0f, 1000.0f, 100.0f);
    addParameter<FloatParameter>("surfaceEpsilon", "Surface Epsilon", "uSurfaceEpsilon", 0.0001f, 0.01f, 0.001f);

}

void ShaderState::reset()
{
    for (auto& param : m_parameters)
    {
        param->reset();
    }

    // Clear uniform cache since we're starting fresh
    m_uniformLocationCache.clear();
}

void ShaderState::drawGui()
{
    ImGui::Separator();
    ImGui::Text("Shader Parameters");

    for (auto& param : m_parameters)
    {
        param->drawGui();
    }

    // Reset button
    if (ImGui::Button("Reset to Defaults"))
    {
        reset();
    }
}

void ShaderState::uploadUniforms(ComputeShader* shader)
{
    if (!shader || !shader->isValid())
        return;

    shader->use();

    for (auto& param : m_parameters)
    {
        const std::string& uniformName = param->getUniformName();

        // Get or cache uniform location
        int location;
        auto it = m_uniformLocationCache.find(uniformName);
        if (it != m_uniformLocationCache.end())
        {
            location = it->second;
        }
        else
        {
            location = shader->getUniformLocation(uniformName.c_str());
            m_uniformLocationCache[uniformName] = location;
        }

        // Let the parameter upload itself
        param->uploadUniform(shader, location);
    }
}

void ShaderState::toJson(nlohmann::json& j) const
{
    j = nlohmann::json::array();
    for (const auto& param : m_parameters)
    {
        nlohmann::json paramJson;
        param->toJson(paramJson);
        j.push_back(paramJson);
    }
}

void ShaderState::fromJson(const nlohmann::json& j)
{
    for (const auto& paramJson : j)
    {
        std::string uniformName = paramJson.value("uniformName", "");
        ShaderParameter* param = getParameter(uniformName);
        if (param)
        {
            param->fromJson(paramJson);
        }
    }
}


ShaderParameter* ShaderState::getParameter(const std::string& name)
{
    auto it = m_parameterMap.find(name);
    if (it != m_parameterMap.end())
    {
        return it->second;
    }
    return nullptr;
}
