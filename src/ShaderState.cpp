#include "ShaderState.h"
#include "shader/ComputeShader.h"
#include <imgui.h>

ShaderState::ShaderState()
{
    m_parameterMap["uBackgroundColor"] = std::make_unique<ColorParameter>( "uBackgroundColor", color(0.0f, 0.0f, 0.0f, 1.0f));

    m_parameterMap["u_paramA"] = std::make_unique<Vec3Parameter>( "u_paramA", -1.0f, 1.0f, vec3(0.0f, 0.0f, 0.0f));
    m_parameterMap["u_paramB"] = std::make_unique<Vec3Parameter>( "u_paramB", -1.0f, 1.0f, vec3(0.0f, 0.0f, 0.0f));
    m_parameterMap["u_paramC"] = std::make_unique<Vec3Parameter>( "u_paramC", -1.0f, 1.0f, vec3(0.0f, 0.0f, 0.0f));
    m_parameterMap["u_paramD"] = std::make_unique<Vec3Parameter>( "u_paramD", -1.0f, 1.0f, vec3(0.0f, 0.0f, 0.0f));

    m_parameterMap["uMaxSteps"] = std::make_unique<IntParameter>( "uMaxSteps", 1, 500, 100);
    m_parameterMap["uMaxDistance"] = std::make_unique<FloatParameter>( "uMaxDistance", 1.0f, 1000.0f, 100.0f);
    m_parameterMap["_LEVELS"] = std::make_unique<FloatParameter>( "_LEVELS", 1.0f, 10.0f, 6.0f);

    m_parameterMap["uStepRatio"] = std::make_unique<FloatParameter>( "uStepRatio", 0.01f, 1.0f, 1.0f);
    m_parameterMap["uSurfaceEpsilon"] = std::make_unique<FloatParameter>( "uSurfaceEpsilon", 1.0f, 10.0f, 4.0f);
}

void ShaderState::reset()
{
    for (auto &nameParam : m_parameterMap)
    {
        nameParam.second->reset();
    }
}

void ShaderState::drawGui()
{
    ImGui::Separator();
    ImGui::Text("Shader Parameters");

    for (auto &nameParam : m_parameterMap)
    {
        nameParam.second->drawGui();
    }

    // Reset button
    if (ImGui::Button("Reset to Defaults"))
    {
        reset();
    }
}

void ShaderState::uploadUniforms(ComputeShader *shader)
{
    if (!shader || !shader->isValid())
        return;

    shader->use();
    for (auto &nameParam : m_parameterMap)
    {
        nameParam.second->uploadUniform(shader);
    }
}

void ShaderState::toJson(nlohmann::json &j) const
{
    j = nlohmann::json::array();
    for (const auto &nameParam : m_parameterMap)
    {
        nlohmann::json paramJson;
        nameParam.second->toJson(paramJson);
        j.push_back(paramJson);
    }
}

void ShaderState::fromJson(const nlohmann::json &j)
{
    for (const auto &paramJson : j)
    {
        std::string displayName = paramJson.value("displayName", "");
        ShaderParameter *param = getParameter(displayName);
        if (param)
        {
            param->fromJson(paramJson);
        }
    }
}

ShaderParameter *ShaderState::getParameter(const std::string &name)
{
    auto it = m_parameterMap.find(name);
    if (it != m_parameterMap.end())
    {
        return it->second.get();
    }
    return nullptr;
}
