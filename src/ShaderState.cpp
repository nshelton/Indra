#include "ShaderState.h"
#include "render/ComputeShader.h"
#include <imgui.h>

ShaderState::ShaderState()
{
    // Add all shader parameters here
    // Each parameter knows how to upload itself and draw its own GUI

    addParameter<ColorParameter>("backgroundColor", "Background Color", "uBackgroundColor", vec3(0.0f, 0.0f, 0.0f));
    addParameter<Vec3Parameter>("spherePosition", "Sphere Position", "uSpherePosition", -10.0f, 10.0f, vec3(0.0f, 0.0f, 0.0f));
    addParameter<FloatParameter>("sphereRadius", "Sphere Radius", "uSphereRadius", 0.1f, 5.0f, 1.0f);
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

ShaderParameter* ShaderState::getParameter(const std::string& name)
{
    auto it = m_parameterMap.find(name);
    if (it != m_parameterMap.end())
    {
        return it->second;
    }
    return nullptr;
}
