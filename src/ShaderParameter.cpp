#include "ShaderParameter.h"
#include "shader/ComputeShader.h"
#include <imgui.h>
#include <glad/glad.h>
#include <algorithm>

// ============================================================================
// FloatParameter
// ============================================================================

FloatParameter::FloatParameter(const char* displayName, const char* uniformName, float min, float max, float defaultVal)
    : m_value(defaultVal)
    , m_default(defaultVal)
    , m_min(min)
    , m_max(max)
{
    m_displayName = displayName;
    m_uniformName = uniformName;
}

void FloatParameter::uploadUniform(Shader* shader) const
{
    int location = shader->getUniformLocationCached(m_uniformName.c_str());
    if (location >= 0)
    {
        glUniform1f(location, m_value);
    }
}

void FloatParameter::drawGui()
{
    ImGui::SliderFloat(m_displayName.c_str(), &m_value, m_min, m_max);
}

void FloatParameter::reset()
{
    m_value = m_default;
}

void FloatParameter::toJson(nlohmann::json& j) const
{
    j["displayName"] = m_displayName;
    j["value"] = m_value;
}

void FloatParameter::fromJson(const nlohmann::json& j)
{
    m_value = j.value("value", m_default);
}

// Vec3Parameter
// ============================================================================

Vec3Parameter::Vec3Parameter(const char* displayName, const char* uniformName, float min, float max, const vec3& defaultVal)
    : m_value(defaultVal)
    , m_default(defaultVal)
    , m_min(min)
    , m_max(max)
{
    m_displayName = displayName;
    m_uniformName = uniformName;
}

void Vec3Parameter::uploadUniform(Shader* shader) const
{
    int location = shader->getUniformLocationCached(m_uniformName.c_str());
    if (location >= 0)
    {
        glUniform3f(location, m_value.x, m_value.y, m_value.z);
    }
}

void Vec3Parameter::drawGui()
{
    ImGui::SliderFloat3(m_displayName.c_str(), &m_value.x, m_min, m_max);
}

void Vec3Parameter::reset()
{
    m_value = m_default;
}

void Vec3Parameter::toJson(nlohmann::json& j) const
{
    j["displayName"] = m_displayName;
    j["value"] = { m_value.x, m_value.y, m_value.z };
}

void Vec3Parameter::fromJson(const nlohmann::json& j)
{
    auto val = j.value("value", std::vector<float>{m_default.x, m_default.y, m_default.z});
    if (val.size() == 3)
    {
        m_value.x = val[0];
        m_value.y = val[1];
        m_value.z = val[2];
    }
}

// ============================================================================
// ColorParameter
// ============================================================================

ColorParameter::ColorParameter(const char* displayName, const char* uniformName, const color& defaultVal)
    : m_value(defaultVal)
    , m_default(defaultVal)
{
    m_displayName = displayName;
    m_uniformName = uniformName;
}

void ColorParameter::uploadUniform(Shader* shader) const
{
    int location = shader->getUniformLocationCached(m_uniformName.c_str());
    if (location >= 0)
    {
        glUniform4f(location, m_value.r, m_value.g, m_value.b, m_value.a);
    }
}

void ColorParameter::drawGui()
{
    ImGui::ColorEdit4(m_displayName.c_str(), &m_value.r);
}

void ColorParameter::reset()
{
    m_value = m_default;
}

void ColorParameter::toJson(nlohmann::json& j) const
{
    j["displayName"] = m_displayName;
    j["value"] = { m_value.r, m_value.g, m_value.b, m_value.a };
}

void ColorParameter::fromJson(const nlohmann::json& j)
{
    auto val = j.value("value", std::vector<float>{m_default.r, m_default.g, m_default.b, m_default.a});
    if (val.size() == 4)
    {
        m_value.r = val[0];
        m_value.g = val[1];
        m_value.b = val[2];
        m_value.a = val[3];
    }
}

// ============================================================================
// IntParameter
// ============================================================================

IntParameter::IntParameter(const char* displayName, const char* uniformName, int min, int max, int defaultVal)
    : m_value(defaultVal)
    , m_default(defaultVal)
    , m_min(min)
    , m_max(max)
{
    m_displayName = displayName;
    m_uniformName = uniformName;
}

void IntParameter::uploadUniform(Shader* shader) const
{
    int location = shader->getUniformLocationCached(m_uniformName.c_str());
    if (location >= 0)
    {
        glUniform1i(location, m_value);
    }
}

void IntParameter::drawGui()
{
    ImGui::SliderInt(m_displayName.c_str(), &m_value, m_min, m_max);
}

void IntParameter::reset()
{
    m_value = m_default;
}

void IntParameter::toJson(nlohmann::json& j) const
{
    j["displayName"] = m_displayName;
    j["value"] = m_value;
}

void IntParameter::fromJson(const nlohmann::json& j)
{
    m_value = j.value("value", m_default);
}
