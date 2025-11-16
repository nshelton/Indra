#include "ShaderParameter.h"
#include "render/ComputeShader.h"
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

void FloatParameter::uploadUniform(ComputeShader* shader, int location) const
{
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

// ============================================================================
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

void Vec3Parameter::uploadUniform(ComputeShader* shader, int location) const
{
    if (location >= 0)
    {
        glUniform3f(location, m_value.x, m_value.y, m_value.z);
    }
}

void Vec3Parameter::drawGui()
{
    ImGui::SliderFloat3(m_displayName.c_str(), &m_value.x, m_min, m_max);
}

void Vec3Parameter::setValue(const vec3& v)
{
    m_value.x = std::clamp(v.x, m_min, m_max);
    m_value.y = std::clamp(v.y, m_min, m_max);
    m_value.z = std::clamp(v.z, m_min, m_max);
}

void Vec3Parameter::reset()
{
    m_value = m_default;
}

// ============================================================================
// ColorParameter
// ============================================================================

ColorParameter::ColorParameter(const char* displayName, const char* uniformName, const vec3& defaultVal)
    : m_value(defaultVal)
    , m_default(defaultVal)
{
    m_displayName = displayName;
    m_uniformName = uniformName;
}

void ColorParameter::uploadUniform(ComputeShader* shader, int location) const
{
    if (location >= 0)
    {
        glUniform3f(location, m_value.x, m_value.y, m_value.z);
    }
}

void ColorParameter::drawGui()
{
    ImGui::ColorEdit3(m_displayName.c_str(), &m_value.x);
}

void ColorParameter::reset()
{
    m_value = m_default;
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

void IntParameter::uploadUniform(ComputeShader* shader, int location) const
{
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
