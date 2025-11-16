#pragma once

#include "core/core.h"
#include <string>
#include <memory>

// Forward declaration
class ComputeShader;

/// @brief Base class for shader parameters that can upload themselves to GPU and draw their own GUI
class ShaderParameter
{
public:
    virtual ~ShaderParameter() = default;

    /// @brief Upload this parameter to the shader
    virtual void uploadUniform(ComputeShader* shader, int location) const = 0;

    /// @brief Draw ImGui widget for this parameter
    virtual void drawGui() = 0;

    /// @brief Reset to default value
    virtual void reset() = 0;

    const std::string& getUniformName() const { return m_uniformName; }
    const std::string& getDisplayName() const { return m_displayName; }

protected:
    std::string m_uniformName;
    std::string m_displayName;
};

/// @brief Float parameter with slider
class FloatParameter : public ShaderParameter
{
public:
    FloatParameter(const char* displayName, const char* uniformName, float min, float max, float defaultVal);

    void uploadUniform(ComputeShader* shader, int location) const override;
    void drawGui() override;
    void reset() override;

    float getValue() const { return m_value; }
    void setValue(float v) { m_value = clamp(v, m_min, m_max); }

private:
    float m_value;
    float m_default;
    float m_min;
    float m_max;
};

/// @brief Vec3 parameter with 3 sliders
class Vec3Parameter : public ShaderParameter
{
public:
    Vec3Parameter(const char* displayName, const char* uniformName, float min, float max, const vec3& defaultVal);

    void uploadUniform(ComputeShader* shader, int location) const override;
    void drawGui() override;
    void reset() override;

    const vec3& getValue() const { return m_value; }
    void setValue(const vec3& v);

private:
    vec3 m_value;
    vec3 m_default;
    float m_min;
    float m_max;
};

/// @brief Color parameter with color picker (vec3 with 0-1 range)
class ColorParameter : public ShaderParameter
{
public:
    ColorParameter(const char* displayName, const char* uniformName, const vec3& defaultVal);

    void uploadUniform(ComputeShader* shader, int location) const override;
    void drawGui() override;
    void reset() override;

    const vec3& getValue() const { return m_value; }
    void setValue(const vec3& v) { m_value = v; }

private:
    vec3 m_value;
    vec3 m_default;
};

/// @brief Integer parameter with slider
class IntParameter : public ShaderParameter
{
public:
    IntParameter(const char* displayName, const char* uniformName, int min, int max, int defaultVal);

    void uploadUniform(ComputeShader* shader, int location) const override;
    void drawGui() override;
    void reset() override;

    int getValue() const { return m_value; }
    void setValue(int v) { m_value = clamp(v, m_min, m_max); }

private:
    int m_value;
    int m_default;
    int m_min;
    int m_max;
};
