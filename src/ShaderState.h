#pragma once

#include "core/core.h"
#include "ShaderParameter.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

// Forward declaration
class ComputeShader;

/// @brief Container for all shader parameters
/// Parameters are stored polymorphically and can upload themselves to GPU and draw their own GUI
struct ShaderState
{
public:
    ShaderState();

    /// @brief Reset all parameters to their default values
    void reset();

    /// @brief Draw ImGui widgets for all parameters
    void drawGui();

    /// @brief Upload all parameters to the compute shader
    void uploadUniforms(ComputeShader* shader);

    /// @brief Serialize to JSON
    void toJson(nlohmann::json& j) const;

    /// @brief Deserialize from JSON
    void fromJson(const nlohmann::json& j);

    /// @brief Get a parameter by name (returns nullptr if not found)
    ShaderParameter* getParameter(const std::string& name);

    /// @brief Template helper to get typed parameter
    template<typename T>
    T* getParameterAs(const std::string& name)
    {
        return dynamic_cast<T*>(getParameter(name));
    }

private:
    int m_associatedShaderRevisionId = -1;
    // Fast lookup by name
    std::unordered_map<std::string, std::unique_ptr<ShaderParameter>> m_parameterMap;
};
