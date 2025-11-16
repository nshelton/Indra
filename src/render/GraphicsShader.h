#pragma once

#include "Shader.h"

/// @brief Graphics shader program for vertex + fragment shader pipelines
/// Handles traditional rendering with vertex and fragment shader stages
class GraphicsShader : public Shader
{
public:
    GraphicsShader();
    ~GraphicsShader() override = default;

    /// @brief Load shaders from files
    /// @param vertexPath Path to vertex shader file
    /// @param fragmentPath Path to fragment shader file
    /// @return true if shaders loaded successfully
    bool loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath);

    /// @brief Load shaders from source strings
    /// @param vertexSource Vertex shader source code
    /// @param fragmentSource Fragment shader source code
    /// @return true if shaders compiled and linked successfully
    bool loadFromSource(const char* vertexSource, const char* fragmentSource);

    /// @brief Set fallback shaders to use if files cannot be loaded
    /// @param vertexSource Fallback vertex shader source
    /// @param fragmentSource Fallback fragment shader source
    void setFallbackSource(const std::string& vertexSource, const std::string& fragmentSource);

    /// @brief Reload shaders from the previously loaded file paths
    /// @return true if reload was successful
    bool reload() override;

    /// @brief Get vertex shader file path
    const std::string& getVertexPath() const { return m_vertexPath; }

    /// @brief Get fragment shader file path
    const std::string& getFragmentPath() const { return m_fragmentPath; }

    /// @brief Check if shader files have been modified since last load
    bool filesModified() const override;

private:
    std::string m_vertexPath;
    std::string m_fragmentPath;
    std::filesystem::file_time_type m_vertexModTime;
    std::filesystem::file_time_type m_fragmentModTime;

    // Fallback sources (if files can't be loaded)
    std::string m_fallbackVertexSource;
    std::string m_fallbackFragmentSource;
};
