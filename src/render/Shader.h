#pragma once

#include <glad/glad.h>
#include <string>
#include <filesystem>

/// @brief Base class for OpenGL shader programs
/// Provides common functionality for shader compilation, linking, and hot-reloading
class Shader
{
public:
    Shader();
    virtual ~Shader();

    // Prevent copying
    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

    /// @brief Reload shaders from the previously loaded file paths
    /// @return true if reload was successful
    virtual bool reload() = 0;

    /// @brief Bind the shader program for rendering
    void use() const;

    /// @brief Get the OpenGL program ID
    GLuint getProgram() const { return m_program; }

    /// @brief Get uniform location by name
    GLint getUniformLocation(const char* name) const;

    /// @brief Check if the program is valid and ready to use
    bool isValid() const { return m_program != 0 && m_isValid; }

    /// @brief Get last compilation/linking error message
    const std::string& getLastError() const { return m_lastError; }

    /// @brief Check if shader source files have been modified since last load
    virtual bool filesModified() const = 0;

protected:
    /// @brief Compile a shader from source code
    /// @param shader The shader object ID
    /// @param source The shader source code
    /// @param shaderName Name for error messages
    /// @return true if compilation succeeded
    bool compileShader(GLuint shader, const char* source, const std::string& shaderName);

    /// @brief Link a shader program
    /// @param program The program object ID
    /// @return true if linking succeeded
    bool linkProgram(GLuint program);

    /// @brief Read a file into a string
    /// @param path Path to the file
    /// @return File contents, or empty string on error
    std::string readFile(const std::string& path);

    /// @brief Get the last modification time of a file
    /// @param path Path to the file
    /// @return File modification time
    std::filesystem::file_time_type getFileModTime(const std::string& path) const;

    GLuint m_program;
    bool m_isValid;
    std::string m_lastError;
};
