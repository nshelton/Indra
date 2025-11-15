#pragma once

#include <glad/glad.h>
#include <string>
#include <filesystem>

class ShaderProgram
{
public:
    ShaderProgram();
    ~ShaderProgram();

    // Load shaders from files
    bool loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath);

    // Load shaders from source strings (fallback)
    bool loadFromSource(const char* vertexSource, const char* fragmentSource);

    // Reload shaders from the previously loaded file paths
    bool reload();

    // Bind the shader program
    void use() const;

    // Get the OpenGL program ID
    GLuint getProgram() const { return m_program; }

    // Get uniform location
    GLint getUniformLocation(const char* name) const;

    // Check if the program is valid
    bool isValid() const { return m_program != 0 && m_isValid; }

    // Get last compilation error
    const std::string& getLastError() const { return m_lastError; }

    // Get shader file paths
    const std::string& getVertexPath() const { return m_vertexPath; }
    const std::string& getFragmentPath() const { return m_fragmentPath; }

    // Check if files have been modified since last load
    bool filesModified() const;

private:
    bool compileShader(GLuint shader, const char* source, const std::string& shaderName);
    bool linkProgram(GLuint program);
    std::string readFile(const std::string& path);
    std::filesystem::file_time_type getFileModTime(const std::string& path) const;

    GLuint m_program;
    bool m_isValid;

    std::string m_vertexPath;
    std::string m_fragmentPath;
    std::filesystem::file_time_type m_vertexModTime;
    std::filesystem::file_time_type m_fragmentModTime;

    std::string m_lastError;

    // Fallback sources (if files can't be loaded)
    std::string m_fallbackVertexSource;
    std::string m_fallbackFragmentSource;
};
