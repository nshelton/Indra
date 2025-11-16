#include "Shader.h"
#include <glog/logging.h>
#include <fstream>
#include <sstream>

Shader::Shader()
    : m_program(0)
    , m_isValid(false)
{
}

Shader::~Shader()
{
    if (m_program != 0)
    {
        glDeleteProgram(m_program);
    }
}

std::string Shader::readFile(const std::string& path)
{
    // Get absolute path for better error messages
    std::filesystem::path absPath = std::filesystem::absolute(path);

    std::ifstream file(path);
    if (!file.is_open())
    {
        m_lastError = "Failed to open file: " + path + "\n  Absolute path: " + absPath.string() +
                      "\n  Current working directory: " + std::filesystem::current_path().string();
        LOG(ERROR) << m_lastError;
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    LOG(INFO) << "Successfully loaded shader from: " << absPath.string();
    return buffer.str();
}

std::filesystem::file_time_type Shader::getFileModTime(const std::string& path) const
{
    try
    {
        return std::filesystem::last_write_time(path);
    }
    catch (const std::filesystem::filesystem_error& e)
    {
        LOG(WARNING) << "Failed to get modification time for " << path << ": " << e.what();
        return std::filesystem::file_time_type{};
    }
}

bool Shader::compileShader(GLuint shader, const char* source, const std::string& shaderName)
{
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        m_lastError = shaderName + " compilation failed:\n" + infoLog;
        LOG(ERROR) << m_lastError;
        return false;
    }

    LOG(INFO) << shaderName << " compiled successfully";
    return true;
}

bool Shader::linkProgram(GLuint program)
{
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success)
    {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        m_lastError = "Program linking failed:\n" + std::string(infoLog);
        LOG(ERROR) << m_lastError;
        return false;
    }

    LOG(INFO) << "Shader program linked successfully";
    return true;
}

void Shader::use() const
{
    if (m_isValid && m_program != 0)
    {
        glUseProgram(m_program);
    }
}

GLint Shader::getUniformLocation(const char* name) const
{
    if (!m_isValid || m_program == 0)
    {
        return -1;
    }
    return glGetUniformLocation(m_program, name);
}
