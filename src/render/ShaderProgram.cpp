#include "ShaderProgram.h"
#include <glog/logging.h>
#include <fstream>
#include <sstream>
#include <filesystem>

ShaderProgram::ShaderProgram()
    : m_program(0)
    , m_isValid(false)
{
}

ShaderProgram::~ShaderProgram()
{
    if (m_program != 0)
    {
        glDeleteProgram(m_program);
    }
}

std::string ShaderProgram::readFile(const std::string& path)
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

std::filesystem::file_time_type ShaderProgram::getFileModTime(const std::string& path) const
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

bool ShaderProgram::compileShader(GLuint shader, const char* source, const std::string& shaderName)
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

bool ShaderProgram::linkProgram(GLuint program)
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

bool ShaderProgram::loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath)
{
    m_vertexPath = vertexPath;
    m_fragmentPath = fragmentPath;

    // Read shader sources
    std::string vertexSource = readFile(vertexPath);
    std::string fragmentSource = readFile(fragmentPath);

    if (vertexSource.empty() || fragmentSource.empty())
    {
        // If files can't be loaded and we have fallback, use it
        if (!m_fallbackVertexSource.empty() && !m_fallbackFragmentSource.empty())
        {
            LOG(WARNING) << "Shader files not found, using fallback embedded shaders";
            return loadFromSource(m_fallbackVertexSource.c_str(), m_fallbackFragmentSource.c_str());
        }
        return false;
    }

    // Store modification times
    m_vertexModTime = getFileModTime(vertexPath);
    m_fragmentModTime = getFileModTime(fragmentPath);

    // Compile and link
    return loadFromSource(vertexSource.c_str(), fragmentSource.c_str());
}

bool ShaderProgram::loadFromSource(const char* vertexSource, const char* fragmentSource)
{
    // Create shaders
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    // Compile vertex shader
    if (!compileShader(vertexShader, vertexSource, "Vertex shader"))
    {
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        m_isValid = false;
        return false;
    }

    // Compile fragment shader
    if (!compileShader(fragmentShader, fragmentSource, "Fragment shader"))
    {
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        m_isValid = false;
        return false;
    }

    // Create and link program
    GLuint newProgram = glCreateProgram();
    glAttachShader(newProgram, vertexShader);
    glAttachShader(newProgram, fragmentShader);

    if (!linkProgram(newProgram))
    {
        glDeleteProgram(newProgram);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        m_isValid = false;
        return false;
    }

    // Clean up shaders (they're linked into the program now)
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Delete old program if it exists
    if (m_program != 0)
    {
        glDeleteProgram(m_program);
    }

    m_program = newProgram;
    m_isValid = true;
    m_lastError.clear();

    LOG(INFO) << "Shader program created successfully (ID: " << m_program << ")";
    return true;
}

bool ShaderProgram::reload()
{
    if (m_vertexPath.empty() || m_fragmentPath.empty())
    {
        m_lastError = "Cannot reload: no file paths set";
        LOG(WARNING) << m_lastError;
        return false;
    }

    LOG(INFO) << "Reloading shaders from " << m_vertexPath << " and " << m_fragmentPath;

    // Store old program in case reload fails
    GLuint oldProgram = m_program;
    bool oldValid = m_isValid;

    // Try to reload
    m_program = 0;  // Temporarily clear so loadFromFiles creates a new one
    bool success = loadFromFiles(m_vertexPath, m_fragmentPath);

    if (!success)
    {
        // Restore old program on failure
        if (oldProgram != 0 && !m_program)
        {
            m_program = oldProgram;
            m_isValid = oldValid;
        }
        LOG(ERROR) << "Shader reload failed, keeping previous version";
        return false;
    }

    // Delete old program on success
    if (oldProgram != 0 && oldProgram != m_program)
    {
        glDeleteProgram(oldProgram);
    }

    LOG(INFO) << "Shader reload successful!";
    return true;
}

void ShaderProgram::use() const
{
    if (m_isValid && m_program != 0)
    {
        glUseProgram(m_program);
    }
}

GLint ShaderProgram::getUniformLocation(const char* name) const
{
    if (!m_isValid || m_program == 0)
    {
        return -1;
    }
    return glGetUniformLocation(m_program, name);
}

bool ShaderProgram::filesModified() const
{
    if (m_vertexPath.empty() || m_fragmentPath.empty())
    {
        return false;
    }

    auto currentVertexTime = getFileModTime(m_vertexPath);
    auto currentFragmentTime = getFileModTime(m_fragmentPath);

    return (currentVertexTime > m_vertexModTime) || (currentFragmentTime > m_fragmentModTime);
}
