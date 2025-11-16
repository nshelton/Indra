#include "Shader.h"
#include <glog/logging.h>
#include <fstream>
#include <sstream>
#include <regex>
#include <set>

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
    std::string source = buffer.str();

    LOG(INFO) << "Successfully loaded shader from: " << absPath.string();

    // Preprocess #include directives
    std::filesystem::path baseDir = absPath.parent_path();
    std::set<std::string> includedFiles;

    // Add the main file to the included set to prevent self-inclusion
    includedFiles.insert(absPath.string());

    std::string preprocessed = preprocessIncludes(source, baseDir, includedFiles);

    return preprocessed;
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

std::string Shader::preprocessIncludes(const std::string& source,
                                       const std::filesystem::path& baseDir,
                                       std::set<std::string>& includedFiles)
{
    // Regular expression to match #include "file.glsl" or #include <file.glsl>
    std::regex includeRegex(R"(^\s*#include\s+[\"<]([^\">\s]+)[\">]\s*$)");

    std::istringstream sourceStream(source);
    std::ostringstream result;
    std::string line;
    int lineNumber = 0;

    while (std::getline(sourceStream, line))
    {
        lineNumber++;
        std::smatch match;

        if (std::regex_match(line, match, includeRegex))
        {
            // Extract the included file path
            std::string includePath = match[1].str();

            // Resolve the full path (relative to baseDir)
            std::filesystem::path fullPath = baseDir / includePath;
            std::string canonicalPath;

            try
            {
                // Get canonical path to handle . and .. in paths
                canonicalPath = std::filesystem::canonical(fullPath).string();
            }
            catch (const std::filesystem::filesystem_error& e)
            {
                // If canonical fails (file doesn't exist), try with absolute
                fullPath = std::filesystem::absolute(baseDir / includePath);
                canonicalPath = fullPath.string();
            }

            // Check for circular includes
            if (includedFiles.find(canonicalPath) != includedFiles.end())
            {
                LOG(WARNING) << "Circular include detected: " << includePath
                           << " (already included from " << canonicalPath << ")";
                result << "// Circular include skipped: " << includePath << "\n";
                continue;
            }

            // Mark this file as included
            includedFiles.insert(canonicalPath);

            // Read the included file
            std::ifstream includeFile(fullPath);
            if (!includeFile.is_open())
            {
                LOG(ERROR) << "Failed to open included file: " << includePath
                          << "\n  Full path: " << fullPath.string()
                          << "\n  Base directory: " << baseDir.string();
                result << "// ERROR: Failed to include: " << includePath << "\n";
                continue;
            }

            std::stringstream includeBuffer;
            includeBuffer << includeFile.rdbuf();
            std::string includeSource = includeBuffer.str();

            LOG(INFO) << "Processing include: " << includePath << " from " << fullPath.string();

            // Add a comment marker for debugging
            result << "// BEGIN INCLUDE: " << includePath << "\n";

            // Recursively process includes in the included file
            std::filesystem::path includeDir = fullPath.parent_path();
            std::string processedInclude = preprocessIncludes(includeSource, includeDir, includedFiles);
            result << processedInclude;

            // Make sure included content ends with newline
            if (!processedInclude.empty() && processedInclude.back() != '\n')
            {
                result << "\n";
            }

            result << "// END INCLUDE: " << includePath << "\n";
        }
        else
        {
            // Regular line, just copy it
            result << line << "\n";
        }
    }

    return result.str();
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
