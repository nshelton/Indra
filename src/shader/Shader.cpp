#include "Shader.h"
#include "core/core.h"
#include <glog/logging.h>
#include <fstream>
#include <sstream>
#include <regex>
#include <set>

Shader::Shader()
    : m_program(0), m_isValid(false)
{
}

Shader::~Shader()
{
    if (m_program != 0)
    {
        glDeleteProgram(m_program);
    }
}

std::string Shader::readFile(const std::string &path)
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

    preprocessed = extractUniformMetadata(preprocessed);

    return preprocessed;
}

std::filesystem::file_time_type Shader::getFileModTime(const std::string &path) const
{
    try
    {
        return std::filesystem::last_write_time(path);
    }
    catch (const std::filesystem::filesystem_error &e)
    {
        LOG(WARNING) << "Failed to get modification time for " << path << ": " << e.what();
        return std::filesystem::file_time_type{};
    }
}

bool parseFloat(const std::string &str, float &outValue)
{
    try
    {
        outValue = std::stof(str);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

std::string Shader::extractUniformMetadata(std::string shaderSrc)
{
    m_uniformMap.clear();

    // Matches: uniform T uVal;
    std::regex uniformRegex(
        R"(^\s*uniform\s+(float|int|vec[234]|mat4)\s+([A-Za-z_][A-Za-z0-9_]*)\b\s*;?.*$)");

    // Matches: uniform float uVal (-1, 1, 0.5);
    std::regex inlineRegex(
        R"(^\s*uniform\s+(float|int|vec[234]|mat4)\s+(\w+)\s*\(\s*([-+]?[\d\.eE]+)\s*,\s*([-+]?[\d\.eE]+)\s*(?:,\s*([-+]?[\d\.eE]+)\s*)?\)\s*;?.*$)");

    std::istringstream in(shaderSrc);
    std::ostringstream out;
    std::string line;

    // match m = [fullstring, type, name, min, max, (optional) default]
    auto emitMetadata = [&](const std::smatch &m)
    {
        float minV = 0.f, maxV = 0.f, defV = 0.f;
        if (!parseFloat(m[3].str(), minV))
            return false;
        if (!parseFloat(m[4].str(), maxV))
            return false;
        if (m[5].matched)
        {
            if (!parseFloat(m[5].str(), defV))
                return false;
        }
        else
        {
            defV = minV;
        }
        auto name = m[2].str();

        m_uniformMap[name] = std::make_unique<ShaderUniform<float>>(name, minV, maxV, defV, 0.0f);
        LOG(INFO) << "Registered float uniform with metadata: " << name << " ["
                  << "min=" << minV << ", max=" << maxV << ", def=" << defV << "]";
        m_uniformMap[name]->hasMetadata = true;
        return true;
    };

    while (std::getline(in, line))
    {
        std::smatch match;
        if (std::regex_match(line, match, inlineRegex))
        {
            if (emitMetadata(match))
            {
                // Keep the uniform but strip metadata
                out << "uniform " << match[1].str() << " " << match[2].str() << ";" << "\n";
            }
            else
            {
                LOG(ERROR) << "Failed to parse uniform metadata in line: " << line;
                out << line << "\n"; // output original line on error
            }
            continue; // skip original line
        }
        else if (std::regex_match(line, match, uniformRegex))
        {
            auto name = match[2].str();
            auto type = match[1].str();

            if (type == "float")
            {
                m_uniformMap[name] = std::make_unique<ShaderUniform<float>>(name);
                LOG(INFO) << "Registered float uniform: " << name;
            }
            else if (type == "int")
            {
                m_uniformMap[name] = std::make_unique<ShaderUniform<int>>(name);
                LOG(INFO) << "Registered int uniform: " << name;
            }
            else if (type == "vec2")
            {
                m_uniformMap[name] = std::make_unique<ShaderUniform<vec2>>(name);
                LOG(INFO) << "Registered vec2 uniform: " << name;
            }
            else if (type == "vec3")
            {
                m_uniformMap[name] = std::make_unique<ShaderUniform<vec3>>(name);
                LOG(INFO) << "Registered vec3 uniform: " << name;
            }
            else if (type == "mat4")
            {
                m_uniformMap[name] = std::make_unique<ShaderUniform<matrix4>>(name);
                LOG(INFO) << "Registered mat4 uniform: " << name;
            }
            else
            {
                LOG(WARNING) << "Unsupported uniform type for metadata extraction: " << type;
            }
        }

        out << line << "\n"; // unchanged line
    }

    return out.str();
}

bool Shader::setUniform(const std::string &name, float value)
{
    int location = getUniformLocation(name.c_str());
    if (location == -1)
    {
        LOG(WARNING) << "Uniform '" << name << "' not found in shader program.";
        return false;
    }
    glUniform1f(location, value);
    return true;
}

bool Shader::setUniform(const std::string &name, int value)
{
    int location = getUniformLocation(name.c_str());
    if (location == -1)
    {
        LOG(WARNING) << "Uniform '" << name << "' not found in shader program.";
        return false;
    }
    glUniform1i(location, value);
    return true;
}

bool Shader::setUniform(const std::string &name, const vec2 &value)
{
    int location = getUniformLocation(name.c_str());
    if (location == -1)
    {
        LOG(WARNING) << "Uniform '" << name << "' not found in shader program.";
        return false;
    }
    glUniform2f(location, value.x, value.y);
    return true;
}

bool Shader::setUniform(const std::string &name, const vec3 &value)
{
    int location = getUniformLocation(name.c_str());
    if (location == -1)
    {
        LOG(WARNING) << "Uniform '" << name << "' not found in shader program.";
        return false;
    }
    glUniform3f(location, value.x, value.y, value.z);
    return true;
}

bool Shader::setUniform(const std::string &name, const matrix4 &value)
{
    int location = getUniformLocation(name.c_str());
    if (location == -1)
    {
        LOG(WARNING) << "Uniform '" << name << "' not found in shader program.";
        return false;
    }
    glUniformMatrix4fv(location, 1, GL_FALSE, &value.m[0][0]);
    return true;
}

bool Shader::set(const std::string &name, float value)
{
    if (m_uniformMap.find(name) != m_uniformMap.end())
    {
        auto uniform = dynamic_cast<ShaderUniform<float> *>(m_uniformMap[name].get());
        if (uniform)
        {
            uniform->value = value;
            return true;
        }
    }
    return false;
}

bool Shader::set(const std::string &name, int value)
{
    if (m_uniformMap.find(name) != m_uniformMap.end())
    {
        auto uniform = dynamic_cast<ShaderUniform<int> *>(m_uniformMap[name].get());
        if (uniform)
        {
            uniform->value = value;
            return true;
        }
    }
    return false;
}

bool Shader::set(const std::string &name, const vec2 &value)
{
    if (m_uniformMap.find(name) != m_uniformMap.end())
    {
        auto uniform = dynamic_cast<ShaderUniform<vec2> *>(m_uniformMap[name].get());
        if (uniform)
        {
            uniform->value = value;
            return true;
        }
    }
    return false;
}

bool Shader::set(const std::string &name, const vec3 &value)
{
    if (m_uniformMap.find(name) != m_uniformMap.end())
    {
        auto uniform = dynamic_cast<ShaderUniform<vec3> *>(m_uniformMap[name].get());
        if (uniform)
        {
            uniform->value = value;
            return true;
        }
    }
    return false;
}

bool Shader::set(const std::string &name, const matrix4 &value)
{
    if (m_uniformMap.find(name) != m_uniformMap.end())
    {
        auto uniform = dynamic_cast<ShaderUniform<matrix4> *>(m_uniformMap[name].get());
        if (uniform)
        {
            uniform->value = value;
            return true;
        }
    }
    return false;
}


bool Shader::compileShader(GLuint shader, const char *source, const std::string &shaderName)
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

        // extract line num from message error line is of the form ({col}){linenum}
        std::regex lineNumRegex(R"(\((\d+)\))");
        std::smatch lineNumMatch;
        int errorLineNum = -1;
        if (std::regex_search(m_lastError, lineNumMatch, lineNumRegex))
        {
            errorLineNum = std::stoi(lineNumMatch[1].str());
        }

        // split source into lines for better debugging
        std::istringstream srcStream(source);
        std::string srcLine;
        int lineNum = 1;
        while (std::getline(srcStream, srcLine))
        {
            if (std::abs(lineNum - errorLineNum) <= 2)
            {
                if (lineNum == errorLineNum)
                {
                    LOG(ERROR) << ">> " << lineNum << ": " << srcLine;
                }
                else
                {
                    LOG(ERROR) << lineNum << ": " << srcLine;
                }
            }
            lineNum++;
        }

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

std::string Shader::preprocessIncludes(const std::string &source,
                                       const std::filesystem::path &baseDir,
                                       std::set<std::string> &includedFiles)
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
            catch (const std::filesystem::filesystem_error &e)
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

GLint Shader::getUniformLocation(const char *name) const
{
    if (!m_isValid || m_program == 0)
    {
        return -1;
    }
    return glGetUniformLocation(m_program, name);
}
