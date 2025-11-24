#pragma once

#include <glad/glad.h>
#include <core/core.h>
#include <string>
#include <filesystem>
#include <set>

// class Shader;

struct IUniform
{
    virtual ~IUniform() = default;
    // virtual void upload(Shader& s) = 0;
    GLuint location;
    bool hasMetadata{false};
    std::string name;
};

template <typename T>
struct ShaderUniform : public IUniform
{
    explicit ShaderUniform(std::string n) {}
    explicit ShaderUniform(std::string n, T min, T max, T def, T val) : minValue(min),
                                                                        maxValue(max),
                                                                        defValue(def),
                                                                        value(val)
    {
        name = std::move(n);
    }

    T value;
    T minValue;
    T maxValue;
    T defValue;
};

typedef std::unordered_map<std::string, std::unique_ptr<IUniform>> UniformMap;

/// @brief Base class for OpenGL shader programs
/// Provides common functionality for shader compilation, linking, and hot-reloading
class Shader
{
public:
    Shader();
    virtual ~Shader();

    // Prevent copying
    Shader(const Shader &) = delete;
    Shader &operator=(const Shader &) = delete;

    /// @brief Reload shaders from the previously loaded file paths
    /// @return true if reload was successful
    virtual bool reload() = 0;

    /// @brief Bind the shader program for rendering
    void use() const;

    /// @brief Get the OpenGL program ID
    GLuint getProgram() const { return m_program; }

    /// @brief Get uniform location by name
    GLint getUniformLocation(const char *name) const;

    /// @brief Check if the program is valid and ready to use
    bool isValid() const { return m_program != 0 && m_isValid; }

    /// @brief Get last compilation/linking error message
    const std::string &getLastError() const { return m_lastError; }

    /// @brief Check if shader source files have been modified since last load
    virtual bool filesModified() const = 0;

    int getUniformLocationCached(const char *name)
    {
        auto it = m_uniformMap.find(name);
        if (it != m_uniformMap.end())
        {
            return it->second.get()->location;
        }
        else
        {
            return -1;
        }
    }

    const UniformMap &uniforms() const { return m_uniformMap; }

    bool setUniform(const std::string &name, float value);
    bool setUniform(const std::string &name, int value);
    bool setUniform(const std::string &name, const vec2 &value);
    bool setUniform(const std::string &name, const vec3 &value);
    bool setUniform(const std::string &name, const matrix4 &value);

    bool set(const std::string &name, float value);
    bool set(const std::string &name, int value);
    bool set(const std::string &name, const vec2 &value);
    bool set(const std::string &name, const vec3 &value);
    bool set(const std::string &name, const matrix4 &value);

    // overlaod [] operator to get uniform by name
    IUniform *operator[](const std::string &name) 
    {
        auto it = m_uniformMap.find(name);
        if (it != m_uniformMap.end())
        {
            return it->second.get();
        }
        else
        {
            return nullptr;
        }
    }

protected:
    /// @brief Compile a shader from source code
    /// @param shader The shader object ID
    /// @param source The shader source code
    /// @param shaderName Name for error messages
    /// @return true if compilation succeeded
    bool compileShader(GLuint shader, const char *source, const std::string &shaderName);

    /// @brief Link a shader program
    /// @param program The program object ID
    /// @return true if linking succeeded
    bool linkProgram(GLuint program);

    /// @brief Read a file into a string
    /// @param path Path to the file
    /// @return File contents, or empty string on error
    std::string readFile(const std::string &path);

    /// @brief Get the last modification time of a file
    /// @param path Path to the file
    /// @return File modification time
    std::filesystem::file_time_type getFileModTime(const std::string &path) const;

    /// @brief Preprocess shader source to handle #include directives
    /// @param source The original shader source code
    /// @param baseDir Directory of the file being processed (for relative includes)
    /// @param includedFiles Set of already included files (prevents circular includes)
    /// @return Preprocessed source code with includes resolved
    std::string preprocessIncludes(const std::string &source,
                                   const std::filesystem::path &baseDir,
                                   std::set<std::string> &includedFiles);

    /// @brief Extract uniform metadata from shader source code
    std::string extractUniformMetadata(std::string shaderSrc);

    GLuint m_program;
    bool m_isValid;
    std::string m_lastError;

    UniformMap m_uniformMap;
};
