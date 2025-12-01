#pragma once

#include <glad/glad.h>
#include <core/core.h>
#include <string>
#include <filesystem>
#include <set>
#include <tuple>
#include <concepts>

// Uniform data for a single type - no inheritance, no virtual
template <typename T>
struct Uniform
{
    std::string name;
    GLuint location{0};
    bool hasMetadata{false};
    T value{};
    T minValue{};
    T maxValue{};
    T defValue{};

    bool needsUpload{true};

    void upload()
    {
        if constexpr (std::same_as<T, float>)
            glUniform1f(location, value);
        else if constexpr (std::same_as<T, int>)
            glUniform1i(location, value);
        else if constexpr (std::same_as<T, vec2>)
            glUniform2f(location, value.x, value.y);
        else if constexpr (std::same_as<T, vec3>)
            glUniform3f(location, value.x, value.y, value.z);
        else if constexpr (std::same_as<T, vec2i>)
            glUniform2i(location, value.x, value.y);
        else if constexpr (std::same_as<T, color>)
            glUniform4f(location, value.r, value.g, value.b, value.a);
        else if constexpr (std::same_as<T, matrix4>)
            glUniformMatrix4fv(location, 1, GL_FALSE, &value.m[0][0]);
        else
            LOG(WARNING) << "Unsupported uniform type for '" << name << "'";

        needsUpload = false;
    }

    void set(const T &val)
    {
        value = val;
        needsUpload = true;
    }
};

// Type-safe uniform store using tuple of maps
template <typename... Ts>
class UniformStore
{
    std::tuple<std::unordered_map<std::string, Uniform<Ts>>...> m_maps;

public:
    // Get the map for a specific type
    template <typename T>
    auto &map()
    {
        return std::get<std::unordered_map<std::string, Uniform<T>>>(m_maps);
    }

    template <typename T>
    const auto &map() const
    {
        return std::get<std::unordered_map<std::string, Uniform<T>>>(m_maps);
    }

    // Get a uniform by name (throws if not found)
    template <typename T>
    Uniform<T> &get(const std::string &name)
    {
        return map<T>().at(name);
    }

    template <typename T>
    const Uniform<T> &get(const std::string &name) const
    {
        return map<T>().at(name);
    }

    // Try to get a uniform, returns nullptr if not found
    template <typename T>
    Uniform<T> *tryGet(const std::string &name)
    {
        auto &m = map<T>();
        auto it = m.find(name);
        return it != m.end() ? &it->second : nullptr;
    }

    // Set value (creates uniform if it doesn't exist)
    template <typename T>
    void set(const std::string &name, const T &val)
    {
        map<T>()[name].value = val;
    }

    // Add a uniform with full metadata
    template <typename T>
    Uniform<T> &add(const std::string &name, T val = {}, T min = {}, T max = {}, T def = {}, bool hasMeta = false)
    {
        auto &u = map<T>()[name];
        u.name = name;
        u.value = val;
        u.minValue = min;
        u.maxValue = max;
        u.defValue = def;
        u.hasMetadata = hasMeta;
        return u;
    }

    // Iterate over ALL uniforms with a lambda
    // Lambda receives: Uniform<T>& for each uniform
    template <typename F>
    void forEach(F &&fn)
    {
        std::apply([&](auto &...maps)
                   { (([&](auto &m)
                       {
                for (auto& [k, v] : m) fn(v); }(maps)),
                      ...); },
                   m_maps);
    }

    template <typename F>
    void forEach(F &&fn) const
    {
        std::apply([&](const auto &...maps)
                   { (([&](const auto &m)
                       {
                for (const auto& [k, v] : m) fn(v); }(maps)),
                      ...); },
                   m_maps);
    }

    // Clear all maps
    void clear()
    {
        std::apply([](auto &...maps)
                   { (maps.clear(), ...); },
                   m_maps);
    }

    // Upload all uniforms to GPU
    void uploadAll() const
    {
        forEach([](const auto &u)
                { u.upload(); });
    }

    // Find a uniform location by name (searches all types)
    // Returns -1 if not found
    GLint getLocationByName(const std::string &name) const
    {
        GLint location = -1;
        std::apply([&](const auto &...maps)
                   {
            (([&](const auto &m) {
                if (location == -1) {
                    auto it = m.find(name);
                    if (it != m.end()) {
                        location = it->second.location;
                    }
                }
            }(maps)), ...); },
                   m_maps);
        return location;
    }
};

// Define supported uniform types
using ShaderUniforms = UniformStore<float, int, vec2, vec2i, vec3, matrix4, color>;

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

    // Access the uniform store
    ShaderUniforms &uniforms() { return m_uniforms; }
    const ShaderUniforms &uniforms() const { return m_uniforms; }

    // Get cached uniform location by name (searches all types)
    GLint getUniformLocationCached(const char *name) const
    {
        return m_uniforms.getLocationByName(name);
    }

    // Upload uniform to GPU immediately
    // bool setUniform(const std::string &name, float value);
    // bool setUniform(const std::string &name, int value);
    // bool setUniform(const std::string &name, const vec2 &value);
    // bool setUniform(const std::string &name, const vec3 &value);
    // bool setUniform(const std::string &name, const matrix4 &value);
    // bool setUniform(const std::string &name, const vec2i &value);

    // Set uniform value in store (call setUniform to upload)
    template <typename T>
    bool set(const std::string &name, const T &value)
    {
        if (auto *u = m_uniforms.tryGet<T>(name))
        {
            u->set(value);
            return true;
        }
        return false;
    }

    // Get uniform pointer (nullptr if not found or wrong type)
    template <typename T>
    Uniform<T> *operator[](const std::string &name)
    {
        return m_uniforms.tryGet<T>(name);
    }

    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json &j);

    /// @brief Restores uniform values from a backup after a reload
    /// @param backup The backed-up uniform store.
    void restoreUniforms(const ShaderUniforms& backup);

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
    std::string m_filename;

    ShaderUniforms m_uniforms;
};
