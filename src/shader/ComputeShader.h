#pragma once

#include "Shader.h"

/// @brief Compute shader program for GPU compute operations
/// Handles a single compute shader stage for general-purpose GPU computing
class ComputeShader : public Shader
{
public:
    ComputeShader();
    ComputeShader(std::string computePath);
    ~ComputeShader() override;

    /// @brief Load compute shader from file
    /// @param computePath Path to compute shader file
    /// @return true if shader loaded successfully
    bool loadFromFile(const std::string& computePath);

    /// @brief Load compute shader from source string
    /// @param computeSource Compute shader source code
    /// @return true if shader compiled and linked successfully
    bool loadFromSource(const char* computeSource);

    /// @brief Set fallback shader to use if file cannot be loaded
    /// @param computeSource Fallback compute shader source
    void setFallbackSource(const std::string& computeSource);

    /// @brief Reload shader from the previously loaded file path
    /// @return true if reload was successful
    bool reload() override;

    /// @brief Get compute shader file path
    const std::string& getComputePath() const { return m_computePath; }

    /// @brief Check if shader file has been modified since last load
    bool filesModified() const override;

    /// @brief Dispatch the compute shader
    /// @param groupsX Number of work groups in X dimension
    /// @param groupsY Number of work groups in Y dimension
    /// @param groupsZ Number of work groups in Z dimension
    void dispatch(GLuint groupsX, GLuint groupsY = 1, GLuint groupsZ = 1);

    /// @brief Get last GPU execution time in milliseconds
    /// @return Execution time in milliseconds (smoothed average)
    float getLastExecutionTimeMs() const { return m_executionTimeMs; }

    /// @brief Get work group size from shader
    /// @param sizeX Output for work group size in X dimension
    /// @param sizeY Output for work group size in Y dimension
    /// @param sizeZ Output for work group size in Z dimension
    void getWorkGroupSize(GLint& sizeX, GLint& sizeY, GLint& sizeZ) const;

    int getShaderRevisionId() const { return m_shaderRevisionId; }

private:
    std::string m_computePath;
    std::filesystem::file_time_type m_computeModTime;

    int m_shaderRevisionId = 0;
    // Fallback source (if file can't be loaded)
    std::string m_fallbackComputeSource;

    // GPU timing
    mutable GLuint m_timerQuery[2] = {0, 0};  // Double-buffered queries
    mutable int m_currentQuery = 0;
    mutable float m_executionTimeMs = 0.0f;
    mutable bool m_firstFrame = true;
};
