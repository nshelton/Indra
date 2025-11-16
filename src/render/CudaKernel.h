#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <cuda.h>

// Runtime-compiled CUDA kernel manager using NVRTC
class CudaKernel
{
public:
    CudaKernel();
    ~CudaKernel();

    // Load and compile kernel from file
    bool loadFromFile(const std::string& kernelPath, const std::string& kernelName);

    // Load and compile kernel from source string
    bool loadFromSource(const std::string& source, const std::string& kernelName);

    // Reload kernel from last file path
    bool reload();

    // Get kernel function
    CUfunction getFunction() const { return m_function; }

    // Check if kernel is valid
    bool isValid() const { return m_function != nullptr; }

    // Get last error message
    const std::string& getLastError() const { return m_lastError; }

    // Check if source file was modified
    bool fileModified() const;

    // Add include path for NVRTC compilation
    void addIncludePath(const std::string& path);

private:
    bool compileSource(const std::string& source, const std::string& kernelName);
    std::filesystem::file_time_type getFileModTime(const std::string& path) const;
    std::string readFile(const std::string& path);

    CUmodule m_module{nullptr};
    CUfunction m_function{nullptr};
    std::string m_kernelPath;
    std::string m_kernelName;
    std::string m_lastError;
    std::filesystem::file_time_type m_lastModTime;
    std::vector<std::string> m_includePaths;
};
