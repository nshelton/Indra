#include "CudaKernel.h"
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <glog/logging.h>

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      m_lastError = std::string("NVRTC error: ") +                \
                    nvrtcGetErrorString(result);                  \
      LOG(ERROR) << m_lastError;                                  \
      return false;                                               \
    }                                                             \
  } while(0)

#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char* errorStr;                                       \
      cuGetErrorString(result, &errorStr);                        \
      m_lastError = std::string("CUDA Driver error: ") +          \
                    errorStr;                                     \
      LOG(ERROR) << m_lastError;                                  \
      return false;                                               \
    }                                                             \
  } while(0)

CudaKernel::CudaKernel()
{
    // Initialize CUDA driver API
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS)
    {
        const char* errorStr;
        cuGetErrorString(err, &errorStr);
        LOG(WARNING) << "Failed to initialize CUDA driver API: " << errorStr;
    }
}

CudaKernel::~CudaKernel()
{
    if (m_module)
    {
        cuModuleUnload(m_module);
        m_module = nullptr;
    }
    m_function = nullptr;
}

std::string CudaKernel::readFile(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        m_lastError = "Failed to open file: " + path;
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::filesystem::file_time_type CudaKernel::getFileModTime(const std::string& path) const
{
    try
    {
        return std::filesystem::last_write_time(path);
    }
    catch (const std::filesystem::filesystem_error& e)
    {
        return std::filesystem::file_time_type{};
    }
}

bool CudaKernel::fileModified() const
{
    if (m_kernelPath.empty())
        return false;

    auto currentModTime = getFileModTime(m_kernelPath);
    return currentModTime > m_lastModTime;
}

bool CudaKernel::loadFromFile(const std::string& kernelPath, const std::string& kernelName)
{
    LOG(INFO) << "Loading CUDA kernel from: " << std::filesystem::absolute(kernelPath);

    std::string source = readFile(kernelPath);
    if (source.empty())
    {
        LOG(ERROR) << "Failed to read kernel file: " << kernelPath;
        return false;
    }

    m_kernelPath = kernelPath;
    m_kernelName = kernelName;
    m_lastModTime = getFileModTime(kernelPath);

    return compileSource(source, kernelName);
}

bool CudaKernel::loadFromSource(const std::string& source, const std::string& kernelName)
{
    m_kernelPath.clear();
    m_kernelName = kernelName;
    return compileSource(source, kernelName);
}

bool CudaKernel::reload()
{
    if (m_kernelPath.empty())
    {
        m_lastError = "Cannot reload: no file path stored";
        LOG(WARNING) << m_lastError;
        return false;
    }

    LOG(INFO) << "Reloading CUDA kernel from: " << m_kernelPath;

    std::string source = readFile(m_kernelPath);
    if (source.empty())
    {
        LOG(ERROR) << "Failed to read kernel file during reload: " << m_kernelPath;
        return false;
    }

    // Store old module in case compilation fails
    CUmodule oldModule = m_module;
    CUfunction oldFunction = m_function;
    m_module = nullptr;
    m_function = nullptr;

    if (!compileSource(source, m_kernelName))
    {
        // Restore old module on failure
        m_module = oldModule;
        m_function = oldFunction;
        LOG(ERROR) << "Kernel reload failed, keeping previous version";
        return false;
    }

    // Cleanup old module
    if (oldModule)
    {
        cuModuleUnload(oldModule);
    }

    m_lastModTime = getFileModTime(m_kernelPath);
    return true;
}

void CudaKernel::addIncludePath(const std::string& path)
{
    m_includePaths.push_back(path);
}

bool CudaKernel::compileSource(const std::string& source, const std::string& kernelName)
{
    nvrtcProgram prog;

    // Create NVRTC program
    NVRTC_SAFE_CALL(nvrtcCreateProgram(
        &prog,
        source.c_str(),
        (kernelName + ".cu").c_str(),
        0,
        nullptr,
        nullptr
    ));

    // Build compilation options
    std::vector<const char*> opts;
    opts.push_back("--gpu-architecture=compute_75");
    opts.push_back("--std=c++17");
    opts.push_back("--use_fast_math");

    // Add include paths
    std::vector<std::string> includeOpts;
    for (const auto& includePath : m_includePaths)
    {
        includeOpts.push_back("-I" + includePath);
    }
    for (const auto& opt : includeOpts)
    {
        opts.push_back(opt.c_str());
    }

    // Compile the program
    nvrtcResult compileResult = nvrtcCompileProgram(prog, opts.size(), opts.data());

    // Get compilation log
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    if (logSize > 1)
    {
        std::string log(logSize, '\0');
        nvrtcGetProgramLog(prog, &log[0]);

        if (compileResult != NVRTC_SUCCESS)
        {
            m_lastError = "NVRTC compilation failed:\n" + log;
            LOG(ERROR) << m_lastError;
            nvrtcDestroyProgram(&prog);
            return false;
        }
        else if (logSize > 10) // Only show non-trivial logs
        {
            LOG(INFO) << "NVRTC compilation log:\n" << log;
        }
    }

    // Get PTX
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));

    std::string ptx(ptxSize, '\0');
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, &ptx[0]));

    // Destroy the program
    nvrtcDestroyProgram(&prog);

    // Load PTX into CUDA module
    CUmodule newModule;
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&newModule, ptx.c_str(), 0, nullptr, nullptr));

    // Get kernel function
    CUfunction newFunction;
    CUDA_SAFE_CALL(cuModuleGetFunction(&newFunction, newModule, kernelName.c_str()));

    m_module = newModule;
    m_function = newFunction;

    LOG(INFO) << "Successfully compiled CUDA kernel: " << kernelName;
    return true;
}
