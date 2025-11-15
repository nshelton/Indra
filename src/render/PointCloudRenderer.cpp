#include "PointCloudRenderer.h"

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glog/logging.h>
#include "PointCloudKernel.cuh"

bool PointCloudRenderer::init()
{
    // Embedded fallback shaders (in case files can't be loaded)
    const char *vsSrc = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;
uniform mat4 uViewProjMat;
uniform float uPointSizePx;
out vec4 vColor;
void main(){
    gl_Position = uViewProjMat * vec4(aPos, 1.0);
    gl_PointSize = uPointSizePx;
    vColor = aColor;
}
)";

    const char *fsSrc = R"(
#version 330 core
in vec4 vColor;
out vec4 FragColor;
void main(){
    // circular point sprite mask
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    FragColor = vColor;
}
)";

    // Create shader program
    m_shaderProgram = std::make_unique<ShaderProgram>();

    // Try to load from source directory first (for hot-reload during development)
    // then fallback to build directory, then embedded shaders
    bool success = m_shaderProgram->loadFromFiles("../../shaders/pointcloud.vert.glsl", "../../shaders/pointcloud.frag.glsl");

    if (!success)
    {
        // Try build directory
        success = m_shaderProgram->loadFromFiles("shaders/pointcloud.vert.glsl", "shaders/pointcloud.frag.glsl");
    }

    if (!success)
    {
        LOG(WARNING) << "Failed to load shader files, using embedded shaders";
        success = m_shaderProgram->loadFromSource(vsSrc, fsSrc);
    }

    if (!success)
    {
        LOG(ERROR) << "Failed to initialize PointCloud shaders: " << m_shaderProgram->getLastError();
        return false;
    }

    // Create vertex array and buffer
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void *)(3 * sizeof(float)));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    LOG(INFO) << "PointCloudRenderer initialized successfully";
    return true;
}

void PointCloudRenderer::shutdown()
{
    // Cleanup CUDA resources before deleting OpenGL resources
    shutdownCudaInterop();

    if (m_vao)
        glDeleteVertexArrays(1, &m_vao);
    if (m_vbo)
        glDeleteBuffers(1, &m_vbo);

    // Shader program is automatically cleaned up by unique_ptr
    m_shaderProgram.reset();

    m_vao = 0;
    m_vbo = 0;
    m_points.clear();
}

void PointCloudRenderer::clear()
{
    m_points.clear();
}

void PointCloudRenderer::addPoint(vec3 p, color c)
{
    m_points.push_back(GLVertex{p.x, p.y, p.z, c.r, c.g, c.b, c.a});
}

void PointCloudRenderer::uploadToGPU()
{
    if (m_points.empty())
        return;

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(m_points.size() * sizeof(GLVertex)),
                 m_points.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

/// @brief Draw the 3D lines using the camera's view-projection matrix
/// @param camera The camera to use for rendering
void PointCloudRenderer::draw(const Camera &camera)
{
    if (m_points.empty() || !m_shaderProgram || !m_shaderProgram->isValid())
        return;

    m_shaderProgram->use();

    // Get uniform locations
    GLint uViewProjMat = m_shaderProgram->getUniformLocation("uViewProjMat");
    GLint uPointSizePx = m_shaderProgram->getUniformLocation("uPointSizePx");

    // Get view-projection matrix from camera
    matrix4 viewProj = camera.getViewProjectionMatrix();
    glUniformMatrix4fv(uViewProjMat, 1, GL_TRUE, &viewProj.m[0][0]);

    float pointDiameterPx = m_pointRadius * 2.0f;
    if (pointDiameterPx <= 0.0f)
    {
        pointDiameterPx = 5.0f; // Default point size
    }

    glEnable(GL_PROGRAM_POINT_SIZE);
    glUniform1f(uPointSizePx, pointDiameterPx);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(m_points.size()));
    glBindVertexArray(0);
}

bool PointCloudRenderer::initCudaInterop()
{
    if (m_cudaInteropInitialized)
    {
        std::cerr << "CUDA interop already initialized" << std::endl;
        return true;
    }

    if (!m_vbo)
    {
        std::cerr << "Cannot initialize CUDA interop: VBO not created yet" << std::endl;
        return false;
    }

    // Initialize CUDA for OpenGL interop
    std::cout << "Initializing CUDA interop..." << std::endl;

    // Check CUDA runtime version
    int runtimeVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "CUDA Runtime Version: " << runtimeVersion << std::endl;

    int driverVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    std::cout << "CUDA Driver Version: " << driverVersion << std::endl;

    // First, find a CUDA device that supports OpenGL interop
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    std::cout << "cudaGetDeviceCount returned: " << cudaGetErrorString(err) << " (code=" << err << "), deviceCount=" << deviceCount << std::endl;

    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " (error code: " << err << ")" << std::endl;
        std::cerr << "This usually means:" << std::endl;
        std::cerr << "  1. CUDA runtime DLLs are not found" << std::endl;
        std::cerr << "  2. CUDA driver is too old for this CUDA version" << std::endl;
        std::cerr << "  3. No CUDA-capable GPU is available" << std::endl;
        return false;
    }

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA devices found (device count = 0)" << std::endl;
        return false;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;

    // Get device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err == cudaSuccess)
    {
        std::cout << "Device 0: " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
    }

    // Register OpenGL VBO with CUDA
    err = cudaGraphicsGLRegisterBuffer(&m_cudaVboResource, m_vbo, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA graphics registration failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    m_cudaInteropInitialized = true;
    std::cout << "CUDA interop initialized successfully" << std::endl;
    return true;
}

void PointCloudRenderer::shutdownCudaInterop()
{
    if (!m_cudaInteropInitialized)
        return;

    if (m_cudaVboResource)
    {
        cudaGraphicsUnregisterResource(m_cudaVboResource);
        m_cudaVboResource = nullptr;
    }

    // Free FFT data buffer only if we own it
    if (m_d_fftData && m_ownsFFTData)
    {
        cudaFree(m_d_fftData);
        m_d_fftData = nullptr;
        m_numFFTBins = 0;
        m_ownsFFTData = false;
    }

    m_cudaInteropInitialized = false;
}

bool PointCloudRenderer::runCudaKernel(float deltaTime)
{
    // Lazy initialization - initialize CUDA on first kernel call when OpenGL context is current
    if (!m_cudaInteropAttempted)
    {
        m_cudaInteropAttempted = true;
        if (!initCudaInterop())
        {
            std::cerr << "CUDA interop initialization failed - GPU processing disabled" << std::endl;
            return false;
        }
    }

    if (!m_cudaInteropInitialized || !m_cudaVboResource)
    {
        return false;  // Silently skip if CUDA not available
    }

    if (m_points.empty())
        return true;

    // Map the OpenGL buffer to CUDA
    cudaError_t err = cudaGraphicsMapResources(1, &m_cudaVboResource, 0);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA graphics map failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Get a device pointer to the mapped buffer
    float* d_vertices = nullptr;
    size_t numBytes = 0;
    err = cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, &numBytes, m_cudaVboResource);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA get mapped pointer failed: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnmapResources(1, &m_cudaVboResource, 0);
        return false;
    }

    // Launch the CUDA kernel to process the vertices with FFT data
    err = launchAnimatePointsKernel(d_vertices, static_cast<int>(m_points.size()), deltaTime,
                                     m_d_fftData, m_numFFTBins);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnmapResources(1, &m_cudaVboResource, 0);
        return false;
    }

    // Unmap the buffer so OpenGL can use it
    err = cudaGraphicsUnmapResources(1, &m_cudaVboResource, 0);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA graphics unmap failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    return true;
}

bool PointCloudRenderer::reloadShaders()
{
    if (!m_shaderProgram)
    {
        LOG(WARNING) << "Cannot reload shaders: shader program not initialized";
        return false;
    }

    LOG(INFO) << "Reloading PointCloud shaders...";

    bool success = m_shaderProgram->reload();

    if (success)
    {
        LOG(INFO) << "PointCloud shaders reloaded successfully!";
    }
    else
    {
        LOG(ERROR) << "Failed to reload PointCloud shaders: " << m_shaderProgram->getLastError();
    }

    return success;
}
