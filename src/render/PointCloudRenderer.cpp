#include "PointCloudRenderer.h"

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda.h>
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

    // Initialize CUDA kernel with NVRTC
    m_cudaKernel = std::make_unique<CudaKernel>();

    // Try to load from source directory first, then build directory
    bool kernelSuccess = m_cudaKernel->loadFromFile("../../kernels/pointcloud_kernel.cu", "animatePointsKernel");
    if (!kernelSuccess)
    {
        kernelSuccess = m_cudaKernel->loadFromFile("kernels/pointcloud_kernel.cu", "animatePointsKernel");
    }

    if (!kernelSuccess)
    {
        LOG(WARNING) << "Failed to load CUDA kernel for hot-reload: " << m_cudaKernel->getLastError();
        LOG(WARNING) << "CUDA kernel hot-reload will not be available";
    }
    else
    {
        LOG(INFO) << "CUDA kernel loaded successfully for hot-reload";
    }

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

    // Free particle data buffer
    if (m_d_particleData)
    {
        cudaFree(m_d_particleData);
        m_d_particleData = nullptr;
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

    // Initialize particle data buffer if needed (lazy allocation)
    int numPoints = static_cast<int>(m_points.size());
    if (!m_d_particleData)
    {
        // Allocate particle data on GPU
        size_t particleDataSize = numPoints * sizeof(ParticleData);
        ParticleData* d_particles = nullptr;
        cudaError_t err = cudaMalloc(&d_particles, particleDataSize);
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to allocate particle data: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        m_d_particleData = d_particles;

        // Initialize particles with random values on CPU, then upload to GPU
        std::vector<ParticleData> h_particles(numPoints);
        for (int i = 0; i < numPoints; ++i)
        {
            // Random velocity in [-0.5, 0.5] for each axis
            h_particles[i].vx = (rand() / float(RAND_MAX)) - 0.5f;
            h_particles[i].vy = (rand() / float(RAND_MAX)) - 0.5f;
            h_particles[i].vz = (rand() / float(RAND_MAX)) - 0.5f;

            // Random age between 0 and 5 seconds
            h_particles[i].maxAge = 2.0f + (rand() / float(RAND_MAX)) * 3.0f;
            h_particles[i].age = (rand() / float(RAND_MAX)) * h_particles[i].maxAge;

            // Random pressure [0, 1]
            h_particles[i].pressure = rand() / float(RAND_MAX);

            // Uniformly distributed UV coordinates in [0, 1]
            h_particles[i].u = rand() / float(RAND_MAX);
            h_particles[i].v = rand() / float(RAND_MAX);
        }

        // Upload to GPU
        err = cudaMemcpy(d_particles, h_particles.data(), particleDataSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to upload particle data: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_particles);
            m_d_particleData = nullptr;
            return false;
        }

        std::cout << "Initialized " << numPoints << " particles with random properties" << std::endl;
    }

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
    // Try to use runtime-compiled kernel if available, otherwise fall back to built-in kernel
    if (m_cudaKernel && m_cudaKernel->isValid())
    {
        // Use CUDA Driver API to launch runtime-compiled kernel
        void* kernelArgs[] = {
            &d_vertices,
            &m_d_particleData,
            &numPoints,
            &deltaTime,
            &m_d_fftData,
            &m_numFFTBins
        };

        int blockSize = 256;
        int numBlocks = (numPoints + blockSize - 1) / blockSize;

        CUresult cuErr = cuLaunchKernel(
            m_cudaKernel->getFunction(),
            numBlocks, 1, 1,    // grid dim
            blockSize, 1, 1,    // block dim
            0,                   // shared mem
            0,                   // stream
            kernelArgs,
            nullptr
        );

        if (cuErr != CUDA_SUCCESS)
        {
            const char* errorStr;
            cuGetErrorString(cuErr, &errorStr);
            std::cerr << "CUDA kernel launch failed (NVRTC): " << errorStr << std::endl;
            cudaGraphicsUnmapResources(1, &m_cudaVboResource, 0);
            return false;
        }

        // Synchronize
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA synchronize failed: " << cudaGetErrorString(err) << std::endl;
            cudaGraphicsUnmapResources(1, &m_cudaVboResource, 0);
            return false;
        }
    }
    else
    {
        // Fall back to built-in kernel
        err = launchAnimatePointsKernel(d_vertices, static_cast<ParticleData*>(m_d_particleData), numPoints, deltaTime,
                                         m_d_fftData, m_numFFTBins);
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            cudaGraphicsUnmapResources(1, &m_cudaVboResource, 0);
            return false;
        }
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

bool PointCloudRenderer::reloadKernel()
{
    if (!m_cudaKernel)
    {
        LOG(WARNING) << "Cannot reload kernel: kernel not initialized";
        return false;
    }

    LOG(INFO) << "Reloading CUDA kernel...";

    bool success = m_cudaKernel->reload();

    if (success)
    {
        LOG(INFO) << "CUDA kernel reloaded successfully!";
    }
    else
    {
        LOG(ERROR) << "Failed to reload CUDA kernel: " << m_cudaKernel->getLastError();
    }

    return success;
}
