#include "ComputeShader.h"
#include <glog/logging.h>

ComputeShader::ComputeShader()
    : Shader()
{
    // Create timer query objects for GPU profiling
    glGenQueries(2, m_timerQuery);
}

ComputeShader::~ComputeShader()
{
    if (m_timerQuery[0] != 0 || m_timerQuery[1] != 0)
    {
        glDeleteQueries(2, m_timerQuery);
    }
}

void ComputeShader::setFallbackSource(const std::string& computeSource)
{
    m_fallbackComputeSource = computeSource;
}

bool ComputeShader::loadFromFile(const std::string& computePath)
{
    m_computePath = computePath;

    // Read shader source
    std::string computeSource = readFile(computePath);

    if (computeSource.empty())
    {
        // If file can't be loaded and we have fallback, use it
        if (!m_fallbackComputeSource.empty())
        {
            LOG(WARNING) << "Compute shader file not found, using fallback embedded shader";
            return loadFromSource(m_fallbackComputeSource.c_str());
        }
        return false;
    }

    // Store modification time
    m_computeModTime = getFileModTime(computePath);

    // Compile and link
    return loadFromSource(computeSource.c_str());
}

bool ComputeShader::loadFromSource(const char* computeSource)
{
    // Create shader
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);

    // Compile compute shader
    if (!compileShader(computeShader, computeSource, "Compute shader"))
    {
        glDeleteShader(computeShader);
        m_isValid = false;
        return false;
    }

    // Create and link program
    GLuint newProgram = glCreateProgram();
    glAttachShader(newProgram, computeShader);

    if (!linkProgram(newProgram))
    {
        glDeleteProgram(newProgram);
        glDeleteShader(computeShader);
        m_isValid = false;
        return false;
    }

    // Clean up shader (it's linked into the program now)
    glDeleteShader(computeShader);

    // Delete old program if it exists
    if (m_program != 0)
    {
        glDeleteProgram(m_program);
    }

    m_program = newProgram;
    m_isValid = true;
    m_lastError.clear();

    LOG(INFO) << "Compute shader program created successfully (ID: " << m_program << ")";
    return true;
}

bool ComputeShader::reload()
{
    if (m_computePath.empty())
    {
        m_lastError = "Cannot reload: no file path set";
        LOG(WARNING) << m_lastError;
        return false;
    }

    LOG(INFO) << "Reloading compute shader from " << m_computePath;

    // Store old program in case reload fails
    GLuint oldProgram = m_program;
    bool oldValid = m_isValid;

    // Try to reload
    m_program = 0;  // Temporarily clear so loadFromFile creates a new one
    bool success = loadFromFile(m_computePath);

    if (!success)
    {
        // Restore old program on failure
        if (oldProgram != 0 && !m_program)
        {
            m_program = oldProgram;
            m_isValid = oldValid;
        }
        LOG(ERROR) << "Compute shader reload failed, keeping previous version";
        return false;
    }

    // Delete old program on success
    if (oldProgram != 0 && oldProgram != m_program)
    {
        glDeleteProgram(oldProgram);
    }

    LOG(INFO) << "Compute shader reload successful!";

    m_shaderRevisionId++;
    return true;
}

bool ComputeShader::filesModified() const
{
    if (m_computePath.empty())
    {
        return false;
    }

    auto currentComputeTime = getFileModTime(m_computePath);
    return currentComputeTime > m_computeModTime;
}

void ComputeShader::dispatch(GLuint groupsX, GLuint groupsY, GLuint groupsZ)
{
    if (!m_isValid || m_program == 0)
        return;

    // Use double-buffered timer queries to avoid GPU stalls
    int queryIndex = m_currentQuery;
    int prevQueryIndex = 1 - m_currentQuery;

    // Start timing this frame
    glBeginQuery(GL_TIME_ELAPSED, m_timerQuery[queryIndex]);
    glDispatchCompute(groupsX, groupsY, groupsZ);
    glEndQuery(GL_TIME_ELAPSED);

    // Retrieve timing from previous frame (if available)
    if (!m_firstFrame)
    {
        GLuint64 elapsed = 0;
        glGetQueryObjectui64v(m_timerQuery[prevQueryIndex], GL_QUERY_RESULT, &elapsed);

        // Convert nanoseconds to milliseconds and smooth with exponential moving average
        float timeMs = elapsed / 1000000.0f;
        m_executionTimeMs = m_executionTimeMs * 0.9f + timeMs * 0.1f;
    }
    else
    {
        m_firstFrame = false;
    }

    // Swap buffers for next frame
    m_currentQuery = prevQueryIndex;
}

void ComputeShader::getWorkGroupSize(GLint& sizeX, GLint& sizeY, GLint& sizeZ) const
{
    if (m_isValid && m_program != 0)
    {
        GLint workGroupSize[3];
        glGetProgramiv(m_program, GL_COMPUTE_WORK_GROUP_SIZE, workGroupSize);
        sizeX = workGroupSize[0];
        sizeY = workGroupSize[1];
        sizeZ = workGroupSize[2];
    }
    else
    {
        sizeX = sizeY = sizeZ = 0;
    }
}
