#include "PostProcessor.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

// Shader compilation helper
static GLuint compileShader(const char *vertSrc, const char *fragSrc)
{
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertSrc, nullptr);
    glCompileShader(vs);

    GLint success;
    glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char log[512];
        glGetShaderInfoLog(vs, 512, nullptr, log);
        std::cerr << "Vertex shader compilation failed:\n"
                  << log << std::endl;
    }

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragSrc, nullptr);
    glCompileShader(fs);

    glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char log[512];
        glGetShaderInfoLog(fs, 512, nullptr, log);
        std::cerr << "Fragment shader compilation failed:\n"
                  << log << std::endl;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        char log[512];
        glGetProgramInfoLog(program, 512, nullptr, log);
        std::cerr << "Shader linking failed:\n"
                  << log << std::endl;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

PostProcessor::PostProcessor()
{
}

PostProcessor::~PostProcessor()
{
    if (m_quadVAO)
        glDeleteVertexArrays(1, &m_quadVAO);
    if (m_quadVBO)
        glDeleteBuffers(1, &m_quadVBO);
    if (m_brightPassShader)
        glDeleteProgram(m_brightPassShader);
    if (m_blurShader)
        glDeleteProgram(m_blurShader);
    if (m_finalShader)
        glDeleteProgram(m_finalShader);
}

void PostProcessor::init(int width, int height)
{
    m_width = width;
    m_height = height;

    setupQuad();
    createShaders();

    // Create bloom framebuffers at half resolution for performance
    int bloomWidth = width / 2;
    int bloomHeight = height / 2;

    m_brightPass = std::make_unique<Framebuffer>();
    m_brightPass->createColorOnly(bloomWidth, bloomHeight, GL_RGBA16F);

    m_blurH = std::make_unique<Framebuffer>();
    m_blurH->createColorOnly(bloomWidth, bloomHeight, GL_RGBA16F);

    m_blurV = std::make_unique<Framebuffer>();
    m_blurV->createColorOnly(bloomWidth, bloomHeight, GL_RGBA16F);
}

void PostProcessor::resize(int width, int height)
{
    if (width == m_width && height == m_height)
        return;

    m_width = width;
    m_height = height;

    int bloomWidth = width / 2;
    int bloomHeight = height / 2;

    if (m_brightPass)
        m_brightPass->resize(bloomWidth, bloomHeight);
    if (m_blurH)
        m_blurH->resize(bloomWidth, bloomHeight);
    if (m_blurV)
        m_blurV->resize(bloomWidth, bloomHeight);
}

void PostProcessor::setupQuad()
{
    // Fullscreen quad: position (xy) and texcoord (zw)
    float quadVertices[] = {
        -1.0f, 1.0f, 0.0f, 1.0f,  // Top-left
        -1.0f, -1.0f, 0.0f, 0.0f, // Bottom-left
        1.0f, -1.0f, 1.0f, 0.0f,  // Bottom-right
        -1.0f, 1.0f, 0.0f, 1.0f,  // Top-left
        1.0f, -1.0f, 1.0f, 0.0f,  // Bottom-right
        1.0f, 1.0f, 1.0f, 1.0f    // Top-right
    };

    glGenVertexArrays(1, &m_quadVAO);
    glGenBuffers(1, &m_quadVBO);

    glBindVertexArray(m_quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));

    glBindVertexArray(0);
}

void PostProcessor::createShaders()
{
    // Simple fullscreen quad vertex shader (used by all passes)
    const char *quadVertShader = R"(
        #version 330 core
        layout(location = 0) in vec2 aPos;
        layout(location = 1) in vec2 aTexCoord;
        out vec2 vTexCoord;
        void main() {
            vTexCoord = aTexCoord;
            gl_Position = vec4(aPos, 0.0, 1.0);
        }
    )";

    // Bright pass: extract bright pixels above threshold
    const char *brightPassFragShader = R"(
        #version 330 core
        in vec2 vTexCoord;
        out vec4 FragColor;
        uniform sampler2D uHdrTexture;
        uniform float uThreshold;

        void main() {
            vec3 color = texture(uHdrTexture, vTexCoord).rgb;
            float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
            if (brightness > uThreshold) {
                FragColor = vec4(color, 1.0);
            } else {
                FragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
    )";

    // Gaussian blur (separable, horizontal/vertical)
    const char *blurFragShader = R"(
        #version 330 core
        in vec2 vTexCoord;
        out vec4 FragColor;
        uniform sampler2D uTexture;
        uniform vec2 uDirection;

        void main() {
            vec2 texelSize = 1.0 / vec2(textureSize(uTexture, 0));
            vec3 result = vec3(0.0);

            // 9-tap Gaussian blur
            float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

            result += texture(uTexture, vTexCoord).rgb * weights[0];
            for(int i = 1; i < 5; ++i) {
                vec2 offset = uDirection * texelSize * float(i);
                result += texture(uTexture, vTexCoord + offset).rgb * weights[i];
                result += texture(uTexture, vTexCoord - offset).rgb * weights[i];
            }

            FragColor = vec4(result, 1.0);
        }
    )";

    // Final composite: tone mapping + bloom
    const char *finalFragShader = R"(
        #version 330 core
        in vec2 vTexCoord;
        out vec4 FragColor;

        uniform sampler2D uHdrTexture;
        uniform sampler2D uBloomTexture;
        uniform float uExposure;
        uniform float uBloomStrength;
        uniform float uTime;

        // Reinhard tone mapping
        vec3 toneMapReinhard(vec3 hdr, float exposure) {
            vec3 mapped = hdr * exposure;
            mapped = mapped / (1.0 + mapped);
            return mapped;
        }

        void main() {
            vec3 hdrColor = texture(uHdrTexture, vTexCoord).rgb;
            vec3 bloomColor = texture(uBloomTexture, vTexCoord).rgb;

            // Add bloom
            vec3 color = hdrColor + bloomColor * uBloomStrength;

            // Tone mapping
            color = toneMapReinhard(color, uExposure);

            // Gamma correction
            color = pow(color, vec3(1.0 / 2.2));

            FragColor = vec4(color, 1.0);
        }
    )";

    m_brightPassShader = compileShader(quadVertShader, brightPassFragShader);
    m_blurShader = compileShader(quadVertShader, blurFragShader);
    m_finalShader = compileShader(quadVertShader, finalFragShader);
}

void PostProcessor::renderQuad()
{
    glBindVertexArray(m_quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void PostProcessor::process(GLuint hdrTexture)
{
    m_time += 0.016f; // Approximate frame time for noise animation

    glDisable(GL_DEPTH_TEST);

    // 1. Bright pass: extract bright pixels
    m_brightPass->bind();
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(m_brightPassShader);
    glUniform1i(glGetUniformLocation(m_brightPassShader, "uHdrTexture"), 0);
    glUniform1f(glGetUniformLocation(m_brightPassShader, "uThreshold"), 1.0f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, hdrTexture);
    renderQuad();

    // 2. Horizontal blur
    m_blurH->bind();
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(m_blurShader);
    glUniform1i(glGetUniformLocation(m_blurShader, "uTexture"), 0);
    glUniform2f(glGetUniformLocation(m_blurShader, "uDirection"), 1.0f, 0.0f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_brightPass->getColorTexture());
    renderQuad();

    // 3. Vertical blur
    m_blurV->bind();
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(m_blurShader);
    glUniform1i(glGetUniformLocation(m_blurShader, "uTexture"), 0);
    glUniform2f(glGetUniformLocation(m_blurShader, "uDirection"), 0.0f, 1.0f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_blurH->getColorTexture());
    renderQuad();

    // 4. Final composite to screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, m_width, m_height);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(m_finalShader);
    glUniform1i(glGetUniformLocation(m_finalShader, "uHdrTexture"), 0);
    glUniform1i(glGetUniformLocation(m_finalShader, "uBloomTexture"), 1);
    glUniform1f(glGetUniformLocation(m_finalShader, "uExposure"), m_exposure);
    glUniform1f(glGetUniformLocation(m_finalShader, "uBloomStrength"), m_bloomStrength);
    glUniform1f(glGetUniformLocation(m_finalShader, "uTime"), m_time);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, hdrTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_blurV->getColorTexture());

    renderQuad();

    glEnable(GL_DEPTH_TEST);
}
