#include "PostProcessor.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

// Shader compilation helper
static GLuint compileShader(const char* vertSrc, const char* fragSrc) {
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertSrc, nullptr);
    glCompileShader(vs);

    GLint success;
    glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(vs, 512, nullptr, log);
        std::cerr << "Vertex shader compilation failed:\n" << log << std::endl;
    }

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragSrc, nullptr);
    glCompileShader(fs);

    glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(fs, 512, nullptr, log);
        std::cerr << "Fragment shader compilation failed:\n" << log << std::endl;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(program, 512, nullptr, log);
        std::cerr << "Shader linking failed:\n" << log << std::endl;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

PostProcessor::PostProcessor() {
}

PostProcessor::~PostProcessor() {
    if (m_quadVAO) glDeleteVertexArrays(1, &m_quadVAO);
    if (m_quadVBO) glDeleteBuffers(1, &m_quadVBO);
    if (m_brightPassShader) glDeleteProgram(m_brightPassShader);
    if (m_blurShader) glDeleteProgram(m_blurShader);
    if (m_finalShader) glDeleteProgram(m_finalShader);
    if (m_blueNoiseTex) glDeleteTextures(1, &m_blueNoiseTex);
}

void PostProcessor::init(int width, int height) {
    m_width = width;
    m_height = height;

    setupQuad();
    createShaders();
    generateBlueNoise();

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

void PostProcessor::resize(int width, int height) {
    if (width == m_width && height == m_height) return;

    m_width = width;
    m_height = height;

    int bloomWidth = width / 2;
    int bloomHeight = height / 2;

    if (m_brightPass) m_brightPass->resize(bloomWidth, bloomHeight);
    if (m_blurH) m_blurH->resize(bloomWidth, bloomHeight);
    if (m_blurV) m_blurV->resize(bloomWidth, bloomHeight);
}

void PostProcessor::setupQuad() {
    // Fullscreen quad: position (xy) and texcoord (zw)
    float quadVertices[] = {
        -1.0f,  1.0f,  0.0f, 1.0f,  // Top-left
        -1.0f, -1.0f,  0.0f, 0.0f,  // Bottom-left
         1.0f, -1.0f,  1.0f, 0.0f,  // Bottom-right
        -1.0f,  1.0f,  0.0f, 1.0f,  // Top-left
         1.0f, -1.0f,  1.0f, 0.0f,  // Bottom-right
         1.0f,  1.0f,  1.0f, 1.0f   // Top-right
    };

    glGenVertexArrays(1, &m_quadVAO);
    glGenBuffers(1, &m_quadVBO);

    glBindVertexArray(m_quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    glBindVertexArray(0);
}

void PostProcessor::createShaders() {
    // Simple fullscreen quad vertex shader (used by all passes)
    const char* quadVertShader = R"(
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
    const char* brightPassFragShader = R"(
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
    const char* blurFragShader = R"(
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

    // Final composite: tone mapping + bloom + blue noise grain
    const char* finalFragShader = R"(
        #version 330 core
        in vec2 vTexCoord;
        out vec4 FragColor;

        uniform sampler2D uHdrTexture;
        uniform sampler2D uBloomTexture;
        uniform sampler2D uBlueNoise;
        uniform float uExposure;
        uniform float uBloomStrength;
        uniform float uGrainAmount;
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

            // Add blue noise film grain
            // Animate by offsetting UV coordinates over time
            vec2 noiseUV = vTexCoord * vec2(1920.0 / 64.0, 1080.0 / 64.0) + vec2(uTime / 0.1, uTime / 0.07);
            float blueNoise = texture(uBlueNoise, noiseUV).r;
            float grain = (blueNoise * 2.0 - 1.0);  // Remap from [0,1] to [-1,1]
            color += grain * uGrainAmount;

            // Gamma correction
            color = pow(color, vec3(1.0 / 2.2));

            FragColor = vec4(color, 1.0);
        }
    )";

    m_brightPassShader = compileShader(quadVertShader, brightPassFragShader);
    m_blurShader = compileShader(quadVertShader, blurFragShader);
    m_finalShader = compileShader(quadVertShader, finalFragShader);
}

void PostProcessor::renderQuad() {
    glBindVertexArray(m_quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void PostProcessor::generateBlueNoise() {
    // Generate a 64x64 blue noise texture using void-and-cluster algorithm
    const int size = 64;
    const int numPixels = size * size;
    std::vector<unsigned char> blueNoise(numPixels);

    // Simple blue noise generation using dithered white noise
    // This creates a spatially distributed noise pattern
    auto hash = [](int x, int y, int seed) -> float {
        int n = x + y * 57 + seed * 131;
        n = (n << 13) ^ n;
        return ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 2147483648.0f;
    };

    // Generate initial white noise
    std::vector<float> noise(numPixels);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int idx = y * size + x;
            noise[idx] = hash(x, y, 12345);
        }
    }

    // Apply multiple passes of spatial filtering to create blue noise characteristics
    for (int pass = 0; pass < 3; ++pass) {
        std::vector<float> filtered(numPixels);

        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                int idx = y * size + x;

                // Sample neighborhood
                float sum = 0.0f;
                int count = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;

                        int nx = (x + dx + size) % size;
                        int ny = (y + dy + size) % size;
                        int nidx = ny * size + nx;

                        sum += noise[nidx];
                        count++;
                    }
                }

                // High-pass filter (emphasize differences from neighbors)
                float avg = sum / count;
                filtered[idx] = noise[idx] - 0.3f * (avg - noise[idx]);
            }
        }

        noise = filtered;
    }

    // Normalize to [0, 255] range
    float minVal = *std::min_element(noise.begin(), noise.end());
    float maxVal = *std::max_element(noise.begin(), noise.end());
    float range = maxVal - minVal;

    for (int i = 0; i < numPixels; ++i) {
        float normalized = (noise[i] - minVal) / range;
        blueNoise[i] = static_cast<unsigned char>(normalized * 255.0f);
    }

    // Create OpenGL texture
    glGenTextures(1, &m_blueNoiseTex);
    glBindTexture(GL_TEXTURE_2D, m_blueNoiseTex);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, size, size, 0, GL_RED, GL_UNSIGNED_BYTE, blueNoise.data());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void PostProcessor::process(GLuint hdrTexture) {
    m_time += 0.016f;  // Approximate frame time for noise animation

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
    glUniform1i(glGetUniformLocation(m_finalShader, "uBlueNoise"), 2);
    glUniform1f(glGetUniformLocation(m_finalShader, "uExposure"), m_exposure);
    glUniform1f(glGetUniformLocation(m_finalShader, "uBloomStrength"), m_bloomStrength);
    glUniform1f(glGetUniformLocation(m_finalShader, "uGrainAmount"), m_grainAmount);
    glUniform1f(glGetUniformLocation(m_finalShader, "uTime"), m_time);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, hdrTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_blurV->getColorTexture());
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, m_blueNoiseTex);

    renderQuad();

    glEnable(GL_DEPTH_TEST);
}
