#include "TextureBlit.h"
#include <glog/logging.h>

// Static member initialization
GLuint TextureBlit::s_quadVAO = 0;
GLuint TextureBlit::s_quadVBO = 0;
GLuint TextureBlit::s_shader = 0;
bool TextureBlit::s_initialized = false;

void TextureBlit::init()
{
    if (s_initialized)
    {
        LOG(WARNING) << "TextureBlit already initialized";
        return;
    }

    // Create fullscreen quad
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    glGenVertexArrays(1, &s_quadVAO);
    glGenBuffers(1, &s_quadVBO);
    glBindVertexArray(s_quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, s_quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);

    // Create simple passthrough shader
    const char* vertexShader = R"(
        #version 460 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoords;
        out vec2 TexCoords;
        void main()
        {
            TexCoords = aTexCoords;
            gl_Position = vec4(aPos, 0.0, 1.0);
        }
    )";

    const char* fragmentShader = R"(
        #version 460 core
        out vec4 FragColor;
        in vec2 TexCoords;
        uniform sampler2D uTexture;
        void main()
        {
            FragColor = texture(uTexture, TexCoords);
        }
    )";

    // Compile vertex shader
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertexShader, nullptr);
    glCompileShader(vs);

    // Compile fragment shader
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragmentShader, nullptr);
    glCompileShader(fs);

    // Link program
    s_shader = glCreateProgram();
    glAttachShader(s_shader, vs);
    glAttachShader(s_shader, fs);
    glLinkProgram(s_shader);

    glDeleteShader(vs);
    glDeleteShader(fs);

    s_initialized = true;
    LOG(INFO) << "TextureBlit initialized";
}

void TextureBlit::shutdown()
{
    if (!s_initialized)
        return;

    if (s_quadVAO != 0)
    {
        glDeleteVertexArrays(1, &s_quadVAO);
        s_quadVAO = 0;
    }
    if (s_quadVBO != 0)
    {
        glDeleteBuffers(1, &s_quadVBO);
        s_quadVBO = 0;
    }
    if (s_shader != 0)
    {
        glDeleteProgram(s_shader);
        s_shader = 0;
    }

    s_initialized = false;
    LOG(INFO) << "TextureBlit shutdown";
}

void TextureBlit::blit(GLuint texture)
{
    if (!s_initialized)
    {
        LOG(ERROR) << "TextureBlit::blit() called before init()";
        return;
    }

    // Enable additive blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);  // Additive blending: src + dst

    glUseProgram(s_shader);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glUniform1i(glGetUniformLocation(s_shader, "uTexture"), 0);

    glBindVertexArray(s_quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    // Disable blending after blit
    glDisable(GL_BLEND);
}
