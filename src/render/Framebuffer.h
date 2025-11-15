#pragma once

#include <glad/glad.h>
#include <stdexcept>

class Framebuffer {
public:
    Framebuffer();
    ~Framebuffer();

    // Disable copy
    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;

    // Create HDR float framebuffer with color and depth
    void createHDR(int width, int height);

    // Create simple color-only framebuffer (for bloom passes)
    void createColorOnly(int width, int height, GLenum internalFormat = GL_RGBA16F);

    void bind() const;
    void unbind() const;

    GLuint getColorTexture() const { return m_colorTex; }
    GLuint getFBO() const { return m_fbo; }

    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }

    void resize(int width, int height);

private:
    void cleanup();

    GLuint m_fbo = 0;
    GLuint m_colorTex = 0;
    GLuint m_depthRbo = 0;

    int m_width = 0;
    int m_height = 0;
    bool m_hasDepth = false;
    GLenum m_internalFormat = GL_RGBA16F;
};
