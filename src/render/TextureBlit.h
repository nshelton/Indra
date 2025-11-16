#pragma once

#include <glad/glad.h>

/// @brief Static utility for blitting textures to framebuffers
/// Manages a fullscreen quad and simple passthrough shader for efficient texture copying
class TextureBlit
{
public:
    /// @brief Initialize the blit utility - must be called once after OpenGL context is ready
    static void init();

    /// @brief Shutdown and cleanup resources
    static void shutdown();

    /// @brief Blit a texture to the currently bound framebuffer
    /// @param texture The texture to blit
    static void blit(GLuint texture);

private:
    TextureBlit() = delete;  // Static class, no instances

    static GLuint s_quadVAO;
    static GLuint s_quadVBO;
    static GLuint s_shader;
    static bool s_initialized;
};
