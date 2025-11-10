#pragma once

#include <glad/glad.h>
#include <vector>
#include "core/core.h"
#include "Camera.h"

class MeshRenderer {
public:
    bool init();
    void shutdown();

    void clear();

    // Render a mesh with its transform
    void renderMesh(const mesh& m, const Camera& camera);

private:
    struct GLVertex {
        float x, y, z;       // Position
        float r, g, b, a;    // Color
    };

    GLuint m_program{0};
    GLuint m_vao{0};
    GLuint m_vbo{0};
    GLuint m_ebo{0};
    GLuint m_uViewProjMat{0};
    GLuint m_uModelMat{0};

    std::vector<GLVertex> m_vertices;
    std::vector<uint32_t> m_indices;

    static GLuint compileShader(GLenum type, const char* src);
    static GLuint linkProgram(GLuint vs, GLuint fs);
};
