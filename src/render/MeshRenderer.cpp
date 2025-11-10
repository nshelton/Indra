#include "MeshRenderer.h"

#include <iostream>

GLuint MeshRenderer::compileShader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        GLint len = 0;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::string log;
        log.resize(static_cast<size_t>(len));
        glGetShaderInfoLog(s, len, nullptr, log.data());
        std::cerr << "MeshRenderer shader compile failed: " << log << std::endl;
        glDeleteShader(s);
        return 0;
    }
    return s;
}

GLuint MeshRenderer::linkProgram(GLuint vs, GLuint fs)
{
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        GLint len = 0;
        glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        std::string log;
        log.resize(static_cast<size_t>(len));
        glGetProgramInfoLog(p, len, nullptr, log.data());
        std::cerr << "MeshRenderer link failed: " << log << std::endl;
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

bool MeshRenderer::init()
{
    // Simple flat color shader with vertex colors
    const char *vsSrc = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;

uniform mat4 uViewProjMat;
uniform mat4 uModelMat;

out vec4 vColor;

void main(){
    gl_Position = uViewProjMat * uModelMat * vec4(aPos, 1.0);
    vColor = aColor;
}
)";

    const char *fsSrc = R"(
#version 330 core
in vec4 vColor;
out vec4 FragColor;

void main(){
    FragColor = vColor;
}
)";

    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
    if (!vs)
        return false;
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
    if (!fs)
    {
        glDeleteShader(vs);
        return false;
    }
    m_program = linkProgram(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);
    if (!m_program)
        return false;

    m_uViewProjMat = glGetUniformLocation(m_program, "uViewProjMat");
    m_uModelMat = glGetUniformLocation(m_program, "uModelMat");

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glGenBuffers(1, &m_ebo);

    glBindVertexArray(m_vao);

    // Setup VBO
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

    // Position attribute (location 0)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void *)0);

    // Color attribute (location 1)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void *)(3 * sizeof(float)));

    // Setup EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return true;
}

void MeshRenderer::shutdown()
{
    if (m_vao)
        glDeleteVertexArrays(1, &m_vao);
    if (m_vbo)
        glDeleteBuffers(1, &m_vbo);
    if (m_ebo)
        glDeleteBuffers(1, &m_ebo);
    if (m_program)
        glDeleteProgram(m_program);
    m_vao = 0;
    m_vbo = 0;
    m_ebo = 0;
    m_program = 0;
    m_vertices.clear();
    m_indices.clear();
}

void MeshRenderer::clear()
{
    m_vertices.clear();
    m_indices.clear();
}

void MeshRenderer::renderMesh(const mesh& m, const Camera& camera)
{
    if (m.positions.empty() || m.indices.empty())
        return;

    // Build vertex buffer from mesh data
    m_vertices.clear();
    m_vertices.reserve(m.positions.size());

    for (size_t i = 0; i < m.positions.size(); ++i)
    {
        const vec3& pos = m.positions[i];
        color col = i < m.colors.size() ? m.colors[i] : color::white();
        m_vertices.push_back(GLVertex{pos.x, pos.y, pos.z, col.r, col.g, col.b, col.a});
    }

    // Get view-projection matrix from camera
    matrix4 viewProj = camera.getViewProjectionMatrix();

    // Get model matrix from mesh transform
    matrix4 model = m.transform.toMatrix();

    // Use shader program
    glUseProgram(m_program);

    // Set uniforms (GL_TRUE transposes from row-major to column-major for OpenGL)
    glUniformMatrix4fv(m_uViewProjMat, 1, GL_TRUE, &viewProj.m[0][0]);
    glUniformMatrix4fv(m_uModelMat, 1, GL_TRUE, &model.m[0][0]);

    // Bind VAO and update buffers
    glBindVertexArray(m_vao);

    // Update VBO with vertex data
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(m_vertices.size() * sizeof(GLVertex)),
        m_vertices.data(), GL_DYNAMIC_DRAW);

    // Update EBO with index data
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(m.indices.size() * sizeof(uint32_t)),
        m.indices.data(), GL_DYNAMIC_DRAW);

    // Enable depth testing for 3D rendering
    glEnable(GL_DEPTH_TEST);

    // Draw the mesh
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(m.indices.size()), GL_UNSIGNED_INT, 0);

    // Cleanup
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}
