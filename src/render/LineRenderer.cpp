#include "LineRenderer.h"

#include <iostream>

GLuint LineRenderer::compileShader(GLenum type, const char *src)
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
        std::cerr << "LineRenderer shader compile failed: " << log << std::endl;
        glDeleteShader(s);
        return 0;
    }
    return s;
}

GLuint LineRenderer::linkProgram(GLuint vs, GLuint fs)
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
        std::cerr << "LineRenderer link failed: " << log << std::endl;
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

bool LineRenderer::init()
{
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
uniform int uIsPointPass;
void main(){
    if (uIsPointPass == 1) {
        // circular point sprite mask
        vec2 d = gl_PointCoord - vec2(0.5);
        if (dot(d, d) > 0.25) discard;
    }
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
    m_uPointSizePx = glGetUniformLocation(m_program, "uPointSizePx");
    m_uIsPointPass = glGetUniformLocation(m_program, "uIsPointPass");

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
    return true;
}

void LineRenderer::shutdown()
{
    if (m_vao)
        glDeleteVertexArrays(1, &m_vao);
    if (m_vbo)
        glDeleteBuffers(1, &m_vbo);
    if (m_program)
        glDeleteProgram(m_program);
    m_vao = 0;
    m_vbo = 0;
    m_program = 0;
    m_vertices.clear();
}

void LineRenderer::clear()
{
    m_vertices.clear();
    m_points.clear();
}

/// @brief Add a 3D line to the render list
/// @param p1 start point in 3D world space
/// @param p2 end point in 3D world space
/// @param c color
void LineRenderer::addLine(vec3 p1, vec3 p2, color c)
{
    m_vertices.push_back(GLVertex{p1.x, p1.y, p1.z, c.r, c.g, c.b, c.a});
    m_vertices.push_back(GLVertex{p2.x, p2.y, p2.z, c.r, c.g, c.b, c.a});
}

void LineRenderer::addPoint(vec3 p, color c)
{
    m_points.push_back(GLVertex{p.x, p.y, p.z, c.r, c.g, c.b, c.a});
}

void LineRenderer::addGrid(vec3 center, float size, int divisions, color gridColor, color centerColor)
{
    float halfSize = size * 0.5f;
    float step = size / static_cast<float>(divisions);

    // Draw lines parallel to X axis (varying Z)
    for (int i = 0; i <= divisions; ++i)
    {
        float z = center.z - halfSize + (i * step);
        vec3 start(center.x - halfSize, center.y, z);
        vec3 end(center.x + halfSize, center.y, z);

        // Use centerColor for the center line (Z=0 in grid space)
        color lineColor = (i == divisions / 2) ? centerColor : gridColor;
        addLine(start, end, lineColor);
    }

    // Draw lines parallel to Z axis (varying X)
    for (int i = 0; i <= divisions; ++i)
    {
        float x = center.x - halfSize + (i * step);
        vec3 start(x, center.y, center.z - halfSize);
        vec3 end(x, center.y, center.z + halfSize);

        // Use centerColor for the center line (X=0 in grid space)
        color lineColor = (i == divisions / 2) ? centerColor : gridColor;
        addLine(start, end, lineColor);
    }
}

/// @brief Draw the 3D lines using the camera's view-projection matrix
/// @param camera The camera to use for rendering
void LineRenderer::draw(const Camera& camera)
{
    if (m_vertices.empty() && m_points.empty())
        return;

    glUseProgram(m_program);

    // Get view-projection matrix from camera
    matrix4 viewProj = camera.getViewProjectionMatrix();
    glUniformMatrix4fv(m_uViewProjMat, 1, GL_TRUE, &viewProj.m[0][0]);
    glUniform1i(m_uIsPointPass, 0);

    // Draw lines
    if (!m_vertices.empty())
    {
        glBindVertexArray(m_vao);
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glBufferData(GL_ARRAY_BUFFER,
            static_cast<GLsizeiptr>(m_vertices.size() * sizeof(GLVertex)),
            m_vertices.data(), GL_DYNAMIC_DRAW);

        if (m_lineWidth > 0.0f)
            glLineWidth(m_lineWidth);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(m_vertices.size()));
    }

    // Draw points as filled circles using point sprites
    if (!m_points.empty())
    {
        float pointDiameterPx = m_pointDiameterPx;
        if (pointDiameterPx <= 0.0f)
        {
            pointDiameterPx = 5.0f; // Default point size
        }

        glEnable(GL_PROGRAM_POINT_SIZE);
        glUniform1f(m_uPointSizePx, pointDiameterPx);
        glUniform1i(m_uIsPointPass, 1);

        glBindVertexArray(m_vao);
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glBufferData(GL_ARRAY_BUFFER,
            static_cast<GLsizeiptr>(m_points.size() * sizeof(GLVertex)),
            m_points.data(), GL_DYNAMIC_DRAW);

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(m_points.size()));

        glUniform1i(m_uIsPointPass, 0);
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // clear points for next frame
    m_points.clear();
}
