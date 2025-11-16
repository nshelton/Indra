#include "Raymarcher.h"
#include <iostream>
#include <glog/logging.h>

bool Raymarcher::init()
{
    m_shaderProgram = std::make_unique<ShaderProgram>();
    return true;
}

void Raymarcher::shutdown()
{
    // Shader program is automatically cleaned up by unique_ptr
    m_shaderProgram.reset();
}

/// @brief Draw the 3D lines using the camera's view-projection matrix
/// @param camera The camera to use for rendering
/// @param scene The scene model to render
void Raymarcher::draw(const Camera &camera, const Scene &scene)
{
    if (!m_shaderProgram || !m_shaderProgram->isValid())
        return;

    m_shaderProgram->use();

    // Get uniform locations
    GLint uViewProjMat = m_shaderProgram->getUniformLocation("uViewProjMat");
    GLint uPointSizePx = m_shaderProgram->getUniformLocation("uPointSizePx");

    // Get view-projection matrix from camera
    matrix4 viewProj = camera.getViewProjectionMatrix();
    glUniformMatrix4fv(uViewProjMat, 1, GL_TRUE, &viewProj.m[0][0]);

    // TODO: render raymarching here

    // upload scene.sphereposition and scene.sphereRadius to shader uniforms
    // upload background color as well

}

bool Raymarcher::reloadShaders()
{
    if (m_shaderProgram)
    {
        return m_shaderProgram->reload();
    }
    return false;
}