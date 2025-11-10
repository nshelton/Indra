#include "Camera.h"
#include <glog/logging.h>

Camera::Camera()
    : m_position(0, 50, 200)
    , m_target(0, 0, 0)
    , m_up(0, 1, 0)
    , m_fov(45.0f)
    , m_aspect(16.0f / 9.0f)
    , m_near(0.1f)
    , m_far(1000.0f)
    , m_viewDirty(true)
    , m_projectionDirty(true)
{
    updateViewMatrix();
}

Camera::Camera(const vec3& position, const vec3& target, const vec3& up)
    : m_position(position)
    , m_target(target)
    , m_up(up)
    , m_fov(45.0f)
    , m_aspect(16.0f / 9.0f)
    , m_near(0.1f)
    , m_far(1000.0f)
    , m_viewDirty(true)
    , m_projectionDirty(true)
{
    updateViewMatrix();
}

void Camera::setPosition(const vec3& position)
{
    m_position = position;
    m_viewDirty = true;
}

void Camera::setTarget(const vec3& target)
{
    m_target = target;
    m_viewDirty = true;
}

void Camera::setUp(const vec3& up)
{
    m_up = up;
    m_viewDirty = true;
}

void Camera::setPerspective(float fov, float aspect, float near, float far)
{
    m_fov = fov;
    m_aspect = aspect;
    m_near = near;
    m_far = far;
    m_projectionDirty = true;
}

void Camera::updateViewMatrix()
{
    // Calculate camera basis vectors
    m_forward = (m_target - m_position).normalized();
    m_right = m_forward.cross(m_up).normalized();
    m_up = m_right.cross(m_forward).normalized();

    // Build view matrix (lookAt matrix)
    m_viewMatrix = matrix4();
    m_viewMatrix.m[0][0] = m_right.x;
    m_viewMatrix.m[0][1] = m_right.y;
    m_viewMatrix.m[0][2] = m_right.z;
    m_viewMatrix.m[0][3] = -m_right.dot(m_position);

    m_viewMatrix.m[1][0] = m_up.x;
    m_viewMatrix.m[1][1] = m_up.y;
    m_viewMatrix.m[1][2] = m_up.z;
    m_viewMatrix.m[1][3] = -m_up.dot(m_position);

    m_viewMatrix.m[2][0] = -m_forward.x;
    m_viewMatrix.m[2][1] = -m_forward.y;
    m_viewMatrix.m[2][2] = -m_forward.z;
    m_viewMatrix.m[2][3] = m_forward.dot(m_position);

    m_viewMatrix.m[3][0] = 0;
    m_viewMatrix.m[3][1] = 0;
    m_viewMatrix.m[3][2] = 0;
    m_viewMatrix.m[3][3] = 1;

    m_viewDirty = false;
}

matrix4 Camera::getViewMatrix() const
{
    if (m_viewDirty)
    {
        const_cast<Camera*>(this)->updateViewMatrix();
    }
    return m_viewMatrix;
}

matrix4 Camera::getProjectionMatrix() const
{
    if (m_projectionDirty)
    {
        float fovRadians = m_fov * 3.14159265359f / 180.0f;
        float tanHalfFov = std::tan(fovRadians / 2.0f);

        matrix4 proj;
        proj.m[0][0] = 1.0f / (m_aspect * tanHalfFov);
        proj.m[0][1] = 0;
        proj.m[0][2] = 0;
        proj.m[0][3] = 0;

        proj.m[1][0] = 0;
        proj.m[1][1] = 1.0f / tanHalfFov;
        proj.m[1][2] = 0;
        proj.m[1][3] = 0;

        proj.m[2][0] = 0;
        proj.m[2][1] = 0;
        proj.m[2][2] = -(m_far + m_near) / (m_far - m_near);
        proj.m[2][3] = -(2.0f * m_far * m_near) / (m_far - m_near);

        proj.m[3][0] = 0;
        proj.m[3][1] = 0;
        proj.m[3][2] = -1;
        proj.m[3][3] = 0;

        const_cast<Camera*>(this)->m_projectionMatrix = proj;
        const_cast<Camera*>(this)->m_projectionDirty = false;
    }
    return m_projectionMatrix;
}

matrix4 Camera::getViewProjectionMatrix() const
{
    return getProjectionMatrix() * getViewMatrix();
}