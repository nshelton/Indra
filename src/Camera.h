#pragma once

#include <glad/glad.h>
#include <core/core.h>
#include <cmath>

class Camera
{
public:
    Camera();
    Camera(const vec3& position, const vec3& target, const vec3& up = vec3(0, 1, 0));

    void setSize(float width, float height) {
        m_aspect = width / height;
        m_projectionDirty = true;
    }

    // Set camera parameters
    void setPosition(const vec3& position);
    void setTarget(const vec3& target);
    void setUp(const vec3& up);
    void setPerspective(float fov, float aspect, float near, float far);

    // Get matrices
    matrix4 getViewMatrix() const;
    matrix4 getProjectionMatrix() const;
    matrix4 getViewProjectionMatrix() const;

    // Get camera vectors
    vec3 getPosition() const { return m_position; }
    vec3 getTarget() const { return m_target; }
    vec3 getForward() const { return m_forward; }
    vec3 getRight() const { return m_right; }
    vec3 getUp() const { return m_up; }

private:
    void updateViewMatrix();

    // Camera position and orientation
    vec3 m_position;
    vec3 m_target;
    vec3 m_up;

    // Derived vectors
    vec3 m_forward;
    vec3 m_right;

    // Projection parameters
    float m_fov;        // Field of view in degrees
    float m_aspect;     // Aspect ratio (width/height)
    float m_near;       // Near plane distance
    float m_far;        // Far plane distance

    // Cached matrices
    matrix4 m_viewMatrix;
    matrix4 m_projectionMatrix;
    bool m_viewDirty;
    bool m_projectionDirty;
};
