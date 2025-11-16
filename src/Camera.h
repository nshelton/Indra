#pragma once

#include <glad/glad.h>
#include <core/core.h>
#include <cmath>
#include <nlohmann/json.hpp>

struct Camera
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

    // Get camera parameters
    float getFov() const { return m_fov; }
    float getAspect() const { return m_aspect; }

    // Draw camera info GUI
    void drawGui();

    void toJson(nlohmann::json& j) const
    {
        j["position"] = { m_position.x, m_position.y, m_position.z };
        j["target"] = { m_target.x, m_target.y, m_target.z };
        j["up"] = { m_up.x, m_up.y, m_up.z };
        j["fov"] = m_fov;
        j["aspect"] = m_aspect;
        j["near"] = m_near;
        j["far"] = m_far;
    };

    void fromJson(const nlohmann::json& j) {
        auto pos = j.value("position", std::vector<float>{0.0f, 0.0f, 0.0f});
        if (pos.size() == 3) {
            m_position = vec3(pos[0], pos[1], pos[2]);
        }
        auto tgt = j.value("target", std::vector<float>{0.0f, 0.0f, -1.0f});
        if (tgt.size() == 3) {
            m_target = vec3(tgt[0], tgt[1], tgt[2]);
        }
        auto upv = j.value("up", std::vector<float>{0.0f, 1.0f, 0.0f});
        if (upv.size() == 3) {
            m_up = vec3(upv[0], upv[1], upv[2]);
        }
        m_fov = j.value("fov", 60.0f);
        m_aspect = j.value("aspect", 1.77778f);
        m_near = j.value("near", 0.1f);
        m_far = j.value("far", 1000.0f);
        m_viewDirty = true;
        m_projectionDirty = true;
        updateViewMatrix();
    }

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
