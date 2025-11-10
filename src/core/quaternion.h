#pragma once

#include "./vec3.h"

struct quaternion
{
    float w, x, y, z;

    quaternion() : w(1), x(0), y(0), z(0) {}
    quaternion(float w, float x, float y, float z) : w(w), x(x), y(y), z(z) {}

    quaternion operator*(const quaternion &other) const {
        return quaternion(
            w * other.w - x * other.x - y * other.y - z * other.z,
            w * other.x + x * other.w + y * other.z - z * other.y,
            w * other.y - x * other.z + y * other.w + z * other.x,
            w * other.z + x * other.y - y * other.x + z * other.w
        );
    }
    quaternion normalized() const {
        float len = std::sqrt(w * w + x * x + y * y + z * z);
        return quaternion(w / len, x / len, y / len, z / len);
    }

    quaternion rotateAxis(const vec3 &axis, float angleRad) const {
        float halfAngle = angleRad * 0.5f;
        float s = std::sin(halfAngle);
        quaternion rot(std::cos(halfAngle), axis.x * s, axis.y * s, axis.z * s);
        return (rot * (*this)).normalized();
    }

    static quaternion fromAxisAngle(const vec3 &axis, float angleRad) {
        float halfAngle = angleRad * 0.5f;
        float s = std::sin(halfAngle);
        return quaternion(std::cos(halfAngle), axis.x * s, axis.y * s, axis.z * s).normalized();
    }

};