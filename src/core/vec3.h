#pragma once
#include <cmath>

struct vec3
{
    float x, y, z;

    vec3() : x(0), y(0), z(0) {}
    vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    vec3 operator+(const vec3 &other) const {
        return vec3(x + other.x, y + other.y, z + other.z);
    }
    vec3 operator-(const vec3 &other) const {
        return vec3(x - other.x, y - other.y, z - other.z);
    }
    vec3 operator*(float scalar) const {
        return vec3(x * scalar, y * scalar, z * scalar);
    }

    vec3 operator*(const vec3 &other) const {
        return vec3(x * other.x, y * other.y, z * other.z);
    }

    vec3 operator/(float scalar) const {
        return vec3(x / scalar, y / scalar, z / scalar);
    }

    vec3 operator/(const vec3 &other) const {
        return vec3(x / other.x, y / other.y, z / other.z);
    }

    float dot(const vec3 &other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    vec3 cross(const vec3 &other) const {
        return vec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    vec3 normalized() const {
        float len = length();
        if (len == 0) return vec3(0, 0, 0);
        return (*this) / len;
    }

};