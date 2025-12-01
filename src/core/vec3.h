#pragma once
#include <cmath>
#include <nlohmann/json.hpp>

struct vec3
{
    float x, y, z;

    vec3() : x(0), y(0), z(0) {}
    vec3(float value) : x(value), y(value), z(value) {}
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

// JSON serialization
inline void to_json(nlohmann::json& j, const vec3& v)
{
    j = nlohmann::json{{"x", v.x}, {"y", v.y}, {"z", v.z}};
}

inline void from_json(const nlohmann::json& j, vec3& v)
{
    v.x = j.value("x", 0.0f);
    v.y = j.value("y", 0.0f);
    v.z = j.value("z", 0.0f);
}