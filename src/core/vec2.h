#pragma once

#include <nlohmann/json.hpp>

struct vec2 {
    float x;
    float y;

    vec2() : x(0), y(0) {}
    vec2(float x, float y) : x(x), y(y) {}

    vec2 operator+(const vec2 &other) const { return vec2(x + other.x, y + other.y); }
    vec2 operator-(const vec2 &other) const { return vec2(x - other.x, y - other.y); }
    vec2 operator*(float scalar) const { return vec2(x * scalar, y * scalar); }
    vec2 operator/(float scalar) const { return vec2(x / scalar, y / scalar); }
};

// JSON serialization
inline void to_json(nlohmann::json& j, const vec2& v)
{
    j = nlohmann::json{{"x", v.x}, {"y", v.y}};
}

inline void from_json(const nlohmann::json& j, vec2& v)
{
    v.x = j.value("x", 0.0f);
    v.y = j.value("y", 0.0f);
}