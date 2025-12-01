#pragma once
#include <nlohmann/json.hpp>

struct vec2i {
    int x;
    int y;

    vec2i() : x(0), y(0) {}
    vec2i(int x, int y) : x(x), y(y) {}

    vec2i operator+(const vec2i &other) const { return vec2i(x + other.x, y + other.y); }
    vec2i operator-(const vec2i &other) const { return vec2i(x - other.x, y - other.y); }
    vec2i operator*(int scalar) const { return vec2i(x * scalar, y * scalar); }
    vec2i operator/(int scalar) const { return vec2i(x / scalar, y / scalar); }
};

// JSON serialization
inline void to_json(nlohmann::json& j, const vec2i& v)
{
    j = nlohmann::json{{"x", v.x}, {"y", v.y}};
}

inline void from_json(const nlohmann::json& j, vec2i& v)
{
    v.x = j.value("x", 0);
    v.y = j.value("y", 0);
}
