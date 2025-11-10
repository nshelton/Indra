#pragma once

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