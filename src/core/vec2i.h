#pragma once

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
