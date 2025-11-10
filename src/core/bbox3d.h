#pragma once

#include "vec3.h"

struct bbox3d
{
    vec3 min;
    vec3 max;

    bbox3d() : min(vec3()), max(vec3()) {}
    bbox3d(const vec3 &min, const vec3 &max) : min(min), max(max) {}

    vec3 center() const {
        return vec3(
            (min.x + max.x) * 0.5f,
            (min.y + max.y) * 0.5f,
            (min.z + max.z) * 0.5f
        );
    }

    vec3 size() const {
        return vec3(
            max.x - min.x,
            max.y - min.y,
            max.z - min.z
        );
    }

    bool contains(const vec3 &point) const {
        return (point.x >= min.x && point.x <= max.x) &&
               (point.y >= min.y && point.y <= max.y) &&
               (point.z >= min.z && point.z <= max.z);
    }

    void expandToInclude(const vec3 &point) {
        if (point.x < min.x) min.x = point.x;
        if (point.y < min.y) min.y = point.y;
        if (point.z < min.z) min.z = point.z;
        if (point.x > max.x) max.x = point.x;
        if (point.y > max.y) max.y = point.y;
        if (point.z > max.z) max.z = point.z;
    }

};