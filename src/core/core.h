#pragma once

#include "./vec2.h"
#include "./vec3.h"
#include "./bbox3d.h"
#include "./quaternion.h"
#include "./transform.h"
#include "./matrix4.h"
#include "./color.h"
#include "./mesh.h"


template<typename T>
T clamp(T v, T min, T max)
{
    return (v < min) ? min : (v > max) ? max : v;
}