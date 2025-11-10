#include <gtest/gtest.h>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include "../src/core/core.h"
#include <type_traits>

inline void dump_mem_usage()
{
    FILE* f = fopen("/proc/self/statm", "rt");
    if (!f) return;
    char   str[300];
    size_t n = fread(str, 1, 200, f);
    str[n]   = 0;
    printf("MEM: %s\n", str);
    fclose(f);
}


TEST(kdtree, Basic)
{
    vec3 p1(1, 2, 3);
    quaternion rot = quaternion::fromAxisAngle(vec3(0, 1, 0), 3.14f / 4.0f);

    matrix4 t1 = matrix4::TRS(p1, rot, vec3(1, 1, 1));
    vec3 p2 = t1.multiplyPoint(vec3(1, 0, 0));

    matrix4 t1_inv = t1.inverse();
    vec3 p1_recovered = t1_inv.multiplyPoint(p2);

    EXPECT_NEAR(p1_recovered.x, 1.0f, 0.001f);
    EXPECT_NEAR(p1_recovered.y, 2.0f, 0.001f);
    EXPECT_NEAR(p1_recovered.z, 3.0f, 0.001f);


}