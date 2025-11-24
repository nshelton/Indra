// ============================================================================
// Distance Functions
// ============================================================================
// Fractal parameters

uniform float uSphereRad (0.1, 5, 1.0);

vec2 mapScene(vec3 p)
{
    float sphereDist = length(p) - uSphereRad;
    return vec2(sphereDist, 1.0);
}