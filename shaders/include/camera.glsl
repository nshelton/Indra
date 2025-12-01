// Camera
uniform vec3 uCameraPos;
uniform vec3 uCameraForward;
uniform vec3 uCameraRight;
uniform vec3 uCameraUp;
uniform float uTanHalfFov;
uniform float uAspect;
// ============================================================================
// Ray Generation
// ============================================================================

vec3 computeRayDirection(ivec2 pixel, ivec2 resolution) {
    vec2 uv = (vec2(pixel) + vec2(0.5)) / vec2(resolution);
    // uv += 0.5 * (uSeed4.xy - 0.5f) / vec2(resolution);
    float ndcX = uv.x * 2.0 - 1.0;
    float ndcY = 1.0 - uv.y * 2.0;

    return normalize(
        uCameraForward +
        ndcX * uTanHalfFov * uAspect * uCameraRight +
        -ndcY * uTanHalfFov * uCameraUp
    );
}
