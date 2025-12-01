// --- Fractal Parameters ---
uniform float uIterations (1, 20, 8);
uniform float uScale (-3.0, 3.0, 2.0);
uniform float uMinRadius2 (0.0, 1.0, 0.25);
uniform float uFixedRadius2 (0.0, 4.0, 1.0);
uniform float uOffsetX (-1.0, 1.0, 1.0);
uniform float uOffsetY (-1.0, 1.0, 1.0);
uniform float uOffsetZ (-1.0, 1.0, 1.0);

float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

vec2 mbox(vec3 p)
{
    vec3 z = p;
    float dr = 1.0; // derivative, starts at 1
    float r2;

    for (int i = 0; i < uIterations; i++) {
        // --- Folds ---
        // 1. Box Fold (mirroring)
        z = clamp(z, -1.0, 1.0) * 2.0 - z; // Fold to a [-1, 1] box

        // --- Scale ---
        z *= uScale;

        // 2. Spherical Fold
        r2 = dot(z,z);
        if (r2 < uMinRadius2) {
            float temp = uFixedRadius2 / uMinRadius2;
            z *= temp;
            dr *= temp;
        } else if (r2 < uFixedRadius2) {
            float temp = uFixedRadius2 / r2;
            z *= temp;
            dr *= temp;
        }

        // --- Translate ---
        // Add the original point (Mandelbrot set style)
        z += p;
        dr = dr * abs(uScale) + 1.0;
    }

    // Final distance estimation (to a sphere)
    float d = (length(z) - 2.0) / dr;
    return vec2(d, 1.0);
}

vec2 mapScene(vec3 p)
{
    float d = sdBox(p, vec3(uOffsetX, uOffsetY, uOffsetZ));
    return vec2(d, 1);
}

vec3 estimateNormal(vec3 p)
{
    const float h = 0.01;
    vec2 e = vec2(0.01, -1.0);
    return normalize(
        e.xyy * mapScene(p + e.xyy * h).x +
        e.yyx * mapScene(p + e.yyx * h).x +
        e.yxy * mapScene(p + e.yxy * h).x +
        e.xxx * mapScene(p + e.xxx * h).x
    );
}