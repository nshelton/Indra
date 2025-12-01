// --- Fractal Parameters ---
uniform float uIterations (1, 20, 8);
uniform float uScale (-3.0, 3.0, 2.0);
uniform float uMinRadius2 (0.0, 1.0, 0.25);
uniform float uFixedRadius2 (0.0, 4.0, 1.0);
uniform float urX (-1.0, 1.0, 1.0);
uniform float urY (-1.0, 1.0, 1.0);
uniform float urZ (-1.0, 1.0, 1.0);

uniform float uSurfaceEpsilon (0.1, 10, 1.0);


float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

// Rotate point p around axis by angle (Rodrigues' rotation formula)
vec3 rotate(vec3 p, vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return vec3(
        oc * axis.x * axis.x + c,
        oc * axis.x * axis.y - axis.z * s,
        oc * axis.z * axis.x + axis.y * s
    ) * p.x +
    vec3(
        oc * axis.x * axis.y + axis.z * s,
        oc * axis.y * axis.y + c,
        oc * axis.y * axis.z - axis.x * s
    ) * p.y +
    vec3(
        oc * axis.z * axis.x - axis.y * s,
        oc * axis.y * axis.z + axis.x * s,
        oc * axis.z * axis.z + c
    ) * p.z;
}

// Simplified rotation functions for common axes
vec3 rotateX(vec3 p, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    return vec3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

vec3 rotateY(vec3 p, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

vec3 rotateZ(vec3 p, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    return vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
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
 

float mapIq( in vec3 p )
{
   float d = sdBox(p,vec3(1.0));
   vec3 res = vec3( d, 1.0, 0.0);

   float s = 1.0;
   for( int m=0; m<4; m++ )
   {
      vec3 a = mod( p*s, 2.0 )-1.0;
      s *= 3.0;
      vec3 r = abs(1.0 - 3.0*abs(a));

      float da = max(r.x,r.y);
      float db = max(r.y,r.z);
      float dc = max(r.z,r.x);
      float c = (min(da,min(db,dc))-1.0)/s;

      if( c>d )
      {
          d = c;
          res = vec3( d, 0.2*da*db*dc, (1.0+float(m))/4.0 );
       }
      p += vec3(urX, urY, urZ);
      p = rotate(p, vec3(0.1, 0.2, 0.3), 1);
   }

   return res.x;
}



vec2 mapScene(vec3 p)
{
    // float d = sdBox(p, );
    float d = mapIq(p);
    return vec2(d, 1);
}

vec3 estimateNormal(vec3 p)
{
    const float h = 0.001;
    vec2 e = vec2(0.001, -1.0);
    return normalize(
        e.xyy * mapScene(p + e.xyy * h).x +
        e.yyx * mapScene(p + e.yyx * h).x +
        e.yxy * mapScene(p + e.yxy * h).x +
        e.xxx * mapScene(p + e.xxx * h).x
    );
}