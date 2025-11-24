// ============================================================================
// Distance Functions
// ============================================================================
// Fractal parameters
uniform vec3 u_paramA;
uniform vec3 u_paramB;
uniform vec3 u_paramC;
uniform vec3 u_paramD;
uniform float _LEVELS;
uniform float uSphereRad (0.1, 5, 1.0);

vec2 pseudo_knightyan(vec3 p)
{
    vec3 CSize = u_paramA.xyz;
    float DEfactor = 1.;
    float orbit = 0;
    for (int i = 0; i < 6; i++)
    {
        p = 2. * clamp(p, -CSize, CSize) - p;
        float r2 = dot(p, p);
        float k = max(u_paramC.y / r2, 1.);
        p *= k;
        DEfactor *= k;
        orbit += abs(p.x);
    }

    float rxy = length(p.xy);
    return vec2(max(rxy - u_paramB.x, abs(rxy * p.z) / length(p)) / DEfactor, orbit);
}

vec2 tglad_variant(vec3 z0)
{
    // z0 = modc(z0, 2.0);
    float mr = u_paramD.x;
    float mxr = u_paramD.y;

    vec4 scale = u_paramD.zzzz;
    vec4 p0 = u_paramA.xyzz;
    vec4 z = vec4(z0, 1.0);
    float orbit = 0;
    for (int n = 0; n < int(_LEVELS); n++   )
    {
        vec4 start = z;
        z.xyz = clamp(z.xyz, -u_paramB.x * 10.0, u_paramB.x * 10.0) * 2.0 - z.xyz;
        z *= scale / clamp(dot(z.xyz, z.xyz), mr, mxr);
        z += p0;
        orbit += length(start - z);
    }
    float dS = (length(max(abs(z.xyz) - u_paramC.xyz, 0.0)) - 0) / z.w;
    return vec2(dS, orbit);
}

// distance function from Hartverdrahtet
// ( http://www.pouet.net/prod.php?which=59086 )
vec2 hartverdrahtet(vec3 f)
{
    vec3 cs = u_paramA.xyz;
    float fs = u_paramC.x;
    vec3 fc = vec3(0);
    float fu = 1.;
    float fd = .3;
    float orbit = 0.0;
    fc.z = -.38;
 
    float v = 1.;
    for (int i = 0; i < _LEVELS; i++)
    {
        vec3 start = f;

        f = 2. * clamp(f, -cs, cs) - f;
        float c = max(fs / dot(f, f), 1.);
        f *= c;
        v *= c;
        f += fc;

        orbit += length(start - f);
    }
    float z = length(f.xy) - fu;
    float d = fd * max(z, abs(length(f.xy) * f.z) / sqrt(dot(f, f))) / abs(v);

    return vec2(d, orbit);
}

// Quaternion multiplication
// http://mathworld.wolfram.com/Quaternion.html
vec4 qmul(vec4 q1, vec4 q2)
{
    return vec4(
        q2.xyz * q1.w + q1.xyz * q2.w + cross(q1.xyz, q2.xyz),
        q1.w * q2.w - dot(q1.xyz, q2.xyz)
    );
}
// Vector rotation with a quaternion
// http://mathworld.wolfram.com/Quaternion.html
vec3 rotate_vector(vec3 v, vec4 r)
{
    vec4 r_c = r * vec4(-1, -1, -1, 1);
    return qmul(r, qmul(vec4(v, 0), r_c)).xyz;
}

vec4 quatFromAA(vec3 axis, float angle)
{
    vec4 q;
    float s = sin(angle / 2.0);
    q.x = axis.x * s;
    q.y = axis.y * s;
    q.z = axis.z * s;
    q.w = cos(angle / 2.0);
    return q;

}

vec2 tglad(vec3 z0)
{
    // z0 = modc(z0, 2.0);

    float limit = u_paramB.x * 10;

    float mr = u_paramC.x , mxr = u_paramC.y + mr;
    vec4 scale = vec4(-6, -6, -6, 6)* u_paramB.y ;
    vec4 p0 = u_paramA.xyzz;
    vec4 z = vec4(z0, 1.0);
    float orbit = 0;

    vec4 q_rot = quatFromAA(normalize(vec3(1, 1, 1)), u_paramD.x);

    for (int n = 0; n < _LEVELS; n++)
    {
        vec3 start = z.xyz;
        z.xyz = rotate_vector(z.xyz, q_rot);

        // boxFold
        z.xyz = clamp(z.xyz, -u_paramB * 20.0, u_paramB * 20.0) * 2.0 - z.xyz;
        z *= scale / clamp(dot(z.xyz, z.xyz), mr, mxr);
        z += p0;
        orbit += length(start - z.xyz);

    }

    float dS = (length(max(abs(z.xyz) - vec3(1.2, 9.0, 1.), 0.0)) - 1) / z.w ;
    return vec2(dS, orbit * 2);
}


void sphereFold(inout vec3 z, inout float dz)
{

    float fixedRadius2 = u_paramA.x * 5;
    float minRadius2 = u_paramA.y * 5;

    float r2 = dot(z, z);
    if (r2 < minRadius2)
    {
		// linear inner scaling
        float temp = (fixedRadius2 / minRadius2);
        z *= temp;
        dz *= temp;
    }
    else if (r2 < fixedRadius2)
    {
		// this is the actual sphere inversion
        float temp = (fixedRadius2 / r2);
        z *= temp;
        dz *= temp;
    }
}

void boxFold(inout vec3 z, inout float dz)
{
    z = clamp(z, -u_paramA.z * 10.0f, u_paramA.z* 10.0f) * 2.0 - z;
}


//----------------------------------------------------------------------------------------
vec2 MBOX(vec3 z)
{
    z.y *= -1.0;
    vec3 offset = z/1;
    float dr = 0.5;

    float Scale = u_paramC.x * 10.0f - (z.y - u_paramD.x * 10.0f) *u_paramD.z* 10.0f;
    float iter = 0.0;

    float orbit = 0;
    vec3 z_prime = z;

    for (int n = 0; n < _LEVELS; n++)
    {
        boxFold(z, dr); // Reflect
        sphereFold(z, dr); // Sphere Inversion

        z = Scale * z + offset; // Scale & Translate
        dr = dr * abs(Scale) + 1.1;
        iter++;
        orbit += length(z_prime - z);
        z_prime = z;

        if (abs(dr) > 10000.)
            break;
    }
    float r = length(z);

    return vec2(r / abs(dr), orbit);
}


vec2 fractal(vec3 p)
{
    vec3  z  = p;
    float dr = 1.0;
    float MANDELBOX_SCALE        = u_paramA.x * 5;
    float MANDELBOX_MIN_RADIUS   = u_paramA.y;
    float MANDELBOX_FIXED_RADIUS = u_paramA.z * 5;

    float minR2   = MANDELBOX_MIN_RADIUS   * MANDELBOX_MIN_RADIUS;
    float fixedR2 = MANDELBOX_FIXED_RADIUS * MANDELBOX_FIXED_RADIUS;

    for (int i = 0; i < _LEVELS; ++i)
    {
        // --- Box fold: reflect into [-1,1]^3 "box"
        z = clamp(z, -1.0, 1.0) * 2.0 - z;

        // --- Sphere fold: push/pull points into a shell
        float r2 = dot(z, z);

        // fold small radii up to fixed radius
        if (r2 < fixedR2) {
            float t = fixedR2 / max(r2, 1e-6);  // avoid div by zero
            z  *= t;
            dr *= t;
        }
        // fold medium radii into min radius sphere
        else if (r2 < minR2) {
            float t = minR2 / r2;
            z  *= t;
            dr *= t;
        }

        // --- Scale and re-center
        z  = z * MANDELBOX_SCALE + p;
        dr = dr * abs(MANDELBOX_SCALE) + 1.0;
    }

    float r = length(z);
    return vec2(r / abs(dr), 0.0);  // distance estimate
}


vec2 mapScene(vec3 p)
{
    float sphereDist = length(p) - uSphereRad;
    return vec2(sphereDist, 1.0);
}