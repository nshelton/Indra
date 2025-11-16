// ============================================================================
// Distance Functions
// ============================================================================

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



vec2 tglad(vec3 z0)
{
    // z0 = modc(z0, 2.0);

    float mr = 0.25, mxr = 1.0;
    vec4 scale = vec4(-3.12, -3.12, -3.12, 3.12), p0 = u_paramA.xyzz;
    vec4 z = vec4(z0, 1.0);
    float orbit = 0;

    for (int n = 0; n < _LEVELS; n++)
    {
        vec3 start = z.xyz;

        z.xyz = clamp(z.xyz, -u_paramB * 3.0, u_paramB * 3.0) * 2.0 - z.xyz;
        z *= scale / clamp(dot(z.xyz, z.xyz), mr, mxr);
        z += p0;
        orbit += length(start - z.xyz);

    }

    float dS = (length(max(abs(z.xyz) - vec3(1.2, 49.0, 1.4), 0.0)) - 0.06) / z.w;
    return vec2(dS, orbit * 2);
}