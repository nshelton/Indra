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
