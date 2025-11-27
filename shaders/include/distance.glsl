// ============================================================================
// Distance Functions
// ============================================================================
// Fractal parameters

uniform vec3 uBoxDim (0.1, 5, 1.0);


float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}


vec2 mapScene(vec3 p)
{
    float d = sdBox(p, uBoxDim);


    return vec2(d, 1.0);
}