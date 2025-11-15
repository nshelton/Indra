#version 330 core
in vec4 vColor;
out vec4 FragColor;
void main(){
    // circular point sprite mask
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    FragColor =  vColor;
}
