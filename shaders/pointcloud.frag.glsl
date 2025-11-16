#version 330 core
in vec4 vColor;
out vec4 FragColor;
void main(){
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.75) discard;
    FragColor =  vColor;
}
