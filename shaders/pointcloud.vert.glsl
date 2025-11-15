#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;
uniform mat4 uViewProjMat;
uniform float uPointSizePx;
out vec4 vColor;
void main(){
    gl_Position = uViewProjMat * vec4(aPos, 1.0);
    gl_PointSize = uPointSizePx;
    vColor = aColor;
}
