#version 330 core
in vec2 vTexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform vec2 uDirection;

void main() {
    vec2 texelSize = 1.0 / vec2(textureSize(uTexture, 0));
    vec3 result = vec3(0.0);

    // 9-tap Gaussian blur
    float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    result += texture(uTexture, vTexCoord).rgb * weights[0];
    for(int i = 1; i < 5; ++i) {
        vec2 offset = uDirection * texelSize * float(i);
        result += texture(uTexture, vTexCoord + offset).rgb * weights[i];
        result += texture(uTexture, vTexCoord - offset).rgb * weights[i];
    }

    FragColor = vec4(result, 1.0);
}