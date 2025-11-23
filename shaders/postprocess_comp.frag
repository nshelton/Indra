#version 330 core
in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uHdrTexture;
uniform sampler2D uBloomTexture;
uniform float uExposure;
uniform float uBloomStrength;
uniform float uTime;

// Reinhard tone mapping
vec3 toneMapReinhard(vec3 hdr, float exposure)
{
    vec3 mapped = hdr * exposure;
    mapped = mapped / (1.0 + mapped);
    return mapped;
}

void main()
{
    vec3 hdrColor = texture(uHdrTexture, vTexCoord).rgb;
    vec3 bloomColor = texture(uBloomTexture, vTexCoord).rgb;

    // Add bloom
    vec3 color = hdrColor + bloomColor * uBloomStrength;

    // Tone mapping
    color = toneMapReinhard(color, uExposure);

    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
}