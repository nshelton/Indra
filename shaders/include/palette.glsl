// Color palette utilities
// Based on Inigo Quilez's palette function

vec3 pal(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d)
{
    return a + b * cos(6.28318 * (c * t + d));
}

vec3 palette(float t)
{
    float _Palette = 2.0;
    vec2 _ColorParam = vec2(0.01, 0.0);
    t = (t * _ColorParam.x) + _ColorParam.y;
    vec3 color = pal(t, vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 0.5), vec3(1.0, 1.0, 1.0), vec3(0.0, 0.33, 0.67));
    if (_Palette > (1.0)) color = pal(t, vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 0.5), vec3(1.0, 1.0, 1.0), vec3(0.0, 0.10, 0.20));
    if (_Palette > (2.0)) color = pal(t, vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 0.5), vec3(1.0, 1.0, 1.0), vec3(0.3, 0.20, 0.20));
    if (_Palette > (3.0)) color = pal(t, vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 0.5), vec3(1.0, 1.0, 0.5), vec3(0.8, 0.90, 0.30));
    if (_Palette > (4.0)) color = pal(t, vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 0.5), vec3(1.0, 0.7, 0.4), vec3(0.0, 0.15, 0.20));
    if (_Palette > (5.0)) color = pal(t, vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 0.5), vec3(2.0, 1.0, 0.0), vec3(0.5, 0.20, 0.25));
    if (_Palette > (5.0)) color = pal(t, vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 0.5), vec3(2.0, 1.0, 0.0), vec3(0.5, 0.20, 0.25));
    if (_Palette > (6.0))
        color = vec3(1.0);

    return color;
}


vec3 turbo(float x)
{
    x = clamp(x, 0.0, 1.0);

    return vec3(
        0.13572137988692035 + x*(4.597363719627905  + x*(-42.327689751912274 + x*( 130.58871182451415 + x*(-150.56663492057857 + x*58.137453451135656)))),
        0.09140261235958302 + x*(2.1856173378635675 + x*( 4.805204796477784  + x*(-14.019450960349728 + x*(  4.210856355081685 + x*2.7747311504638876)))),
        0.10667330048674728 + x*(12.592563476453211 + x*(-60.10967551582361  + x*( 109.07449945380961 + x*( -88.50658250648611 + x*26.818260967511673))))
    );
}