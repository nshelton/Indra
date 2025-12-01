Fractal raymarcher

- audio analysis with cuda fft and miniaudio (currently not actually hooked up)
- hierarchial raymarcher
- TAA with reprojection
filter needs some work

tonemapper (also needs some work)

TODO
- move tonemapper and postFX shaders to use the shader reloader system
- serialize Inteeraciton module (movement parameters)
     - maybe make "serializable module" thing that serializes "state" from every module. 
     - Module has some kind of abstract parameter object? 


put some header in the shader to autogenerate the shaderstate,
it gets parsed on compile and then generates parameters on the fly
this would be pretty sick, the gui would update when the shader header changes

Animaiton ideas
1. simple: presets, tween between presets
2. timeline? 

either way need to serialize presets and support tweening through states

audioreactivity ?


Okay I'm lazy - procedural animation
- some kind of seed + procedural movement
- find good seeds and save them
- fragile though

----

## Next steps
1. improve controls
2. hierarchy working agian
3. path tracing

Check out serializaiton.deserializaiton

clean up postproc - use same shader layout

in Raymarchersimple - clean up the way shaders are handled
- audit uniforms
- generic gui setup
- generic init/reload/destroy

uniforms should work right on a failed compilation
