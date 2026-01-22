#version 450 core
#extension GL_ARB_separate_shader_objects : enable

// OpenGL Superbible 7th Edition style fragment shader
// Samples color texture and applies atmospheric fog

layout(location = 0) out vec4 FragColor;

layout(binding = 0) uniform mvpMatrix {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 color;
} uMVP;

// Color texture sampler (landscape texture)
layout(binding = 3) uniform sampler2D tex_color;

// Fog parameters
layout(binding = 4) uniform FogParams {
    int enable_fog;
    float fog_density;
    float fog_start;
    float fog_end;
    vec4 fog_color;
} fogParams;

layout(location = 0) in TES_OUT {
    vec2 tc;
    vec3 world_coord;
    vec3 eye_coord;
} fs_in;

// Fog calculation function based on OpenGL Superbible
vec4 fog(vec4 c) {
    float z = length(fs_in.eye_coord);

    // Height-based fog density
    float de = 0.025 * smoothstep(0.0, 6.0, 10.0 - fs_in.world_coord.y);
    float di = 0.045 * (smoothstep(0.0, 40.0, 20.0 - fs_in.world_coord.y));

    // Exponential extinction and inscattering
    float extinction = exp(-z * de);
    float inscattering = exp(-z * di);

    // Blend between original color and fog color
    return c * extinction + fogParams.fog_color * (1.0 - inscattering);
}

void main(void) {
    // Sample the landscape color texture
    vec4 landscape = texture(tex_color, fs_in.tc);

    // Apply fog if enabled
    if (fogParams.enable_fog != 0) {
        FragColor = fog(landscape);
    } else {
        FragColor = landscape;
    }
}
