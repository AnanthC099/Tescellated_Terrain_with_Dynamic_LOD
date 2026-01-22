#version 450 core
#extension GL_ARB_separate_shader_objects : enable

// OpenGL Superbible 7th Edition style tessellation evaluation shader
// Samples displacement texture and applies height displacement

layout(quads, fractional_odd_spacing, ccw) in;

layout(binding = 0) uniform mvpMatrix {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 color;
} uMVP;

// Displacement texture sampler (heightmap)
layout(binding = 1) uniform sampler2D tex_displacement;

// Displacement depth uniform
layout(binding = 2) uniform DisplacementParams {
    float dmap_depth;
    float padding1;
    float padding2;
    float padding3;
} dispParams;

layout(location = 0) in TCS_OUT {
    vec2 tc;
} tes_in[];

layout(location = 0) out TES_OUT {
    vec2 tc;
    vec3 world_coord;
    vec3 eye_coord;
} tes_out;

void main(void) {
    // Bilinear interpolation of texture coordinates
    vec2 tc1 = mix(tes_in[0].tc, tes_in[1].tc, gl_TessCoord.x);
    vec2 tc2 = mix(tes_in[2].tc, tes_in[3].tc, gl_TessCoord.x);
    vec2 tc = mix(tc1, tc2, gl_TessCoord.y);

    // Bilinear interpolation of position
    vec4 p1 = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);
    vec4 p2 = mix(gl_in[2].gl_Position, gl_in[3].gl_Position, gl_TessCoord.x);
    vec4 p = mix(p1, p2, gl_TessCoord.y);

    // Sample displacement map and apply to Y coordinate
    // The displacement map stores height values in the red channel
    p.y += texture(tex_displacement, tc).r * dispParams.dmap_depth;

    // Calculate world coordinate
    vec4 world_pos = uMVP.modelMatrix * p;
    tes_out.world_coord = world_pos.xyz;

    // Calculate eye coordinate (view space)
    vec4 eye_pos = uMVP.viewMatrix * world_pos;
    tes_out.eye_coord = eye_pos.xyz;

    // Output texture coordinate
    tes_out.tc = tc;

    // Transform to clip space
    gl_Position = uMVP.projectionMatrix * eye_pos;
}
