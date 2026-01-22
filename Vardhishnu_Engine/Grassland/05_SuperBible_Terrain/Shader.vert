#version 450 core
#extension GL_ARB_separate_shader_objects : enable

// OpenGL Superbible 7th Edition style vertex shader
// Generates patch vertices using instance ID

layout(binding = 0) uniform mvpMatrix {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 color;
} uMVP;

layout(location = 0) out VS_OUT {
    vec2 tc;
} vs_out;

void main(void) {
    // Define a single patch quad (unit square in xz plane, centered at origin)
    const vec4 vertices[4] = vec4[4](
        vec4(-0.5, 0.0, -0.5, 1.0),
        vec4( 0.5, 0.0, -0.5, 1.0),
        vec4(-0.5, 0.0,  0.5, 1.0),
        vec4( 0.5, 0.0,  0.5, 1.0)
    );

    // Calculate grid position from instance ID (64x64 grid)
    int x = gl_InstanceIndex & 63;
    int y = gl_InstanceIndex >> 6;
    vec2 offs = vec2(x, y);

    // Calculate texture coordinates for this vertex
    vs_out.tc = (vertices[gl_VertexIndex].xz + offs + vec2(0.5)) / 64.0;

    // Calculate world position (offset patch to create grid centered at origin)
    gl_Position = vertices[gl_VertexIndex] + vec4(float(x - 32), 0.0, float(y - 32), 0.0);
}
