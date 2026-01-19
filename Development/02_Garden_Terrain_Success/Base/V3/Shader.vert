#version 450 core
#extension GL_ARB_separate_shader_objects : enable

// Vertex Shader for Tessellated Terrain
// Passes vertex positions to tessellation control shader

layout(location = 0) in vec4 vPosition;

layout(binding = 0) uniform mvpMatrix {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 color;
} uMVP;

layout(location = 0) out vec4 outPosition;

void main(void) {
    // Pass vertex position to tessellation control shader
    outPosition = vPosition;
}
