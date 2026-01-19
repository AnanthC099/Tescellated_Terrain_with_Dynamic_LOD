#version 450 core
#extension GL_ARB_separate_shader_objects : enable

// Tessellation Evaluation Shader for Flat Terrain
// Evaluates tessellated vertex positions

layout(quads, equal_spacing, ccw) in;

layout(binding = 0) uniform mvpMatrix {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 color;
} uMVP;

layout(location = 0) in vec4 inPosition[];

void main() {
    // Get tessellation coordinates
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;

    // Bilinear interpolation of patch corners for quad
    vec4 p0 = inPosition[0];
    vec4 p1 = inPosition[1];
    vec4 p2 = inPosition[2];
    vec4 p3 = inPosition[3];

    // Interpolate position across the quad patch
    vec4 pos1 = mix(p0, p1, u);
    vec4 pos2 = mix(p3, p2, u);
    vec4 position = mix(pos1, pos2, v);

    // Flat terrain in x-z plane: y = 0
    position.y = 0.0;
    position.w = 1.0;

    // Transform to clip space
    gl_Position = uMVP.projectionMatrix * uMVP.viewMatrix * uMVP.modelMatrix * position;
}
