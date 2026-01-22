#version 450 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in float inWaterMask;  // Water body indicator from tessellation shader

layout(location = 0) out vec4 FragColor;

layout(binding = 0) uniform mvpMatrix {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 color;
} uMVP;

// Water color for wireframe visualization (blue tint)
const vec4 waterColor = vec4(0.2, 0.5, 0.9, 1.0);

void main(void) {
    // Blend between terrain color and water color based on water mask
    vec4 finalColor = mix(uMVP.color, waterColor, inWaterMask);
    FragColor = finalColor;
}
