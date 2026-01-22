#version 450 core
#extension GL_ARB_separate_shader_objects : enable

layout(vertices = 4) out;

layout(binding = 0) uniform mvpMatrix {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 color;
} uMVP;

layout(location = 0) in vec4 inPosition[];
layout(location = 0) out vec4 outPosition[];

const float minTessLevel = 1.0;
const float maxTessLevel = 16.0;
const float minDistance = 50.0;
const float maxDistance = 600.0;

float calculateTessLevel(vec4 p0, vec4 p1) {
    vec4 worldMid = (p0 + p1) * 0.5;
    vec4 viewMid = uMVP.viewMatrix * uMVP.modelMatrix * worldMid;

    float distance = length(viewMid.xyz);

    float t = clamp((distance - minDistance) / (maxDistance - minDistance), 0.0, 1.0);

    return mix(maxTessLevel, minTessLevel, t);
}

void main() {
    outPosition[gl_InvocationID] = inPosition[gl_InvocationID];

    if (gl_InvocationID == 0) {
        float e0 = calculateTessLevel(inPosition[0], inPosition[1]);
        float e1 = calculateTessLevel(inPosition[1], inPosition[2]);
        float e2 = calculateTessLevel(inPosition[2], inPosition[3]);
        float e3 = calculateTessLevel(inPosition[3], inPosition[0]);

        gl_TessLevelOuter[0] = e3;  // left edge (3→0)
        gl_TessLevelOuter[1] = e0;  // bottom edge (0→1)
        gl_TessLevelOuter[2] = e1;  // right edge (1→2)
        gl_TessLevelOuter[3] = e2;  // top edge (2→3)

        gl_TessLevelInner[0] = (e0 + e2) * 0.5;  // u-direction (horizontal)
        gl_TessLevelInner[1] = (e1 + e3) * 0.5;  // v-direction (vertical)
    }
}
