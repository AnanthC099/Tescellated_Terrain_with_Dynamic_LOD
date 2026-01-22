#version 450 core
#extension GL_ARB_separate_shader_objects : enable

// Tessellation Control Shader for Dynamic LOD
// Adjusts tessellation levels based on distance from camera

layout(vertices = 4) out;

layout(binding = 0) uniform mvpMatrix {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 color;
} uMVP;

layout(location = 0) in vec4 inPosition[];
layout(location = 0) out vec4 outPosition[];

// LOD parameters
const float minTessLevel = 1.0;
const float maxTessLevel = 64.0;
const float minDistance = 0.5;   // Distance for maximum tessellation
const float maxDistance = 10.0;  // Distance for minimum tessellation

float calculateTessLevel(vec4 p0, vec4 p1) {
    // Transform patch edge midpoint to view space
    vec4 worldMid = (p0 + p1) * 0.5;
    vec4 viewMid = uMVP.viewMatrix * uMVP.modelMatrix * worldMid;

    // Distance from camera (camera is at origin in view space)
    float distance = length(viewMid.xyz);

    // Linearly interpolate tessellation level based on distance
    float t = clamp((distance - minDistance) / (maxDistance - minDistance), 0.0, 1.0);

    // Higher tessellation when closer, lower when farther
    return mix(maxTessLevel, minTessLevel, t);
}

void main() {
    // Pass through vertex position
    outPosition[gl_InvocationID] = inPosition[gl_InvocationID];

    // Only first invocation sets tessellation levels (optimization)
    if (gl_InvocationID == 0) {
        // Calculate tessellation level for each edge based on distance
        float e0 = calculateTessLevel(inPosition[0], inPosition[1]);
        float e1 = calculateTessLevel(inPosition[1], inPosition[2]);
        float e2 = calculateTessLevel(inPosition[2], inPosition[3]);
        float e3 = calculateTessLevel(inPosition[3], inPosition[0]);

        // Outer tessellation levels (edges)
        gl_TessLevelOuter[0] = e0;
        gl_TessLevelOuter[1] = e1;
        gl_TessLevelOuter[2] = e2;
        gl_TessLevelOuter[3] = e3;

        // Inner tessellation levels (average of edges)
        gl_TessLevelInner[0] = (e1 + e3) * 0.5;
        gl_TessLevelInner[1] = (e0 + e2) * 0.5;
    }
}
