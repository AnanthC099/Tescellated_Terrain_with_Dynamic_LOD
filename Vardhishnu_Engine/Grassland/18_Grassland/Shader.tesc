#version 450 core
#extension GL_ARB_separate_shader_objects : enable

layout(vertices = 4) out;

layout(binding = 0) uniform mvpMatrix {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 color;
} uMVP;

// CUDA-computed tessellation factors (one per patch)
layout(std430, binding = 6) readonly buffer TessFactorsBuffer {
    float tessFactors[];
};

// Frustum culling visibility mask (1 = visible, 0 = culled)
layout(std430, binding = 7) readonly buffer VisibilityBuffer {
    int visibilityMask[];
};

layout(location = 0) in vec4 inPosition[];
layout(location = 0) out vec4 outPosition[];

// Grid size must match PATCH_GRID_SIZE in VK.cu
const int PATCH_GRID_SIZE = 64;

void main() {
    outPosition[gl_InvocationID] = inPosition[gl_InvocationID];

    if (gl_InvocationID == 0) {
        // Calculate patch index from primitive ID
        int patchIdx = gl_PrimitiveID;

        // Check visibility from CUDA frustum culling
        if (patchIdx < PATCH_GRID_SIZE * PATCH_GRID_SIZE && visibilityMask[patchIdx] == 0) {
            // Patch is culled - set tessellation to 0 to discard
            gl_TessLevelOuter[0] = 0.0;
            gl_TessLevelOuter[1] = 0.0;
            gl_TessLevelOuter[2] = 0.0;
            gl_TessLevelOuter[3] = 0.0;
            gl_TessLevelInner[0] = 0.0;
            gl_TessLevelInner[1] = 0.0;
        } else {
            // Use CUDA-computed tessellation factor
            float tessFactor = 8.0; // Default fallback
            if (patchIdx < PATCH_GRID_SIZE * PATCH_GRID_SIZE) {
                tessFactor = tessFactors[patchIdx];
            }

            // Apply tessellation factor to all edges
            gl_TessLevelOuter[0] = tessFactor;
            gl_TessLevelOuter[1] = tessFactor;
            gl_TessLevelOuter[2] = tessFactor;
            gl_TessLevelOuter[3] = tessFactor;
            gl_TessLevelInner[0] = tessFactor;
            gl_TessLevelInner[1] = tessFactor;
        }
    }
}
