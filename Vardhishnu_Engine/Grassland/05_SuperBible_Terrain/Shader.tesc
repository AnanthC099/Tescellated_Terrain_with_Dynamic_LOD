#version 450 core
#extension GL_ARB_separate_shader_objects : enable

// OpenGL Superbible 7th Edition style tessellation control shader
// Screen-space adaptive tessellation based on projected edge lengths

layout(vertices = 4) out;

layout(binding = 0) uniform mvpMatrix {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 color;
} uMVP;

layout(location = 0) in VS_OUT {
    vec2 tc;
} tcs_in[];

layout(location = 0) out TCS_OUT {
    vec2 tc;
} tcs_out[];

void main(void) {
    // Pass through texture coordinates
    if (gl_InvocationID == 0) {
        // Calculate MVP matrix
        mat4 mvp = uMVP.projectionMatrix * uMVP.viewMatrix * uMVP.modelMatrix;

        // Project all 4 vertices to clip space
        vec4 p0 = mvp * gl_in[0].gl_Position;
        vec4 p1 = mvp * gl_in[1].gl_Position;
        vec4 p2 = mvp * gl_in[2].gl_Position;
        vec4 p3 = mvp * gl_in[3].gl_Position;

        // Back-face / frustum culling: if any vertex is behind the camera, cull the patch
        // (check if all vertices are behind the near plane)
        if (p0.z <= 0.0 && p1.z <= 0.0 && p2.z <= 0.0 && p3.z <= 0.0) {
            // Cull the patch by setting tessellation levels to 0
            gl_TessLevelOuter[0] = 0.0;
            gl_TessLevelOuter[1] = 0.0;
            gl_TessLevelOuter[2] = 0.0;
            gl_TessLevelOuter[3] = 0.0;
            gl_TessLevelInner[0] = 0.0;
            gl_TessLevelInner[1] = 0.0;
        } else {
            // Convert to normalized device coordinates (perspective divide)
            p0 /= p0.w;
            p1 /= p1.w;
            p2 /= p2.w;
            p3 /= p3.w;

            // Calculate edge lengths in screen space
            // Scale factor: larger value = more tessellation
            float scale = 16.0;

            // Edge 0: p0 -> p2 (left edge in our quad layout)
            float l0 = length(p2.xy - p0.xy) * scale + 1.0;
            // Edge 1: p0 -> p1 (bottom edge)
            float l1 = length(p1.xy - p0.xy) * scale + 1.0;
            // Edge 2: p1 -> p3 (right edge)
            float l2 = length(p3.xy - p1.xy) * scale + 1.0;
            // Edge 3: p2 -> p3 (top edge)
            float l3 = length(p3.xy - p2.xy) * scale + 1.0;

            // Clamp tessellation levels to valid range
            float maxTess = 64.0;
            l0 = clamp(l0, 1.0, maxTess);
            l1 = clamp(l1, 1.0, maxTess);
            l2 = clamp(l2, 1.0, maxTess);
            l3 = clamp(l3, 1.0, maxTess);

            // Set outer tessellation levels
            // For quads: outer[0]=left, outer[1]=bottom, outer[2]=right, outer[3]=top
            gl_TessLevelOuter[0] = l0;
            gl_TessLevelOuter[1] = l1;
            gl_TessLevelOuter[2] = l2;
            gl_TessLevelOuter[3] = l3;

            // Set inner tessellation levels (average of opposite edges)
            gl_TessLevelInner[0] = min(l1, l3);
            gl_TessLevelInner[1] = min(l0, l2);
        }
    }

    // Pass through vertex position and texture coordinate
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    tcs_out[gl_InvocationID].tc = tcs_in[gl_InvocationID].tc;
}
