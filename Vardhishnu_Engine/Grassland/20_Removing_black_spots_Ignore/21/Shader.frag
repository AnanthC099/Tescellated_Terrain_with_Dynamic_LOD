#version 450 core
#extension GL_ARB_separate_shader_objects : enable

// Inputs from tessellation evaluation shader
layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inTangent;
layout(location = 3) in float inBitangentSign;

layout(location = 0) out vec4 FragColor;

// Uniform buffer for matrices and optional color
layout(binding = 0) uniform mvpMatrix {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 color;
} uMVP;

void main(void) {
    // Use CUDA-computed normal for basic shading
    // Safeguard against NaN/zero vectors - use default up normal if invalid
    float normalLen = length(inNormal);
    // NaN check: normalLen != normalLen is true only for NaN
    vec3 normal = (normalLen > 0.0001 && normalLen == normalLen)
                  ? inNormal / normalLen
                  : vec3(0.0, 1.0, 0.0);

    // Simple directional light from upper-right
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));

    // Basic diffuse lighting
    float diffuse = max(dot(normal, lightDir), 0.0);

    // Ambient term to prevent completely black areas
    float ambient = 0.2;

    // Combine lighting
    float lighting = ambient + diffuse * 0.8;

    // Final safeguard: check for NaN in lighting (catches any upstream NaN propagation)
    // NaN check: NaN != NaN is always true
    if (lighting != lighting || isinf(lighting)) {
        lighting = ambient;  // Fall back to ambient-only lighting
    }

    // Output lit white color for terrain visualization
    // The normal-based shading shows terrain shape
    FragColor = vec4(vec3(lighting), 1.0);
}
