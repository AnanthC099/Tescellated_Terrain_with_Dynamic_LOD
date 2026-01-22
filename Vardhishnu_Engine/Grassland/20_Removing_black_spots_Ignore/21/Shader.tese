#version 450 core
#extension GL_ARB_separate_shader_objects : enable

layout(quads, equal_spacing, ccw) in;

layout(binding = 0) uniform mvpMatrix {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 color;
} uMVP;

// ============================================================================
// CUDA-GENERATED TERRAIN DATA
// Complete terrain pre-computed by CUDA using cuRAND and parallel streams
// All noise generation happens in CUDA - shader only samples the results
// ============================================================================

#define HEIGHTMAP_SIZE 2048
#define TERRAIN_SIZE 800.0

// Storage buffer for CUDA-generated heightmap (binding 3)
layout(std430, binding = 3) readonly buffer HeightmapBuffer {
    float heights[HEIGHTMAP_SIZE * HEIGHTMAP_SIZE];
} cudaHeightmap;

// Storage buffer for CUDA-generated normal map (binding 4)
// Each normal is stored as vec4(nx, ny, nz, slopeMagnitude)
layout(std430, binding = 4) readonly buffer NormalMapBuffer {
    vec4 normals[HEIGHTMAP_SIZE * HEIGHTMAP_SIZE];
} cudaNormalMap;

// Storage buffer for CUDA-generated tangent space (binding 5)
// Each tangent is stored as vec4(tx, ty, tz, bitangentSign)
layout(std430, binding = 5) readonly buffer TangentMapBuffer {
    vec4 tangents[HEIGHTMAP_SIZE * HEIGHTMAP_SIZE];
} cudaTangentMap;

// ============================================================================
// HEIGHTMAP SAMPLING FUNCTIONS
// Bilinear interpolation with NaN/Inf safeguards
// ============================================================================

// Sample heightmap with bilinear interpolation
float sampleHeightmap(vec2 worldPos) {
    // Convert world position to heightmap UV coordinates [0, 1]
    vec2 uv = (worldPos / TERRAIN_SIZE) + 0.5;

    // Clamp UV to valid range
    uv = clamp(uv, 0.0, 1.0);

    // Convert to pixel coordinates
    vec2 pixelCoord = uv * float(HEIGHTMAP_SIZE - 1);

    // Get integer and fractional parts for bilinear interpolation
    ivec2 p00 = ivec2(floor(pixelCoord));
    ivec2 p11 = min(p00 + ivec2(1, 1), ivec2(HEIGHTMAP_SIZE - 1));
    ivec2 p10 = ivec2(p11.x, p00.y);
    ivec2 p01 = ivec2(p00.x, p11.y);

    vec2 f = fract(pixelCoord);

    // Sample four neighboring heights
    float h00 = cudaHeightmap.heights[p00.y * HEIGHTMAP_SIZE + p00.x];
    float h10 = cudaHeightmap.heights[p10.y * HEIGHTMAP_SIZE + p10.x];
    float h01 = cudaHeightmap.heights[p01.y * HEIGHTMAP_SIZE + p01.x];
    float h11 = cudaHeightmap.heights[p11.y * HEIGHTMAP_SIZE + p11.x];

    // Safeguard against NaN in heightmap data (prevents black spots)
    // NaN check: x != x is true only for NaN
    if (h00 != h00 || isinf(h00)) h00 = 0.0;
    if (h10 != h10 || isinf(h10)) h10 = 0.0;
    if (h01 != h01 || isinf(h01)) h01 = 0.0;
    if (h11 != h11 || isinf(h11)) h11 = 0.0;

    // Bilinear interpolation
    float h0 = mix(h00, h10, f.x);
    float h1 = mix(h01, h11, f.x);
    float result = mix(h0, h1, f.y);

    // Final safeguard after interpolation
    return (result != result || isinf(result)) ? 0.0 : result;
}

// Sample normal map with bilinear interpolation
vec3 sampleNormalMap(vec2 worldPos) {
    // Convert world position to heightmap UV coordinates [0, 1]
    vec2 uv = (worldPos / TERRAIN_SIZE) + 0.5;
    uv = clamp(uv, 0.0, 1.0);

    // Convert to pixel coordinates
    vec2 pixelCoord = uv * float(HEIGHTMAP_SIZE - 1);

    // Get integer and fractional parts for bilinear interpolation
    ivec2 p00 = ivec2(floor(pixelCoord));
    ivec2 p11 = min(p00 + ivec2(1, 1), ivec2(HEIGHTMAP_SIZE - 1));
    ivec2 p10 = ivec2(p11.x, p00.y);
    ivec2 p01 = ivec2(p00.x, p11.y);

    vec2 f = fract(pixelCoord);

    // Sample four neighboring normals
    vec3 n00 = cudaNormalMap.normals[p00.y * HEIGHTMAP_SIZE + p00.x].xyz;
    vec3 n10 = cudaNormalMap.normals[p10.y * HEIGHTMAP_SIZE + p10.x].xyz;
    vec3 n01 = cudaNormalMap.normals[p01.y * HEIGHTMAP_SIZE + p01.x].xyz;
    vec3 n11 = cudaNormalMap.normals[p11.y * HEIGHTMAP_SIZE + p11.x].xyz;

    // Safeguard against NaN in normal map data (prevents black spots)
    vec3 defaultNormal = vec3(0.0, 1.0, 0.0);
    if (any(isnan(n00)) || any(isinf(n00))) n00 = defaultNormal;
    if (any(isnan(n10)) || any(isinf(n10))) n10 = defaultNormal;
    if (any(isnan(n01)) || any(isinf(n01))) n01 = defaultNormal;
    if (any(isnan(n11)) || any(isinf(n11))) n11 = defaultNormal;

    // Bilinear interpolation
    vec3 n0 = mix(n00, n10, f.x);
    vec3 n1 = mix(n01, n11, f.x);
    vec3 result = mix(n0, n1, f.y);

    // Final safeguard: check result and normalize
    if (any(isnan(result)) || any(isinf(result))) {
        return defaultNormal;
    }
    float len = length(result);
    return len > 0.0001 ? result / len : defaultNormal;
}

// Sample tangent with bilinear interpolation
vec4 sampleTangentMap(vec2 worldPos) {
    // Convert world position to heightmap UV coordinates [0, 1]
    vec2 uv = (worldPos / TERRAIN_SIZE) + 0.5;
    uv = clamp(uv, 0.0, 1.0);

    // Convert to pixel coordinates
    vec2 pixelCoord = uv * float(HEIGHTMAP_SIZE - 1);

    // Get integer and fractional parts for bilinear interpolation
    ivec2 p00 = ivec2(floor(pixelCoord));
    ivec2 p11 = min(p00 + ivec2(1, 1), ivec2(HEIGHTMAP_SIZE - 1));
    ivec2 p10 = ivec2(p11.x, p00.y);
    ivec2 p01 = ivec2(p00.x, p11.y);

    vec2 f = fract(pixelCoord);

    // Sample four neighboring tangents
    vec4 t00 = cudaTangentMap.tangents[p00.y * HEIGHTMAP_SIZE + p00.x];
    vec4 t10 = cudaTangentMap.tangents[p10.y * HEIGHTMAP_SIZE + p10.x];
    vec4 t01 = cudaTangentMap.tangents[p01.y * HEIGHTMAP_SIZE + p01.x];
    vec4 t11 = cudaTangentMap.tangents[p11.y * HEIGHTMAP_SIZE + p11.x];

    // Safeguard against NaN in tangent map data (prevents black spots)
    vec4 defaultTangent = vec4(1.0, 0.0, 0.0, 1.0);
    if (any(isnan(t00)) || any(isinf(t00))) t00 = defaultTangent;
    if (any(isnan(t10)) || any(isinf(t10))) t10 = defaultTangent;
    if (any(isnan(t01)) || any(isinf(t01))) t01 = defaultTangent;
    if (any(isnan(t11)) || any(isinf(t11))) t11 = defaultTangent;

    // Bilinear interpolation for tangent direction (xyz)
    vec3 tangent0 = mix(t00.xyz, t10.xyz, f.x);
    vec3 tangent1 = mix(t01.xyz, t11.xyz, f.x);
    vec3 tangentResult = mix(tangent0, tangent1, f.y);

    // Final safeguard: check result and normalize
    if (any(isnan(tangentResult)) || any(isinf(tangentResult))) {
        return defaultTangent;
    }
    float tangentLen = length(tangentResult);
    vec3 tangent = tangentLen > 0.0001 ? tangentResult / tangentLen : vec3(1.0, 0.0, 0.0);

    // Take bitangent sign from nearest sample (validate it too)
    float bitangentSign = t00.w;
    if (isnan(bitangentSign) || isinf(bitangentSign)) bitangentSign = 1.0;

    return vec4(tangent, bitangentSign);
}

// ============================================================================
// VERTEX I/O
// ============================================================================

layout(location = 0) in vec4 inPosition[];
layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outTangent;
layout(location = 3) out float outBitangentSign;

// ============================================================================
// CONSTANTS
// ============================================================================

const float heightScale = 35.0;  // Max height range in world units

// ============================================================================
// MAIN
// All terrain data is pre-computed by CUDA using cuRAND
// Shader only performs bilinear sampling from CUDA-generated buffers
// ============================================================================

void main() {
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;

    vec4 p0 = inPosition[0];
    vec4 p1 = inPosition[1];
    vec4 p2 = inPosition[2];
    vec4 p3 = inPosition[3];

    vec4 pos1 = mix(p0, p1, u);
    vec4 pos2 = mix(p3, p2, u);
    vec4 position = mix(pos1, pos2, v);

    vec2 worldPos = position.xz;

    // Sample height from CUDA-generated heightmap (computed with cuRAND)
    float height = sampleHeightmap(worldPos);

    position.y = height * heightScale;
    position.w = 1.0;

    // Sample normal from CUDA-generated normal map
    vec3 normal = sampleNormalMap(worldPos);

    // Sample tangent from CUDA-generated tangent map
    vec4 tangentData = sampleTangentMap(worldPos);

    // Output world position and TBN data for fragment shader
    outWorldPos = position.xyz;
    outNormal = normal;
    outTangent = tangentData.xyz;
    outBitangentSign = tangentData.w;

    gl_Position = uMVP.projectionMatrix * uMVP.viewMatrix * uMVP.modelMatrix * position;
}
