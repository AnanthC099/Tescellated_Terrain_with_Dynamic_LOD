#version 450 core
#extension GL_ARB_separate_shader_objects : enable

// Tessellation Evaluation Shader for Bumpy Terrain (Ubhad Khabad)
// Evaluates tessellated vertex positions with procedural height

layout(quads, equal_spacing, ccw) in;

layout(binding = 0) uniform mvpMatrix {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 color;
} uMVP;

layout(location = 0) in vec4 inPosition[];

// ============================================
// Procedural Noise Functions for Bumpy Terrain
// ============================================

// Hash function for pseudo-random values
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

// Perlin-style gradient noise
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    // Smooth interpolation curve (quintic)
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    // Four corners
    float a = dot(hash2(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0));
    float b = dot(hash2(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0));
    float c = dot(hash2(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0));
    float d = dot(hash2(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0));

    // Bilinear interpolation
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractal Brownian Motion - multiple layers of noise for natural terrain
float fbm(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise(p * frequency);
        amplitude *= 0.5;   // Persistence - each octave contributes less
        frequency *= 2.0;   // Lacunarity - each octave has higher frequency
    }
    return value;
}

// ============================================
// Terrain Height Parameters
// ============================================
const float heightScale = 0.4;      // Overall height of bumps
const float noiseScale = 2.5;       // Frequency of terrain features
const int numOctaves = 4;           // Detail levels (more = finer detail)

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

    // ============================================
    // Generate Bumpy Terrain Height (Ubhad Khabad)
    // ============================================
    vec2 noiseCoord = position.xz * noiseScale;

    // Base terrain using fractal noise
    float height = fbm(noiseCoord, numOctaves);

    // Add some gentle rolling hills
    height += 0.3 * sin(position.x * 0.8) * cos(position.z * 0.6);

    // Add smaller bumps for roughness
    height += 0.15 * noise(noiseCoord * 3.0);

    // Apply height scale
    position.y = height * heightScale;
    position.w = 1.0;

    // Transform to clip space
    gl_Position = uMVP.projectionMatrix * uMVP.viewMatrix * uMVP.modelMatrix * position;
}
