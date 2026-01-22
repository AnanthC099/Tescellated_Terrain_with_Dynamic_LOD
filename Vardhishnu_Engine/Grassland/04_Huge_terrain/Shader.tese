#version 450 core
#extension GL_ARB_separate_shader_objects : enable

layout(quads, equal_spacing, ccw) in;

layout(binding = 0) uniform mvpMatrix {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 color;
} uMVP;

layout(location = 0) in vec4 inPosition[];

vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    float a = dot(hash2(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0));
    float b = dot(hash2(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0));
    float c = dot(hash2(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0));
    float d = dot(hash2(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}

// Ridged noise for sharp, jagged features
float ridgedNoise(vec2 p) {
    float n = noise(p);
    n = 1.0 - abs(n);  // Creates sharp ridges
    return n * n;      // Sharpen the ridges further
}

// Ridged FBM for rugged terrain
float ridgedFbm(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    float weight = 1.0;

    for (int i = 0; i < octaves; i++) {
        float n = ridgedNoise(p * frequency);
        n *= weight;
        weight = clamp(n * 2.0, 0.0, 1.0);  // Weight successive octaves by previous
        value += amplitude * n;
        amplitude *= 0.5;
        frequency *= 2.2;  // Slightly higher frequency multiplier for more variation
    }
    return value;
}

const float heightScale = 20.0;   // Reduced for less dramatic but more uneven terrain
const float noiseScale = 0.03;    // Slightly higher for more features
const int numOctaves = 8;         // More octaves for finer detail

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

    vec2 noiseCoord = position.xz * noiseScale;

    // Base terrain using regular FBM
    float height = fbm(noiseCoord, numOctaves) * 0.4;

    // Add ridged noise for rugged, uneven features
    height += ridgedFbm(noiseCoord * 1.5, 5) * 0.35;

    // Multiple high-frequency noise layers for roughness
    height += 0.12 * noise(noiseCoord * 4.0);   // Medium detail
    height += 0.08 * noise(noiseCoord * 8.0);   // Fine detail
    height += 0.05 * noise(noiseCoord * 16.0);  // Very fine detail

    // Small random bumps for surface roughness
    height += 0.03 * ridgedNoise(noiseCoord * 12.0);

    position.y = height * heightScale;
    position.w = 1.0;

    gl_Position = uMVP.projectionMatrix * uMVP.viewMatrix * uMVP.modelMatrix * position;
}
