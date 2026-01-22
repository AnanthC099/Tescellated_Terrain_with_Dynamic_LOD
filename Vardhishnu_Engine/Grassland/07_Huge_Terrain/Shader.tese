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
layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;

// Domain rotation matrix to eliminate directional artifacts between octaves
const mat2 ROT = mat2(0.80, 0.60, -0.60, 0.80);

// High-quality hash functions for gradient generation
float hash1(vec2 p) {
    p = fract(p * vec2(443.897, 441.423));
    p += dot(p, p.yx + 19.19);
    return fract((p.x + p.y) * p.x);
}

vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

vec3 hash3(vec2 p) {
    vec3 q = vec3(dot(p, vec2(127.1, 311.7)),
                  dot(p, vec2(269.5, 183.3)),
                  dot(p, vec2(419.2, 371.9)));
    return fract(sin(q) * 43758.5453);
}

// Perlin noise with quintic interpolation
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

// Simplex-like noise for smoother results
float simplexNoise(vec2 p) {
    const float K1 = 0.366025404; // (sqrt(3)-1)/2
    const float K2 = 0.211324865; // (3-sqrt(3))/6

    vec2 i = floor(p + (p.x + p.y) * K1);
    vec2 a = p - i + (i.x + i.y) * K2;
    float m = step(a.y, a.x);
    vec2 o = vec2(m, 1.0 - m);
    vec2 b = a - o + K2;
    vec2 c = a - 1.0 + 2.0 * K2;

    vec3 h = max(0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
    vec3 n = h * h * h * h * vec3(dot(a, hash2(i)),
                                   dot(b, hash2(i + o)),
                                   dot(c, hash2(i + 1.0)));
    return dot(n, vec3(70.0));
}

// Cellular (Worley) noise for valleys and erosion patterns
float cellularNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    float minDist = 1.0;
    float secondMin = 1.0;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 point = hash3(i + neighbor).xy;
            vec2 diff = neighbor + point - f;
            float dist = length(diff);

            if (dist < minDist) {
                secondMin = minDist;
                minDist = dist;
            } else if (dist < secondMin) {
                secondMin = dist;
            }
        }
    }
    return secondMin - minDist; // Edge distance for valley patterns
}

// FBM with domain rotation to break directional artifacts
float fbmRotated(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
        p = ROT * p; // Rotate domain between octaves
    }
    return value;
}

// Ridged multifractal for mountain ranges
float ridgedMultifractal(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.6;
    float frequency = 1.0;
    float weight = 1.0;
    float offset = 1.0;

    for (int i = 0; i < octaves; i++) {
        float signal = noise(p * frequency);
        signal = offset - abs(signal);
        signal *= signal;
        signal *= weight;
        weight = clamp(signal * 2.0, 0.0, 1.0);
        value += amplitude * signal;
        amplitude *= 0.5;
        frequency *= 2.1;
        p = ROT * p;
    }
    return value;
}

// Swiss noise for smooth, billowy terrain
float swissNoise(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    vec2 dsum = vec2(0.0);

    for (int i = 0; i < octaves; i++) {
        vec2 pf = p * frequency;
        float n = noise(pf + dsum * 1.5);
        float d = noise(pf + vec2(5.2, 1.3));
        dsum += vec2(d, d) * amplitude * 0.5;
        value += amplitude * n / (1.0 + dot(dsum, dsum));
        amplitude *= 0.5;
        frequency *= 2.0;
        p = ROT * p;
    }
    return value;
}

// Domain warping for organic, non-uniform shapes
vec2 domainWarp(vec2 p, float strength) {
    float warpX = fbmRotated(p + vec2(0.0, 0.0), 4);
    float warpY = fbmRotated(p + vec2(5.2, 1.3), 4);
    return p + vec2(warpX, warpY) * strength;
}

// Double domain warp for even more organic patterns
vec2 doubleDomainWarp(vec2 p, float strength) {
    vec2 q = vec2(fbmRotated(p + vec2(0.0, 0.0), 3),
                  fbmRotated(p + vec2(5.2, 1.3), 3));
    vec2 r = vec2(fbmRotated(p + 4.0 * q + vec2(1.7, 9.2), 3),
                  fbmRotated(p + 4.0 * q + vec2(8.3, 2.8), 3));
    return p + r * strength;
}

// Regional variation - creates different terrain characteristics across the map
float getRegionType(vec2 p) {
    return noise(p * 0.002 + vec2(100.0, 200.0)) * 0.5 + 0.5;
}

float getErosionFactor(vec2 p) {
    return smoothstep(0.0, 1.0, noise(p * 0.005 + vec2(50.0, 150.0)) * 0.5 + 0.5);
}

float getMountainMask(vec2 p) {
    float mask = fbmRotated(p * 0.003 + vec2(200.0, 100.0), 4);
    return smoothstep(-0.2, 0.4, mask);
}

// Hydraulic erosion simulation using cellular noise
float hydraulicErosion(vec2 p, float baseHeight) {
    float erosion = cellularNoise(p * 0.08) * 0.3;
    erosion += cellularNoise(p * 0.15) * 0.15;
    float flowFactor = smoothstep(0.2, 0.6, baseHeight);
    return erosion * flowFactor;
}

// Thermal erosion - slope-dependent weathering
float thermalErosion(vec2 p, float slope) {
    float erosion = noise(p * 2.0) * 0.1;
    return erosion * smoothstep(0.3, 0.8, slope);
}

const float heightScale = 35.0;
const float noiseScale = 0.015;
const int numOctaves = 8;

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
    vec2 noiseCoord = worldPos * noiseScale;

    // Get regional characteristics for variation
    float regionType = getRegionType(worldPos);
    float mountainMask = getMountainMask(worldPos);
    float erosionFactor = getErosionFactor(worldPos);

    // Apply domain warping for organic terrain shapes
    vec2 warpedCoord = domainWarp(noiseCoord, 0.8 + regionType * 0.4);
    vec2 doubleWarpedCoord = doubleDomainWarp(noiseCoord * 0.7, 0.5);

    // Continental-scale base terrain with domain warping
    float continentalBase = fbmRotated(doubleWarpedCoord * 0.5, 5) * 0.3;

    // Regional terrain - blend between terrain types based on region
    float rollingHills = swissNoise(warpedCoord, 6) * 0.4;
    float mountains = ridgedMultifractal(warpedCoord * 1.2, 6) * 0.5;
    float plains = fbmRotated(noiseCoord * 0.8, 4) * 0.15;

    // Blend terrain types based on regional masks
    float regionalTerrain = mix(plains, rollingHills, smoothstep(0.3, 0.6, regionType));
    regionalTerrain = mix(regionalTerrain, mountains, mountainMask);

    // Combine base and regional terrain
    float height = continentalBase + regionalTerrain;

    // Add medium-scale features with warped coordinates
    height += fbmRotated(warpedCoord * 2.0, 5) * 0.15 * (1.0 - mountainMask * 0.5);

    // Add ridged details in mountainous regions
    float ridgeDetail = ridgedMultifractal(noiseCoord * 3.0, 4) * 0.12;
    height += ridgeDetail * mountainMask;

    // Apply erosion effects
    float hydraulic = hydraulicErosion(worldPos, height);
    height -= hydraulic * erosionFactor * 0.4;

    // Fine detail layers - vary based on region
    float detailScale = 0.8 + regionType * 0.4;
    height += noise(warpedCoord * 4.0) * 0.08 * detailScale;
    height += noise(warpedCoord * 8.0) * 0.04 * detailScale;
    height += simplexNoise(noiseCoord * 12.0) * 0.025;

    // Micro-detail for surface roughness
    float microDetail = noise(noiseCoord * 20.0) * 0.015;
    microDetail += noise(noiseCoord * 35.0) * 0.008;
    height += microDetail * (0.5 + mountainMask * 0.5);

    // Cellular patterns for rocky areas
    float rockPattern = cellularNoise(noiseCoord * 6.0) * 0.03;
    height += rockPattern * mountainMask;

    position.y = height * heightScale;
    position.w = 1.0;

    // Calculate normal for lighting
    float eps = 0.5;
    vec2 dx = vec2(eps, 0.0);
    vec2 dz = vec2(0.0, eps);

    // Sample neighboring heights for normal calculation
    vec2 ncX1 = (worldPos + dx) * noiseScale;
    vec2 ncX2 = (worldPos - dx) * noiseScale;
    vec2 ncZ1 = (worldPos + dz) * noiseScale;
    vec2 ncZ2 = (worldPos - dz) * noiseScale;

    // Simplified height samples for normal (using main contributors)
    float hX1 = (fbmRotated(domainWarp(ncX1, 0.8), 5) + ridgedMultifractal(ncX1, 4) * mountainMask) * heightScale;
    float hX2 = (fbmRotated(domainWarp(ncX2, 0.8), 5) + ridgedMultifractal(ncX2, 4) * mountainMask) * heightScale;
    float hZ1 = (fbmRotated(domainWarp(ncZ1, 0.8), 5) + ridgedMultifractal(ncZ1, 4) * mountainMask) * heightScale;
    float hZ2 = (fbmRotated(domainWarp(ncZ2, 0.8), 5) + ridgedMultifractal(ncZ2, 4) * mountainMask) * heightScale;

    vec3 normal = normalize(vec3(hX2 - hX1, 2.0 * eps, hZ2 - hZ1));

    outWorldPos = position.xyz;
    outNormal = normal;

    gl_Position = uMVP.projectionMatrix * uMVP.viewMatrix * uMVP.modelMatrix * position;
}
