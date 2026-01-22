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

// ============================================================================
// GARDEN/LAND TERRAIN GENERATOR
// Implements layered terrain: Macro → Meso → Micro with feature primitives
// For 800×800 unit terrain with heightScale=35
// ============================================================================

// Domain rotation matrix to eliminate directional artifacts between octaves
const mat2 ROT = mat2(0.80, 0.60, -0.60, 0.80);

// Additional rotation matrices for breaking coherence
const mat2 ROT2 = mat2(0.95, 0.31, -0.31, 0.95);
const mat2 ROT3 = mat2(0.70, 0.71, -0.71, 0.70);

// ============================================================================
// HASH FUNCTIONS - High-quality gradient generation with seed support
// ============================================================================

// Seeded hash functions - different seeds produce completely different sequences
float hash1(vec2 p) {
    p = fract(p * vec2(443.897, 441.423));
    p += dot(p, p.yx + 19.19);
    return fract((p.x + p.y) * p.x);
}

float hash1Seeded(vec2 p, float seed) {
    // Mix seed into coordinates using prime multipliers
    p = p + seed * vec2(17.31, 23.57);
    p = fract(p * vec2(443.897 + seed * 0.731, 441.423 + seed * 0.619));
    p += dot(p, p.yx + 19.19 + seed);
    return fract((p.x + p.y) * p.x + seed * 0.1731);
}

vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

vec2 hash2Seeded(vec2 p, float seed) {
    // Incorporate seed into the dot product vectors using primes
    vec2 k1 = vec2(127.1 + seed * 7.31, 311.7 + seed * 11.17);
    vec2 k2 = vec2(269.5 + seed * 13.37, 183.3 + seed * 17.93);
    p = vec2(dot(p, k1), dot(p, k2));
    return -1.0 + 2.0 * fract(sin(p) * (43758.5453123 + seed * 1.618));
}

vec3 hash3(vec2 p) {
    vec3 q = vec3(dot(p, vec2(127.1, 311.7)),
                  dot(p, vec2(269.5, 183.3)),
                  dot(p, vec2(419.2, 371.9)));
    return fract(sin(q) * 43758.5453);
}

vec3 hash3Seeded(vec2 p, float seed) {
    vec2 k1 = vec2(127.1 + seed * 7.31, 311.7 + seed * 11.17);
    vec2 k2 = vec2(269.5 + seed * 13.37, 183.3 + seed * 17.93);
    vec2 k3 = vec2(419.2 + seed * 19.41, 371.9 + seed * 23.59);
    vec3 q = vec3(dot(p, k1), dot(p, k2), dot(p, k3));
    return fract(sin(q) * (43758.5453 + seed * 1.618));
}

// ============================================================================
// BASE NOISE FUNCTIONS - With seeded variants for decorrelation
// ============================================================================

// Perlin noise with quintic interpolation (smoother than cubic)
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

// Seeded Perlin noise - produces completely different pattern per seed
float noiseSeeded(vec2 p, float seed) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    float a = dot(hash2Seeded(i + vec2(0.0, 0.0), seed), f - vec2(0.0, 0.0));
    float b = dot(hash2Seeded(i + vec2(1.0, 0.0), seed), f - vec2(1.0, 0.0));
    float c = dot(hash2Seeded(i + vec2(0.0, 1.0), seed), f - vec2(0.0, 1.0));
    float d = dot(hash2Seeded(i + vec2(1.0, 1.0), seed), f - vec2(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Simplex-like noise for smoother results
float simplexNoise(vec2 p) {
    const float K1 = 0.366025404;
    const float K2 = 0.211324865;

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

// Seeded simplex noise
float simplexNoiseSeeded(vec2 p, float seed) {
    const float K1 = 0.366025404;
    const float K2 = 0.211324865;

    vec2 i = floor(p + (p.x + p.y) * K1);
    vec2 a = p - i + (i.x + i.y) * K2;
    float m = step(a.y, a.x);
    vec2 o = vec2(m, 1.0 - m);
    vec2 b = a - o + K2;
    vec2 c = a - 1.0 + 2.0 * K2;

    vec3 h = max(0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
    vec3 n = h * h * h * h * vec3(dot(a, hash2Seeded(i, seed)),
                                   dot(b, hash2Seeded(i + o, seed)),
                                   dot(c, hash2Seeded(i + 1.0, seed)));
    return dot(n, vec3(70.0));
}

// Cellular (Worley) noise for erosion patterns
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
    return secondMin - minDist;
}

// Seeded cellular noise
float cellularNoiseSeeded(vec2 p, float seed) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    float minDist = 1.0;
    float secondMin = 1.0;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 point = hash3Seeded(i + neighbor, seed).xy;
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
    return secondMin - minDist;
}

// ============================================================================
// FBM VARIANTS - Different characteristics for different layers
// Seeded versions ensure each layer produces independent patterns
// ============================================================================

// Standard FBM with rotation (breaks directional artifacts)
float fbmRotated(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
        p = ROT * p;
    }
    return value;
}

// Seeded FBM - completely independent pattern per seed
float fbmRotatedSeeded(vec2 p, int octaves, float seed) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * noiseSeeded(p * frequency, seed + float(i) * 7.31);
        amplitude *= 0.5;
        frequency *= 2.0;
        p = ROT * p;
    }
    return value;
}

// Billowy FBM - softer, more organic for mounds
float fbmBillowy(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < octaves; i++) {
        float n = noise(p * frequency);
        n = n * n; // Square to create billowy effect
        value += amplitude * n;
        amplitude *= 0.45;
        frequency *= 2.0;
        p = ROT2 * p;
    }
    return value;
}

// Seeded billowy FBM
float fbmBillowySeeded(vec2 p, int octaves, float seed) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < octaves; i++) {
        float n = noiseSeeded(p * frequency, seed + float(i) * 11.17);
        n = n * n;
        value += amplitude * n;
        amplitude *= 0.45;
        frequency *= 2.0;
        p = ROT2 * p;
    }
    return value;
}

// Ridged FBM - for subtle ridges (NOT sharp mountains)
float fbmRidged(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    float weight = 1.0;

    for (int i = 0; i < octaves; i++) {
        float signal = 1.0 - abs(noise(p * frequency));
        signal = signal * signal * weight;
        weight = clamp(signal * 1.5, 0.0, 1.0);
        value += amplitude * signal;
        amplitude *= 0.4;
        frequency *= 2.2;
        p = ROT3 * p;
    }
    return value * 0.7; // Reduce amplitude
}

// Seeded ridged FBM
float fbmRidgedSeeded(vec2 p, int octaves, float seed) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    float weight = 1.0;

    for (int i = 0; i < octaves; i++) {
        float signal = 1.0 - abs(noiseSeeded(p * frequency, seed + float(i) * 13.37));
        signal = signal * signal * weight;
        weight = clamp(signal * 1.5, 0.0, 1.0);
        value += amplitude * signal;
        amplitude *= 0.4;
        frequency *= 2.2;
        p = ROT3 * p;
    }
    return value * 0.7;
}

// ============================================================================
// DOMAIN WARPING - Breaks repetition, creates organic shapes
// Each warp function uses unique seeds for decorrelation
// ============================================================================

// Light domain warp for macro forms (seeds: 3.0, 7.0)
vec2 domainWarpLight(vec2 p, float strength) {
    float warpX = fbmRotatedSeeded(p, 3, 3.0);
    float warpY = fbmRotatedSeeded(p, 3, 7.0);
    return p + vec2(warpX, warpY) * strength;
}

// Strong domain warp for breaking repetition (seeds: 11.0, 13.0, 17.0, 19.0)
vec2 domainWarpStrong(vec2 p, float strength) {
    vec2 q = vec2(fbmRotatedSeeded(p, 3, 11.0),
                  fbmRotatedSeeded(p, 3, 13.0));
    vec2 r = vec2(fbmRotatedSeeded(p + 3.0 * q, 3, 17.0),
                  fbmRotatedSeeded(p + 3.0 * q, 3, 19.0));
    return p + r * strength;
}

// ============================================================================
// CONTROL MASKS - Vary roughness/features by region
// Each mask uses a unique prime seed to ensure independence
// ============================================================================

// Roughness mask M(x,z): high = rough detail, low = smooth (seed: 23.0)
float getRoughnessMask(vec2 p) {
    float n = fbmRotatedSeeded(p * 0.003, 4, 23.0);
    return smoothstep(-0.3, 0.5, n);
}

// Mound/bed zone mask - where raised beds are (seed: 29.0)
float getMoundMask(vec2 p) {
    // Create 3-6 broad mound regions
    float n = fbmBillowySeeded(p * 0.004, 3, 29.0);
    return smoothstep(0.1, 0.5, n);
}

// Basin mask - where low spots/sediment areas are (seed: 31.0)
float getBasinMask(vec2 p) {
    float n = fbmRotatedSeeded(p * 0.005, 3, 31.0);
    return smoothstep(0.4, 0.1, n); // Inverted - basins are where noise is LOW
}

// Path mask - creates curving path ribbons
float getPathMask(vec2 worldPos) {
    // Create 2-3 meandering paths using warped coordinates
    vec2 warpedPos = domainWarpStrong(worldPos * 0.008, 0.6);

    // Main path - serpentine curve
    float pathWidth = 15.0; // Path width in world units
    float mainPath = abs(sin(warpedPos.x * 0.5 + warpedPos.y * 0.3) * 60.0 - worldPos.y);
    float path1 = 1.0 - smoothstep(0.0, pathWidth, mainPath);

    // Secondary path - crosses at angle
    float secondPath = abs(sin(warpedPos.y * 0.4 - warpedPos.x * 0.2) * 50.0 - worldPos.x * 0.7 + worldPos.y * 0.3);
    float path2 = 1.0 - smoothstep(0.0, pathWidth * 0.8, secondPath);

    return max(path1, path2 * 0.8);
}

// Raised bed edge mask - creates ridge borders around beds
float getBedEdgeMask(vec2 p) {
    // Create edge where mound mask transitions using manual finite differences
    float eps = 2.0;
    float moundCenter = getMoundMask(p);
    float moundRight = getMoundMask(p + vec2(eps, 0.0));
    float moundUp = getMoundMask(p + vec2(0.0, eps));

    float gradX = abs(moundRight - moundCenter) / eps;
    float gradY = abs(moundUp - moundCenter) / eps;
    float edgeGradient = gradX + gradY;

    return smoothstep(0.001, 0.02, edgeGradient);
}

// Drainage swale mask - shallow channels (seed: 37.0)
float getSwaleMask(vec2 worldPos) {
    vec2 warpedPos = domainWarpLight(worldPos * 0.006, 0.4);
    // Create branching drainage patterns using cellular noise
    float swalePattern = cellularNoiseSeeded(warpedPos * 0.8, 37.0);
    return smoothstep(0.0, 0.15, swalePattern);
}

// Lawn/flat zone mask (seed: 41.0)
float getLawnMask(vec2 p) {
    float n = fbmBillowySeeded(p * 0.006, 3, 41.0);
    // Areas where we want flat plateaus
    return smoothstep(0.2, 0.5, n);
}

// ============================================================================
// NATURAL IMPERFECTION FUNCTIONS - Real-world terrain details
// Each function uses unique prime seeds to prevent correlation
// ============================================================================

// Soil heterogeneity mask - rocky vs sandy vs clay areas (seeds: 43.0, 47.0)
float getSoilHeterogeneityMask(vec2 p) {
    // Create patchy soil variation zones
    float coarse = fbmRotatedSeeded(p * 0.012, 3, 43.0);
    float fine = noiseSeeded(p * 0.025, 47.0);
    // Combine for irregular patches
    return smoothstep(-0.2, 0.6, coarse + fine * 0.3);
}

// Procedural pebble/stone field - scattered point displacements
// Uses seeds: 53.0 (large stones), 59.0 (small pebbles), 61.0 (gravel)
float getPebbleHeight(vec2 worldPos) {
    float pebbles = 0.0;

    // Layer 1: Larger stones (sparse) - seed 53.0
    vec2 stoneCoord = worldPos * 0.15;
    vec2 stoneCell = floor(stoneCoord);
    vec2 stoneFract = fract(stoneCoord);

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec3 randVal = hash3Seeded(stoneCell + neighbor, 53.0);

            // Only place stone if random threshold met (sparse)
            if (randVal.z > 0.7) {
                vec2 stonePos = neighbor + randVal.xy * 0.8 + 0.1;
                float dist = length(stoneFract - stonePos);
                float stoneRadius = 0.15 + randVal.z * 0.1;
                // Rounded stone profile
                float stone = 1.0 - smoothstep(0.0, stoneRadius, dist);
                stone = stone * stone * (3.0 - 2.0 * stone); // Smooth falloff
                pebbles += stone * 0.025 * randVal.z;
            }
        }
    }

    // Layer 2: Small pebbles (denser) - seed 59.0
    vec2 pebbleCoord = worldPos * 0.5;
    vec2 pebbleCell = floor(pebbleCoord);
    vec2 pebbleFract = fract(pebbleCoord);

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec3 randVal = hash3Seeded(pebbleCell + neighbor, 59.0);

            if (randVal.z > 0.5) {
                vec2 pebblePos = neighbor + randVal.xy * 0.9 + 0.05;
                float dist = length(pebbleFract - pebblePos);
                float pebbleRadius = 0.08 + randVal.z * 0.06;
                float pebble = 1.0 - smoothstep(0.0, pebbleRadius, dist);
                pebble = pebble * pebble;
                pebbles += pebble * 0.012 * randVal.z;
            }
        }
    }

    // Layer 3: Tiny gravel (very dense, subtle) - seed 61.0
    vec2 gravelCoord = worldPos * 1.5;
    vec2 gravelCell = floor(gravelCoord);
    vec2 gravelFract = fract(gravelCoord);
    vec3 gravelRand = hash3Seeded(gravelCell, 61.0);
    vec2 gravelPos = gravelRand.xy;
    float gravelDist = length(gravelFract - gravelPos);
    float gravel = 1.0 - smoothstep(0.0, 0.12, gravelDist);
    pebbles += gravel * 0.005 * gravelRand.z;

    return pebbles;
}

// Micro-erosion channels - tiny water-carved gullies
// Uses seeds: 67.0 (primary channels), 71.0 (fine channels), 73.0 (branching)
float getMicroErosionHeight(vec2 worldPos) {
    float erosion = 0.0;

    // Primary micro-channels using cellular noise - seed 67.0
    vec2 erosionCoord = worldPos * 0.04;
    vec2 warpedCoord = domainWarpLight(erosionCoord, 0.3);
    float channels = cellularNoiseSeeded(warpedCoord * 2.0, 67.0);
    // Carve where cell boundaries meet (edges)
    erosion -= smoothstep(0.1, 0.0, channels) * 0.015;

    // Secondary finer channels - seed 71.0
    vec2 fineCoord = worldPos * 0.08;
    float fineChannels = cellularNoiseSeeded(fineCoord * 3.0, 71.0);
    erosion -= smoothstep(0.08, 0.0, fineChannels) * 0.008;

    // Branching erosion pattern - seed 73.0
    float branchNoise = noiseSeeded(worldPos * 0.06, 73.0);
    float branchPattern = abs(sin(worldPos.x * 0.03 + branchNoise * 3.0) *
                              cos(worldPos.y * 0.025 + branchNoise * 2.5));
    erosion -= (1.0 - smoothstep(0.0, 0.15, branchPattern)) * 0.01;

    return erosion;
}

// High-frequency surface noise - soil granularity
// Uses seeds: 79.0, 83.0, 89.0, 97.0 (freq layers), 101.0, 103.0 (clumps)
float getHighFrequencyBumps(vec2 worldPos, float heterogeneityMask) {
    float bumps = 0.0;

    // Very high frequency noise for soil texture - each frequency has unique seed
    float freq1 = noiseSeeded(worldPos * 0.12, 79.0) * 0.008;
    float freq2 = noiseSeeded(worldPos * 0.25, 83.0) * 0.005;
    float freq3 = simplexNoiseSeeded(worldPos * 0.4, 89.0) * 0.003;
    float freq4 = noiseSeeded(worldPos * 0.6, 97.0) * 0.002;

    // Combine frequencies
    bumps = freq1 + freq2 + freq3 + freq4;

    // Modulate by soil heterogeneity (more bumps in rough soil areas)
    bumps *= (0.5 + heterogeneityMask * 0.8);

    // Add occasional larger soil clumps - seeds 101.0, 103.0
    float clumpNoise = fbmRotatedSeeded(worldPos * 0.02, 2, 101.0);
    float clumps = smoothstep(0.3, 0.5, clumpNoise) * noiseSeeded(worldPos * 0.08, 103.0) * 0.012;
    bumps += clumps * heterogeneityMask;

    return bumps;
}

// Root bumps - subtle surface undulations from underground roots
// Uses seeds: 107.0, 109.0 (root patterns), 113.0 (root mask)
float getRootBumps(vec2 worldPos) {
    float roots = 0.0;

    // Simulate root paths as elongated ridges
    vec2 rootCoord = worldPos * 0.02;

    // Main root directions (radial from imaginary tree positions) - seeds 107.0, 109.0
    float rootPattern1 = sin(rootCoord.x * 5.0 + noiseSeeded(rootCoord * 2.0, 107.0) * 4.0);
    float rootPattern2 = sin(rootCoord.y * 4.5 + noiseSeeded(rootCoord * 2.5, 109.0) * 3.5);

    // Create ridge-like bumps
    float ridge1 = 1.0 - abs(rootPattern1);
    float ridge2 = 1.0 - abs(rootPattern2);

    ridge1 = pow(ridge1, 4.0) * 0.015;
    ridge2 = pow(ridge2, 4.0) * 0.012;

    // Sparse placement using mask - seed 113.0
    float rootMask = smoothstep(0.2, 0.5, fbmRotatedSeeded(worldPos * 0.008, 2, 113.0));
    roots = (ridge1 + ridge2) * rootMask;

    return roots;
}

// Footprint/compression zones - subtle depressions (seed: 127.0)
float getCompressionDepth(vec2 worldPos) {
    float compression = 0.0;

    // Scattered compression points (old footprints, animal trails)
    vec2 compCoord = worldPos * 0.08;
    vec2 compCell = floor(compCoord);
    vec2 compFract = fract(compCoord);

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec3 randVal = hash3Seeded(compCell + neighbor, 127.0);

            // Sparse placement
            if (randVal.z > 0.75) {
                vec2 compPos = neighbor + randVal.xy * 0.7 + 0.15;
                float dist = length(compFract - compPos);
                // Elongated depression (elliptical)
                vec2 diff = compFract - compPos;
                float angle = randVal.x * 6.28;
                vec2 rotDiff = vec2(
                    diff.x * cos(angle) - diff.y * sin(angle),
                    diff.x * sin(angle) + diff.y * cos(angle)
                );
                float ellipseDist = length(rotDiff * vec2(1.0, 0.6));
                float depression = 1.0 - smoothstep(0.0, 0.2, ellipseDist);
                compression -= depression * 0.008 * randVal.z;
            }
        }
    }

    return compression;
}

// ============================================================================
// COMBINED NATURAL IMPERFECTIONS - All real-world terrain details
// ============================================================================

float computeNaturalImperfections(vec2 worldPos, float distanceFade, float pathMask) {
    float imperfections = 0.0;

    // Get soil heterogeneity for this location
    float soilMask = getSoilHeterogeneityMask(worldPos);

    // 1) High-frequency surface bumps (soil granularity)
    float surfaceBumps = getHighFrequencyBumps(worldPos, soilMask);
    imperfections += surfaceBumps;

    // 2) Pebbles and stones (reduced on paths)
    float pebbles = getPebbleHeight(worldPos);
    pebbles *= (1.0 - pathMask * 0.8); // Fewer stones on paths
    pebbles *= soilMask; // More stones in rocky soil areas
    imperfections += pebbles;

    // 3) Micro-erosion channels
    float microErosion = getMicroErosionHeight(worldPos);
    microErosion *= (1.0 - pathMask * 0.5); // Less erosion on maintained paths
    imperfections += microErosion;

    // 4) Root bumps (not on paths)
    float roots = getRootBumps(worldPos);
    roots *= (1.0 - pathMask);
    imperfections += roots;

    // 5) Compression/footprint zones
    float compression = getCompressionDepth(worldPos);
    imperfections += compression;

    // Apply distance fade - imperfections less visible far away
    // Keep some detail even at distance for realism
    float fadedImperfections = imperfections * (0.3 + distanceFade * 0.7);

    return fadedImperfections;
}

// ============================================================================
// MACRO LAYER - Dominates the look (3-10× mid amplitude)
// Gentle overall slope, 2-6 broad mounds, 1-3 shallow basins
// Uses seeds: 131.0 (mounds), 137.0 (basins)
// ============================================================================

float computeMacroHeight(vec2 worldPos, out float overallSlope) {
    vec2 macroCoord = worldPos * 0.0015; // Very low frequency

    // Apply domain warp to break any repetition
    vec2 warpedCoord = domainWarpStrong(macroCoord, 0.3);

    // 1) Gentle overall slope (even tiny helps realism)
    overallSlope = worldPos.x * 0.0003 + worldPos.y * 0.0002;
    overallSlope += sin(worldPos.x * 0.002) * 0.01; // Slight undulation

    // 2) 2-6 broad mounds (landscape beds) - seed 131.0
    float broadMounds = fbmBillowySeeded(warpedCoord, 4, 131.0) * 0.6;
    broadMounds = max(broadMounds, 0.0); // Only raise, not lower for mounds

    // 3) 1-3 shallow basins (low spots - sediment feel) - seed 137.0
    float basins = fbmRotatedSeeded(warpedCoord * 0.8, 3, 137.0);
    basins = min(basins, 0.0) * 0.4; // Only lower for basins

    // Combine macro elements
    float macro = overallSlope + broadMounds + basins;

    return macro;
}

// ============================================================================
// MESO LAYER - Localized irregularity (1-3× mid amplitude)
// Clumpy noise at larger wavelength, varies with roughness mask
// Uses seeds: 139.0 (lumps), 149.0 (ridges)
// ============================================================================

float computeMesoHeight(vec2 worldPos, float roughnessMask) {
    vec2 mesoCoord = worldPos * 0.008; // Medium-low frequency

    // Apply light domain warp
    vec2 warpedCoord = domainWarpLight(mesoCoord, 0.5);

    // Clumpy irregular noise (not uniform ripples) - seed 139.0
    float irregularLumps = fbmRotatedSeeded(warpedCoord, 5, 139.0) * 0.2;

    // Add some ridged character but soft - seed 149.0
    float softRidges = fbmRidgedSeeded(warpedCoord * 1.5, 4, 149.0) * 0.12;

    // Combine and apply roughness mask
    float meso = (irregularLumps + softRidges) * roughnessMask;

    return meso;
}

// ============================================================================
// MICRO LAYER - Small bumps (0.1-0.5× mid amplitude)
// Subtle in wireframe, uses squared mask for selective application
// Uses seeds: 151.0, 157.0, 163.0
// ============================================================================

float computeMicroHeight(vec2 worldPos, float roughnessMask, float distanceFade) {
    vec2 microCoord = worldPos * 0.03; // Higher frequency

    // Multiple subtle detail frequencies - each with unique seed
    float detail1 = noiseSeeded(microCoord, 151.0) * 0.03;
    float detail2 = noiseSeeded(microCoord * 2.5, 157.0) * 0.015;
    float detail3 = simplexNoiseSeeded(microCoord * 4.0, 163.0) * 0.008;

    // Apply squared roughness mask (concentrates detail in rough areas)
    float maskSquared = roughnessMask * roughnessMask;

    // Apply distance fade (near = detailed, far = smoother)
    float micro = (detail1 + detail2 + detail3) * maskSquared * distanceFade;

    return micro;
}

// ============================================================================
// FEATURE PRIMITIVES - Sells the "human/real" look
// ============================================================================

float computeFeatures(vec2 worldPos, float baseHeight, float roughnessMask) {
    float features = 0.0;

    // 1) Path ribbon - lower + smoothed strip that curves
    float pathMask = getPathMask(worldPos);
    float pathDepth = -0.12; // Lower than surroundings
    features += pathMask * pathDepth;

    // 2) Raised bed edges - slight ridge border
    float moundMask = getMoundMask(worldPos);
    float bedEdge = getBedEdgeMask(worldPos);
    features += bedEdge * 0.04; // Small ridge at bed boundaries

    // 3) Drainage swale - shallow channel
    float swaleMask = getSwaleMask(worldPos);
    float swaleDepth = -0.06;
    features += swaleMask * swaleDepth * (1.0 - pathMask); // Don't dig swales in paths

    // 4) Raised bed tops - slightly convex
    features += moundMask * 0.05; // Beds are raised

    return features;
}

// ============================================================================
// THERMAL EROSION - Cheap but powerful realism
// Clamp steep slopes, smooth high-frequency, natural talus
// Uses seed: 167.0 for slope computation
// ============================================================================

float applyThermalErosion(vec2 worldPos, float height, float roughnessMask) {
    // Approximate slope from noise derivatives - seed 167.0
    float eps = 2.0;
    vec2 mesoCoord = worldPos * 0.008;
    float hCenter = fbmRotatedSeeded(mesoCoord, 5, 167.0);
    float hRight = fbmRotatedSeeded((worldPos + vec2(eps, 0.0)) * 0.008, 5, 167.0);
    float hUp = fbmRotatedSeeded((worldPos + vec2(0.0, eps)) * 0.008, 5, 167.0);

    float slopeX = abs(hRight - hCenter);
    float slopeZ = abs(hUp - hCenter);
    float slope = sqrt(slopeX * slopeX + slopeZ * slopeZ);

    // If slope exceeds talus threshold, reduce detail
    float talusThreshold = 0.15;
    float erosionFactor = smoothstep(talusThreshold, talusThreshold * 2.0, slope);

    // Reduce height in steep areas (material moved downhill)
    float erosion = erosionFactor * 0.02;

    // Also smooth in low areas (sediment fill feel)
    float basinMask = getBasinMask(worldPos);
    erosion += basinMask * 0.01;

    return -erosion;
}

// ============================================================================
// FLAT REGIONS - Gardens always have them
// Lawns, plateaus, compressed paths
// ============================================================================

float applyFlatRegions(float height, vec2 worldPos) {
    // Lawn mask for flat plateaus
    float lawnMask = getLawnMask(worldPos);

    // Target plane for flattening (average height in region)
    float targetHeight = 0.1; // Slightly above zero

    // Soft flatten operator: lerp toward target
    float flattenAlpha = lawnMask * 0.5; // 50% flatten in lawn areas
    float flattenedHeight = mix(height, targetHeight, flattenAlpha);

    // Path smoothing - paths should be flatter and compressed
    float pathMask = getPathMask(worldPos);
    float pathTargetHeight = -0.05; // Slightly below grade
    flattenedHeight = mix(flattenedHeight, pathTargetHeight, pathMask * 0.6);

    return flattenedHeight;
}

// ============================================================================
// DISTANCE FADE - Reduce detail with distance for wireframe realism
// ============================================================================

float computeDistanceFade(vec3 cameraPos, vec2 worldPos) {
    float dist = length(cameraPos.xz - worldPos);

    // Fade detail from near to far
    float nearDist = 50.0;
    float farDist = 400.0;

    return 1.0 - smoothstep(nearDist, farDist, dist);
}

// ============================================================================
// TERRAIN HEIGHT CALCULATION - Main function combining all layers
// ============================================================================

float calculateTerrainHeight(vec2 worldPos, vec3 cameraPos) {
    // Get control masks
    float roughnessMask = getRoughnessMask(worldPos);
    float distanceFade = computeDistanceFade(cameraPos, worldPos);
    float pathMask = getPathMask(worldPos);

    // LAYER A: Macro shape (dominates the look)
    float overallSlope;
    float macroHeight = computeMacroHeight(worldPos, overallSlope);

    // LAYER B: Meso shape (localized irregularity)
    float mesoHeight = computeMesoHeight(worldPos, roughnessMask);

    // LAYER C: Micro detail (small bumps) with distance fade
    float microHeight = computeMicroHeight(worldPos, roughnessMask, distanceFade);

    // LAYER D: Feature primitives (paths, beds, swales)
    float baseHeight = macroHeight + mesoHeight;
    float features = computeFeatures(worldPos, baseHeight, roughnessMask);

    // LAYER E: Natural imperfections (real-world terrain details)
    // Adds: pebbles/stones, micro-erosion, soil granularity, roots, footprints
    float naturalImperfections = computeNaturalImperfections(worldPos, distanceFade, pathMask);

    // Combine base layers
    float height = macroHeight + mesoHeight + microHeight + features + naturalImperfections;

    // Apply thermal erosion
    height += applyThermalErosion(worldPos, height, roughnessMask);

    // Apply flat regions (lawns, paths)
    height = applyFlatRegions(height, worldPos);

    // Apply distance fade to meso+micro (keep macro even at distance)
    // This makes far terrain smoother (like real LOD)
    float detailContrib = (mesoHeight + microHeight) * (1.0 - distanceFade) * 0.3;
    height -= detailContrib;

    return height;
}

// ============================================================================
// CONSTANTS
// ============================================================================

const float heightScale = 35.0;  // Max height range in world units

// ============================================================================
// MAIN
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

    // Extract camera position from view matrix (inverse of view matrix translation)
    vec3 cameraPos = -vec3(uMVP.viewMatrix[3][0], uMVP.viewMatrix[3][1], uMVP.viewMatrix[3][2]);
    // Correct extraction from view matrix
    mat3 viewRotation = mat3(uMVP.viewMatrix);
    cameraPos = -viewRotation * vec3(uMVP.viewMatrix[3]);

    // Calculate terrain height using layered system
    float height = calculateTerrainHeight(worldPos, cameraPos);

    position.y = height * heightScale;
    position.w = 1.0;

    outWorldPos = position.xyz;

    gl_Position = uMVP.projectionMatrix * uMVP.viewMatrix * uMVP.modelMatrix * position;
}
