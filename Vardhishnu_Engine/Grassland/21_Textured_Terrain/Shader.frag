#version 450 core
#extension GL_ARB_separate_shader_objects : enable

// ============================================================================
// REALISTIC PBR TERRAIN FRAGMENT SHADER
// Features:
//   - Multi-layer procedural textures (grass, rock, dirt, sand, snow)
//   - Advanced noise: Simplex, Ridged Multifractal, Voronoi, Turbulence
//   - Realistic material details:
//       * Grass: blade patterns, moss, clover patches
//       * Rock: stratification, weathering, lichen, quartz veins
//       * Dirt: pebbles, stones, roots, moisture variation
//       * Sand: wind ripples, shell debris, water pooling
//       * Snow: sastrugi drifts, ice crust, crystal sparkle
//   - Height-based depth blending for natural transitions
//   - Slope and height-based material blending
//   - Tri-planar mapping for steep surfaces
//   - Cook-Torrance BRDF with GGX distribution
//   - Multi-layer procedural detail normal mapping
// ============================================================================

// Inputs from tessellation evaluation shader
layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inTangent;
layout(location = 3) in float inBitangentSign;
layout(location = 4) in vec2 inUV;
layout(location = 5) in float inHeight;

layout(location = 0) out vec4 FragColor;

// Uniform buffer for matrices and camera data
layout(binding = 0) uniform mvpMatrix {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 color;
} uMVP;

// ============================================================================
// CONSTANTS
// ============================================================================

const float PI = 3.14159265359;
const float TERRAIN_SIZE = 800.0;
const float HEIGHT_SCALE = 35.0;

// Material layer height thresholds (normalized 0-1 height)
const float WATER_LEVEL = 0.05;
const float SAND_LEVEL = 0.12;
const float GRASS_LEVEL = 0.45;
const float ROCK_LEVEL = 0.70;
const float SNOW_LEVEL = 0.85;

// Slope thresholds (dot product with up vector)
const float CLIFF_SLOPE = 0.5;   // Below this = cliff/rock
const float STEEP_SLOPE = 0.7;  // Below this = mostly rock

// ============================================================================
// PROCEDURAL NOISE FUNCTIONS
// GPU-friendly hash-based noise for realistic terrain texturing
// ============================================================================

// Hash functions for procedural noise
vec3 hash3(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
             dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));
    return fract(sin(p) * 43758.5453123);
}

vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453123);
}

float hash1(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float hash1_3d(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453123);
}

// Better hash for higher quality noise (avoids sin artifacts)
vec2 hash2_hq(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

vec3 hash3_hq(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

// Simplex noise for smoother, more natural patterns
vec3 mod289_3(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289_2(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289_3(((x * 34.0) + 1.0) * x); }

float simplexNoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                        -0.577350269189626, 0.024390243902439);
    vec2 i = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289_2(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

// Ridged multifractal noise - great for rocky/mountainous details
float ridgedNoise(vec2 p, int octaves, float lacunarity, float gain) {
    float sum = 0.0;
    float freq = 1.0;
    float amp = 0.5;
    float prev = 1.0;
    for (int i = 0; i < octaves; i++) {
        float n = abs(simplexNoise(p * freq));
        n = 1.0 - n; // Ridge
        n = n * n;   // Sharpen
        sum += n * amp * prev;
        prev = n;
        freq *= lacunarity;
        amp *= gain;
    }
    return sum;
}

// Turbulence noise - sum of absolute values
float turbulence(vec2 p, int octaves) {
    float sum = 0.0;
    float freq = 1.0;
    float amp = 1.0;
    for (int i = 0; i < octaves; i++) {
        sum += abs(simplexNoise(p * freq)) * amp;
        freq *= 2.0;
        amp *= 0.5;
    }
    return sum;
}

// Billowed noise - inverse ridged for puffy/cloud patterns
float billowedNoise(vec2 p, int octaves) {
    float sum = 0.0;
    float freq = 1.0;
    float amp = 0.5;
    for (int i = 0; i < octaves; i++) {
        float n = abs(simplexNoise(p * freq));
        sum += n * amp;
        freq *= 2.0;
        amp *= 0.5;
    }
    return sum;
}

// Value noise with smooth interpolation (quintic curve for C2 continuity)
float valueNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    // Quintic interpolation for smoother derivatives
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    float a = hash1(i);
    float b = hash1(i + vec2(1.0, 0.0));
    float c = hash1(i + vec2(0.0, 1.0));
    float d = hash1(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// 3D value noise for tri-planar mapping
float valueNoise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    float n000 = hash1_3d(i);
    float n100 = hash1_3d(i + vec3(1, 0, 0));
    float n010 = hash1_3d(i + vec3(0, 1, 0));
    float n110 = hash1_3d(i + vec3(1, 1, 0));
    float n001 = hash1_3d(i + vec3(0, 0, 1));
    float n101 = hash1_3d(i + vec3(1, 0, 1));
    float n011 = hash1_3d(i + vec3(0, 1, 1));
    float n111 = hash1_3d(i + vec3(1, 1, 1));

    float n00 = mix(n000, n100, f.x);
    float n10 = mix(n010, n110, f.x);
    float n01 = mix(n001, n101, f.x);
    float n11 = mix(n011, n111, f.x);

    float n0 = mix(n00, n10, f.y);
    float n1 = mix(n01, n11, f.y);

    return mix(n0, n1, f.z);
}

// Gradient noise using simplex for base, returns 0-1 range
float gradientNoise(vec2 p) {
    return simplexNoise(p) * 0.5 + 0.5;
}

// Fractional Brownian Motion using simplex noise
float fbm(vec2 p, int octaves, float lacunarity, float gain) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    float maxValue = 0.0;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * (simplexNoise(p * frequency) * 0.5 + 0.5);
        maxValue += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }
    return value / maxValue;
}

// FBM with rotation to reduce grid artifacts
float fbmRotated(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));

    for (int i = 0; i < octaves; i++) {
        value += amplitude * (simplexNoise(p) * 0.5 + 0.5);
        p = rot * p * 2.0 + vec2(100.0);
        amplitude *= 0.5;
    }
    return value;
}

// Voronoi noise with multiple output modes
struct VoronoiResult {
    float dist1;      // Distance to closest cell
    float dist2;      // Distance to second closest
    vec2 cellCenter;  // Center of closest cell
    float cellId;     // Unique ID for closest cell
};

VoronoiResult voronoiExt(vec2 p) {
    vec2 n = floor(p);
    vec2 f = fract(p);

    float dist1 = 8.0;
    float dist2 = 8.0;
    vec2 cellCenter = vec2(0.0);
    float cellId = 0.0;

    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = hash2(n + g);
            vec2 r = g + o - f;
            float d = dot(r, r);

            if (d < dist1) {
                dist2 = dist1;
                dist1 = d;
                cellCenter = n + g + o;
                cellId = hash1(n + g);
            } else if (d < dist2) {
                dist2 = d;
            }
        }
    }
    VoronoiResult result;
    result.dist1 = sqrt(dist1);
    result.dist2 = sqrt(dist2);
    result.cellCenter = cellCenter;
    result.cellId = cellId;
    return result;
}

float voronoi(vec2 p) {
    return voronoiExt(p).dist1;
}

// Voronoi edge detection
float voronoiEdge(vec2 p) {
    VoronoiResult v = voronoiExt(p);
    return v.dist2 - v.dist1;
}

// Crackle pattern for dried earth/rocks
float crackle(vec2 p) {
    VoronoiResult v = voronoiExt(p);
    return smoothstep(0.0, 0.1, v.dist2 - v.dist1);
}

// Domain warped noise for organic variation
float warpedNoise(vec2 p, float warpStrength) {
    vec2 q = vec2(fbm(p, 4, 2.0, 0.5),
                  fbm(p + vec2(5.2, 1.3), 4, 2.0, 0.5));
    return fbm(p + warpStrength * q, 4, 2.0, 0.5);
}

// Swiss cheese noise for porous surfaces
float swissNoise(vec2 p, int octaves) {
    float sum = 0.0;
    float freq = 1.0;
    float amp = 1.0;
    float warp = 0.0;
    for (int i = 0; i < octaves; i++) {
        float n = simplexNoise(p * freq + warp);
        n = 1.0 - abs(n);
        sum += n * n * amp;
        warp = n * 2.0;
        freq *= 2.0;
        amp *= 0.5;
    }
    return sum;
}

// Erosion pattern simulation
float erosion(vec2 p, float detail) {
    float e = fbm(p * detail, 5, 2.2, 0.45);
    float r = ridgedNoise(p * detail * 0.5, 4, 2.0, 0.6);
    return mix(e, r, 0.5);
}

// ============================================================================
// PROCEDURAL TEXTURE FUNCTIONS
// Generate realistic terrain materials procedurally
// ============================================================================

// Procedural grass blade pattern
float grassBlades(vec2 uv) {
    // Directional noise for blade orientation
    float angle = fbm(uv * 5.0, 2, 2.0, 0.5) * 6.28;
    vec2 dir = vec2(cos(angle), sin(angle));

    // Create blade-like streaks
    float bladeFreq = 150.0;
    float blade1 = sin(dot(uv * bladeFreq, dir) + fbm(uv * 80.0, 2, 2.0, 0.5) * 3.0);
    float blade2 = sin(dot(uv * bladeFreq * 1.3, dir * 1.1) + fbm(uv * 90.0, 2, 2.0, 0.5) * 2.5);
    float blade3 = sin(dot(uv * bladeFreq * 0.7, dir * 0.9) + fbm(uv * 70.0, 2, 2.0, 0.5) * 3.5);

    float blades = (blade1 * 0.5 + 0.5) * 0.4 + (blade2 * 0.5 + 0.5) * 0.35 + (blade3 * 0.5 + 0.5) * 0.25;
    return blades;
}

// Procedural clover/small plant patches
float cloverPatches(vec2 uv) {
    VoronoiResult v = voronoiExt(uv * 25.0);
    float cloverPatch = smoothstep(0.15, 0.0, v.dist1);
    float patchVariation = hash1(v.cellCenter);
    return cloverPatch * step(0.7, patchVariation);
}

// Procedural moss pattern
float mossPattern(vec2 uv) {
    float moss = swissNoise(uv * 30.0, 4);
    float mossClusters = voronoi(uv * 8.0);
    moss *= smoothstep(0.6, 0.3, mossClusters);
    return moss;
}

// Procedural grass texture - enhanced with realistic detail
vec3 grassAlbedo(vec2 uv, float variation) {
    // Multi-octave base pattern
    float baseNoise = fbmRotated(uv * 40.0, 5);
    float detailNoise = fbm(uv * 200.0, 4, 2.0, 0.5);
    float microDetail = simplexNoise(uv * 500.0) * 0.5 + 0.5;

    // Blade pattern for texture
    float blades = grassBlades(uv);

    // Enhanced green color palette (based on real grass photography)
    vec3 grassDeep = vec3(0.08, 0.22, 0.04);      // Deep shadow green
    vec3 grassDark = vec3(0.12, 0.32, 0.06);      // Dark green
    vec3 grassMid = vec3(0.22, 0.45, 0.10);       // Medium green
    vec3 grassLight = vec3(0.35, 0.58, 0.15);     // Light green
    vec3 grassYellow = vec3(0.50, 0.55, 0.18);    // Yellow-green highlights
    vec3 grassDry = vec3(0.55, 0.50, 0.25);       // Dry/dead grass
    vec3 mossGreen = vec3(0.15, 0.35, 0.12);      // Moss color
    vec3 cloverGreen = vec3(0.10, 0.40, 0.15);    // Clover patches

    // Build up color in layers
    vec3 grassColor = mix(grassDark, grassMid, baseNoise);

    // Add blade variation
    grassColor = mix(grassColor, grassLight, blades * 0.4);

    // Yellow tips/highlights
    float yellowHighlight = smoothstep(0.7, 0.9, blades * detailNoise);
    grassColor = mix(grassColor, grassYellow, yellowHighlight * 0.3);

    // Shadows between blades
    float shadowDetail = smoothstep(0.3, 0.0, blades);
    grassColor = mix(grassColor, grassDeep, shadowDetail * 0.4);

    // Moss patches in shaded areas
    float moss = mossPattern(uv);
    float mossAmount = moss * (1.0 - variation) * smoothstep(0.6, 0.3, baseNoise);
    grassColor = mix(grassColor, mossGreen, mossAmount * 0.5);

    // Clover/small plant patches
    float clover = cloverPatches(uv);
    grassColor = mix(grassColor, cloverGreen, clover * 0.6);

    // Dry patches variation
    float dryPatch = fbm(uv * 8.0, 3, 2.0, 0.5);
    dryPatch = smoothstep(0.55, 0.75, dryPatch);
    grassColor = mix(grassColor, grassDry, dryPatch * variation * 0.6);

    // Micro-scale detail (individual blade tips)
    grassColor *= 0.88 + 0.24 * microDetail;

    // Subtle color temperature variation
    float warmCool = fbm(uv * 3.0, 2, 2.0, 0.5);
    grassColor = mix(grassColor, grassColor * vec3(1.05, 1.0, 0.95), warmCool * 0.15);

    return grassColor;
}

// Procedural sedimentary stratification layers
float stratificationLayers(vec3 worldPos, vec2 uv) {
    // Multiple layer frequencies for realistic stratification
    float largeLayer = sin(worldPos.y * 2.0 + fbm(uv * 5.0, 2, 2.0, 0.5) * 2.0) * 0.5 + 0.5;
    float medLayer = sin(worldPos.y * 8.0 + fbm(uv * 10.0, 2, 2.0, 0.5) * 1.5) * 0.5 + 0.5;
    float fineLayer = sin(worldPos.y * 25.0 + fbm(uv * 20.0, 2, 2.0, 0.5)) * 0.5 + 0.5;

    return largeLayer * 0.5 + medLayer * 0.3 + fineLayer * 0.2;
}

// Weathering and erosion patterns
float weatheringPattern(vec2 uv) {
    float erosion1 = ridgedNoise(uv * 20.0, 4, 2.0, 0.5);
    float erosion2 = turbulence(uv * 40.0, 4);
    float cracks = 1.0 - crackle(uv * 15.0);

    return mix(erosion1, erosion2, 0.5) * 0.7 + cracks * 0.3;
}

// Lichen/moss growth on rocks
vec3 lichenPattern(vec2 uv, vec3 baseColor) {
    VoronoiResult v = voronoiExt(uv * 12.0);
    float lichenMask = smoothstep(0.25, 0.0, v.dist1);
    lichenMask *= smoothstep(0.5, 0.8, hash1(v.cellCenter)); // Only some cells

    // Multiple lichen colors
    vec3 lichenGreen = vec3(0.25, 0.35, 0.18);
    vec3 lichenYellow = vec3(0.55, 0.50, 0.20);
    vec3 lichenOrange = vec3(0.60, 0.40, 0.15);
    vec3 lichenGray = vec3(0.50, 0.52, 0.48);

    float colorSelect = hash1(v.cellCenter + vec2(0.5));
    vec3 lichenColor = lichenGreen;
    if (colorSelect > 0.75) lichenColor = lichenYellow;
    else if (colorSelect > 0.5) lichenColor = lichenOrange;
    else if (colorSelect > 0.25) lichenColor = lichenGray;

    return mix(baseColor, lichenColor, lichenMask * 0.8);
}

// Procedural rock texture - enhanced with realistic geological detail
vec3 rockAlbedo(vec2 uv, vec3 worldPos, float variation) {
    // Multi-scale base noise
    float baseNoise = warpedNoise(uv * 12.0, 0.6);
    float mediumNoise = fbmRotated(uv * 40.0, 4);
    float detailNoise = fbm(uv * 120.0, 4, 2.0, 0.5);
    float microDetail = simplexNoise(uv * 400.0) * 0.5 + 0.5;

    // Fracture and crack patterns
    float fractures = voronoiEdge(uv * 8.0);
    float microCracks = voronoiEdge(uv * 40.0);

    // Weathering
    float weathering = weatheringPattern(uv);

    // Stratification
    float strata = stratificationLayers(worldPos, uv);

    // Enhanced rock color palette (based on real geological samples)
    vec3 rockBlack = vec3(0.12, 0.11, 0.10);       // Deep crevices
    vec3 rockDarkGray = vec3(0.22, 0.21, 0.20);    // Dark granite
    vec3 rockMidGray = vec3(0.42, 0.40, 0.38);     // Medium stone
    vec3 rockLightGray = vec3(0.58, 0.56, 0.54);   // Light surface
    vec3 rockWarm = vec3(0.50, 0.42, 0.35);        // Sandstone tones
    vec3 rockCool = vec3(0.38, 0.42, 0.45);        // Bluish slate
    vec3 rockBrown = vec3(0.40, 0.32, 0.25);       // Iron-stained
    vec3 rockOxide = vec3(0.55, 0.35, 0.25);       // Rust/oxidation
    vec3 rockQuartz = vec3(0.72, 0.70, 0.68);      // Quartz veins

    // Build base color from multiple layers
    vec3 rockColor = mix(rockDarkGray, rockMidGray, baseNoise);

    // Add stratification color banding
    vec3 strataColor = mix(rockMidGray, rockWarm, strata);
    rockColor = mix(rockColor, strataColor, 0.4);

    // Cool/warm color variation based on position
    float tempVar = fbm(worldPos.xz * 0.01, 2, 2.0, 0.5);
    rockColor = mix(rockColor, rockCool, tempVar * 0.25);
    rockColor = mix(rockColor, rockWarm, (1.0 - tempVar) * 0.2);

    // Fracture darkening (cracks are darker)
    float crackDark = smoothstep(0.15, 0.0, fractures);
    rockColor = mix(rockColor, rockBlack, crackDark * 0.6);

    // Micro-crack surface detail
    float microCrackDetail = smoothstep(0.1, 0.0, microCracks);
    rockColor = mix(rockColor, rockDarkGray, microCrackDetail * 0.3);

    // Weathering highlights (exposed surfaces are lighter)
    rockColor = mix(rockColor, rockLightGray, weathering * 0.35);

    // Iron oxide staining
    float oxideMask = fbm(uv * 15.0 + vec2(worldPos.y * 0.1), 3, 2.0, 0.5);
    oxideMask = smoothstep(0.55, 0.75, oxideMask);
    rockColor = mix(rockColor, rockOxide, oxideMask * variation * 0.4);

    // Quartz vein intrusions
    float quartzVein = ridgedNoise(uv * 6.0 + vec2(worldPos.y * 0.05), 3, 2.0, 0.5);
    quartzVein = smoothstep(0.78, 0.85, quartzVein);
    rockColor = mix(rockColor, rockQuartz, quartzVein * 0.5);

    // Add lichen on exposed surfaces
    rockColor = lichenPattern(uv, rockColor);

    // Fine surface detail and texture
    rockColor *= 0.82 + 0.36 * detailNoise;

    // Micro-grain detail
    rockColor *= 0.94 + 0.12 * microDetail;

    // Subtle color variation for realism
    vec3 tint = vec3(
        1.0 + (hash1(floor(uv * 10.0)) - 0.5) * 0.08,
        1.0 + (hash1(floor(uv * 10.0) + vec2(1.0)) - 0.5) * 0.06,
        1.0 + (hash1(floor(uv * 10.0) + vec2(2.0)) - 0.5) * 0.08
    );
    rockColor *= tint;

    return rockColor;
}

// Procedural pebble pattern
float pebblePattern(vec2 uv, out float pebbleId) {
    VoronoiResult v = voronoiExt(uv * 60.0);
    float pebble = smoothstep(0.12, 0.02, v.dist1);
    // Only some cells are pebbles
    float isPebble = step(0.6, hash1(v.cellCenter));
    pebbleId = v.cellId;
    return pebble * isPebble;
}

// Larger stones scattered in dirt
float stonePattern(vec2 uv, out float stoneId) {
    VoronoiResult v = voronoiExt(uv * 15.0);
    float stone = smoothstep(0.2, 0.05, v.dist1);
    float isStone = step(0.85, hash1(v.cellCenter));
    stoneId = v.cellId;
    return stone * isStone;
}

// Root/organic debris pattern
float rootPattern(vec2 uv) {
    // Winding root-like lines using ridged noise
    float root1 = ridgedNoise(uv * 8.0, 3, 2.0, 0.5);
    float root2 = ridgedNoise(uv * 12.0 + vec2(5.0), 3, 2.0, 0.5);

    root1 = smoothstep(0.7, 0.85, root1);
    root2 = smoothstep(0.75, 0.88, root2);

    return max(root1, root2 * 0.7);
}

// Footprint/disturbed soil pattern
float disturbedSoil(vec2 uv) {
    float disturb = fbm(uv * 20.0, 4, 2.0, 0.5);
    float clumps = voronoi(uv * 35.0);
    return disturb * (1.0 - clumps * 0.5);
}

// Procedural dirt texture - enhanced with realistic organic detail
vec3 dirtAlbedo(vec2 uv, float variation) {
    // Multi-scale noise layers
    float baseNoise = warpedNoise(uv * 25.0, 0.4);
    float mediumNoise = fbmRotated(uv * 60.0, 4);
    float detailNoise = fbm(uv * 150.0, 4, 2.0, 0.5);
    float microDetail = simplexNoise(uv * 400.0) * 0.5 + 0.5;

    // Clump/aggregate pattern
    float clumps = voronoi(uv * 40.0);
    float fineClumps = voronoi(uv * 100.0);

    // Moisture variation
    float moisture = fbm(uv * 10.0, 3, 2.0, 0.5);
    moisture = smoothstep(0.3, 0.7, moisture);

    // Pattern elements
    float pebbleId;
    float pebbles = pebblePattern(uv, pebbleId);
    float stoneId;
    float stones = stonePattern(uv, stoneId);
    float roots = rootPattern(uv);
    float disturbed = disturbedSoil(uv);

    // Enhanced dirt color palette (based on real soil samples)
    vec3 dirtBlack = vec3(0.10, 0.08, 0.06);       // Very dark humus
    vec3 dirtDark = vec3(0.20, 0.15, 0.10);        // Dark rich soil
    vec3 dirtBrown = vec3(0.35, 0.25, 0.17);       // Medium brown
    vec3 dirtTan = vec3(0.50, 0.40, 0.28);         // Tan/light brown
    vec3 dirtRed = vec3(0.45, 0.28, 0.18);         // Reddish clay
    vec3 dirtYellow = vec3(0.55, 0.48, 0.30);      // Sandy/yellow
    vec3 dirtWet = vec3(0.15, 0.12, 0.08);         // Wet dark dirt
    vec3 dirtGray = vec3(0.40, 0.38, 0.35);         // Gray mineral
    vec3 pebbleColor = vec3(0.45, 0.42, 0.40);     // Pebble gray
    vec3 rootColor = vec3(0.25, 0.18, 0.12);       // Dead root brown
    vec3 stoneColor = vec3(0.50, 0.48, 0.45);      // Stone gray

    // Build base dirt color
    vec3 dirtColor = mix(dirtDark, dirtBrown, baseNoise);

    // Add clay/iron content variation
    float clayContent = fbm(uv * 8.0, 2, 2.0, 0.5);
    dirtColor = mix(dirtColor, dirtRed, clayContent * 0.35);

    // Sandy patches
    float sandyPatch = smoothstep(0.6, 0.8, fbm(uv * 12.0 + vec2(10.0), 3, 2.0, 0.5));
    dirtColor = mix(dirtColor, dirtYellow, sandyPatch * 0.4);

    // Clump shading - darker in crevices
    float clumpShade = smoothstep(0.3, 0.0, clumps);
    dirtColor = mix(dirtColor, dirtBlack, clumpShade * 0.4);

    // Clump highlights - lighter on tops
    float clumpHighlight = smoothstep(0.5, 0.7, clumps);
    dirtColor = mix(dirtColor, dirtTan, clumpHighlight * 0.3);

    // Add pebbles
    vec3 thisPebbleColor = mix(pebbleColor, pebbleColor * vec3(0.9, 0.95, 1.0), hash1(vec2(pebbleId)));
    thisPebbleColor *= 0.8 + 0.4 * hash1(vec2(pebbleId) + vec2(1.0));
    dirtColor = mix(dirtColor, thisPebbleColor, pebbles);

    // Add larger stones
    vec3 thisStoneColor = mix(stoneColor, stoneColor * vec3(1.0, 0.95, 0.9), hash1(vec2(stoneId)));
    thisStoneColor *= 0.85 + 0.3 * hash1(vec2(stoneId) + vec2(1.0));
    dirtColor = mix(dirtColor, thisStoneColor, stones * 0.8);

    // Add root debris
    dirtColor = mix(dirtColor, rootColor, roots * 0.7);

    // Moisture darkening
    float wetAmount = moisture * variation;
    dirtColor = mix(dirtColor, dirtWet, wetAmount * 0.5);

    // Surface texture and detail
    dirtColor *= 0.85 + 0.3 * mediumNoise;
    dirtColor *= 0.88 + 0.24 * detailNoise;

    // Micro grain texture
    dirtColor *= 0.94 + 0.12 * microDetail;

    // Organic matter dark specks
    float organicSpecks = step(0.92, simplexNoise(uv * 300.0) * 0.5 + 0.5);
    dirtColor = mix(dirtColor, dirtBlack, organicSpecks * 0.4);

    return dirtColor;
}

// Wind-driven sand ripple pattern
float sandRipples(vec2 uv, float windAngle) {
    // Rotate UV based on wind direction
    float c = cos(windAngle);
    float s = sin(windAngle);
    vec2 rotUV = vec2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);

    // Multi-frequency ripples with wavelength variation
    float largeDune = sin(rotUV.x * 15.0 + fbm(uv * 3.0, 2, 2.0, 0.5) * 4.0) * 0.5 + 0.5;
    float medRipple = sin(rotUV.x * 50.0 + fbm(uv * 10.0, 2, 2.0, 0.5) * 3.0) * 0.5 + 0.5;
    float fineRipple = sin(rotUV.x * 120.0 + simplexNoise(uv * 30.0) * 2.0) * 0.5 + 0.5;

    // Asymmetric ripple shape (steep lee, gentle windward)
    medRipple = pow(medRipple, 0.7);
    fineRipple = pow(fineRipple, 0.6);

    return largeDune * 0.3 + medRipple * 0.45 + fineRipple * 0.25;
}

// Sand grain sparkle effect
float sandSparkle(vec2 uv, float viewDot) {
    float grain = simplexNoise(uv * 800.0);
    float sparkle = smoothstep(0.85, 1.0, grain) * viewDot;
    return sparkle;
}

// Shell/debris fragments in sand
float shellDebris(vec2 uv, out float shellId) {
    VoronoiResult v = voronoiExt(uv * 80.0);
    float shell = smoothstep(0.08, 0.02, v.dist1);
    float isShell = step(0.92, hash1(v.cellCenter));
    shellId = v.cellId;
    return shell * isShell;
}

// Wet sand water pooling
float waterPooling(vec2 uv) {
    float pool = fbm(uv * 15.0, 4, 2.0, 0.5);
    pool = smoothstep(0.45, 0.55, pool);
    return pool;
}

// Procedural sand texture - enhanced with realistic beach/desert detail
vec3 sandAlbedo(vec2 uv, float variation) {
    // Multi-scale noise
    float baseNoise = warpedNoise(uv * 30.0, 0.3);
    float mediumNoise = fbmRotated(uv * 80.0, 4);
    float detailNoise = fbm(uv * 200.0, 3, 2.0, 0.5);
    float microGrain = simplexNoise(uv * 600.0) * 0.5 + 0.5;

    // Wind direction varies across terrain
    float windAngle = 0.3 + fbm(uv * 2.0, 2, 2.0, 0.5) * 0.4;
    float ripples = sandRipples(uv, windAngle);

    // Shell debris
    float shellId;
    float shells = shellDebris(uv, shellId);

    // Water pooling (for wet areas)
    float pools = waterPooling(uv);

    // Enhanced sand color palette (based on real beach/desert sand)
    vec3 sandPale = vec3(0.92, 0.88, 0.78);        // Pale beach sand
    vec3 sandLight = vec3(0.88, 0.80, 0.65);       // Light golden
    vec3 sandGold = vec3(0.82, 0.72, 0.55);        // Golden sand
    vec3 sandDark = vec3(0.70, 0.60, 0.45);        // Darker sand
    vec3 sandOrange = vec3(0.85, 0.65, 0.45);      // Orange/red sand
    vec3 sandGray = vec3(0.65, 0.62, 0.58);        // Mineral gray
    vec3 sandWet = vec3(0.50, 0.45, 0.35);         // Wet sand
    vec3 sandVeryWet = vec3(0.40, 0.38, 0.32);     // Saturated sand
    vec3 shellWhite = vec3(0.90, 0.88, 0.85);      // Shell fragments
    vec3 shellPink = vec3(0.88, 0.82, 0.80);       // Pink shells

    // Build base sand color
    vec3 sandColor = mix(sandGold, sandLight, baseNoise);

    // Regional color variation
    float colorRegion = fbm(uv * 5.0, 2, 2.0, 0.5);
    sandColor = mix(sandColor, sandOrange, colorRegion * 0.2);

    // Mineral content variation (darker streaks)
    float mineralStreak = fbm(uv * 20.0 + vec2(5.0), 3, 2.0, 0.5);
    mineralStreak = smoothstep(0.55, 0.7, mineralStreak);
    sandColor = mix(sandColor, sandGray, mineralStreak * 0.25);

    // Ripple shading - ridges are lighter, troughs darker
    float rippleShadow = 1.0 - ripples;
    sandColor = mix(sandColor, sandDark, rippleShadow * 0.25);
    sandColor = mix(sandColor, sandPale, ripples * 0.15);

    // Wind-blown lighter surface layer
    float surfaceLayer = smoothstep(0.4, 0.6, ripples);
    sandColor = mix(sandColor, sandPale, surfaceLayer * 0.1);

    // Shell fragments
    vec3 thisShellColor = mix(shellWhite, shellPink, hash1(vec2(shellId)));
    thisShellColor *= 0.9 + 0.2 * hash1(vec2(shellId) + vec2(1.0));
    sandColor = mix(sandColor, thisShellColor, shells);

    // Wet sand variation
    float wetMask = variation;

    // Water pooling in low areas when wet
    float poolMask = pools * wetMask;
    sandColor = mix(sandColor, sandVeryWet, poolMask * 0.6);

    // General wet darkening
    sandColor = mix(sandColor, sandWet, wetMask * (1.0 - pools) * 0.5);

    // Wet sand has slight sheen (subtle reflectivity approximation)
    float wetSheen = wetMask * ripples * 0.1;
    sandColor += vec3(wetSheen);

    // Surface grain detail
    sandColor *= 0.92 + 0.16 * detailNoise;

    // Micro-grain texture
    sandColor *= 0.96 + 0.08 * microGrain;

    // Individual grain color variation (subtle)
    float grainColorVar = simplexNoise(uv * 400.0) * 0.03;
    sandColor += vec3(grainColorVar, grainColorVar * 0.8, grainColorVar * 0.5);

    // Sparkle effect (simulates quartz grains catching light)
    float sparkle = sandSparkle(uv, 0.5);
    sandColor += vec3(sparkle * 0.15);

    return sandColor;
}

// Wind-carved sastrugi pattern (snow dune ridges)
float sastrugi(vec2 uv, float windAngle) {
    float c = cos(windAngle);
    float s = sin(windAngle);
    vec2 rotUV = vec2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);

    // Elongated ridges in wind direction
    float ridge1 = ridgedNoise(vec2(rotUV.x * 8.0, rotUV.y * 2.0), 3, 2.0, 0.5);
    float ridge2 = ridgedNoise(vec2(rotUV.x * 15.0, rotUV.y * 4.0) + vec2(10.0), 3, 2.0, 0.5);

    return ridge1 * 0.6 + ridge2 * 0.4;
}

// Ice crust patches
float iceCrust(vec2 uv) {
    VoronoiResult v = voronoiExt(uv * 6.0);
    float crust = smoothstep(0.35, 0.15, v.dist1);
    float isCrust = smoothstep(0.4, 0.7, hash1(v.cellCenter));
    return crust * isCrust;
}

// Wind-packed snow pattern
float windPack(vec2 uv) {
    float pack = warpedNoise(uv * 12.0, 0.5);
    pack = smoothstep(0.3, 0.7, pack);
    return pack;
}

// Snow crystal sparkle (glitter effect)
float snowSparkle(vec2 uv) {
    // Multiple sparkle frequencies
    float spark1 = simplexNoise(uv * 500.0);
    float spark2 = simplexNoise(uv * 800.0 + vec2(100.0));
    float spark3 = simplexNoise(uv * 1200.0 + vec2(200.0));

    float sparkle = smoothstep(0.88, 1.0, spark1) +
                    smoothstep(0.90, 1.0, spark2) * 0.7 +
                    smoothstep(0.92, 1.0, spark3) * 0.5;

    return sparkle;
}

// Footprint/track disturbance pattern
float snowDisturbance(vec2 uv) {
    VoronoiResult v = voronoiExt(uv * 20.0);
    float disturb = smoothstep(0.15, 0.05, v.dist1);
    float isDisturbed = step(0.9, hash1(v.cellCenter));
    return disturb * isDisturbed;
}

// Procedural snow texture - enhanced with realistic alpine/arctic detail
vec3 snowAlbedo(vec2 uv, float variation) {
    // Multi-scale noise
    float baseNoise = warpedNoise(uv * 15.0, 0.4);
    float mediumNoise = fbmRotated(uv * 50.0, 4);
    float detailNoise = fbm(uv * 150.0, 3, 2.0, 0.5);
    float microNoise = simplexNoise(uv * 400.0) * 0.5 + 0.5;

    // Wind direction varies slightly
    float windAngle = 0.5 + fbm(uv * 1.5, 2, 2.0, 0.5) * 0.3;

    // Pattern elements
    float drifts = sastrugi(uv, windAngle);
    float ice = iceCrust(uv);
    float packed = windPack(uv);
    float sparkle = snowSparkle(uv);
    float disturbed = snowDisturbance(uv);

    // Enhanced snow color palette
    vec3 snowBright = vec3(0.98, 0.99, 1.0);       // Fresh powder
    vec3 snowWhite = vec3(0.94, 0.96, 0.98);       // Clean snow
    vec3 snowCream = vec3(0.95, 0.94, 0.92);       // Warm sunlit
    vec3 snowBlue = vec3(0.85, 0.90, 0.98);        // Blue shadow
    vec3 snowDeepBlue = vec3(0.70, 0.78, 0.92);    // Deep shadow
    vec3 snowGray = vec3(0.82, 0.84, 0.86);        // Overcast
    vec3 snowIce = vec3(0.88, 0.94, 0.98);         // Ice patches
    vec3 snowIceBlue = vec3(0.78, 0.88, 0.96);     // Blue ice
    vec3 snowDirty = vec3(0.78, 0.76, 0.72);       // Dirty snow
    vec3 snowPink = vec3(0.95, 0.90, 0.92);        // Algae-tinted
    vec3 snowPacked = vec3(0.88, 0.90, 0.94);      // Wind-packed

    // Build base snow color
    vec3 snowColor = mix(snowBlue, snowWhite, baseNoise);

    // Drift shading - ridges catch light, troughs in shadow
    float driftLight = drifts;
    float driftShadow = 1.0 - drifts;
    snowColor = mix(snowColor, snowBright, driftLight * 0.2);
    snowColor = mix(snowColor, snowDeepBlue, driftShadow * 0.3);

    // Wind-packed snow areas (firn)
    snowColor = mix(snowColor, snowPacked, packed * 0.25);

    // Ice crust patches
    vec3 iceColor = mix(snowIce, snowIceBlue, fbm(uv * 20.0, 2, 2.0, 0.5));
    snowColor = mix(snowColor, iceColor, ice * 0.7);

    // Surface texture variation
    float surfaceVar = fbm(uv * 30.0, 3, 2.0, 0.5);
    snowColor = mix(snowColor, snowCream, surfaceVar * 0.08);

    // Disturbed snow (footprints, etc.) - shows underlayers
    snowColor = mix(snowColor, snowGray, disturbed * 0.3);

    // Dirty snow at lower elevations (transition zones)
    float dirtyMask = variation * smoothstep(0.3, 0.0, baseNoise);
    snowColor = mix(snowColor, snowDirty, dirtyMask * 0.5);

    // Occasional pink algae patches (watermelon snow)
    float algaePatch = fbm(uv * 8.0 + vec2(50.0), 3, 2.0, 0.5);
    algaePatch = smoothstep(0.7, 0.85, algaePatch) * variation;
    snowColor = mix(snowColor, snowPink, algaePatch * 0.15);

    // Surface grain detail
    snowColor *= 0.96 + 0.08 * mediumNoise;
    snowColor *= 0.98 + 0.04 * detailNoise;

    // Micro crystal texture
    snowColor *= 0.98 + 0.04 * microNoise;

    // Sparkle effect (ice crystals catching light)
    snowColor += vec3(sparkle * 0.12);

    // Add very subtle blue ambient in shadows
    float shadowAmount = 1.0 - baseNoise;
    snowColor = mix(snowColor, snowColor + vec3(-0.02, 0.0, 0.03), shadowAmount * 0.3);

    return snowColor;
}

// ============================================================================
// PROCEDURAL NORMAL MAP GENERATION
// Compute detail normals from noise derivatives
// ============================================================================

// Basic detail normal from FBM
vec3 computeDetailNormal(vec2 uv, float scale, float strength) {
    float eps = 0.001;

    float h0 = fbm(uv * scale, 4, 2.0, 0.5);
    float hx = fbm((uv + vec2(eps, 0.0)) * scale, 4, 2.0, 0.5);
    float hy = fbm((uv + vec2(0.0, eps)) * scale, 4, 2.0, 0.5);

    float dx = (hx - h0) / eps;
    float dy = (hy - h0) / eps;

    vec3 detailNormal = normalize(vec3(-dx * strength, 1.0, -dy * strength));
    return detailNormal;
}

// Multi-octave detail normal with different noise types
vec3 computeRichDetailNormal(vec2 uv, float baseScale, float strength, int noiseType) {
    float eps = 0.0008;
    float h0, hx, hy;

    if (noiseType == 0) {
        // Standard FBM - good for general terrain
        h0 = fbmRotated(uv * baseScale, 5);
        hx = fbmRotated((uv + vec2(eps, 0.0)) * baseScale, 5);
        hy = fbmRotated((uv + vec2(0.0, eps)) * baseScale, 5);
    }
    else if (noiseType == 1) {
        // Ridged for rocky surfaces
        h0 = ridgedNoise(uv * baseScale, 4, 2.0, 0.5);
        hx = ridgedNoise((uv + vec2(eps, 0.0)) * baseScale, 4, 2.0, 0.5);
        hy = ridgedNoise((uv + vec2(0.0, eps)) * baseScale, 4, 2.0, 0.5);
    }
    else if (noiseType == 2) {
        // Voronoi for cellular/cracked surfaces
        h0 = voronoi(uv * baseScale);
        hx = voronoi((uv + vec2(eps, 0.0)) * baseScale);
        hy = voronoi((uv + vec2(0.0, eps)) * baseScale);
    }
    else {
        // Turbulence for rough surfaces
        h0 = turbulence(uv * baseScale, 4);
        hx = turbulence((uv + vec2(eps, 0.0)) * baseScale, 4);
        hy = turbulence((uv + vec2(0.0, eps)) * baseScale, 4);
    }

    float dx = (hx - h0) / eps;
    float dy = (hy - h0) / eps;

    return normalize(vec3(-dx * strength, 1.0, -dy * strength));
}

// Combine base normal with detail normal in tangent space
vec3 blendNormals(vec3 baseNormal, vec3 detailNormal) {
    // Reoriented Normal Mapping blend
    vec3 t = baseNormal + vec3(0.0, 0.0, 1.0);
    vec3 u = detailNormal * vec3(-1.0, -1.0, 1.0);
    return normalize(t * dot(t, u) - u * t.z);
}

// Multi-layer detail normal combining different scales
vec3 computeLayeredDetailNormal(vec2 uv, float strength) {
    vec3 largeDetail = computeDetailNormal(uv, 30.0, strength * 0.5);
    vec3 medDetail = computeDetailNormal(uv, 80.0, strength * 0.7);
    vec3 fineDetail = computeDetailNormal(uv, 200.0, strength);

    // Blend normals progressively
    vec3 combined = blendNormals(largeDetail, medDetail);
    combined = blendNormals(combined, fineDetail);
    return combined;
}

// ============================================================================
// TRI-PLANAR MAPPING
// Avoid texture stretching on steep surfaces
// ============================================================================

vec3 triplanarSample(vec3 worldPos, vec3 normal, float scale, int materialType) {
    // Calculate blending weights based on normal
    vec3 blend = abs(normal);
    blend = pow(blend, vec3(4.0)); // Sharpen blend
    blend /= (blend.x + blend.y + blend.z + 0.0001);

    // Sample on each axis
    vec2 uvX = worldPos.zy * scale;
    vec2 uvY = worldPos.xz * scale;
    vec2 uvZ = worldPos.xy * scale;

    vec3 colorX, colorY, colorZ;
    float variation = fbm(worldPos.xz * 0.01, 2, 2.0, 0.5);

    // Sample based on material type
    if (materialType == 0) { // Grass
        colorX = grassAlbedo(uvX, variation);
        colorY = grassAlbedo(uvY, variation);
        colorZ = grassAlbedo(uvZ, variation);
    } else if (materialType == 1) { // Rock
        colorX = rockAlbedo(uvX, worldPos, variation);
        colorY = rockAlbedo(uvY, worldPos, variation);
        colorZ = rockAlbedo(uvZ, worldPos, variation);
    } else if (materialType == 2) { // Dirt
        colorX = dirtAlbedo(uvX, variation);
        colorY = dirtAlbedo(uvY, variation);
        colorZ = dirtAlbedo(uvZ, variation);
    } else if (materialType == 3) { // Sand
        colorX = sandAlbedo(uvX, variation);
        colorY = sandAlbedo(uvY, variation);
        colorZ = sandAlbedo(uvZ, variation);
    } else { // Snow
        colorX = snowAlbedo(uvX, variation);
        colorY = snowAlbedo(uvY, variation);
        colorZ = snowAlbedo(uvZ, variation);
    }

    // Blend based on normal direction
    return colorX * blend.x + colorY * blend.y + colorZ * blend.z;
}

// ============================================================================
// PBR MATERIAL STRUCTURE
// ============================================================================

struct PBRMaterial {
    vec3 albedo;
    float roughness;
    float metallic;
    float ao;
    vec3 normal;
};

// Get material properties for each terrain type
PBRMaterial getGrassMaterial(vec2 uv, vec3 worldPos, vec3 geometryNormal) {
    PBRMaterial mat;
    float variation = fbm(worldPos.xz * 0.01, 2, 2.0, 0.5);

    mat.albedo = grassAlbedo(uv, variation);

    // Grass roughness varies with blade direction and moisture
    float bladeRoughness = fbmRotated(uv * 100.0, 3);
    mat.roughness = 0.70 + 0.25 * bladeRoughness;

    mat.metallic = 0.0;

    // AO based on blade density patterns
    float densityAO = fbm(uv * 40.0, 3, 2.0, 0.5);
    mat.ao = 0.80 + 0.20 * densityAO;

    // Multi-layer detail normal for grass texture
    vec3 baseNormal = computeDetailNormal(uv, 60.0, 0.12);
    vec3 fineNormal = computeDetailNormal(uv, 150.0, 0.08);
    mat.normal = blendNormals(baseNormal, fineNormal);

    return mat;
}

PBRMaterial getRockMaterial(vec2 uv, vec3 worldPos, vec3 geometryNormal) {
    PBRMaterial mat;
    float variation = fbm(worldPos.xz * 0.01, 2, 2.0, 0.5);

    // Use tri-planar for rock on steep surfaces
    float steepness = 1.0 - abs(dot(geometryNormal, vec3(0.0, 1.0, 0.0)));
    if (steepness > 0.5) {
        mat.albedo = triplanarSample(worldPos, geometryNormal, 0.05, 1);
    } else {
        mat.albedo = rockAlbedo(uv, worldPos, variation);
    }

    // Rock roughness varies with weathering and mineral content
    float weatherRoughness = voronoi(uv * 25.0);
    float mineralRoughness = fbm(uv * 50.0, 3, 2.0, 0.5);
    mat.roughness = 0.50 + 0.40 * mix(weatherRoughness, mineralRoughness, 0.5);

    // Slight metallic for mineral inclusions
    float mineralShine = fbm(uv * 30.0, 2, 2.0, 0.5);
    mat.metallic = 0.01 + 0.04 * smoothstep(0.6, 0.8, mineralShine);

    // AO based on crack patterns
    float crackAO = voronoiEdge(uv * 15.0);
    mat.ao = 0.55 + 0.45 * crackAO;

    // Rich detail normal with ridged noise for rocky surface
    vec3 baseNormal = computeRichDetailNormal(uv, 25.0, 0.35, 1); // Ridged
    vec3 crackNormal = computeRichDetailNormal(uv, 50.0, 0.25, 2); // Voronoi
    vec3 fineNormal = computeDetailNormal(uv, 120.0, 0.15);
    mat.normal = blendNormals(blendNormals(baseNormal, crackNormal), fineNormal);

    return mat;
}

PBRMaterial getDirtMaterial(vec2 uv, vec3 worldPos, vec3 geometryNormal) {
    PBRMaterial mat;
    float variation = fbm(worldPos.xz * 0.01, 2, 2.0, 0.5);

    mat.albedo = dirtAlbedo(uv, variation);

    // Dirt is generally rough with some moisture variation
    float moistureRoughness = fbm(uv * 30.0, 3, 2.0, 0.5);
    float clumpRoughness = voronoi(uv * 60.0);
    mat.roughness = 0.80 + 0.18 * mix(moistureRoughness, clumpRoughness, 0.4);

    mat.metallic = 0.0;

    // AO from clumps and pebbles
    float clumpAO = fbm(uv * 50.0, 3, 2.0, 0.5);
    mat.ao = 0.70 + 0.30 * clumpAO;

    // Layered normal for dirt texture
    vec3 clumpNormal = computeRichDetailNormal(uv, 40.0, 0.18, 2); // Voronoi for clumps
    vec3 grainNormal = computeDetailNormal(uv, 100.0, 0.12);
    mat.normal = blendNormals(clumpNormal, grainNormal);

    return mat;
}

PBRMaterial getSandMaterial(vec2 uv, vec3 worldPos, vec3 geometryNormal) {
    PBRMaterial mat;
    float variation = inHeight < WATER_LEVEL + 0.02 ? 0.8 : 0.0; // Wet near water

    mat.albedo = sandAlbedo(uv, variation);

    // Sand roughness: wet is smoother, dry is rougher
    float rippleRoughness = fbm(uv * 80.0, 2, 2.0, 0.5);
    float baseRoughness = 0.55 + 0.40 * rippleRoughness;
    mat.roughness = mix(0.35, baseRoughness, 1.0 - variation); // Wet sand is smoother

    mat.metallic = 0.0;

    // AO from ripple patterns
    float rippleAO = fbm(uv * 60.0, 2, 2.0, 0.5);
    mat.ao = 0.88 + 0.12 * rippleAO;

    // Subtle ripple normals
    vec3 rippleNormal = computeDetailNormal(uv, 70.0, 0.06);
    vec3 grainNormal = computeDetailNormal(uv, 200.0, 0.04);
    mat.normal = blendNormals(rippleNormal, grainNormal);

    return mat;
}

PBRMaterial getSnowMaterial(vec2 uv, vec3 worldPos, vec3 geometryNormal) {
    PBRMaterial mat;
    float variation = 1.0 - smoothstep(ROCK_LEVEL, SNOW_LEVEL, inHeight);

    mat.albedo = snowAlbedo(uv, variation);

    // Snow roughness varies: fresh powder is rough, packed/icy is smooth
    float packLevel = fbm(uv * 20.0, 3, 2.0, 0.5);
    float iceLevel = voronoi(uv * 8.0);
    mat.roughness = 0.25 + 0.50 * packLevel - 0.15 * smoothstep(0.3, 0.1, iceLevel);
    mat.roughness = clamp(mat.roughness, 0.15, 0.8);

    mat.metallic = 0.0;

    // AO subtle for snow
    mat.ao = 0.92 + 0.08 * fbm(uv * 30.0, 2, 2.0, 0.5);

    // Soft normals for snow surface
    vec3 driftNormal = computeDetailNormal(uv, 25.0, 0.08);
    vec3 surfaceNormal = computeDetailNormal(uv, 80.0, 0.05);
    mat.normal = blendNormals(driftNormal, surfaceNormal);

    return mat;
}

// ============================================================================
// MATERIAL BLENDING
// Advanced blending based on height, slope, and noise with depth-based mixing
// ============================================================================

// Simple linear material blend
PBRMaterial blendMaterials(PBRMaterial a, PBRMaterial b, float t) {
    PBRMaterial result;
    result.albedo = mix(a.albedo, b.albedo, t);
    result.roughness = mix(a.roughness, b.roughness, t);
    result.metallic = mix(a.metallic, b.metallic, t);
    result.ao = mix(a.ao, b.ao, t);
    result.normal = normalize(mix(a.normal, b.normal, t));
    return result;
}

// Height-based blend (materials with height variation blend more naturally)
PBRMaterial heightBlendMaterials(PBRMaterial a, PBRMaterial b, float heightA, float heightB, float t, float blendDepth) {
    // Compute heights with blend factor
    float ha = heightA + (1.0 - t);
    float hb = heightB + t;

    // Soft depth blend
    float ma = max(0.0, ha - max(hb - blendDepth, 0.0));
    float mb = max(0.0, hb - max(ha - blendDepth, 0.0));
    float blendSum = ma + mb + 0.0001;

    float finalBlend = mb / blendSum;

    return blendMaterials(a, b, finalBlend);
}

// Compute blend weight with noise perturbation
float computeBlendWeight(float value, float low, float high, vec2 uv, float noiseScale, float noiseStrength) {
    float noise = fbm(uv * noiseScale, 4, 2.0, 0.5) * 2.0 - 1.0;
    float noisyValue = value + noise * noiseStrength;
    return smoothstep(low, high, noisyValue);
}

// Transition zone width based on material types
float getTransitionWidth(int fromMat, int toMat) {
    // Wider transitions between dissimilar materials
    // 0=sand, 1=dirt, 2=grass, 3=rock, 4=snow
    if (fromMat == 0 && toMat == 1) return 0.06;  // Sand to dirt - medium
    if (fromMat == 1 && toMat == 2) return 0.08;  // Dirt to grass - wider
    if (fromMat == 2 && toMat == 3) return 0.10;  // Grass to rock - widest
    if (fromMat == 3 && toMat == 4) return 0.07;  // Rock to snow - medium
    return 0.05;  // Default
}

PBRMaterial getTerrainMaterial(vec2 uv, vec3 worldPos, vec3 geometryNormal, float height, float slope) {
    // Multi-octave noise for natural-looking transitions
    float largeNoise = warpedNoise(worldPos.xz * 0.015, 0.5) * 0.15 - 0.075;
    float medNoise = fbm(worldPos.xz * 0.04, 3, 2.0, 0.5) * 0.10 - 0.05;
    float fineNoise = simplexNoise(worldPos.xz * 0.1) * 0.05;
    float noiseOffset = largeNoise + medNoise + fineNoise;
    float heightWithNoise = height + noiseOffset;

    // Calculate slope factor (1.0 = flat, 0.0 = vertical)
    float slopeFactor = slope;

    // Additional detail noise for material variation
    float detailNoise = fbm(uv * 50.0, 3, 2.0, 0.5);
    float microNoise = simplexNoise(uv * 200.0) * 0.5 + 0.5;

    // Get only sand and dirt materials
    PBRMaterial dirt = getDirtMaterial(uv, worldPos, geometryNormal);
    PBRMaterial sand = getSandMaterial(uv, worldPos, geometryNormal);

    // Create pseudo-height values for height-blending
    float sandHeight = 0.35 + detailNoise * 0.2;
    float dirtHeight = 0.45 + detailNoise * 0.25;

    PBRMaterial result;

    // Simple height-based sand to dirt transition
    // Lower areas = more sand, higher areas = more dirt
    float transitionMid = 0.4; // Middle point of transition
    float t = computeBlendWeight(heightWithNoise, 0.15, 0.65, uv, 20.0, 0.08);

    result = heightBlendMaterials(sand, dirt, sandHeight, dirtHeight, t, 0.4);

    // Slope-based variation: steeper slopes get more dirt (erosion exposes soil)
    float slopeBlend = 1.0 - smoothstep(CLIFF_SLOPE, STEEP_SLOPE, slopeFactor);
    slopeBlend *= fbm(worldPos.xz * 0.06, 3, 2.0, 0.5);
    result = blendMaterials(result, dirt, slopeBlend * 0.4);

    // Add sand patches in lower/flatter areas
    float sandPatch = smoothstep(0.5, 0.2, heightWithNoise);
    sandPatch *= smoothstep(0.6, 0.9, slopeFactor); // Flatter areas
    sandPatch *= fbm(worldPos.xz * 0.08, 4, 2.0, 0.5);
    result = blendMaterials(result, sand, sandPatch * 0.35);

    // Add dirt erosion channels
    float erosionPattern = ridgedNoise(worldPos.xz * 0.05, 4, 2.0, 0.5);
    float erosion = smoothstep(0.6, 0.8, erosionPattern);
    result = blendMaterials(result, dirt, erosion * 0.25);

    // Scattered sand deposits in dirt areas
    float sandDeposit = voronoi(worldPos.xz * 0.15);
    sandDeposit = smoothstep(0.35, 0.15, sandDeposit);
    sandDeposit *= fbm(worldPos.xz * 0.1, 3, 2.0, 0.5);
    result = blendMaterials(result, sand, sandDeposit * 0.2);

    return result;
}

// ============================================================================
// COOK-TORRANCE BRDF
// Physically-based specular reflection model
// ============================================================================

// Fresnel-Schlick approximation
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Fresnel-Schlick with roughness for IBL
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// GGX/Trowbridge-Reitz Normal Distribution Function
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / max(denom, 0.0001);
}

// Smith's Geometry Function (Schlick-GGX)
float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / max(denom, 0.0001);
}

// Smith's method for geometry obstruction
float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// ============================================================================
// LIGHTING CALCULATIONS
// ============================================================================

// Main sun light
const vec3 SUN_DIRECTION = normalize(vec3(0.4, 0.8, 0.3));
const vec3 SUN_COLOR = vec3(1.0, 0.95, 0.85) * 3.0;

// Sky/ambient light approximation
const vec3 SKY_COLOR = vec3(0.4, 0.6, 0.9);
const vec3 GROUND_COLOR = vec3(0.3, 0.25, 0.2);

vec3 getAmbientLight(vec3 normal) {
    // Hemisphere lighting: sky above, ground below
    float skyFactor = normal.y * 0.5 + 0.5;
    return mix(GROUND_COLOR, SKY_COLOR, skyFactor) * 0.3;
}

vec3 calculatePBRLighting(vec3 albedo, vec3 normal, float roughness, float metallic, float ao, vec3 worldPos) {
    // Get view direction (camera position from inverse view matrix)
    mat4 invView = inverse(uMVP.viewMatrix);
    vec3 camPos = invView[3].xyz;
    vec3 V = normalize(camPos - worldPos);

    // Calculate reflectance at normal incidence
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // Main directional light
    vec3 L = SUN_DIRECTION;
    vec3 H = normalize(V + L);

    float NdotL = max(dot(normal, L), 0.0);
    float NdotV = max(dot(normal, V), 0.0);

    // Cook-Torrance BRDF
    float NDF = distributionGGX(normal, H, roughness);
    float G = geometrySmith(normal, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.0001;
    vec3 specular = numerator / denominator;

    // Energy conservation
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;

    // Direct lighting
    vec3 Lo = (kD * albedo / PI + specular) * SUN_COLOR * NdotL;

    // Ambient lighting (simplified IBL)
    vec3 ambient = getAmbientLight(normal) * albedo * ao;

    // Add slight fresnel rim light
    vec3 fresnel = fresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 rimLight = fresnel * SKY_COLOR * 0.15 * (1.0 - roughness);

    vec3 color = ambient + Lo + rimLight;

    return color;
}

// ============================================================================
// ATMOSPHERIC EFFECTS
// ============================================================================

vec3 applyFog(vec3 color, float distance, vec3 viewDir) {
    // Exponential fog
    float fogDensity = 0.002;
    float fogFactor = 1.0 - exp(-distance * fogDensity);
    fogFactor = clamp(fogFactor, 0.0, 1.0);

    // Fog color varies with view angle (lighter looking at sky)
    vec3 fogColor = mix(vec3(0.6, 0.65, 0.75), vec3(0.8, 0.85, 0.9), max(viewDir.y, 0.0));

    // Sun inscattering
    float sunAmount = max(dot(viewDir, SUN_DIRECTION), 0.0);
    fogColor = mix(fogColor, vec3(1.0, 0.9, 0.7), pow(sunAmount, 8.0) * 0.3);

    return mix(color, fogColor, fogFactor);
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

void main(void) {
    // Validate normal (safeguard against NaN)
    float normalLen = length(inNormal);
    vec3 geometryNormal = (normalLen > 0.0001 && normalLen == normalLen)
                          ? inNormal / normalLen
                          : vec3(0.0, 1.0, 0.0);

    // Calculate slope (dot with up vector)
    float slope = max(dot(geometryNormal, vec3(0.0, 1.0, 0.0)), 0.0);

    // Get world-space UV for texturing
    vec2 uv = inUV;

    // Validate UV
    if (any(isnan(uv)) || any(isinf(uv))) {
        uv = vec2(0.5);
    }

    // Get terrain material based on height, slope, and position
    PBRMaterial material = getTerrainMaterial(uv, inWorldPos, geometryNormal, inHeight, slope);

    // Construct TBN matrix for normal mapping
    vec3 T = normalize(inTangent);
    vec3 N = geometryNormal;
    vec3 B = cross(N, T) * inBitangentSign;
    mat3 TBN = mat3(T, B, N);

    // Transform detail normal to world space and blend with geometry normal
    vec3 worldDetailNormal = normalize(TBN * material.normal);
    vec3 finalNormal = normalize(mix(geometryNormal, worldDetailNormal, 0.6));

    // Calculate PBR lighting
    vec3 color = calculatePBRLighting(
        material.albedo,
        finalNormal,
        material.roughness,
        material.metallic,
        material.ao,
        inWorldPos
    );

    // Apply atmospheric fog
    mat4 invView = inverse(uMVP.viewMatrix);
    vec3 camPos = invView[3].xyz;
    float viewDist = length(inWorldPos - camPos);
    vec3 viewDir = normalize(inWorldPos - camPos);
    color = applyFog(color, viewDist, -viewDir);

    // Tone mapping (ACES approximation)
    color = color / (color + vec3(1.0));

    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));

    // Final NaN check
    if (any(isnan(color)) || any(isinf(color))) {
        color = vec3(0.5); // Fallback gray
    }

    FragColor = vec4(color, 1.0);
}
