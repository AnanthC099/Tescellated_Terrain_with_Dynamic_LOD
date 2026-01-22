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
// HASH FUNCTIONS - High-quality gradient generation
// ============================================================================

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

// ============================================================================
// BASE NOISE FUNCTIONS
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

// ============================================================================
// FBM VARIANTS - Different characteristics for different layers
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

// ============================================================================
// DOMAIN WARPING - Breaks repetition, creates organic shapes
// ============================================================================

// Light domain warp for macro forms
vec2 domainWarpLight(vec2 p, float strength) {
    float warpX = fbmRotated(p + vec2(0.0, 0.0), 3);
    float warpY = fbmRotated(p + vec2(5.2, 1.3), 3);
    return p + vec2(warpX, warpY) * strength;
}

// Strong domain warp for breaking repetition
vec2 domainWarpStrong(vec2 p, float strength) {
    vec2 q = vec2(fbmRotated(p + vec2(0.0, 0.0), 3),
                  fbmRotated(p + vec2(5.2, 1.3), 3));
    vec2 r = vec2(fbmRotated(p + 3.0 * q + vec2(1.7, 9.2), 3),
                  fbmRotated(p + 3.0 * q + vec2(8.3, 2.8), 3));
    return p + r * strength;
}

// ============================================================================
// CONTROL MASKS - Vary roughness/features by region
// ============================================================================

// Roughness mask M(x,z): high = rough detail, low = smooth
float getRoughnessMask(vec2 p) {
    float n = fbmRotated(p * 0.003 + vec2(100.0, 200.0), 4);
    return smoothstep(-0.3, 0.5, n);
}

// Mound/bed zone mask - where raised beds are
float getMoundMask(vec2 p) {
    // Create 3-6 broad mound regions
    float n = fbmBillowy(p * 0.004 + vec2(50.0, 80.0), 3);
    return smoothstep(0.1, 0.5, n);
}

// Basin mask - where low spots/sediment areas are
float getBasinMask(vec2 p) {
    float n = fbmRotated(p * 0.005 + vec2(200.0, 150.0), 3);
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

// Drainage swale mask - shallow channels
float getSwaleMask(vec2 worldPos) {
    vec2 warpedPos = domainWarpLight(worldPos * 0.006, 0.4);
    // Create branching drainage patterns using cellular noise
    float swalePattern = cellularNoise(warpedPos * 0.8);
    return smoothstep(0.0, 0.15, swalePattern);
}

// Lawn/flat zone mask
float getLawnMask(vec2 p) {
    float n = fbmBillowy(p * 0.006 + vec2(300.0, 100.0), 3);
    // Areas where we want flat plateaus
    return smoothstep(0.2, 0.5, n);
}

// ============================================================================
// WATER BODIES - Ponds and garden water features
// Creates distinctive 3D geometry for wireframe visualization:
// - Embankment ridges around edges
// - Sloped banks leading to water
// - Concentric ripples on water surface
// ============================================================================

// Water level constant (normalized, will be multiplied by heightScale)
const float waterLevel = -0.08;  // Below average terrain level

// Main pond mask - creates a natural organic pond shape
float getPondMask(vec2 worldPos) {
    // Domain warp for organic pond edges
    vec2 warpedPos = domainWarpStrong(worldPos * 0.004, 0.4);

    // Primary pond - larger central water feature
    float primaryPond = fbmBillowy(warpedPos + vec2(400.0, 250.0), 4);
    float primary = smoothstep(0.38, 0.55, primaryPond);

    // Secondary smaller pond using different offset
    float secondaryPond = fbmRotated(warpedPos * 1.3 + vec2(150.0, 350.0), 3);
    float secondary = smoothstep(0.42, 0.58, secondaryPond) * 0.7;

    return max(primary, secondary);
}

// Small decorative pools - for garden features like fountains, bird baths
float getSmallPoolMask(vec2 worldPos) {
    vec2 warpedPos = domainWarpLight(worldPos * 0.008, 0.3);

    // Cellular pattern creates isolated small pools
    float poolPattern = cellularNoise(warpedPos * 1.5 + vec2(80.0, 120.0));
    float pools = smoothstep(0.02, 0.06, poolPattern) * 0.5;

    return pools;
}

// Stream/channel mask - narrow water channels connecting features
float getStreamMask(vec2 worldPos) {
    vec2 warpedPos = domainWarpStrong(worldPos * 0.006, 0.5);

    // Meandering stream path
    float streamWidth = 8.0;
    float streamPath = abs(sin(warpedPos.x * 0.3 + warpedPos.y * 0.2) * 40.0 - worldPos.y * 0.5 + worldPos.x * 0.3);
    float stream = 1.0 - smoothstep(0.0, streamWidth, streamPath);

    return stream * 0.6;
}

// Combined water body mask for garden
float getWaterBodyMask(vec2 worldPos) {
    // Get basin mask - water bodies form in low-lying areas
    float basinMask = getBasinMask(worldPos);

    // Combine different water features
    float pondMask = getPondMask(worldPos);
    float poolMask = getSmallPoolMask(worldPos);
    float streamMask = getStreamMask(worldPos);

    // Ponds and pools prefer basin areas, streams can cross terrain
    float waterMask = max(pondMask * basinMask, poolMask * basinMask);
    waterMask = max(waterMask, streamMask * 0.5);

    // Add isolated small ponds that don't need basins (man-made features)
    float isolatedPonds = fbmBillowy(worldPos * 0.012 + vec2(500.0, 300.0), 3);
    isolatedPonds = smoothstep(0.5, 0.65, isolatedPonds) * 0.4;
    waterMask = max(waterMask, isolatedPonds);

    return clamp(waterMask, 0.0, 1.0);
}

// Calculate embankment ridge around water edges
float getEmbankmentRidge(vec2 worldPos, float waterMask) {
    // Detect edge of water body using gradient
    float eps = 3.0;
    float maskRight = getWaterBodyMask(worldPos + vec2(eps, 0.0));
    float maskLeft = getWaterBodyMask(worldPos - vec2(eps, 0.0));
    float maskUp = getWaterBodyMask(worldPos + vec2(0.0, eps));
    float maskDown = getWaterBodyMask(worldPos - vec2(0.0, eps));

    float gradX = abs(maskRight - maskLeft) / (2.0 * eps);
    float gradY = abs(maskUp - maskDown) / (2.0 * eps);
    float edgeGradient = sqrt(gradX * gradX + gradY * gradY);

    // Create ridge at water edge (embankment)
    float ridgeStrength = smoothstep(0.01, 0.08, edgeGradient);

    // Add organic variation to ridge height
    float ridgeVariation = noise(worldPos * 0.05) * 0.3 + 0.7;

    return ridgeStrength * ridgeVariation * 0.15;  // Ridge height
}

// Calculate sloped bank leading down to water
float getBankSlope(vec2 worldPos, float waterMask) {
    // Create transition zone around water (bank area)
    float bankWidth = 0.4;  // Width of slope transition
    float bankZone = smoothstep(0.0, bankWidth, waterMask) * smoothstep(0.8, bankWidth, waterMask);

    // Calculate slope based on distance from water center
    float slopeDepth = (1.0 - waterMask) * 0.08;  // Deeper toward center

    return -slopeDepth * bankZone;
}

// Calculate concentric ripples on water surface
float getWaterRipples(vec2 worldPos, float waterMask) {
    if (waterMask < 0.3) return 0.0;  // No ripples outside water

    // Find approximate center of water body for ripple origin
    vec2 warpedPos = domainWarpLight(worldPos * 0.003, 0.3);

    // Distance-based concentric rings
    float dist = length(warpedPos);
    float rippleFreq = 0.8;  // Frequency of ripples
    float ripples = sin(dist * rippleFreq * 6.28318) * 0.5 + 0.5;

    // Multiple ripple sources for natural look
    vec2 offset1 = vec2(50.0, 30.0);
    vec2 offset2 = vec2(-40.0, 60.0);
    float dist1 = length(worldPos * 0.02 + offset1);
    float dist2 = length(worldPos * 0.025 + offset2);

    float ripple1 = sin(dist1 * 4.0) * 0.5 + 0.5;
    float ripple2 = sin(dist2 * 5.0) * 0.5 + 0.5;

    // Combine ripple patterns
    float combinedRipples = (ripples * 0.4 + ripple1 * 0.3 + ripple2 * 0.3);

    // Fade ripples toward edges
    float edgeFade = smoothstep(0.3, 0.6, waterMask);

    // Very subtle height variation for ripples
    return combinedRipples * edgeFade * 0.012;
}

// Calculate stepped terraces around pond (like rice paddy edges)
float getPondTerraces(vec2 worldPos, float waterMask) {
    if (waterMask < 0.1 || waterMask > 0.7) return 0.0;

    // Create 2-3 terrace steps around pond
    float terraceCount = 3.0;
    float terraceZone = waterMask * terraceCount;
    float terraceStep = floor(terraceZone) / terraceCount;

    // Add slight variation to terrace heights
    float variation = noise(worldPos * 0.02) * 0.02;

    return (terraceStep * 0.06) + variation;
}

// Apply water bodies with distinctive 3D features
float applyWaterBodies(float height, vec2 worldPos) {
    float waterMask = getWaterBodyMask(worldPos);

    if (waterMask < 0.01) return height;

    // 1) Add embankment ridge around water edge
    float embankment = getEmbankmentRidge(worldPos, waterMask);

    // 2) Add sloped banks leading to water
    float bankSlope = getBankSlope(worldPos, waterMask);

    // 3) Add terraced edges around ponds
    float terraces = getPondTerraces(worldPos, waterMask);

    // Apply embankment and bank features to terrain around water
    if (waterMask < 0.5) {
        // Bank/edge zone - add embankment ridge and slope
        height += embankment;
        height += bankSlope;
        height -= terraces;
    }

    // 4) Water surface with ripples
    if (waterMask > 0.3) {
        float waterBlend = smoothstep(0.3, 0.6, waterMask);

        // Base water surface
        float waterSurface = waterLevel;

        // Add concentric ripples to water surface
        float ripples = getWaterRipples(worldPos, waterMask);
        waterSurface += ripples;

        // Add subtle random surface variation
        vec2 waterCoord = worldPos * 0.02;
        float surfaceNoise = simplexNoise(waterCoord + vec2(100.0, 200.0)) * 0.005;
        waterSurface += surfaceNoise;

        // Blend to water surface
        height = mix(height, waterSurface, waterBlend);
    }

    return height;
}

// ============================================================================
// MACRO LAYER - Dominates the look (3-10× mid amplitude)
// Gentle overall slope, 2-6 broad mounds, 1-3 shallow basins
// ============================================================================

float computeMacroHeight(vec2 worldPos, out float overallSlope) {
    vec2 macroCoord = worldPos * 0.0015; // Very low frequency

    // Apply domain warp to break any repetition
    vec2 warpedCoord = domainWarpStrong(macroCoord, 0.3);

    // 1) Gentle overall slope (even tiny helps realism)
    overallSlope = worldPos.x * 0.0003 + worldPos.y * 0.0002;
    overallSlope += sin(worldPos.x * 0.002) * 0.01; // Slight undulation

    // 2) 2-6 broad mounds (landscape beds)
    float broadMounds = fbmBillowy(warpedCoord + vec2(10.0, 20.0), 4) * 0.6;
    broadMounds = max(broadMounds, 0.0); // Only raise, not lower for mounds

    // 3) 1-3 shallow basins (low spots - sediment feel)
    float basins = fbmRotated(warpedCoord * 0.8 + vec2(50.0, 30.0), 3);
    basins = min(basins, 0.0) * 0.4; // Only lower for basins

    // Combine macro elements
    float macro = overallSlope + broadMounds + basins;

    return macro;
}

// ============================================================================
// MESO LAYER - Localized irregularity (1-3× mid amplitude)
// Clumpy noise at larger wavelength, varies with roughness mask
// ============================================================================

float computeMesoHeight(vec2 worldPos, float roughnessMask) {
    vec2 mesoCoord = worldPos * 0.008; // Medium-low frequency

    // Apply light domain warp
    vec2 warpedCoord = domainWarpLight(mesoCoord, 0.5);

    // Clumpy irregular noise (not uniform ripples)
    float irregularLumps = fbmRotated(warpedCoord, 5) * 0.2;

    // Add some ridged character but soft
    float softRidges = fbmRidged(warpedCoord * 1.5, 4) * 0.12;

    // Combine and apply roughness mask
    float meso = (irregularLumps + softRidges) * roughnessMask;

    return meso;
}

// ============================================================================
// MICRO LAYER - Small bumps (0.1-0.5× mid amplitude)
// Subtle in wireframe, uses squared mask for selective application
// ============================================================================

float computeMicroHeight(vec2 worldPos, float roughnessMask, float distanceFade) {
    vec2 microCoord = worldPos * 0.03; // Higher frequency

    // Multiple subtle detail frequencies
    float detail1 = noise(microCoord) * 0.03;
    float detail2 = noise(microCoord * 2.5) * 0.015;
    float detail3 = simplexNoise(microCoord * 4.0) * 0.008;

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
// ============================================================================

float applyThermalErosion(vec2 worldPos, float height, float roughnessMask) {
    // Approximate slope from noise derivatives
    float eps = 2.0;
    vec2 mesoCoord = worldPos * 0.008;
    float hCenter = fbmRotated(mesoCoord, 5);
    float hRight = fbmRotated((worldPos + vec2(eps, 0.0)) * 0.008, 5);
    float hUp = fbmRotated((worldPos + vec2(0.0, eps)) * 0.008, 5);

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

    // Combine base layers
    float height = macroHeight + mesoHeight + microHeight + features;

    // Apply thermal erosion
    height += applyThermalErosion(worldPos, height, roughnessMask);

    // Apply flat regions (lawns, paths)
    height = applyFlatRegions(height, worldPos);

    // Apply distance fade to meso+micro (keep macro even at distance)
    // This makes far terrain smoother (like real LOD)
    float detailContrib = (mesoHeight + microHeight) * (1.0 - distanceFade) * 0.3;
    height -= detailContrib;

    // Apply water bodies - flatten terrain to water level for ponds and streams
    height = applyWaterBodies(height, worldPos);

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
