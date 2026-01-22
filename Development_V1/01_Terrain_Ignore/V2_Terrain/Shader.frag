#version 450 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in float inHeight;

layout(location = 0) out vec4 FragColor;

layout(binding = 0) uniform mvpMatrix
{
	mat4 modelMatrix;
	mat4 viewMatrix;
	mat4 projectionMatrix;
	vec4 color;
}uMVP;

layout(binding = 1) uniform TessParams
{
	vec4 cameraPos;
	float minTessLevel;
	float maxTessLevel;
	float minDistance;
	float maxDistance;
}uTess;

// ============================================================================
// Procedural Noise Functions for Texture Detail
// ============================================================================

// Simple hash function for procedural texturing
float hash(vec2 p) {
	return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// 2D noise function
float noise(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);

	// Smooth interpolation
	vec2 u = f * f * (3.0 - 2.0 * f);

	// Four corners
	float a = hash(i);
	float b = hash(i + vec2(1.0, 0.0));
	float c = hash(i + vec2(0.0, 1.0));
	float d = hash(i + vec2(1.0, 1.0));

	return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractional Brownian Motion for texture detail
float fbm(vec2 p, int octaves) {
	float value = 0.0;
	float amplitude = 0.5;
	float frequency = 1.0;

	for (int i = 0; i < octaves; i++) {
		value += amplitude * noise(p * frequency);
		frequency *= 2.0;
		amplitude *= 0.5;
	}

	return value;
}

// ============================================================================
// Terrain Material Colors (Unigine Valley Style)
// ============================================================================

// Grass colors - lush green valley grass
vec3 getGrassColor(vec2 uv, float variation) {
	vec3 grassBase = vec3(0.22, 0.38, 0.12);      // Dark grass
	vec3 grassLight = vec3(0.35, 0.52, 0.18);     // Light grass
	vec3 grassYellow = vec3(0.45, 0.48, 0.15);    // Dry grass patches

	float grassNoise = fbm(uv * 40.0, 4);
	float grassPattern = fbm(uv * 15.0 + vec2(100.0), 3);

	vec3 grass = mix(grassBase, grassLight, grassNoise);
	grass = mix(grass, grassYellow, grassPattern * 0.3 * variation);

	return grass;
}

// Dirt/soil colors
vec3 getDirtColor(vec2 uv) {
	vec3 dirtDark = vec3(0.28, 0.22, 0.15);       // Dark soil
	vec3 dirtLight = vec3(0.42, 0.35, 0.25);      // Light soil

	float dirtNoise = fbm(uv * 50.0, 3);
	return mix(dirtDark, dirtLight, dirtNoise);
}

// Rock colors - gray mountain rock
vec3 getRockColor(vec2 uv, float wetness) {
	vec3 rockDark = vec3(0.32, 0.32, 0.34);       // Dark rock
	vec3 rockLight = vec3(0.52, 0.50, 0.48);      // Light rock
	vec3 rockWet = vec3(0.25, 0.26, 0.28);        // Wet rock (darker)

	float rockNoise = fbm(uv * 30.0, 4);
	float rockPattern = fbm(uv * 8.0 + vec2(50.0), 3);

	vec3 rock = mix(rockDark, rockLight, rockNoise * 0.7 + rockPattern * 0.3);
	rock = mix(rock, rockWet, wetness * 0.5);

	return rock;
}

// Snow colors
vec3 getSnowColor(vec2 uv) {
	vec3 snowWhite = vec3(0.95, 0.97, 1.0);       // Pure snow
	vec3 snowShadow = vec3(0.75, 0.82, 0.92);     // Snow in shadow

	float snowNoise = fbm(uv * 60.0, 3);
	return mix(snowShadow, snowWhite, snowNoise * 0.5 + 0.5);
}

// Forest/vegetation dark color for tree coverage simulation
vec3 getForestColor(vec2 uv) {
	vec3 forestDark = vec3(0.12, 0.22, 0.08);     // Dense forest
	vec3 forestLight = vec3(0.18, 0.30, 0.12);    // Forest edge

	float forestNoise = fbm(uv * 25.0, 4);
	return mix(forestDark, forestLight, forestNoise);
}

// ============================================================================
// Main Fragment Shader
// ============================================================================

void main(void)
{
	// Normalize inputs
	vec3 normal = normalize(inNormal);
	vec2 uv = inTexCoord;

	// ==========================================================================
	// SLOPE CALCULATION
	// ==========================================================================
	// Calculate slope from normal (dot product with up vector)
	// slope = 0 means flat, slope = 1 means vertical cliff
	float slope = 1.0 - normal.y;
	slope = clamp(slope, 0.0, 1.0);

	// Steepness factor (exponential for more dramatic cliffs)
	float steepness = pow(slope, 1.5);

	// ==========================================================================
	// HEIGHT-BASED TERRAIN ZONES (Unigine Valley Style)
	// ==========================================================================
	// Normalize height for material blending
	// Valley floor: -0.3 to -0.1
	// Grasslands: -0.1 to 0.15
	// Tree line: 0.15 to 0.25
	// Alpine: 0.25 to 0.4
	// Snow: > 0.4

	float h = inHeight;

	// Height thresholds (tuned for valley terrain)
	float waterLevel = -0.25;
	float grassStart = -0.15;
	float treeLine = 0.12;
	float alpineStart = 0.22;
	float snowStart = 0.32;

	// ==========================================================================
	// MATERIAL WEIGHTS
	// ==========================================================================

	// Water/river areas (very low)
	float waterWeight = smoothstep(grassStart, waterLevel, h);

	// Grass weight - dominant in valley
	float grassWeight = smoothstep(waterLevel, grassStart, h) *
	                    smoothstep(alpineStart, treeLine, h);
	grassWeight *= (1.0 - steepness * 0.8);  // Less grass on steep slopes

	// Forest weight - mid elevations
	float forestWeight = smoothstep(grassStart, treeLine * 0.5, h) *
	                     smoothstep(alpineStart, treeLine, h);
	forestWeight *= (1.0 - steepness * 0.9);  // Forest doesn't grow on cliffs

	// Add some noise to forest distribution
	float forestNoise = fbm(uv * 12.0, 3);
	forestWeight *= smoothstep(0.3, 0.6, forestNoise);

	// Dirt weight - transitions and bare ground
	float dirtWeight = steepness * 0.5 + (1.0 - grassWeight - forestWeight) * 0.3;
	dirtWeight *= smoothstep(snowStart, alpineStart, h);

	// Rock weight - steep slopes and high altitude
	float rockWeight = steepness * 0.9;
	rockWeight += smoothstep(treeLine, alpineStart, h) * (1.0 - steepness * 0.5);
	rockWeight *= (1.0 - smoothstep(snowStart, snowStart + 0.15, h) * 0.5);

	// Snow weight - high altitude and flat areas
	float snowWeight = smoothstep(snowStart, snowStart + 0.1, h);
	snowWeight *= (1.0 - steepness * 0.7);  // Less snow on steep cliffs

	// Snow accumulation on north-facing slopes (simulate)
	float northFacing = max(0.0, -normal.z);
	snowWeight += northFacing * smoothstep(alpineStart, snowStart, h) * 0.3;

	// ==========================================================================
	// SAMPLE MATERIAL COLORS
	// ==========================================================================

	// Add variation based on position
	float variation = fbm(uv * 5.0, 2);

	vec3 grassColor = getGrassColor(uv, variation);
	vec3 forestColor = getForestColor(uv);
	vec3 dirtColor = getDirtColor(uv);
	vec3 rockColor = getRockColor(uv, 1.0 - h);  // Lower = wetter
	vec3 snowColor = getSnowColor(uv);

	// Water color for river
	vec3 waterColor = vec3(0.15, 0.25, 0.35);

	// ==========================================================================
	// BLEND MATERIALS
	// ==========================================================================

	// Normalize weights
	float totalWeight = grassWeight + forestWeight + dirtWeight + rockWeight + snowWeight + waterWeight;
	totalWeight = max(totalWeight, 0.001);

	grassWeight /= totalWeight;
	forestWeight /= totalWeight;
	dirtWeight /= totalWeight;
	rockWeight /= totalWeight;
	snowWeight /= totalWeight;
	waterWeight /= totalWeight;

	// Blend all materials
	vec3 terrainColor = grassColor * grassWeight +
	                    forestColor * forestWeight +
	                    dirtColor * dirtWeight +
	                    rockColor * rockWeight +
	                    snowColor * snowWeight +
	                    waterColor * waterWeight;

	// ==========================================================================
	// LIGHTING (Enhanced for Valley Style)
	// ==========================================================================

	// Sun direction (warm afternoon sun)
	vec3 sunDir = normalize(vec3(0.4, 0.8, 0.3));
	vec3 sunColor = vec3(1.0, 0.95, 0.85);

	// Secondary light (sky/ambient from above)
	vec3 skyDir = vec3(0.0, 1.0, 0.0);
	vec3 skyColor = vec3(0.5, 0.6, 0.8);

	// View direction for specular
	vec3 viewDir = normalize(uTess.cameraPos.xyz - inWorldPos);

	// Ambient lighting (hemisphere)
	float skyAmount = normal.y * 0.5 + 0.5;
	vec3 ambient = mix(vec3(0.15, 0.12, 0.1), skyColor * 0.4, skyAmount);

	// Diffuse lighting
	float NdotL = max(dot(normal, sunDir), 0.0);
	vec3 diffuse = sunColor * NdotL * 0.75;

	// Soft shadows in valleys (ambient occlusion approximation)
	float ao = smoothstep(-0.3, 0.1, h) * 0.3 + 0.7;

	// Specular for wet/snow surfaces
	vec3 halfVec = normalize(sunDir + viewDir);
	float NdotH = max(dot(normal, halfVec), 0.0);

	// Snow is more specular
	float specPower = mix(32.0, 64.0, snowWeight);
	float specIntensity = mix(0.1, 0.4, snowWeight + waterWeight * 0.5);
	vec3 specular = sunColor * pow(NdotH, specPower) * specIntensity * NdotL;

	// ==========================================================================
	// ATMOSPHERIC FOG (Distance-based)
	// ==========================================================================

	float distToCamera = length(uTess.cameraPos.xyz - inWorldPos);

	// Fog parameters
	float fogStart = 1.5;
	float fogEnd = 5.0;
	float fogDensity = 0.4;

	// Exponential fog
	float fogFactor = 1.0 - exp(-fogDensity * max(0.0, distToCamera - fogStart));
	fogFactor = clamp(fogFactor, 0.0, 0.85);

	// Height-based fog (thicker in valleys)
	float heightFog = smoothstep(0.2, -0.2, h) * 0.3;
	fogFactor = max(fogFactor, heightFog);

	// Fog color (bluish atmospheric haze)
	vec3 fogColor = vec3(0.6, 0.7, 0.85);

	// Add slight color variation to fog based on sun
	fogColor = mix(fogColor, sunColor * 0.8, 0.2);

	// ==========================================================================
	// FINAL COLOR COMPOSITION
	// ==========================================================================

	// Combine lighting
	vec3 litColor = terrainColor * (ambient * ao + diffuse) + specular;

	// Apply fog
	vec3 finalColor = mix(litColor, fogColor, fogFactor);

	// Slight color grading for cinematic look
	finalColor = pow(finalColor, vec3(0.95));  // Slight gamma adjustment

	// Output
	FragColor = vec4(finalColor, 1.0);
}
