#version 450 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 FragColor;

layout(binding = 0) uniform mvpMatrix
{
	mat4 modelMatrix;
	mat4 viewMatrix;
	mat4 projectionMatrix;
	vec4 color;
	vec4 cameraPos;      // Camera position (xyz)
	vec4 resolution;     // Resolution (xy), time (z)
} uMVP;

// ============================================================================
// NOISE FUNCTIONS
// ============================================================================

// Hash function for pseudo-random numbers
float hash(vec3 p)
{
	p = fract(p * 0.3183099 + 0.1);
	p *= 17.0;
	return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

// 3D Value Noise
float noise(vec3 p)
{
	vec3 i = floor(p);
	vec3 f = fract(p);

	// Quintic interpolation for smoother results
	vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

	// Sample 8 corners of the cube
	float a = hash(i + vec3(0.0, 0.0, 0.0));
	float b = hash(i + vec3(1.0, 0.0, 0.0));
	float c = hash(i + vec3(0.0, 1.0, 0.0));
	float d = hash(i + vec3(1.0, 1.0, 0.0));
	float e = hash(i + vec3(0.0, 0.0, 1.0));
	float f1 = hash(i + vec3(1.0, 0.0, 1.0));
	float g = hash(i + vec3(0.0, 1.0, 1.0));
	float h = hash(i + vec3(1.0, 1.0, 1.0));

	// Trilinear interpolation
	return mix(mix(mix(a, b, u.x), mix(c, d, u.x), u.y),
	           mix(mix(e, f1, u.x), mix(g, h, u.x), u.y), u.z);
}

// Fractal Brownian Motion - multiple octaves of noise
float fbm(vec3 p, int octaves)
{
	float value = 0.0;
	float amplitude = 0.5;
	float frequency = 1.0;
	float maxValue = 0.0;

	for (int i = 0; i < octaves; i++)
	{
		value += amplitude * noise(p * frequency);
		maxValue += amplitude;
		amplitude *= 0.5;
		frequency *= 2.0;
	}

	return value / maxValue;
}

// ============================================================================
// CUMULUS CLOUD SHAPE
// ============================================================================

// Cloud parameters
const vec3 CLOUD_CENTER = vec3(0.0, 0.0, 0.0);
const float CLOUD_RADIUS = 1.5;
const float CLOUD_HEIGHT = 1.2;
const float CLOUD_FLAT_BOTTOM = -0.3;  // Y position of flat bottom

// Cumulus cloud shape function
// Returns base density [0,1] based on position within cloud shape
float cloudShape(vec3 p)
{
	vec3 localP = p - CLOUD_CENTER;

	// Flat bottom cutoff - cumulus clouds have flat bases
	if (localP.y < CLOUD_FLAT_BOTTOM)
		return 0.0;

	// Distance from vertical axis
	float horizontalDist = length(localP.xz);

	// Vertical profile - puffy top, narrower at bottom
	// Use exponential falloff from top
	float topY = CLOUD_HEIGHT;
	float verticalFactor = 1.0 - pow(max(0.0, (localP.y - CLOUD_FLAT_BOTTOM) / (topY - CLOUD_FLAT_BOTTOM)), 0.5);

	// Radius varies with height - wider at middle, narrower at top and bottom
	float heightNorm = (localP.y - CLOUD_FLAT_BOTTOM) / (topY - CLOUD_FLAT_BOTTOM);
	float radiusAtHeight = CLOUD_RADIUS * (0.6 + 0.4 * sin(heightNorm * 3.14159));

	// Horizontal falloff
	float horizontalFalloff = 1.0 - smoothstep(0.0, radiusAtHeight, horizontalDist);

	// Combine for base shape
	float shape = horizontalFalloff * verticalFactor;

	return max(0.0, shape);
}

// Sample cloud density at a point
float sampleCloudDensity(vec3 p, float time)
{
	// Get base shape
	float shape = cloudShape(p);
	if (shape <= 0.0)
		return 0.0;

	// Add noise detail for cloud texture
	vec3 noisePos = p * 2.0 + vec3(time * 0.1, 0.0, time * 0.05);

	// Large-scale noise for cloud billows
	float largeNoise = fbm(noisePos * 0.8, 4);

	// Medium-scale noise for detail
	float mediumNoise = fbm(noisePos * 1.5, 3);

	// Small-scale noise for fine detail
	float smallNoise = fbm(noisePos * 3.0, 2);

	// Combine noises
	float noiseValue = largeNoise * 0.6 + mediumNoise * 0.3 + smallNoise * 0.1;

	// Apply noise to shape - erode edges more than center
	float edgeFactor = 1.0 - shape;  // More erosion at edges
	float density = shape - (noiseValue - 0.5) * (0.3 + edgeFactor * 0.4);

	// Add some billowy protrusions
	float billowNoise = fbm(p * 1.2 + vec3(time * 0.05), 3);
	density += (billowNoise - 0.4) * 0.3 * shape;

	return max(0.0, density);
}

// ============================================================================
// RAY MARCHING
// ============================================================================

// Ray-sphere intersection for bounding volume
vec2 intersectSphere(vec3 ro, vec3 rd, vec3 center, float radius)
{
	vec3 oc = ro - center;
	float b = dot(oc, rd);
	float c = dot(oc, oc) - radius * radius;
	float h = b * b - c;

	if (h < 0.0)
		return vec2(-1.0);  // No intersection

	h = sqrt(h);
	return vec2(-b - h, -b + h);  // Near and far intersection distances
}

// Ray march through cloud
vec4 raymarchCloud(vec3 ro, vec3 rd, float time)
{
	// Bounding sphere for the cloud (slightly larger than cloud)
	float boundingRadius = CLOUD_RADIUS + CLOUD_HEIGHT + 0.5;
	vec2 tBounds = intersectSphere(ro, rd, CLOUD_CENTER, boundingRadius);

	if (tBounds.x < 0.0 && tBounds.y < 0.0)
		return vec4(0.0);  // Ray misses bounding volume

	// Clamp to valid range
	float tMin = max(0.0, tBounds.x);
	float tMax = tBounds.y;

	if (tMin >= tMax)
		return vec4(0.0);

	// Ray marching parameters
	const int MAX_STEPS = 64;
	const float STEP_SIZE = 0.05;

	float transmittance = 1.0;
	float totalDensity = 0.0;

	float t = tMin;

	for (int i = 0; i < MAX_STEPS; i++)
	{
		if (t > tMax || transmittance < 0.01)
			break;

		vec3 pos = ro + rd * t;

		// Sample density
		float density = sampleCloudDensity(pos, time);

		if (density > 0.0)
		{
			// Accumulate density
			float absorption = density * STEP_SIZE * 3.0;
			transmittance *= exp(-absorption);
			totalDensity += density * STEP_SIZE;
		}

		t += STEP_SIZE;
	}

	// Cloud opacity (1 - transmittance)
	float cloudAlpha = 1.0 - transmittance;

	return vec4(totalDensity, totalDensity, totalDensity, cloudAlpha);
}

// ============================================================================
// MAIN
// ============================================================================

void main(void)
{
	vec2 uv = fragUV;
	vec2 resolution = uMVP.resolution.xy;
	float time = uMVP.resolution.z;

	// Calculate ray direction from UV
	vec2 screenPos = (uv * 2.0 - 1.0) * vec2(resolution.x / resolution.y, 1.0);

	// Camera setup
	vec3 ro = uMVP.cameraPos.xyz;  // Ray origin (camera position)
	vec3 lookAt = CLOUD_CENTER;
	vec3 forward = normalize(lookAt - ro);
	vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
	vec3 up = cross(right, forward);

	// Ray direction with field of view
	float fov = 1.0;  // Field of view factor
	vec3 rd = normalize(forward * fov + right * screenPos.x + up * screenPos.y);

	// Sky gradient background
	vec3 skyColor = mix(vec3(0.4, 0.6, 0.9), vec3(0.7, 0.85, 1.0), uv.y);

	// Ray march the cloud
	vec4 cloudResult = raymarchCloud(ro, rd, time);

	// Cloud color (white for now, no lighting)
	vec3 cloudColor = vec3(1.0, 1.0, 1.0);

	// Blend cloud with sky based on density
	// Use density to modulate cloud brightness slightly
	float densityFactor = clamp(cloudResult.x * 2.0, 0.0, 1.0);
	cloudColor = mix(vec3(0.9, 0.92, 0.95), vec3(1.0, 1.0, 1.0), densityFactor);

	// Final composite
	vec3 finalColor = mix(skyColor, cloudColor, cloudResult.a);

	FragColor = vec4(finalColor, 1.0);
}
