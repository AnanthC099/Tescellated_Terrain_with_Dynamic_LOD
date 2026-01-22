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

void main(void)
{
	// Height-based terrain coloring (OpenGL Insights style)
	// Map height from [-0.5, 0.5] to [0, 1] for color interpolation
	float h = clamp((inHeight + 0.5) * 1.0, 0.0, 1.0);

	// Terrain color gradient based on elevation
	vec3 deepColor = vec3(0.2, 0.4, 0.1);    // Dark green (valleys)
	vec3 midColor = vec3(0.4, 0.6, 0.2);     // Green (mid elevation)
	vec3 highColor = vec3(0.6, 0.5, 0.3);    // Brown (hills)
	vec3 peakColor = vec3(0.9, 0.9, 0.95);   // Snow white (peaks)

	vec3 terrainColor;
	if (h < 0.33)
	{
		terrainColor = mix(deepColor, midColor, h * 3.0);
	}
	else if (h < 0.66)
	{
		terrainColor = mix(midColor, highColor, (h - 0.33) * 3.0);
	}
	else
	{
		terrainColor = mix(highColor, peakColor, (h - 0.66) * 3.0);
	}

	// Simple directional lighting
	vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
	vec3 normal = normalize(inNormal);

	// Ambient + Diffuse lighting
	float ambient = 0.3;
	float diffuse = max(dot(normal, lightDir), 0.0) * 0.7;

	vec3 finalColor = terrainColor * (ambient + diffuse);

	FragColor = vec4(finalColor, 1.0);
}
