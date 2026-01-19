#version 450 core
#extension GL_ARB_separate_shader_objects : enable

// 3 control points per patch (triangle)
layout(vertices = 3) out;

layout(binding = 0) uniform mvpMatrix
{
	mat4 modelMatrix;
	mat4 viewMatrix;
	mat4 projectionMatrix;
	vec4 color;
}uMVP;

layout(binding = 1) uniform TessParams
{
	vec4 cameraPos;      // Camera world position (xyz) + padding (w)
	float minTessLevel;  // Minimum tessellation level (e.g., 1.0)
	float maxTessLevel;  // Maximum tessellation level (e.g., 64.0)
	float minDistance;   // Distance for maximum tessellation
	float maxDistance;   // Distance for minimum tessellation
}uTess;

layout(location = 0) in vec4 inPosition[];
layout(location = 0) out vec4 outPosition[];

// Calculate tessellation level based on edge distance from camera
float calcTessLevel(vec4 p0, vec4 p1)
{
	// Transform to world space
	vec4 worldP0 = uMVP.modelMatrix * p0;
	vec4 worldP1 = uMVP.modelMatrix * p1;

	// Calculate edge midpoint
	vec4 midpoint = (worldP0 + worldP1) * 0.5;

	// Distance from camera to edge midpoint
	float dist = distance(midpoint.xyz, uTess.cameraPos.xyz);

	// Linear interpolation based on distance
	// Close = maxTessLevel, Far = minTessLevel
	float t = clamp((dist - uTess.minDistance) / (uTess.maxDistance - uTess.minDistance), 0.0, 1.0);

	return mix(uTess.maxTessLevel, uTess.minTessLevel, t);
}

void main()
{
	// Pass through control point position to TES
	outPosition[gl_InvocationID] = inPosition[gl_InvocationID];

	// Only the first invocation sets tessellation levels for the patch
	if (gl_InvocationID == 0)
	{
		// Calculate tessellation levels for each edge based on camera distance
		// Triangle patch edges: edge 0 (v1-v2), edge 1 (v2-v0), edge 2 (v0-v1)
		float tessEdge0 = calcTessLevel(inPosition[1], inPosition[2]);
		float tessEdge1 = calcTessLevel(inPosition[2], inPosition[0]);
		float tessEdge2 = calcTessLevel(inPosition[0], inPosition[1]);

		// Set outer tessellation levels (3 edges for triangle)
		gl_TessLevelOuter[0] = tessEdge0;
		gl_TessLevelOuter[1] = tessEdge1;
		gl_TessLevelOuter[2] = tessEdge2;

		// Set inner tessellation level (1 for triangle interior)
		// Use average of all edges for smooth interior subdivision
		gl_TessLevelInner[0] = (tessEdge0 + tessEdge1 + tessEdge2) / 3.0;
	}
}
