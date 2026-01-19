#version 450 core
#extension GL_ARB_separate_shader_objects : enable

// Triangle tessellation with equal spacing and counter-clockwise winding
layout(triangles, equal_spacing, ccw) in;

layout(binding = 0) uniform mvpMatrix
{
	mat4 modelMatrix;
	mat4 viewMatrix;
	mat4 projectionMatrix;
	vec4 color;
}uMVP;

layout(binding = 2) uniform sampler2D heightMap;

layout(location = 0) in vec4 inPosition[];

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outTexCoord;
layout(location = 3) out float outHeight;

void main()
{
	// Get barycentric coordinates for triangle tessellation
	// gl_TessCoord.x + gl_TessCoord.y + gl_TessCoord.z = 1.0
	float u = gl_TessCoord.x;
	float v = gl_TessCoord.y;
	float w = gl_TessCoord.z;

	// Barycentric interpolation of the 3 triangle vertices
	// Triangle vertices: 0, 1, 2
	vec4 position = inPosition[0] * u + inPosition[1] * v + inPosition[2] * w;

	// Calculate texture coordinates from world XZ position
	// Assuming terrain spans [-1, 1] in XZ, map to [0, 1] for texture
	vec2 texCoord = vec2(
		(position.x + 1.0) * 0.5,
		(position.z + 1.0) * 0.5
	);

	// Sample height from CUDA-generated height map
	float height = texture(heightMap, texCoord).r;

	// Apply height displacement to Y coordinate
	position.y = height;

	// ========================================================================
	// IMPROVED NORMAL CALCULATION FOR DETAILED TERRAIN
	// ========================================================================
	// Use Sobel-like filter for smoother normals on detailed terrain
	float texelSize = 1.0 / 4096.0;  // Height map resolution (4096x4096 for 8km terrain)

	// Sample 3x3 neighborhood for better normal estimation
	float h00 = texture(heightMap, texCoord + vec2(-texelSize, -texelSize)).r;
	float h10 = texture(heightMap, texCoord + vec2(0.0, -texelSize)).r;
	float h20 = texture(heightMap, texCoord + vec2(texelSize, -texelSize)).r;

	float h01 = texture(heightMap, texCoord + vec2(-texelSize, 0.0)).r;
	// h11 is our center height (already sampled as 'height')
	float h21 = texture(heightMap, texCoord + vec2(texelSize, 0.0)).r;

	float h02 = texture(heightMap, texCoord + vec2(-texelSize, texelSize)).r;
	float h12 = texture(heightMap, texCoord + vec2(0.0, texelSize)).r;
	float h22 = texture(heightMap, texCoord + vec2(texelSize, texelSize)).r;

	// Sobel filter for X gradient (horizontal)
	// [ -1  0  1 ]
	// [ -2  0  2 ]
	// [ -1  0  1 ]
	float gradX = (h20 - h00) + 2.0 * (h21 - h01) + (h22 - h02);
	gradX /= 8.0;

	// Sobel filter for Z gradient (vertical)
	// [ -1 -2 -1 ]
	// [  0  0  0 ]
	// [  1  2  1 ]
	float gradZ = (h02 - h00) + 2.0 * (h12 - h10) + (h22 - h20);
	gradZ /= 8.0;

	// Scale gradients to world space
	// Terrain spans 2 units in XZ (-1 to 1), so texelSize in world = 2/4096
	float worldTexelSize = 2.0 / 4096.0;

	// Construct normal from gradients
	// Normal = normalize((-dH/dX, 1, -dH/dZ))
	vec3 normal = normalize(vec3(
		-gradX / worldTexelSize,
		1.0,
		-gradZ / worldTexelSize
	));

	// Transform position to world space
	vec4 worldPos = uMVP.modelMatrix * position;

	// Transform normal to world space (using normal matrix)
	mat3 normalMatrix = mat3(uMVP.modelMatrix);
	vec3 worldNormal = normalize(normalMatrix * normal);

	// Output to fragment shader
	outWorldPos = worldPos.xyz;
	outNormal = worldNormal;
	outTexCoord = texCoord;
	outHeight = height;

	// Final clip space position
	gl_Position = uMVP.projectionMatrix * uMVP.viewMatrix * worldPos;
}
