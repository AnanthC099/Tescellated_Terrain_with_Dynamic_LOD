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

	// Calculate normal using central differences (finite differences)
	// Sample neighboring heights for gradient calculation
	float texelSize = 1.0 / 1024.0;  // Height map resolution

	float hLeft  = texture(heightMap, texCoord + vec2(-texelSize, 0.0)).r;
	float hRight = texture(heightMap, texCoord + vec2( texelSize, 0.0)).r;
	float hDown  = texture(heightMap, texCoord + vec2(0.0, -texelSize)).r;
	float hUp    = texture(heightMap, texCoord + vec2(0.0,  texelSize)).r;

	// Calculate normal from height gradients
	// Using the cross product of tangent vectors
	vec3 tangentX = vec3(2.0 * texelSize, hRight - hLeft, 0.0);
	vec3 tangentZ = vec3(0.0, hUp - hDown, 2.0 * texelSize);
	vec3 normal = normalize(cross(tangentZ, tangentX));

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
