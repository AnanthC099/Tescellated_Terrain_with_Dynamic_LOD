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

void main(void)
{
	// Simple wireframe output - no lighting or effects
	// Use a single color for the wireframe
	vec3 wireframeColor = vec3(0.0, 1.0, 0.0);  // Green wireframe

	FragColor = vec4(wireframeColor, 1.0);
}
