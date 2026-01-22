#version 450 core
#extension GL_ARB_separate_shader_objects : enable

// Full-screen triangle vertex shader for ray marching
// Uses vertex ID to generate positions - no vertex buffer needed

layout(location = 0) out vec2 fragUV;

void main(void)
{
	// Generate full-screen triangle from vertex ID (0, 1, 2)
	// This creates a triangle that covers the entire screen
	vec2 positions[3] = vec2[](
		vec2(-1.0, -1.0),  // Bottom-left
		vec2( 3.0, -1.0),  // Bottom-right (extends past screen)
		vec2(-1.0,  3.0)   // Top-left (extends past screen)
	);

	vec2 pos = positions[gl_VertexIndex];
	gl_Position = vec4(pos, 0.0, 1.0);

	// Convert from NDC [-1,1] to UV [0,1]
	fragUV = pos * 0.5 + 0.5;
}
