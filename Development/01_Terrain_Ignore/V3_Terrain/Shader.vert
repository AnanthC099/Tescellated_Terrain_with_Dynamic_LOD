#version 450 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 vPosition;

layout(location = 0) out vec4 outPosition;

void main(void)
{
	// Pass position directly to Tessellation Control Shader
	// MVP transformation will be applied in TES after tessellation
	outPosition = vPosition;
}

