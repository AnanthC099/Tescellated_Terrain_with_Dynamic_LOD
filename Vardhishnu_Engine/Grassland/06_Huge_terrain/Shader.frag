#version 450 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec4 FragColor;

layout(binding = 0) uniform mvpMatrix {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 color;
} uMVP;

// Simple hash for variation
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main(void) {
    vec3 normal = normalize(inNormal);
    float height = inWorldPos.y;

    // Calculate slope from normal
    float slope = 1.0 - normal.y;

    // Sun direction
    vec3 sunDir = normalize(vec3(0.5, 0.8, 0.3));
    float sunLight = max(dot(normal, sunDir), 0.0);

    // Ambient light
    float ambient = 0.3;

    // Height-based coloring with smooth transitions
    vec3 terrainColor;

    // Color palette
    vec3 deepValley = vec3(0.15, 0.22, 0.12);   // Dark green
    vec3 lowland = vec3(0.25, 0.38, 0.18);      // Forest green
    vec3 midland = vec3(0.35, 0.42, 0.22);      // Light green
    vec3 highland = vec3(0.45, 0.40, 0.30);     // Brown-green
    vec3 rock = vec3(0.40, 0.38, 0.35);         // Gray-brown rock
    vec3 peak = vec3(0.55, 0.52, 0.50);         // Light gray

    // Normalized height (assuming heightScale of 35)
    float normalizedHeight = height / 35.0;

    // Height-based color blending
    if (normalizedHeight < 0.15) {
        terrainColor = mix(deepValley, lowland, smoothstep(0.0, 0.15, normalizedHeight));
    } else if (normalizedHeight < 0.35) {
        terrainColor = mix(lowland, midland, smoothstep(0.15, 0.35, normalizedHeight));
    } else if (normalizedHeight < 0.55) {
        terrainColor = mix(midland, highland, smoothstep(0.35, 0.55, normalizedHeight));
    } else if (normalizedHeight < 0.75) {
        terrainColor = mix(highland, rock, smoothstep(0.55, 0.75, normalizedHeight));
    } else {
        terrainColor = mix(rock, peak, smoothstep(0.75, 1.0, normalizedHeight));
    }

    // Slope-based rock blending (steep areas show more rock)
    vec3 rockColor = vec3(0.42, 0.40, 0.38);
    terrainColor = mix(terrainColor, rockColor, smoothstep(0.4, 0.7, slope));

    // Add subtle color variation based on position
    float variation = hash(floor(inWorldPos.xz * 0.1)) * 0.08;
    terrainColor *= (0.96 + variation);

    // Apply lighting
    float lighting = ambient + sunLight * 0.7;
    vec3 finalColor = terrainColor * lighting;

    // Slight atmospheric perspective for distant terrain
    float dist = length(inWorldPos.xz);
    float fog = 1.0 - exp(-dist * 0.0008);
    vec3 fogColor = vec3(0.6, 0.65, 0.7);
    finalColor = mix(finalColor, fogColor, fog * 0.3);

    // Gamma correction
    finalColor = pow(finalColor, vec3(1.0 / 2.2));

    FragColor = vec4(finalColor, 1.0);
}
