#version 450

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragWorldPos;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 normal = normalize(fragNormal);

    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    vec3 viewDir  = normalize(-fragWorldPos);
    vec3 halfDir  = normalize(lightDir + viewDir);

    float ambient  = 0.15;
    float diffuse  = max(dot(normal, lightDir), 0.0) * 0.7;
    float specular = pow(max(dot(normal, halfDir), 0.0), 32.0) * 0.15;

    // Toggle: comment/uncomment to switch between texture and solid color
    //vec3 baseColor = texture(texSampler, fragTexCoord).rgb;
    vec3 baseColor = vec3(0.7, 0.5, 0.35);

    vec3 color = baseColor * (ambient + diffuse) + vec3(specular);
    outColor = vec4(color, 1.0);
}
