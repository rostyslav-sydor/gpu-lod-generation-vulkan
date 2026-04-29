#version 450

layout(push_constant) uniform PushConst {
    uint mode;
} pc;

layout(location = 0) flat in vec3 fragColor;
layout(location = 1) in vec3 fragWorldPos;

layout(location = 0) out vec4 outColor;

void main() {
    if (pc.mode == 2u) {
        outColor = vec4(fragColor * 0.4, 1.0);
        return;
    }
    outColor = vec4(fragColor, 1.0);
}
