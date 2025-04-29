CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi -lglm -lassimp -lmeshoptimizer
SHADER_DBG_FLAGS = -O

VulkanBasics: main.cpp simplifier.cpp
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
	g++ $(CFLAGS) -o VulkanBasics main.cpp $(LDFLAGS)

recomp_shaders: 
	glslc -fshader-stage=comp $(SHADER_DBG_FLAGS) shaders/simpify.glsl -o shaders/comp.spv
	glslc -fshader-stage=vert $(SHADER_DBG_FLAGS) shaders/vertex.glsl -c -o shaders/vert.spv
	glslc -fshader-stage=frag $(SHADER_DBG_FLAGS) shaders/fragment.glsl -o shaders/frag.spv

.PHONY: test clean recomp_shaders

all: recomp_shaders VulkanBasics test

test: VulkanBasics
	VK_LAYER_PRINTF_BUFFER_SIZE=8192 ./VulkanBasics

clean:
	rm -f VulkanBasics
