CFLAGS = -std=c++17 -O2 -I./include/ -I/usr/local/include
LDFLAGS = -L/usr/local/lib -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi -lglm -lassimp -lmeshoptimizer
SHADER_DBG_FLAGS = -O
SOURCES := $(wildcard src/*.cpp)

DECIMATION_DIR = shaders2/mesh_decimation
DECIMATION_COMPS = $(sort $(wildcard $(DECIMATION_DIR)/*.comp))
DECIMATION_SPVS = $(DECIMATION_COMPS:.comp=.spv)

VulkanLOD: $(SOURCES)
	g++ $(CFLAGS) -o VulkanLOD $(SOURCES) $(LDFLAGS) -Wl,-rpath,/usr/local/lib

recomp_shaders: 
	glslc -fshader-stage=comp $(SHADER_DBG_FLAGS) shaders/simpify.glsl -o shaders/comp.spv
	glslc -fshader-stage=vert $(SHADER_DBG_FLAGS) shaders/vertex.glsl -c -o shaders/vert.spv
	glslc -fshader-stage=frag $(SHADER_DBG_FLAGS) shaders/fragment.glsl -o shaders/frag.spv

$(DECIMATION_DIR)/%.spv: $(DECIMATION_DIR)/%.comp $(DECIMATION_DIR)/common.glsl $(DECIMATION_DIR)/bindings.glsl
	glslc -fshader-stage=comp --target-env=vulkan1.1 -I $(DECIMATION_DIR) $(SHADER_DBG_FLAGS) $< -o $@

recomp_decimation_shaders: $(DECIMATION_SPVS)

.PHONY: test clean recomp_shaders recomp_decimation_shaders

all: recomp_shaders recomp_decimation_shaders VulkanLOD test

test: VulkanLOD
	VK_LAYER_PRINTF_BUFFER_SIZE=8192 ./VulkanLOD

# Use lavapipe (software Vulkan) - useful on WSL2 where dzn may crash
test-wsl: VulkanLOD
	VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json VK_LAYER_PRINTF_BUFFER_SIZE=8192 ./VulkanLOD

clean:
	rm -f VulkanLOD
