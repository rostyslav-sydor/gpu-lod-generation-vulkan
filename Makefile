CFLAGS = -std=c++17 -O2 -I./include/
SHADER_DBG_FLAGS = -O
VULKAN_ENV ?= vulkan1.1
SOURCES := $(wildcard src/*.cpp)
HEADERS := $(wildcard include/*.hpp)

ifeq ($(OS),Windows_NT)
  # MSYS2/MinGW on Windows
  LDFLAGS = -lglfw3 -lvulkan-1 -lassimp -lmeshoptimizer -lgdi32 -lshell32
else
  CFLAGS += -I/usr/local/include
  LDFLAGS = -L/usr/local/lib -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi -lassimp -lmeshoptimizer -Wl,-rpath,/usr/local/lib
endif

DECIMATION_DIR = shaders2/mesh_decimation
DECIMATION_COMPS = $(sort $(wildcard $(DECIMATION_DIR)/*.comp) $(wildcard shaders/*.comp))
DECIMATION_SPVS = $(DECIMATION_COMPS:.comp=.spv)

.DEFAULT_GOAL := build
.PHONY: build test clean recomp_decimation_shaders

build: recomp_decimation_shaders VulkanLOD

VulkanLOD: $(SOURCES) $(HEADERS)
	g++ $(CFLAGS) -o VulkanLOD $(SOURCES) $(LDFLAGS)


shaders/vert.spv: shaders/vertex.glsl
	glslc -fshader-stage=vert --target-env=$(VULKAN_ENV) $(SHADER_DBG_FLAGS) $< -o $@

shaders/frag.spv: shaders/fragment.glsl
	glslc -fshader-stage=frag --target-env=$(VULKAN_ENV) $(SHADER_DBG_FLAGS) $< -o $@

shaders/comp.spv: shaders/simpify.glsl
	glslc -fshader-stage=comp --target-env=$(VULKAN_ENV) $(SHADER_DBG_FLAGS) $< -o $@

$(DECIMATION_DIR)/%.spv: $(DECIMATION_DIR)/%.comp $(DECIMATION_DIR)/common.glsl $(DECIMATION_DIR)/bindings.glsl
	glslc -fshader-stage=comp --target-env=$(VULKAN_ENV) -I $(DECIMATION_DIR) $(SHADER_DBG_FLAGS) $< -o $@

RENDER_SPVS = shaders/vert.spv shaders/frag.spv shaders/comp.spv
recomp_decimation_shaders: $(RENDER_SPVS) $(DECIMATION_SPVS)

all: recomp_decimation_shaders VulkanLOD test

test: VulkanLOD
	VK_LAYER_PRINTF_BUFFER_SIZE=8192 ./VulkanLOD

test-wsl: VulkanLOD
	VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json VK_LAYER_PRINTF_BUFFER_SIZE=8192 ./VulkanLOD

clean:
	rm -f VulkanLOD
	rm -f shaders/*.spv
	rm -f $(DECIMATION_DIR)/*.spv
