# Efficient GPU-based Dynamic LOD Generation via Vulkan API

This is code for Bachelors thesis "Efficient GPU-based Dynamic LOD Generation via Vulkan AP" for Ukrainian Catholic University Computer Scince program.

In this project we explored impact of Vulkan-specific optimizations for generating LODs dynamically on GPU.

### Requirements

Vulkan SDK

[GLFW](https://github.com/glfw/glfw)

[GLM](https://github.com/g-truc/glm)

[Assimp](https://github.com/assimp/assimp)

[Meshoptimizer](https://github.com/zeux/meshoptimizer)

### Compilation
```
git clone https://github.com/rostyslav-sydor/gpu-lod-generation-vulkan vulkan_lod
cd vulkan_lod
make all
```
### Attribution

[stb](https://github.com/nothings/stb) library is used to load texture data

[Assimp](https://github.com/assimp/assimp) library is used to handle model loading

Meshlet generation is implemented by [meshoptimizer library](https://github.com/zeux/meshoptimizer)

Most of the renderer is based on code by [vulkan-tutorial](https://vulkan-tutorial.com/)

The testing bunny model used is ["Stanford Bunny PBR"](https://sketchfab.com/3d-models/stanford-bunny-pbr-42c9bdc4d27a418daa19b2d5ff690095) by [hackmans](https://sketchfab.com/hackmans) licensed under [CC-BY-4.0](http://creativecommons.org/licenses/by/4.0/)
