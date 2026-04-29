# Investigating the Performance of Fully Parallel GPU-Based 3D Mesh Decimation Algorithms

Bachelor's thesis project for the Computer Science program at Ukrainian Catholic University (2026).

This project implements a fully parallel mesh decimation pipeline using Vulkan compute shaders. Multiple edges are collapsed simultaneously each iteration using an independent-set selection via `atomicMin`, with no CPU–GPU synchronization between iterations. The pipeline supports three cost models — QEM (quadric error metric with optimal vertex placement), discrete Gaussian curvature, and edge length — and introduces a deferred compaction optimization that skips costly topology rebuilds on "light" iterations.

## Algorithm Pipeline

Each decimation iteration runs as a sequence of compute shader dispatches recorded into a single Vulkan command buffer:

| # | Shader | Purpose |
|---|--------|---------|
| 0 | `01_hash_vertices` | Attribute-level vertex deduplication via hash map |
| 1 | `02_dedup_indices` | Remap index buffer, flag position discontinuities |
| 2 | `03_build_adjacency` | Per-vertex adjacency linked lists + valence counts |
| 3 | `04_build_edges` | Unique edge extraction + edge–triangle pairing via hash map |
| 4 | `04b_flag_boundary` | Boundary vertex detection, boundary quadrics for QEM |
| 5 | `05_compute_quadrics` | Plane quadrics (iteration 0) + initialize triangle descriptors |
| 6 | `06_compute_cost_and_scatter` | Edge cost evaluation + `atomicMin` scatter to triangles (independent set) |
| 7 | `09_collapse_edges` | Collapse winning edges, update topology |
| 8 | `10_mark_degenerate` | Flag degenerate triangles after collapse |
| 9 | `11_compact` | Subgroup-based compaction of alive triangles |
| 10 | `12_copy_back` | Copy compacted data back, update triangle count |

**Deferred compaction** skips passes 3–5 and 9–10 on "light" iterations, reusing stale edge/cost data and only performing a full rebuild every *n* iterations (configurable via `DECIM_LIGHT`).

## Building

### Prerequisites

- [Vulkan SDK](https://vulkan.lunarg.com/) (with `glslc`)
- g\+\+ with C\+\+17 support
- [GLFW](https://github.com/glfw/glfw)
- [GLM](https://github.com/g-truc/glm)
- [Assimp](https://github.com/assimp/assimp)
- [meshoptimizer](https://github.com/zeux/meshoptimizer)
- X11 development libraries (Linux)

### Compile & Run

```bash
git clone https://github.com/rostyslav-sydor/gpu-lod-generation-vulkan
cd gpu-lod-generation-vulkan
make all
./VulkanLOD
```

## Configuration

All parameters are set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `scene.gltf` | Path to input 3D model |
| `DECIM_RATIO` | `0.1` | Target triangle ratio (0.0–1.0) |
| `DECIM_COST` | `0.01` | Cost threshold for stopping |
| `DECIM_NUM` | `100` | Maximum decimation iterations |
| `DECIM_MODE` | `0` | Cost mode: 0 = QEM, 1 = curvature, 2 = edge length |
| `DECIM_LIGHT` | `5` | Full rebuild frequency (light iterations between rebuilds) |
| `CPU_DECIM` | `0` | Set to `1` to also run CPU decimation (meshoptimizer) |
| `HEADLESS` | `0` | Set to `1` to skip window creation / rendering |
| `DECIM_LOG` | `0` | Set to `1` to enable per-iteration CSV logging |

Example — decimate Armadillo to 1% with edge-length cost, headless:

```bash
MODEL_PATH=armadillo.obj DECIM_RATIO=0.01 DECIM_MODE=2 HEADLESS=1 ./VulkanLOD
```

## Attribution

- [stb](https://github.com/nothings/stb) — texture loading
- [Assimp](https://github.com/assimp/assimp) — model loading
- [meshoptimizer](https://github.com/zeux/meshoptimizer) — vertex cache/fetch optimization + CPU decimation baseline
- [vulkan-tutorial.com](https://vulkan-tutorial.com/) — base renderer implementation
- ["Stanford Bunny PBR"](https://sketchfab.com/3d-models/stanford-bunny-pbr-42c9bdc4d27a418daa19b2d5ff690095) by [hackmans](https://sketchfab.com/hackmans), licensed under [CC-BY-4.0](http://creativecommons.org/licenses/by/4.0/)
