#ifndef BINDINGS_GLSL
#define BINDINGS_GLSL

// ============================================================================
// Shared SSBO Bindings for Mesh Decimation Pipeline
// ============================================================================

// B0: Vertex data (position + normal + texcoord interleaved as 3x vec4 per vertex)
// Layout: [pos.x, pos.y, pos.z, 0, norm.x, norm.y, norm.z, 0, uv.x, uv.y, 0, 0] per vertex
layout(set = 0, binding = 0) buffer VertexBuffer {
    vec4 vertexData[];   // 3 vec4s per vertex: [0]=pos, [1]=normal, [2]=texcoord
};

// B1: Index buffer — 3 uints per triangle
layout(set = 0, binding = 1) buffer IndexBuffer {
    uint indices[];
};

// B2: Position-only index buffer (PB) — 3 uints per triangle
layout(set = 0, binding = 2) buffer PosIndexBuffer {
    uint posIndices[];
};

// B3: Per-vertex flags
layout(set = 0, binding = 3) buffer VertexFlagsBuffer {
    uint vertexFlags[];
};

// B4: Per-vertex adjacency linked-list head
// adjHead[v] = index into triAdjNext[] of first entry, or NONE
layout(set = 0, binding = 4) buffer AdjHeadBuffer {
    uint adjHead[];
};

// B5: Linked-list next pointers — triAdjNext[idx] = next entry index, or NONE
// triIdx = idx / 3, slot = idx % 3
layout(set = 0, binding = 5) buffer TriAdjNextBuffer {
    uint triAdjNext[];
};

// B6: Unique edge list — uvec2(v0, v1) per edge, using position indices
layout(set = 0, binding = 6) buffer EdgeBuffer {
    uvec2 edges[];
};

// B7: Per-edge adjacent triangles — 2 uints per edge (flat for atomic safety)
layout(set = 0, binding = 7) buffer EdgeTriBuffer {
    uint edgeTriangles[];  // [edgeIdx*2+0] = tri0, [edgeIdx*2+1] = tri1
};

// B8: Per-vertex valence (triangle count). Built in Pass 3, read in Pass 6.
layout(set = 0, binding = 8) buffer ValenceBuffer {
    uint valence[];
};

// B9: Per-vertex quadric — 11 ints per vertex (fixed-point for atomic accumulation)
// Fields 0-9: symmetric 4x4 quadric, field 10: accumulated weight (for normalization)
#define QUADRIC_SCALE 1048576.0
#define QUADRIC_FIELDS 11
layout(set = 0, binding = 9) buffer QuadricBuffer {
    int quadricsData[];  // [vertIdx*QUADRIC_FIELDS + 0..10], fixed-point encoded
};

// B10: Per-edge QEM cost
layout(set = 0, binding = 10) buffer EdgeCostBuffer {
    float edgeCost[];
};

// B11: Per-edge flags
layout(set = 0, binding = 11) buffer EdgeFlagBuffer {
    uint edgeFlags[];
};

// B12: Per-edge optimal collapse target (packed as 3x vec4)
layout(set = 0, binding = 12) buffer EdgeTargetBuffer {
    vec4 edgeTarget[];   // [edgeIdx*3+0]=pos, [edgeIdx*3+1]=normal, [edgeIdx*3+2]=texcoord
};

// B13: Per-triangle 64-bit edge descriptor for atomic-min race
layout(set = 0, binding = 13) buffer TriDescriptorBuffer {
    uint64_t triDescriptor[];
};

// B14: Edge hash map — open addressing table (used by Pass 4)
layout(set = 0, binding = 14) buffer EdgeHashMapBuffer {
    uvec4 edgeHashMap[];
};

// B15: Atomic counters
layout(set = 0, binding = 15) buffer CounterBuffer {
    uint counters[];
};

// B16: Vertex canonical mapping (vertex dedup result)
layout(set = 1, binding = 0) buffer VertexMapBuffer {
    uint vertexMap[];    // vertexMap[i] = canonical index of vertex i
};

// B17: Position canonical mapping (position dedup result)
layout(set = 1, binding = 1) buffer PosMapBuffer {
    uint posMap[];       // posMap[i] = canonical position index of vertex i
};

// B18: Prefix sum / compaction scratch
layout(set = 1, binding = 2) buffer ScanBuffer {
    uint scanData[];
};

// B19: Triangle alive flags for compaction
layout(set = 1, binding = 3) buffer AliveBuffer {
    uint aliveFlags[];
};

// B20: Vertex hash map — open addressing table (used by Pass 1)
layout(set = 1, binding = 4) buffer VertexHashMapBuffer {
    uvec4 vertexHashMap[];
};

// B21: Position hash map — open addressing table (used by Pass 2)
layout(set = 1, binding = 5) buffer PositionHashMapBuffer {
    uvec4 positionHashMap[];
};

// ============================================================================
// Push constants for per-dispatch parameters
// ============================================================================

layout(push_constant) uniform PushConstants {
    uint vertexCount;
    uint triangleCount;
    uint edgeCount;
    uint hashMapSize;
    float costThreshold;
    uint iteration;
    uint costMode;
    uint costQuantBits;
    uint targetTriCount;
};

// ============================================================================
// Vertex accessors
// ============================================================================

Vertex loadVertex(uint idx) {
    Vertex v;
    v.pos      = vertexData[idx * 3 + 0].xyz;
    v.normal   = vertexData[idx * 3 + 1].xyz;
    v.texCoord = vertexData[idx * 3 + 2].xy;
    return v;
}

void storeVertex(uint idx, Vertex v) {
    vertexData[idx * 3 + 0] = vec4(v.pos, 0.0);
    vertexData[idx * 3 + 1] = vec4(v.normal, 0.0);
    vertexData[idx * 3 + 2] = vec4(v.texCoord, 0.0, 0.0);
}

// ============================================================================
// Quadric accessors
// ============================================================================

Quadric loadQuadric(uint idx) {
    Quadric qr;
    uint base = idx * QUADRIC_FIELDS;
    for (int i = 0; i < QUADRIC_FIELDS; i++) {
        qr.q[i] = float(quadricsData[base + uint(i)]) / QUADRIC_SCALE;
    }
    return qr;
}

void clearQuadric(uint idx) {
    uint base = idx * QUADRIC_FIELDS;
    for (int i = 0; i < QUADRIC_FIELDS; i++) {
        quadricsData[base + uint(i)] = 0;
    }
}

void atomicAddQuadric(uint idx, Quadric qr) {
    uint base = idx * QUADRIC_FIELDS;
    for (int i = 0; i < QUADRIC_FIELDS; i++) {
        int ival = int(round(qr.q[i] * QUADRIC_SCALE));
        atomicAdd(quadricsData[base + uint(i)], ival);
    }
}

#endif // BINDINGS_GLSL
