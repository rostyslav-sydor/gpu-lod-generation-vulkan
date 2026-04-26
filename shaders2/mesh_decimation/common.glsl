#ifndef COMMON_GLSL
#define COMMON_GLSL

#extension GL_ARB_gpu_shader_int64 : require

// ============================================================================
// Parallel GPU Mesh Decimation — Shared Definitions
// ============================================================================

#define WORKGROUP_SIZE 256
#define NONE 0xFFFFFFFFu
#define MAX_PROBES 128
#define HASHMAP_EMPTY     0xFFFFFFFFu
#define HASHMAP_TOMBSTONE 0xFFFFFFFEu

// Counter indices in the counters[] SSBO
#define COUNTER_EDGE_COUNT      0
#define COUNTER_COLLAPSE_COUNT  1
#define COUNTER_TRIANGLE_COUNT  2
#define COUNTER_VERTEX_COUNT    3
#define COUNTER_COMPACT_COUNT   4

// Vertex flags
#define FLAG_DISCONTINUITY  (1u << 0)
#define FLAG_BOUNDARY       (1u << 1)
#define FLAG_COLLAPSED      (1u << 2)

// Cost function modes (matches DecimationPushConstants.costMode)
#define COST_MODE_QEM       0u
#define COST_MODE_PAPER     1u
#define COST_MODE_EDGELEN   2u

// Edge flags
#define EDGE_ELIGIBLE       (1u << 0)
#define EDGE_BOUNDARY       (1u << 1)

// ============================================================================
// Structures
// ============================================================================

struct Vertex {
    vec3 pos;
    vec3 normal;
    vec2 texCoord;
};

// Symmetric 4x4 quadric stored as 10 floats + 1 weight:
//   | q0  q1  q2  q3 |
//   | q1  q4  q5  q6 |
//   | q2  q5  q7  q8 |
//   | q3  q6  q8  q9 |
// q[10] = w (accumulated weight)
struct Quadric {
    float q[11];
};

// ============================================================================
// Hash Functions (MurmurHash3-inspired)
// ============================================================================

uint murmur_mix(uint h) {
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

uint hash_combine(uint seed, uint value) {
    return seed ^ (murmur_mix(value) + 0x9e3779b9u + (seed << 6) + (seed >> 2));
}

uint hash_float(float f) {
    uint bits = floatBitsToUint(f);
    // Treat -0.0 as 0.0
    if (bits == 0x80000000u) bits = 0u;
    return bits;
}

uint hash_position(vec3 pos) {
    uint h = 0u;
    h = hash_combine(h, hash_float(pos.x));
    h = hash_combine(h, hash_float(pos.y));
    h = hash_combine(h, hash_float(pos.z));
    return h;
}

uint hash_vertex(Vertex v) {
    uint h = hash_position(v.pos);
    h = hash_combine(h, hash_float(v.normal.x));
    h = hash_combine(h, hash_float(v.normal.y));
    h = hash_combine(h, hash_float(v.normal.z));
    return h;
}

uint hash_edge(uint v0, uint v1) {
    uint a = min(v0, v1);
    uint b = max(v0, v1);
    uint h = 0u;
    h = hash_combine(h, a);
    h = hash_combine(h, b);
    return h;
}

// ============================================================================
// Quadric Math
// ============================================================================

Quadric quadric_zero() {
    Quadric qr;
    for (int i = 0; i < 11; i++) qr.q[i] = 0.0;
    return qr;
}

// Build fundamental quadric from plane equation (a, b, c, d)
// where ax + by + cz + d = 0
Quadric quadric_from_plane(vec4 plane) {
    float a = plane.x, b = plane.y, c = plane.z, d = plane.w;
    Quadric qr;
    qr.q[0] = a * a;  qr.q[1] = a * b;  qr.q[2] = a * c;  qr.q[3] = a * d;
    qr.q[4] = b * b;  qr.q[5] = b * c;  qr.q[6] = b * d;
    qr.q[7] = c * c;  qr.q[8] = c * d;
    qr.q[9] = d * d;
    qr.q[10] = 0.0;
    return qr;
}

Quadric quadric_add(Quadric a, Quadric b) {
    Quadric qr;
    for (int i = 0; i < 11; i++) qr.q[i] = a.q[i] + b.q[i];
    return qr;
}

// Evaluate v^T * Q * v for homogeneous coordinate v = (x, y, z, 1)
float quadric_evaluate(Quadric qr, vec3 v) {
    float x = v.x, y = v.y, z = v.z;
    return qr.q[0]*x*x + 2.0*qr.q[1]*x*y + 2.0*qr.q[2]*x*z + 2.0*qr.q[3]*x
         + qr.q[4]*y*y + 2.0*qr.q[5]*y*z + 2.0*qr.q[6]*y
         + qr.q[7]*z*z + 2.0*qr.q[8]*z
         + qr.q[9];
}

// Compute optimal vertex position minimizing quadric error.
// Returns true if the 3x3 submatrix is invertible.
// Solves:  | q0 q1 q2 |       | q3 |
//          | q1 q4 q5 | * v = -| q6 |
//          | q2 q5 q7 |       | q8 |
bool quadric_optimal(Quadric qr, out vec3 result) {
    // 3x3 matrix (upper-left of the 4x4 quadric)
    float a00 = qr.q[0], a01 = qr.q[1], a02 = qr.q[2];
    float a11 = qr.q[4], a12 = qr.q[5];
    float a22 = qr.q[7];

    // Cofactors
    float c00 = a11 * a22 - a12 * a12;
    float c01 = a02 * a12 - a01 * a22;
    float c02 = a01 * a12 - a02 * a11;

    float det = a00 * c00 + a01 * c01 + a02 * c02;

    // Scale-relative threshold: reject if det is tiny relative to matrix diagonal
    float scale = max(abs(a00), max(abs(a11), abs(a22)));
    if (abs(det) < 1e-6 * scale * scale * scale + 1e-20) {
        return false;
    }

    float invDet = 1.0 / det;

    float c11 = a00 * a22 - a02 * a02;
    float c12 = a01 * a02 - a00 * a12;
    float c22 = a00 * a11 - a01 * a01;

    // Right-hand side
    float bx = -qr.q[3];
    float by = -qr.q[6];
    float bz = -qr.q[8];

    result.x = (c00 * bx + c01 * by + c02 * bz) * invDet;
    result.y = (c01 * bx + c11 * by + c12 * bz) * invDet;
    result.z = (c02 * bx + c12 * by + c22 * bz) * invDet;

    return true;
}

// Build a quadric that penalizes deviation from a point (vertex regularization).
// The quadric evaluates to |v - p|^2 when scaled by weight.
Quadric quadric_from_point(vec3 p, float weight) {
    Quadric qr;
    qr.q[0] = weight;       qr.q[1] = 0.0;         qr.q[2] = 0.0;         qr.q[3] = -weight * p.x;
    qr.q[4] = weight;       qr.q[5] = 0.0;         qr.q[6] = -weight * p.y;
    qr.q[7] = weight;       qr.q[8] = -weight * p.z;
    qr.q[9] = weight * dot(p, p);
    qr.q[10] = weight;
    return qr;
}

// Build a border-edge quadric matching meshoptimizer's quadricFromTriangleEdge.
// Creates a wall plane perpendicular to edge p0->p1, pointing toward p2.
// Weight is scaled by edge length (matching meshopt's linear edge scaling).
Quadric quadric_from_triangle_edge(vec3 p0, vec3 p1, vec3 p2, float weight) {
    vec3 p10 = p1 - p0;
    float lengthsq = dot(p10, p10);
    float edgeLen = sqrt(lengthsq);

    vec3 p20 = p2 - p0;
    float p20p = dot(p20, p10);

    // Perpendicular from p2 to line p0-p1 (in the triangle plane)
    vec3 perp = p20 * lengthsq - p10 * p20p;
    float perpLen = length(perp);
    if (perpLen < 1e-12) return quadric_zero();
    perp /= perpLen;

    float d = -dot(perp, p0);
    float w = edgeLen * weight;

    Quadric qr = quadric_from_plane(vec4(perp, d));
    for (int i = 0; i < 10; i++) qr.q[i] *= w;
    qr.q[10] = w;
    return qr;
}

// ============================================================================
// GPU Hash Map Protocol (open-addressing, linear probing)
//
// Three separate SSBOs, one per use case:
//   vertexHashMap[]   — Pass 1: vertex dedup.    Slot: (key, vertexIdx, -, -)
//   positionHashMap[] — Pass 2: position dedup.  Slot: (key, vertexIdx, -, -)
//   edgeHashMap[]     — Pass 4: edge dedup.      Slot: (vMin, vMax, edgeIdx, -)
//
// Insert protocol (passes 1-2):
//   1. slot = key & (mapSize - 1), linear probe on collision
//   2. atomicCompSwap(slot.x, HASHMAP_EMPTY, key) to claim
//   3. If won: write value to .y
//   4. If lost: check if .x matches key, verify with full equality check
//
// Insert protocol (pass 4):
//   Same, but key is two vertex indices (.x = vMin, .y = vMax),
//   value (.z) is the allocated edge index.
// ============================================================================

// ============================================================================
// Cost quantization for parallel collapse
// ============================================================================

uint quantizeCostBits(float cost, uint mantissaBits) {
    uint bits = floatBitsToUint(cost);
    uint shift = 23u - min(mantissaBits, 23u);
    return (bits >> shift) << shift;
}

// ============================================================================
// Utility
// ============================================================================

// Adjacency packing: triangle index in upper 30 bits, vertex slot (0-2) in lower 2 bits
#define ADJ_PACK(triIdx, slot)   (((triIdx) << 2) | (slot))
#define ADJ_TRI(packed)          ((packed) >> 2)
#define ADJ_SLOT(packed)         ((packed) & 3u)

// Pack two uint16s into one uint32
uint pack_u16(uint a, uint b) {
    return (a & 0xFFFFu) | ((b & 0xFFFFu) << 16);
}

// Unpack two uint16s from one uint32
uvec2 unpack_u16(uint packed) {
    return uvec2(packed & 0xFFFFu, (packed >> 16) & 0xFFFFu);
}

bool vertices_equal(Vertex a, Vertex b) {
    return a.pos == b.pos && a.normal == b.normal;
}

bool positions_equal(vec3 a, vec3 b) {
    return a == b;
}

#endif // COMMON_GLSL
