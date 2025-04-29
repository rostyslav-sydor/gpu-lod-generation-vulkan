#version 450
#extension GL_EXT_debug_printf : enable

layout (local_size_x = 1) in;

struct inVertex {
  vec3 position;
  vec3 color;
  vec2 texCoord;
};

struct inMeshlet {
    uint voffset;
    uint toffset;
    uint vcount;
    uint tcount;
};

struct inMeshletVertex {
    uint i;
};

struct inFace {
    uvec3 v;
};

layout(std140, binding = 0) buffer VertexBuffer {
   inVertex inVertices[ ];
};

layout(std140, binding = 1) buffer MeshletsBuffer {
   inMeshlet meshlets[ ];
};

layout(std140, binding = 2) buffer MeshletVertices {
    inMeshletVertex meshletVertices[ ];
};

layout(std140, binding = 3) buffer MeshletTriangles {
    inFace meshletFaces[ ];
};

uint meshletID = gl_WorkGroupID.x;
inMeshlet meshlet = meshlets[meshletID];

#define print1(txt, v1) if(meshletID == 0) debugPrintfEXT((txt),(v1));
#define print2(txt, v1, v2) if(meshletID == 0) debugPrintfEXT((txt),(v1),(v2));
#define print3(txt, v1, v2, v3) if(meshletID == 0) debugPrintfEXT((txt),(v1),(v2),(v3));

struct Edge {
    uint v1;
    uint v2;
    bool deleted;
    bool boundary;
    bool dirty;
	float cost;
	vec3 newPos;
};
Edge edges[512];
uint eCnt = 0;

struct Face {
    mat4 quadric;
    uvec3 v;
    bool deleted;
};
Face faces[256];
uint fCnt = meshlet.tcount;

struct Vertex {
    vec3 pos;
    vec3 col;
    vec2 texCoord;
    mat4 quadric;
    uint edges[16];
    uint faces[16];
    uint edgesNum;
    uint facesNum;
};
Vertex vertices[256];
uint vCnt = meshlet.vcount;

// TODO
struct priority_queue {
    uint a;
};

// hash
uint hash(vec2 first, vec2 second) {
    return 1;
}

uint getVIDFromMV(uint id) {
    return meshletVertices[meshlet.voffset + id].i;
}

uvec3 getVIDsFromFaces(uint id) {
    uvec3 mVIndices = meshletFaces[meshlet.toffset + id].v;
    uvec3 ret;
    ret[0] = getVIDFromMV(mVIndices[0]);
    ret[1] = getVIDFromMV(mVIndices[1]);
    ret[2] = getVIDFromMV(mVIndices[2]);
    return ret;
}

void calcFaceQuadric(uint id) {
    if (meshletID == 0) {
        debugPrintfEXT("Checking indices: vcount = %d, toffset = %d, tcount = %d, faces[0].v = %u, %u, %u\n", 
                        meshlet.vcount, meshlet.toffset, meshlet.tcount, 
                        faces[0].v.x, faces[0].v.y, faces[0].v.z);
    }

    // uvec3 v = faces[id].v;
    // if (v.x >= vCnt || v.y >= vCnt || v.z >= vCnt) {
    //     if (meshletID == 0) debugPrintfEXT("Error: Vertex index out of bounds: %u, %u, %u\n", v.x, v.y, v.z);
    //     return;  // Exit early if there's an invalid vertex index
    // }

    // uvec3 v = faces[id].v;
    // vec3 v1 = vertices[v[0]].pos;
    // vec3 v2 = vertices[v[1]].pos;
    // vec3 v3 = vertices[v[2]].pos;

    // vec3 n = normalize(cross(v2-v1, v3-v1));

    // vec4 plane = vec4(n, -dot(n, v1));
    // faces[id].quadric = outerProduct(plane, plane);
}

void getPairs() {
    // pairs = set{}
    // foreach face
    //  pairs.add(face.v1, face.v2);
    //  pairs.add(face.v2, face.v3);
    //  pairs.add(face.v3, face.v1);
}

void calcPairs() {
    // foreach pair in pairs
    // mat4 q_sum = pair.v1.q + pair.v2.q
    // vec4 new_pos;
    //
    // mat4 invq_sum = q_sum;
    // invq_sum[3] = (0,0,0,1)
    // 
    // if (det(invq_sum) > 1e-5f)
    //  pair.new_pos = inverse(q2) * (0,0,0,1)
    // else
    //  pair.new_pos = ((v1 + v2) / 2, 1.0f )
    //
    // pair.cost = dot(pair.new_pos, q_sum * new_pos);
    // pair.dirty = true;
}

void calcCostAndPos(uint edgeInd) {
    Vertex newVert;
    uint v1 = edges[edgeInd].v1;
    uint v2 = edges[edgeInd].v2;
    vec4 newPos;
    mat4 newQ = vertices[v1].quadric + vertices[v2].quadric;

    mat4 q2 = newQ;
	q2[0][3] = 0;
	q2[1][3] = 0;
	q2[2][3] = 0;
	q2[3][3] = 1;

    if (determinant(q2) > 1e-2f)
		newPos = inverse(q2) * vec4(0,0,0,1);
	else 
        newPos = vec4((vertices[v1].pos + vertices[v2].pos) / 2, 1);

	edges[edgeInd].newPos = vec3(newPos);
	edges[edgeInd].cost = dot(newPos, newQ * newPos);

	edges[edgeInd].dirty = true;
}


void addPairsToHeap() {
    //foreach pair
    //  heap.add(pair.cost,pairInd)
    //heapify();
}

void build() {
    // init local vertex array
    for (uint i = 0; i < vCnt; ++i) {
        uint vID = getVIDFromMV(i);
        vertices[i].pos = inVertices[vID].position;
        vertices[i].col = inVertices[vID].color;
        vertices[i].texCoord = inVertices[vID].texCoord;
        vertices[i].facesNum = 0;
        vertices[i].edgesNum = 0;
        vertices[i].quadric = mat4(0);
        // print1("opapa %v4f", vertices[i].quadric[0]);
    }

    // calculate face quadrics and update vertex.faces lists
    for (int i = 0; i < fCnt; ++i) {
        // vec3 vIDs = getVIDsFromFaces(i);

        faces[i].v = meshletFaces[meshlet.toffset + i].v;
        faces[i].deleted = false;
        calcFaceQuadric(i);

        // print1("opapa %v3i", faces[i].v);
        // print1("opapa %v3i", faces[i].deleted);
        Face f = faces[i];
        // print1("opapa %i", vertices[f.v[0]].facesNum);
        for (int j = 0; j < 1; ++j) {

            // uint vInd = f.v[j];

            // uint vFNum = vertices[vInd].facesNum;
            // print1("opapa %i", vFNum);
            // print1("ind: %i", vertices[vInd].facesNum)
            // vertices[vInd].faces[vFNum] = i;
            // vertices[vInd].facesNum++;
        }
        // print1("opapa %v3i", faces[i].v);
    }
    // calculate quadrics for each vertex
    // for (int i = 0; i < vCnt; ++i) {
    //     Vertex v = vertices[i];
    //     for (int j = 0; j < v.facesNum; ++j) {
    //         vertices[i].quadric += faces[v.faces[j]].quadric;
    //     }
    // }

    // //  populate edges array
    // //  todo no comparison min()
    // for (int i = 0; i < meshlet.tcount; ++i) {
    //     Face f = faces[i];
    //     for(int j = 0; j < 3; ++j) {
    //         uint v1 = min(f.v[j], f.v[(j+1)%3]);
    //         uint v2 = max(f.v[j], f.v[(j+1)%3]);

    //         // uint edgeInd = findEdge(v1,v2);
    //         uint edgeInd = 0;
    //         if (edgeInd == -1) {
    //             edgeInd = eCnt++;
    //             edges[edgeInd].v1 = v1;
    //             edges[edgeInd].v2 = v2;
    //             edges[edgeInd].boundary = true;
    //             edges[edgeInd].deleted = false;
    //             edges[edgeInd].dirty = false;
    //             calcCostAndPos(edgeInd);
    //             vertices[v1].edges[vertices[v1].edgesNum++] = edgeInd;
    //         } else {
    //             edges[edgeInd].boundary = false;
    //         }
    //     }
    // }
}

void extract() {
    // write all vertices back to original SSBO
    for (int i = 0; i < vCnt; ++i) {
        // vec3 v = vertices[i].pos;
        // uint VID = getVIDFromMV(i);
        // inVertices[VID].position = vertices[i].pos;
        // inVertices[VID].color = vertices[i].col;
        // inVertices[VID].texCoord = vertices[i].texCoord;
    }

    // for (int i = 0; i < fCnt; ++i) {
    //     // TODO check for deleted;
    //     meshletFaces[meshlet.toffset + i].v = faces[i].v;
    //     // meshletFaces[i].v = faces[i].v;
    // }
    // print1("init cnt %i", meshlet.vcount)
    // print1("final cnt %i", vCnt)
    // print1("init cnt %i", meshlet.tcount)
    // print1("final cnt %i", fCnt)
    // update vertex and face count
    meshlets[meshletID].vcount = vCnt;
    meshlets[meshletID].tcount = fCnt;
}

void main() {
    uint verticesNum = meshlet.vcount;
    uint trianglesNum = meshlet.tcount;
    // print1("opa %u",meshletVertices[0].i);
    // if (meshletID == 0)
        // debugPrintfEXT("meshlet data is %i %i %i %i\n", meshlet.voffset, meshlet.toffset, meshlet.vcount, meshlet.tcount);
        // debugPrintfEXT("postion is is %v3f\n", vertices[meshletVertices[meshlet.voffset]].position);

    
    build();

    

    // do shit

    // extract();

    // for (uint i = meshlet.voffset; i < meshlet.vcount + meshlet.voffset; ++i) {
    //     debugPrintfEXT("meshletVertices[i] = %i\n", meshletVertices[i]);
    //     vertices[meshletVertices[i]].position *= 2;
    // }

}