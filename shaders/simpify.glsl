#version 450
// #extension GL_EXT_debug_printf : enable

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

layout(std430, binding = 1) buffer MeshletsBuffer {
   inMeshlet meshlets[ ];
};

layout(std430, binding = 2) buffer MeshletVertices {
    inMeshletVertex meshletVertices[ ];
};

layout(std430, binding = 3) buffer MeshletTriangles {
    inFace meshletFaces[ ];
};

uint meshletID = gl_WorkGroupID.x;
inMeshlet meshlet = meshlets[meshletID];

// #define print1(txt, v1) if(meshletID == 0) debugPrintfEXT((txt),(v1));
// #define print2(txt, v1, v2) if(meshletID == 0) debugPrintfEXT((txt),(v1),(v2));
// #define print3(txt, v1, v2, v3) if(meshletID == 0) debugPrintfEXT((txt),(v1),(v2),(v3));

struct Edge {
    uint v1;
    uint v2;
    bool deleted;
    bool boundary;
    bool dirty;
	float cost;
	vec3 newPos;
};
Edge edges[2048];
uint eCnt = 0;

uint findEdge(uint v1, uint v2) {
    for (uint i = 0; i < eCnt; ++i) {
        if(edges[i].v1 == v1 && edges[i].v2 == v2)
            return i;
    }
    return -1;
}

struct Face {
    mat4 quadric;
    uvec3 v;
    bool deleted;
};
Face faces[1024];
uint fCnt = meshlet.tcount;
uint finalFCnt = fCnt;

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
Vertex vertices[512];
uint vCnt = meshlet.vcount;
uint finalVCnt = vCnt;


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
    uvec3 v = faces[id].v;
    vec3 v1 = vertices[v[0]].pos;
    vec3 v2 = vertices[v[1]].pos;
    vec3 v3 = vertices[v[2]].pos;

    vec3 n = normalize(cross(v2-v1, v3-v1));
    vec4 plane = vec4(n, -dot(n, v1));

    faces[id].quadric = outerProduct(plane, plane);
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

void buildData() {
    // init local vertex array
    for (uint i = 0; i < vCnt; ++i) {
        uint vID = getVIDFromMV(i);
        vertices[i].pos = inVertices[vID].position;
        vertices[i].col = inVertices[vID].color;
        vertices[i].texCoord = inVertices[vID].texCoord;
        vertices[i].facesNum = 0;
        vertices[i].edgesNum = 0;
        vertices[i].quadric = mat4(0);
    }

    // calculate face quadrics and update vertex.faces lists
    for (int i = 0; i < fCnt; ++i) {
        faces[i].v = meshletFaces[meshlet.toffset + i].v;
        faces[i].deleted = false;
        calcFaceQuadric(i);


        Face f = faces[i];
        for (int j = 0; j < 3; ++j) {
            uint vInd = f.v[j];
            uint vFNum = vertices[vInd].facesNum;
            vertices[vInd].faces[vFNum] = i;
            vertices[vInd].facesNum++;
        }
    }

    // calculate quadrics for each vertex
    for (int i = 0; i < vCnt; ++i) {
        Vertex v = vertices[i];
        for (int j = 0; j < v.facesNum; ++j) {
            vertices[i].quadric += faces[v.faces[j]].quadric;
        }
    }

    //  populate edges array
    for (int i = 0; i < meshlet.tcount; ++i) {
        Face f = faces[i];
        for(int j = 0; j < 3; ++j) {
            // TODO compareless minmax
            uint v1 = min(f.v[j], f.v[(j+1)%3]);
            uint v2 = max(f.v[j], f.v[(j+1)%3]);

            uint edgeInd = findEdge(v1,v2);
            if (edgeInd == -1) {
                edgeInd = eCnt++;
                edges[edgeInd].v1 = v1;
                edges[edgeInd].v2 = v2;
                edges[edgeInd].boundary = true;
                edges[edgeInd].deleted = false;
                edges[edgeInd].dirty = false;
                calcCostAndPos(edgeInd);
                vertices[v1].edges[vertices[v1].edgesNum++] = edgeInd;
            } else {
                edges[edgeInd].boundary = false;
            }
        }
    }
}

bool edgeCmp(Edge e1, Edge e2) {
    return e1.v1 == e2.v1 && e1.v2 == e2.v2;
}

bool isDegenerate(Face f){
    return f.v[0] == f.v[1] || f.v[0] == f.v[2] || f.v[1] == f.v[2];
}

void edgeCollapse(int eInd) {
    // int eInd = -1;
    // for(int i = 0; i < eCnt; ++i) {
    //     if(edges[i].boundary == false){
    //         eInd = i;
    //         break;
    //     }
    // }
    // if (eInd == -1) {
    //     debugPrintfEXT("oh man :(");
    //     return;
    // }

    Edge e1 = edges[eInd];
    Vertex v1 = vertices[e1.v1];
    Vertex v2 = vertices[e1.v2];

    edges[eInd].deleted = true;
    
    // reconnect all v2 incident edges to v1
    for(int i = 0; i < v2.edgesNum; ++i) {
        uint e2Ind = v2.edges[i];
        Edge e2 = edges[e2Ind];

        if (e1.v2 == e2.v2)
            edges[e2Ind].v2 = e1.v1;
        else
            edges[e2Ind].v1 = e1.v1;
        
        vertices[e1.v1].edges[vertices[e1.v1].edgesNum++] = e2Ind;
    }

    // remap incident faces
    for(int i = 0; i < v2.facesNum; ++i) {
        Face f = faces[v2.faces[i]];
        for (int j = 0; j < 3; ++j) {
            if (f.v[j] == e1.v2)
                faces[v2.faces[i]].v[j] = e1.v1;
        }

        vertices[e1.v1].faces[vertices[e1.v1].facesNum++] = v2.faces[i];
    }

    // apply newPosition
    vertices[e1.v1].pos = e1.newPos;

    // remove degenerate faces
    for(int i = 0; i < vertices[e1.v1].facesNum; ++i) {
        Face f = faces[vertices[e1.v1].faces[i]];
        if (isDegenerate(f) && !f.deleted && finalFCnt > 0) {
            faces[vertices[e1.v1].faces[i]].deleted = true;
            --finalFCnt;
        }
    }
}

void extract() {
    // write all vertices back to original SSBO
    for (int i = 0; i < vCnt; ++i) {
        uint vId = getVIDFromMV(i);
        inVertices[vId].position = vertices[i].pos;
        inVertices[vId].color = vertices[i].col;
        inVertices[vId].texCoord = vertices[i].texCoord;
    }

    //write all faces back to original SSBO
    uint currFCnt = 0;
    for (int i = 0; i < fCnt; ++i) {
        if(faces[i].deleted)
            continue;
        meshletFaces[meshlet.toffset + currFCnt++].v = faces[i].v;
    }

    // update vertex and face count
    meshlets[meshletID].vcount = finalVCnt;
    meshlets[meshletID].tcount = finalFCnt;
}

void main() {
    buildData();

    for(int i = 0; i < meshlet.vcount / 2; ++i) {
        edgeCollapse(i);
    }

    extract();
}