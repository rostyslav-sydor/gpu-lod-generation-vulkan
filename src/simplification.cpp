#include "renderer.hpp"

void App::loadModel() {
    vertices.clear();
    indices.clear();
    Timer tWhole;
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(modelPath,
        aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices);

    if (!scene || !scene->mRootNode || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) {
        throw std::runtime_error(importer.GetErrorString());
    }

    auto mesh = scene->mMeshes[0];

    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        Vertex v;

        const auto& vec = mesh->mVertices[i];
        v.pos = { vec.x, vec.y, vec.z };

        if (mesh->HasNormals()) {
            auto norm = mesh->mNormals[i];
            v.normal = glm::vec3(norm.x, norm.y, norm.z);
        }
        else
            v.normal = glm::vec3(0);

        if (mesh->mTextureCoords[0]) {
            v.texCoord = { mesh->mTextureCoords[0][i].x, -mesh->mTextureCoords[0][i].y };
        }
        else {
            v.texCoord = { 0, 0 };
        }
        vertices.emplace_back(v);
    }

    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; ++j) {
            indices.push_back(face.mIndices[j]);
        }
        std::array<unsigned, 3> inds = { face.mIndices[0],face.mIndices[1],face.mIndices[2] };
    }
    std::cout << "verts and inds before simplification: " << vertices.size() << ' ' << indices.size() << '\n';

    {
        glm::vec3 bbMin(FLT_MAX), bbMax(-FLT_MAX);
        for (auto& v : vertices) {
            bbMin = glm::min(bbMin, v.pos);
            bbMax = glm::max(bbMax, v.pos);
        }
        glm::vec3 center = (bbMin + bbMax) * 0.5f;
        float extent = glm::max(bbMax.x - bbMin.x, glm::max(bbMax.y - bbMin.y, bbMax.z - bbMin.z));
        float scale = (extent > 1e-12f) ? (2.0f / extent) : 1.0f;
        for (auto& v : vertices) {
            v.pos = (v.pos - center) * scale;
        }
    }

    meshopt_optimizeVertexCache(indices.data(), indices.data(), indices.size(), vertices.size());
    std::vector<uint32_t> remap(vertices.size());
    size_t uniqueVerts = meshopt_optimizeVertexFetchRemap(remap.data(), indices.data(), indices.size(), vertices.size());
    meshopt_remapIndexBuffer(indices.data(), indices.data(), indices.size(), remap.data());
    meshopt_remapVertexBuffer(vertices.data(), vertices.data(), vertices.size(), sizeof(Vertex), remap.data());
    std::cout << "Mesh reordered for cache locality\n";

    timesLoad.push_back(tWhole.getTime());

    meshSnapshots[RENDER_ORIGINAL].verts = vertices;
    meshSnapshots[RENDER_ORIGINAL].inds = indices;
    meshSnapshots[RENDER_ORIGINAL].valid = true;

    Timer tAlgo;

    if (simplify) {
        simplifyMesh();
    } else {
        if (useGPUDecimation) {
            Timer tGPU;
            runDecimation();
            long long gpuTime = tGPU.getTime();
            meshSnapshots[RENDER_GPU].verts = vertices;
            meshSnapshots[RENDER_GPU].inds = indices;
            meshSnapshots[RENDER_GPU].valid = true;
            std::cout << "GPU Decimation: " << meshSnapshots[RENDER_ORIGINAL].inds.size()/3
                      << " -> " << indices.size()/3
                      << " triangles (" << gpuTime << " us)\n";
        }

        if (useCPUDecimation) {
            vertices = meshSnapshots[RENDER_ORIGINAL].verts;
            indices = meshSnapshots[RENDER_ORIGINAL].inds;
            runCPUDecimation();
            meshSnapshots[RENDER_CPU].verts = vertices;
            meshSnapshots[RENDER_CPU].inds = indices;
            meshSnapshots[RENDER_CPU].valid = true;
        }
    }

    timesAlgo.push_back(tAlgo.getTime());

    if (meshSnapshots[RENDER_GPU].valid)
        activeRenderMode = RENDER_GPU;
    else if (meshSnapshots[RENDER_CPU].valid)
        activeRenderMode = RENDER_CPU;
    else
        activeRenderMode = RENDER_ORIGINAL;

    vertices = meshSnapshots[activeRenderMode].verts;
    indices = meshSnapshots[activeRenderMode].inds;

    printDecimationMetrics();

    timesWhole.push_back(tWhole.getTime());
}

void App::splitMesh(std::vector<meshopt_Meshlet>& meshlets,
               std::vector<uint32_t>& meshletVertices, std::vector<Triangle>& meshletTriangles) {
    const size_t kMaxVertices = 32;
    const size_t kMaxTriangles = 64;
    const float  kConeWeight = 0.0f;
    std::vector<uint8_t> packedTriangles;

    const size_t maxMeshlets = meshopt_buildMeshletsBound(indices.size(), kMaxVertices, kMaxTriangles);

    meshlets.resize(maxMeshlets);
    meshletVertices.resize(maxMeshlets * kMaxVertices);
    packedTriangles.resize(maxMeshlets * kMaxTriangles * 3);

    size_t meshletCount = meshopt_buildMeshlets(
        meshlets.data(),
        meshletVertices.data(),
        packedTriangles.data(),
        indices.data(),
        indices.size(),
        reinterpret_cast<float*>(vertices.data()),
        vertices.size(),
        sizeof(Vertex),
        kMaxVertices,
        kMaxTriangles,
        kConeWeight);

    auto& last = meshlets[meshletCount - 1];
    meshletVertices.resize(last.vertex_offset + last.vertex_count);
    packedTriangles.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));
    meshlets.resize(meshletCount);

    for (auto& m: meshlets) {
        // Repack to uint32_t
        uint32_t triangleOffset = static_cast<uint32_t>(meshletTriangles.size());

        for (uint32_t i = 0; i < m.triangle_count; i++) {
            Triangle t{packedTriangles[i*3 + m.triangle_offset], packedTriangles[i*3+1 + m.triangle_offset], packedTriangles[i*3+2 + m.triangle_offset]};
            meshletTriangles.push_back(std::move(t));
        }

        // Update triangle offset for current meshlet
        m.triangle_offset = triangleOffset;
    }
}

void App::retrieveDataLocal(std::vector<meshopt_Meshlet>& meshlets,
    std::vector<uint32_t>& meshletVertices,
    std::vector<Triangle>& meshletTriangles) {

    VkDeviceSize bufferSize = std::max(compVertexBufferSize, compMeshletsBufferSize);
    bufferSize = std::max(bufferSize, compMeshletVerticesBufferSize);                                   
    bufferSize = std::max(bufferSize, compMeshletTrianglesBufferSize);

    void* data;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    auto copyFromBuf = [&](VkDeviceSize bufSize, void* dstData, VkBuffer& srcBuf){
        copyBuffer(srcBuf, stagingBuffer, bufSize);
        vkMapMemory(device, stagingBufferMemory, 0, bufSize, 0, &data);
        memcpy(dstData, data, (size_t)bufSize);
        vkUnmapMemory(device, stagingBufferMemory);
    };
    auto copyFromBuf2 = [&](VkCommandBuffer cmdBuf, VkDeviceSize bufSize, void* dstData, VkBuffer& srcBuf, VkDeviceSize srcOff){
        copyBuffer2(cmdBuf, srcBuf, stagingBuffer, bufSize, srcOff, 0);
        vkMapMemory(device, stagingBufferMemory, 0, bufSize, 0, &data);
        memcpy(dstData, data, (size_t)bufSize);
        vkUnmapMemory(device, stagingBufferMemory);
    };

    if(singleBuffer) {
        VkCommandBuffer cmdBuf = beginSingleTimeCommands();
        copyFromBuf2(cmdBuf, compVertexBufferSize, vertices.data(), totalBuffer, 0);
        copyFromBuf2(cmdBuf, compMeshletsBufferSize, meshlets.data(), totalBuffer, meshletsOffset);
        copyFromBuf2(cmdBuf, compMeshletVerticesBufferSize, meshletVertices.data(), totalBuffer, meshletVerticesOffset);
        copyFromBuf2(cmdBuf, compMeshletTrianglesBufferSize, meshletTriangles.data(), totalBuffer, meshletTrianglesOffset);
        endSingleTimeCommands(cmdBuf);
    } else {
        copyFromBuf(compVertexBufferSize, vertices.data(), compVertexBuffer);
        copyFromBuf(compMeshletsBufferSize, meshlets.data(), compMeshletsBuffer);
        copyFromBuf(compMeshletVerticesBufferSize, meshletVertices.data(), compMeshletVerticesBuffer);
        copyFromBuf(compMeshletTrianglesBufferSize, meshletTriangles.data(), compMeshletTrianglesBuffer);
    }

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void App::retrieveData(std::vector<meshopt_Meshlet>& meshlets,
    std::vector<uint32_t>& meshletVertices,
    std::vector<Triangle>& meshletTriangles) {
    void* data;

    auto copyFromBuf = [&](VkDeviceSize bufSize, void* dstData, VkDeviceMemory& srcMem, VkDeviceSize offset = 0){
        vkMapMemory(device, srcMem, offset, bufSize, 0, &data);
        memcpy(dstData, data, (size_t)bufSize);
        vkUnmapMemory(device, srcMem);
    };

    
    if(singleBuffer) {
        copyFromBuf(compVertexBufferSize, vertices.data(), totalBufferMemory, 0);
        copyFromBuf(compMeshletsBufferSize, meshlets.data(), totalBufferMemory, meshletsOffset);
        copyFromBuf(compMeshletVerticesBufferSize, meshletVertices.data(), totalBufferMemory, meshletVerticesOffset);
        copyFromBuf(compMeshletTrianglesBufferSize, meshletTriangles.data(), totalBufferMemory, meshletTrianglesOffset);
    } else {
        copyFromBuf(compVertexBufferSize, vertices.data(), compVertexBufferMemory);
        copyFromBuf(compMeshletsBufferSize, meshlets.data(), compMeshletsMemory);
        copyFromBuf(compMeshletVerticesBufferSize, meshletVertices.data(), compMeshletVerticesMemory);
        copyFromBuf(compMeshletTrianglesBufferSize, meshletTriangles.data(), compMeshletTrianglesMemory);
    }

}

void App::simplifyMesh() {
    std::vector<meshopt_Meshlet> meshlets;
    std::vector<uint32_t>        meshletVertices;
    std::vector<Triangle>        meshletTriangles;

    splitMesh(meshlets, meshletVertices, meshletTriangles);

    // initialize only once
    if (compVertexBuffer == VK_NULL_HANDLE) {
        compVertexBufferSize = vertices.size() * sizeof(Vertex);
        compMeshletsBufferSize = meshlets.size() * sizeof(meshopt_Meshlet);
        compMeshletVerticesBufferSize = meshletVertices.size() * sizeof(uint32_t);
        compMeshletTrianglesBufferSize = meshletTriangles.size() * sizeof(Triangle);
        meshletsOffset = compVertexBufferSize;
        meshletVerticesOffset = meshletsOffset + compMeshletsBufferSize;
        meshletTrianglesOffset = meshletVerticesOffset + compMeshletVerticesBufferSize;
        totalBufferSize = compVertexBufferSize + compMeshletsBufferSize + compMeshletVerticesBufferSize + compMeshletTrianglesBufferSize;
    
        if (deviceLocalBuffer)
            createComputeBuffersLocal(meshlets, meshletVertices, meshletTriangles);
        else
            createComputeBuffers(meshlets, meshletVertices, meshletTriangles);

        createComputeDescriptorSet();
    }

    if(deviceLocalBuffer)
        copyComputeBuffersLocal(meshlets, meshletVertices, meshletTriangles);
    else
        copyComputeBuffers(meshlets, meshletVertices, meshletTriangles);

    vkResetCommandBuffer(computeCommandBuffer, 0);
    recordComputeCommandBuffer(computeCommandBuffer, meshlets.size());

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &computeCommandBuffer;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = nullptr;

    vkResetFences(device, 1, &computeFence);

    Timer tShader;
    if (vkQueueSubmit(computeQueue, 1, &submitInfo, computeFence) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit compute command buffer!");
    };

    // wait until shaders run
    vkWaitForFences(device, 1, &computeFence, 1, INT64_MAX);
    timesShader.push_back(tShader.getTime());

    // get data back
    if (deviceLocalBuffer)
        retrieveDataLocal(meshlets, meshletVertices, meshletTriangles);
    else
        retrieveData(meshlets, meshletVertices, meshletTriangles);

    // reconstruct data from meshlets
    indices.clear();
    std::vector<Vertex> finalVertices;
    finalVertices.reserve(vertices.size());
    std::unordered_map<size_t, uint32_t> posIndexMap;
    
    // separate unique vertices and track each ones' index based on position
    for (auto& meshlet: meshlets) {
        for(int i = meshlet.vertex_offset; i < meshlet.vertex_offset + meshlet.vertex_count; ++i) {
            const size_t vertInd = meshletVertices[i];
            auto hash = std::hash<Vertex>{}(vertices[vertInd]);
            if(posIndexMap.find(hash) == posIndexMap.end()) {
                posIndexMap[hash] = finalVertices.size();
                finalVertices.push_back(vertices[vertInd]);
            }
        }
    }
    
    for (auto& meshlet: meshlets) {
        for (int i = 0; i < meshlet.triangle_count; ++i) {
            Triangle t = meshletTriangles[meshlet.triangle_offset + i];
            for(int j = 0; j < 3; ++j) {
                const size_t vertInd = t.v[j] + meshlet.vertex_offset;
                const auto& vert = vertices[vertInd];
                indices.push_back(posIndexMap[std::hash<Vertex>{}(vert)]);
            }
        } 
    }
    
    vertices = std::move(finalVertices);
    
    std::cout << "mesh simplified and reconstructed\n";
}

void App::runCPUDecimation() {
    uint32_t vertCount = static_cast<uint32_t>(vertices.size());
    uint32_t triCount = static_cast<uint32_t>(indices.size() / 3);
    size_t targetIndexCount = std::max((size_t)3, (size_t)(triCount * decimationTargetRatio) * 3);

    std::vector<uint32_t> result(indices.size());
    float resultError = 0.0f;

    Timer t;
    size_t newIndexCount = meshopt_simplify(
        result.data(), indices.data(), indices.size(),
        &vertices[0].pos.x, vertCount, sizeof(Vertex),
        targetIndexCount, std::numeric_limits<float>::max(),
        0, &resultError);
    long long elapsed = t.getTime();

    result.resize(newIndexCount);
    indices = std::move(result);

    uint32_t newTriCount = static_cast<uint32_t>(indices.size() / 3);

    for (uint32_t i = 0; i < vertCount; i++)
        vertices[i].normal = {0.0f, 0.0f, 0.0f};
    for (uint32_t ti = 0; ti < newTriCount; ti++) {
        uint32_t i0 = indices[ti * 3 + 0], i1 = indices[ti * 3 + 1], i2 = indices[ti * 3 + 2];
        if (i0 >= vertCount || i1 >= vertCount || i2 >= vertCount) continue;
        glm::vec3 fn = glm::cross(vertices[i1].pos - vertices[i0].pos,
                                   vertices[i2].pos - vertices[i0].pos);
        vertices[i0].normal += fn;
        vertices[i1].normal += fn;
        vertices[i2].normal += fn;
    }
    for (uint32_t i = 0; i < vertCount; i++) {
        float len = glm::length(vertices[i].normal);
        if (len > 1e-8f) vertices[i].normal /= len;
    }

    logCpuUs = elapsed;
    std::cout << "CPU Decimation (meshoptimizer): " << triCount << " -> " << newTriCount
              << " triangles (" << elapsed << " us)\n";
}

static float pointToTriDist(glm::vec3 p, glm::vec3 a, glm::vec3 b, glm::vec3 c) {
    glm::vec3 ab = b - a, ac = c - a, ap = p - a;
    float d1 = glm::dot(ab, ap), d2 = glm::dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return glm::length(p - a);

    glm::vec3 bp = p - b;
    float d3 = glm::dot(ab, bp), d4 = glm::dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return glm::length(p - b);

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        float v = d1 / (d1 - d3);
        return glm::length(p - (a + v * ab));
    }

    glm::vec3 cp = p - c;
    float d5 = glm::dot(ab, cp), d6 = glm::dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return glm::length(p - c);

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        float w = d2 / (d2 - d6);
        return glm::length(p - (a + w * ac));
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return glm::length(p - (b + w * (c - b)));
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom, w = vc * denom;
    return glm::length(p - (a + v * ab + w * ac));
}

static float triAngle(glm::vec3 a, glm::vec3 b, glm::vec3 c) {
    glm::vec3 ab = b - a, ac = c - a;
    float labac = glm::length(ab) * glm::length(ac);
    if (labac < 1e-12f) return 0.0f;
    return glm::degrees(std::acos(glm::clamp(glm::dot(ab, ac) / labac, -1.0f, 1.0f)));
}

static float triAspectRatio(glm::vec3 a, glm::vec3 b, glm::vec3 c) {
    float lab = glm::length(b - a);
    float lbc = glm::length(c - b);
    float lca = glm::length(a - c);
    float longest = std::max({lab, lbc, lca});
    float area = glm::length(glm::cross(b - a, c - a)) * 0.5f;
    if (area < 1e-12f) return 1e6f;
    float altitude = 2.0f * area / longest;
    return longest / altitude;
}

struct MeshMetrics {
    uint32_t triCount = 0;
    float reductionRatio = 1.0f;
    float minAngleDeg = 0.0f;
    float avgMinAngleDeg = 0.0f;
    float maxAspectRatio = 0.0f;
    float avgAspectRatio = 0.0f;
    float hausdorffDist = 0.0f;
    float avgVertDist = 0.0f;
    float avgNormalDevDeg = 0.0f;
};

struct TriGrid {
    glm::vec3 bbMin;
    float cellSize;
    int res;
    std::vector<std::vector<uint32_t>> cells;

    TriGrid(const std::vector<Vertex>& verts, const std::vector<uint32_t>& inds) {
        uint32_t triCount = static_cast<uint32_t>(inds.size() / 3);
        bbMin = glm::vec3(FLT_MAX);
        glm::vec3 bbMax(-FLT_MAX);
        for (uint32_t t = 0; t < triCount; t++) {
            for (int k = 0; k < 3; k++) {
                glm::vec3 p = verts[inds[t*3+k]].pos;
                bbMin = glm::min(bbMin, p);
                bbMax = glm::max(bbMax, p);
            }
        }
        glm::vec3 extent = bbMax - bbMin;
        float maxExt = std::max({extent.x, extent.y, extent.z, 1e-12f});
        res = std::min(128, std::max(1, (int)std::cbrt(triCount * 0.5)));
        cellSize = maxExt / res + 1e-12f;
        cells.resize(res * res * res);

        for (uint32_t t = 0; t < triCount; t++) {
            glm::vec3 tMin(FLT_MAX), tMax(-FLT_MAX);
            for (int k = 0; k < 3; k++) {
                glm::vec3 p = verts[inds[t*3+k]].pos;
                tMin = glm::min(tMin, p);
                tMax = glm::max(tMax, p);
            }
            glm::ivec3 lo = glm::clamp(glm::ivec3((tMin - bbMin) / cellSize), glm::ivec3(0), glm::ivec3(res-1));
            glm::ivec3 hi = glm::clamp(glm::ivec3((tMax - bbMin) / cellSize), glm::ivec3(0), glm::ivec3(res-1));
            for (int z = lo.z; z <= hi.z; z++)
                for (int y = lo.y; y <= hi.y; y++)
                    for (int x = lo.x; x <= hi.x; x++)
                        cells[z*res*res + y*res + x].push_back(t);
        }
    }

    float nearestDist(glm::vec3 p, const std::vector<Vertex>& verts, const std::vector<uint32_t>& inds,
                      glm::vec3* outNormal = nullptr) const {
        glm::ivec3 center = glm::clamp(glm::ivec3((p - bbMin) / cellSize), glm::ivec3(0), glm::ivec3(res-1));
        float bestDist = FLT_MAX;
        glm::vec3 bestNorm(0,0,1);

        for (int radius = 0; radius < res; radius++) {
            if (bestDist < cellSize * radius) break;
            glm::ivec3 lo = glm::max(center - glm::ivec3(radius), glm::ivec3(0));
            glm::ivec3 hi = glm::min(center + glm::ivec3(radius), glm::ivec3(res-1));
            for (int z = lo.z; z <= hi.z; z++)
                for (int y = lo.y; y <= hi.y; y++)
                    for (int x = lo.x; x <= hi.x; x++) {
                        if (radius > 0 && x > lo.x && x < hi.x && y > lo.y && y < hi.y && z > lo.z && z < hi.z)
                            continue;
                        for (uint32_t t : cells[z*res*res + y*res + x]) {
                            glm::vec3 a = verts[inds[t*3+0]].pos;
                            glm::vec3 b = verts[inds[t*3+1]].pos;
                            glm::vec3 c = verts[inds[t*3+2]].pos;
                            float d = pointToTriDist(p, a, b, c);
                            if (d < bestDist) {
                                bestDist = d;
                                if (outNormal) {
                                    glm::vec3 n = glm::cross(b-a, c-a);
                                    if (glm::length(n) > 1e-12f) bestNorm = glm::normalize(n);
                                }
                            }
                        }
                    }
        }
        if (outNormal) *outNormal = bestNorm;
        return bestDist;
    }
};

static MeshMetrics computeMetrics(
    const std::vector<Vertex>& verts,
    const std::vector<uint32_t>& inds,
    uint32_t originalTriCount,
    const std::vector<Vertex>* origVerts,
    const std::vector<uint32_t>* origInds)
{
    MeshMetrics m;
    m.triCount = static_cast<uint32_t>(inds.size() / 3);
    m.reductionRatio = (originalTriCount > 0) ? (float)m.triCount / originalTriCount : 1.0f;

    float sumMinAngle = 0.0f, sumAspect = 0.0f;
    uint32_t validTris = 0;
    m.minAngleDeg = 180.0f;
    m.maxAspectRatio = 0.0f;

    for (uint32_t t = 0; t < m.triCount; t++) {
        uint32_t i0 = inds[t*3+0], i1 = inds[t*3+1], i2 = inds[t*3+2];
        if (i0 >= verts.size() || i1 >= verts.size() || i2 >= verts.size()) continue;
        glm::vec3 a = verts[i0].pos, b = verts[i1].pos, c = verts[i2].pos;

        float area = glm::length(glm::cross(b - a, c - a));
        if (area < 1e-12f) continue;

        float a0 = triAngle(a, b, c);
        float a1 = triAngle(b, c, a);
        float a2 = triAngle(c, a, b);
        float minA = std::min({a0, a1, a2});
        m.minAngleDeg = std::min(m.minAngleDeg, minA);
        sumMinAngle += minA;

        float ar = triAspectRatio(a, b, c);
        m.maxAspectRatio = std::max(m.maxAspectRatio, ar);
        sumAspect += ar;
        validTris++;
    }
    if (validTris > 0) {
        m.avgMinAngleDeg = sumMinAngle / validTris;
        m.avgAspectRatio = sumAspect / validTris;
    }

    if (origVerts && origInds && !origInds->empty()) {
        // Forward: simplified vertices -> original surface
        TriGrid gridOrig(*origVerts, *origInds);

        std::unordered_set<uint32_t> usedVerts(inds.begin(), inds.end());
        float fwdMax = 0.0f, sumDist = 0.0f;
        uint32_t distCount = 0;
        for (uint32_t vi : usedVerts) {
            if (vi >= verts.size()) continue;
            float d = gridOrig.nearestDist(verts[vi].pos, *origVerts, *origInds);
            fwdMax = std::max(fwdMax, d);
            sumDist += d;
            distCount++;
        }

        // Reverse: original vertices -> simplified surface
        TriGrid gridSimp(verts, inds);

        std::unordered_set<uint32_t> origUsed(origInds->begin(), origInds->end());
        float revMax = 0.0f;
        for (uint32_t vi : origUsed) {
            if (vi >= origVerts->size()) continue;
            float d = gridSimp.nearestDist((*origVerts)[vi].pos, verts, inds);
            revMax = std::max(revMax, d);
        }

        m.hausdorffDist = std::max(fwdMax, revMax);
        m.avgVertDist = (distCount > 0) ? sumDist / distCount : 0.0f;

        float sumNormalDev = 0.0f;
        uint32_t normalCount = 0;
        for (uint32_t t = 0; t < m.triCount; t++) {
            uint32_t i0 = inds[t*3+0], i1 = inds[t*3+1], i2 = inds[t*3+2];
            if (i0 >= verts.size() || i1 >= verts.size() || i2 >= verts.size()) continue;
            glm::vec3 fn = glm::cross(verts[i1].pos - verts[i0].pos,
                                       verts[i2].pos - verts[i0].pos);
            if (glm::length(fn) < 1e-12f) continue;
            fn = glm::normalize(fn);
            glm::vec3 centroid = (verts[i0].pos + verts[i1].pos + verts[i2].pos) / 3.0f;

            glm::vec3 bestNormal;
            gridOrig.nearestDist(centroid, *origVerts, *origInds, &bestNormal);
            float cosA = glm::clamp(glm::dot(fn, bestNormal), -1.0f, 1.0f);
            sumNormalDev += glm::degrees(std::acos(cosA));
            normalCount++;
        }
        m.avgNormalDevDeg = (normalCount > 0) ? sumNormalDev / normalCount : 0.0f;
    }

    return m;
}

void App::printDecimationMetrics() {
    auto& orig = meshSnapshots[RENDER_ORIGINAL];
    if (!orig.valid) return;

    uint32_t origTriCount = static_cast<uint32_t>(orig.inds.size() / 3);
    bool hasGPU = meshSnapshots[RENDER_GPU].valid;
    bool hasCPU = meshSnapshots[RENDER_CPU].valid;
    if (!hasGPU && !hasCPU) return;

    std::cout << "Computing metrics for original..." << std::flush;
    auto origM = computeMetrics(orig.verts, orig.inds, origTriCount, nullptr, nullptr);
    std::cout << " done" << std::endl;

    MeshMetrics gpuM, cpuM;
    if (hasGPU) {
        std::cout << "Computing metrics for GPU mesh..." << std::flush;
        gpuM = computeMetrics(meshSnapshots[RENDER_GPU].verts, meshSnapshots[RENDER_GPU].inds,
                              origTriCount, &orig.verts, &orig.inds);
        std::cout << " done" << std::endl;
    }
    if (hasCPU) {
        std::cout << "Computing metrics for CPU mesh..." << std::flush;
        cpuM = computeMetrics(meshSnapshots[RENDER_CPU].verts, meshSnapshots[RENDER_CPU].inds,
                              origTriCount, &orig.verts, &orig.inds);
        std::cout << " done" << std::endl;
    }

    auto printRow = [&](const char* label, auto origVal, auto gpuVal, auto cpuVal, const char* fmt) {
        char buf[256];
        std::string line;
        snprintf(buf, sizeof(buf), "  %-24s", label); line += buf;
        snprintf(buf, sizeof(buf), fmt, origVal); line += buf;
        if (hasGPU) { snprintf(buf, sizeof(buf), fmt, gpuVal); line += buf; }
        if (hasCPU) { snprintf(buf, sizeof(buf), fmt, cpuVal); line += buf; }
        std::cout << line << "\n";
    };

    std::cout << "\n  === Decimation Quality Comparison ===\n";
    char hdr[256];
    snprintf(hdr, sizeof(hdr), "  %-24s%12s", "Metric", "Original");
    std::string header = hdr;
    if (hasGPU) header += "         GPU";
    if (hasCPU) header += "   CPU(mopt)";
    std::cout << header << "\n";
    std::cout << "  " << std::string(24 + 12 + (hasGPU ? 12 : 0) + (hasCPU ? 12 : 0), '-') << "\n";

    printRow("Triangles",        origM.triCount,        gpuM.triCount,        cpuM.triCount,        "%12u");
    printRow("Reduction ratio",  origM.reductionRatio,  gpuM.reductionRatio,  cpuM.reductionRatio,  "%12.4f");
    printRow("Min angle (deg)",  origM.minAngleDeg,     gpuM.minAngleDeg,     cpuM.minAngleDeg,     "%12.2f");
    printRow("Avg min angle",    origM.avgMinAngleDeg,  gpuM.avgMinAngleDeg,  cpuM.avgMinAngleDeg,  "%12.2f");
    printRow("Max aspect ratio", origM.maxAspectRatio,  gpuM.maxAspectRatio,  cpuM.maxAspectRatio,  "%12.2f");
    printRow("Avg aspect ratio", origM.avgAspectRatio,  gpuM.avgAspectRatio,  cpuM.avgAspectRatio,  "%12.2f");
    if (hasGPU || hasCPU) {
        auto printDistRow = [&](const char* label, float gpuVal, float cpuVal, const char* fmt) {
            char buf[256];
            std::string line;
            snprintf(buf, sizeof(buf), "  %-24s%12s", label, "—  ");
            line += buf;
            if (hasGPU) { snprintf(buf, sizeof(buf), fmt, gpuVal); line += buf; }
            if (hasCPU) { snprintf(buf, sizeof(buf), fmt, cpuVal); line += buf; }
            std::cout << line << "\n";
        };
        printDistRow("Hausdorff dist",  gpuM.hausdorffDist,   cpuM.hausdorffDist,   "%12.4f");
        printDistRow("Avg vertex dist", gpuM.avgVertDist,     cpuM.avgVertDist,     "%12.4f");
        printDistRow("Avg normal dev",  gpuM.avgNormalDevDeg, cpuM.avgNormalDevDeg, "%12.2f");
    }
    std::cout << "\n";

    if (hasGPU || hasCPU)
        std::cout << "  Keys: [G]PU  [C]PU  [O]riginal\n\n";

    if (hasGPU) {
        std::string baseName = modelPath;
        auto slash = baseName.find_last_of("/\\");
        if (slash != std::string::npos) baseName = baseName.substr(slash + 1);

        auto now = std::chrono::system_clock::now();
        auto epoch = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
        std::string runId = std::to_string(epoch);

        {
            std::string path = "decim_runs.csv";
            bool exists = std::ifstream(path).good();
            std::ofstream csv(path, std::ios::app);
            if (!exists) {
                csv << "run_id,experiment,model,vertices,orig_triangles,"
                       "target_ratio,cost_mode,cost_threshold,quant_bits,max_iterations,full_rebuild_freq,"
                       "gpu_final_tris,gpu_ms,gpu_total_ms,"
                       "gpu_hausdorff,gpu_avg_vert_dist,gpu_avg_normal_dev,"
                       "gpu_min_angle,gpu_avg_min_angle,gpu_max_aspect,gpu_avg_aspect,"
                       "cpu_final_tris,cpu_ms,"
                       "cpu_hausdorff,cpu_avg_vert_dist,cpu_avg_normal_dev,"
                       "cpu_min_angle,cpu_avg_min_angle,cpu_max_aspect,cpu_avg_aspect\n";
            }

            auto writeMetrics = [&](std::ofstream& f, const MeshMetrics& m) {
                f << std::setprecision(6)
                  << m.hausdorffDist << ","
                  << m.avgVertDist << ","
                  << m.avgNormalDevDeg << ","
                  << m.minAngleDeg << ","
                  << m.avgMinAngleDeg << ","
                  << m.maxAspectRatio << ","
                  << m.avgAspectRatio;
            };

            csv << runId << ","
                << experimentTag << ","
                << baseName << ","
                << static_cast<uint32_t>(orig.verts.size()) << ","
                << origTriCount << ","
                << decimationTargetRatio << ","
                << decimationCostMode << ","
                << decimationCostThreshold << ","
                << decimationCostQuantBits << ","
                << maxDecimationIterations << ","
                << decimationFullRebuildFreq << ","
                << std::fixed
                << logFinalTriCount << ","
                << std::setprecision(2) << logGpuMs << ","
                << (logTotalUs / 1000) << ",";
            writeMetrics(csv, gpuM);
            csv << ",";
            if (hasCPU) {
                csv << cpuM.triCount << ","
                    << std::setprecision(2) << (logCpuUs / 1000.0) << ",";
                writeMetrics(csv, cpuM);
            } else {
                csv << ",,,,,,,,";
            }
            csv << std::defaultfloat << "\n";
            csv.close();
            std::cout << "Run summary -> decim_runs.csv (run " << runId << ")\n";
        }

        // --- decim_iterations.csv (only with DECIM_LOG=1) ---
        if (decimationLogEnabled) {
            std::string path = "decim_iterations.csv";
            bool exists = std::ifstream(path).good();
            std::ofstream csv(path, std::ios::app);
            if (!exists) {
                csv << "run_id,iteration,edges,eligible,collapses,tri_after,avg_valence,"
                       "iter_gpu_ms,"
                       "build_adj_ms,build_edges_ms,quadrics_ms,cost_scatter_ms,"
                       "collapse_ms,mark_degen_ms,compact_ms,copyback_ms\n";
            }
            for (size_t i = 0; i < logIterData.size(); i++) {
                csv << runId << ","
                    << i << ","
                    << logIterData[i].edges << ","
                    << logIterData[i].eligible << ","
                    << logIterData[i].collapses << ","
                    << logIterData[i].triangles << ","
                    << std::fixed << std::setprecision(2) << logIterData[i].avgValence << ","
                    << std::setprecision(4) << logIterData[i].gpu_ms;
                for (int p = 0; p < 8; p++)
                    csv << "," << std::setprecision(4) << logIterData[i].pass_ms[p];
                csv << std::defaultfloat << "\n";
            }
            csv.close();
            std::cout << "Per-iteration data -> decim_iterations.csv (" << logIterData.size() << " rows)\n";
        }
    }
}


static uint32_t divUp(uint32_t n, uint32_t d) { return (n + d - 1) / d; }

void App::runDecimation() {
    const uint32_t WORKGROUP_SIZE = 256;
    uint32_t vertCount = static_cast<uint32_t>(vertices.size());
    uint32_t triCount  = static_cast<uint32_t>(indices.size() / 3);
    uint32_t originalTriCount = triCount;
    uint32_t maxEdges = triCount * 3;

    auto np2 = [](uint32_t v) { v--; v|=v>>1; v|=v>>2; v|=v>>4; v|=v>>8; v|=v>>16; v++; return v; };
    uint32_t hashMapSize = np2(std::max(vertCount, maxEdges) * 2);

    const char* modeNames[] = {"QEM", "Curvature", "Edge length"};
    std::cout << "Decimation: " << vertCount << " vertices, " << triCount << " triangles, hashMap=" << hashMapSize
              << ", costMode=" << decimationCostMode << " (" << modeNames[std::min(decimationCostMode, 2u)] << ")"
              << ", rounds=" << decimationInnerRounds << std::endl;

    std::cout << "  allocating buffers..." << std::flush;
    allocateDecimationBuffers(vertCount, triCount);
    std::cout << " done\n  writing descriptors..." << std::flush;
    writeDecimationDescriptorSets();
    std::cout << " done" << std::endl;

    std::cout << "  uploading data..." << std::flush;
    {
        auto writeVertices = [&](void* data) {
            float* dst = static_cast<float*>(data);
            for (uint32_t i = 0; i < vertCount; i++) {
                dst[i * 12 + 0]  = vertices[i].pos.x;
                dst[i * 12 + 1]  = vertices[i].pos.y;
                dst[i * 12 + 2]  = vertices[i].pos.z;
                dst[i * 12 + 3]  = 0.0f;
                dst[i * 12 + 4]  = vertices[i].normal.x;
                dst[i * 12 + 5]  = vertices[i].normal.y;
                dst[i * 12 + 6]  = vertices[i].normal.z;
                dst[i * 12 + 7]  = 0.0f;
                dst[i * 12 + 8]  = vertices[i].texCoord.x;
                dst[i * 12 + 9]  = vertices[i].texCoord.y;
                dst[i * 12 + 10] = 0.0f;
                dst[i * 12 + 11] = 0.0f;
            }
        };

        VkDeviceSize vertBufSize = decimationBufSizes[DB_VERTEX];
        if (decimationUseDeviceLocal) {
            VkBuffer stagingBuf;
            VkDeviceMemory stagingMem;
            createBuffer(vertBufSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                stagingBuf, stagingMem);
            void* data;
            vkMapMemory(device, stagingMem, 0, vertBufSize, 0, &data);
            writeVertices(data);
            vkUnmapMemory(device, stagingMem);
            copyBuffer(stagingBuf, decimationBufs[DB_VERTEX], vertBufSize);
            vkDestroyBuffer(device, stagingBuf, nullptr);
            vkFreeMemory(device, stagingMem, nullptr);
        } else {
            void* data;
            vkMapMemory(device, decimationMem[DB_VERTEX], 0, vertBufSize, 0, &data);
            writeVertices(data);
            vkUnmapMemory(device, decimationMem[DB_VERTEX]);
        }
    }

    {
        VkDeviceSize idxBufSize = (VkDeviceSize)triCount * 3 * sizeof(uint32_t);
        if (decimationUseDeviceLocal) {
            VkBuffer stagingBuf;
            VkDeviceMemory stagingMem;
            createBuffer(idxBufSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                stagingBuf, stagingMem);
            void* data;
            vkMapMemory(device, stagingMem, 0, idxBufSize, 0, &data);
            memcpy(data, indices.data(), idxBufSize);
            vkUnmapMemory(device, stagingMem);
            copyBuffer(stagingBuf, decimationBufs[DB_INDEX], idxBufSize);
            vkDestroyBuffer(device, stagingBuf, nullptr);
            vkFreeMemory(device, stagingMem, nullptr);
        } else {
            void* data;
            vkMapMemory(device, decimationMem[DB_INDEX], 0, idxBufSize, 0, &data);
            memcpy(data, indices.data(), idxBufSize);
            vkUnmapMemory(device, decimationMem[DB_INDEX]);
        }
    }
    std::cout << " done" << std::endl;

    auto bindDescSets = [&](VkCommandBuffer cmd) {
        VkDescriptorSet sets[] = { decimationDescSet0, decimationDescSet1 };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            decimationPipelineLayout, 0, 2, sets, 0, nullptr);
    };

    auto computeBarrier = [&](VkCommandBuffer cmd) {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    };

    auto transferToComputeBarrier = [&](VkCommandBuffer cmd) {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    };

    auto dispatchPass = [&](VkCommandBuffer cmd, uint32_t passIdx, const DecimationPushConstants& pc, uint32_t workgroups) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, decimationPipelines[passIdx]);
        vkCmdPushConstants(cmd, decimationPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(DecimationPushConstants), &pc);
        vkCmdDispatch(cmd, workgroups, 1, 1);
    };

    auto submitAndWait = [&](VkCommandBuffer cmd) {
        vkEndCommandBuffer(cmd);
        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        vkResetFences(device, 1, &computeFence);
        if (vkQueueSubmit(computeQueue, 1, &submitInfo, computeFence) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit decimation command buffer!");
        }
        VkResult fenceResult = vkWaitForFences(device, 1, &computeFence, VK_TRUE, 5000000000ULL); // 5s timeout
        if (fenceResult == VK_TIMEOUT) {
            std::cerr << "ERROR: GPU hung (fence timeout after 5s)!\n";
            vkDeviceWaitIdle(device);
        } else if (fenceResult == VK_ERROR_DEVICE_LOST) {
            std::cerr << "ERROR: GPU device lost!\n";
        }
    };

    auto beginCmd = [&]() -> VkCommandBuffer {
        vkResetCommandBuffer(computeCommandBuffer, 0);
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(computeCommandBuffer, &beginInfo);
        bindDescSets(computeCommandBuffer);
        return computeCommandBuffer;
    };

    auto readCounter = [&](uint32_t index) -> uint32_t {
        if (decimationUseDeviceLocal) {
            return static_cast<uint32_t*>(counterReadbackMapped)[index];
        } else {
            void* data;
            vkMapMemory(device, decimationMem[DB_COUNTER], 0, decimationBufSizes[DB_COUNTER], 0, &data);
            uint32_t val = static_cast<uint32_t*>(data)[index];
            vkUnmapMemory(device, decimationMem[DB_COUNTER]);
            return val;
        }
    };

    std::cout << "passes 1-2" << std::flush;
    {
        VkCommandBuffer cmd = beginCmd();

        vkCmdFillBuffer(cmd, decimationBufs[DB_HASHMAP_VERTEX], 0, decimationBufSizes[DB_HASHMAP_VERTEX], 0xFFFFFFFF);
        vkCmdFillBuffer(cmd, decimationBufs[DB_HASHMAP_POSITION], 0, decimationBufSizes[DB_HASHMAP_POSITION], 0xFFFFFFFF);
        vkCmdFillBuffer(cmd, decimationBufs[DB_HASHMAP_EDGE], 0, decimationBufSizes[DB_HASHMAP_EDGE], 0xFFFFFFFF);
        // clear vertex flags
        vkCmdFillBuffer(cmd, decimationBufs[DB_VERTEX_FLAGS], 0, decimationBufSizes[DB_VERTEX_FLAGS], 0);
        vkCmdFillBuffer(cmd, decimationBufs[DB_ALIVE], 0, decimationBufSizes[DB_ALIVE], 1);
        vkCmdFillBuffer(cmd, decimationBufs[DB_POS_MAP], 0, decimationBufSizes[DB_POS_MAP], 0);
        vkCmdFillBuffer(cmd, decimationBufs[DB_COUNTER], 0, decimationBufSizes[DB_COUNTER], 0);

        transferToComputeBarrier(cmd);

        DecimationPushConstants pc{};
        pc.vertexCount = vertCount;
        pc.triangleCount = triCount;
        pc.edgeCount = 0;
        pc.hashMapSize = hashMapSize;
        pc.costThreshold = decimationCostThreshold;
        pc.iteration = 0;
        pc.costMode = decimationCostMode;
        pc.costQuantBits = decimationCostQuantBits;
        pc.targetTriCount = std::max(1u, (uint32_t)(triCount * decimationTargetRatio));

        // pass 1
        dispatchPass(cmd, 0, pc, divUp(vertCount, WORKGROUP_SIZE));
        computeBarrier(cmd);

        // pass 2
        dispatchPass(cmd, 1, pc, divUp(triCount, WORKGROUP_SIZE));
        computeBarrier(cmd);

        submitAndWait(cmd);
    }
    std::cout << " done" << std::endl;

    VkDeviceSize edgeHashMapSize = decimationBufSizes[DB_HASHMAP_EDGE];

    uint32_t triDispatchWGs = divUp(triCount, WORKGROUP_SIZE);
    uint32_t edgeDispatchWGs = divUp(maxEdges, WORKGROUP_SIZE);

    const uint32_t COUNTERS_PER_ITER = 7;
    const uint32_t PASSES_PER_ITER = 9;
    const uint32_t TS_PER_ITER = PASSES_PER_ITER + 1;
    VkBuffer iterStatsBuf = VK_NULL_HANDLE;
    VkDeviceMemory iterStatsMem = VK_NULL_HANDLE;
    VkQueryPool iterTimestampPool = VK_NULL_HANDLE;
    if (decimationLogEnabled) {
        VkDeviceSize iterStatsBufSize = (VkDeviceSize)maxDecimationIterations * COUNTERS_PER_ITER * sizeof(uint32_t);
        createBuffer(iterStatsBufSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            iterStatsBuf, iterStatsMem);

        VkQueryPoolCreateInfo qpInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpInfo.queryCount = maxDecimationIterations * TS_PER_ITER;
        vkCreateQueryPool(device, &qpInfo, nullptr, &iterTimestampPool);
    }

    Timer gpuTimer;
    VkCommandBuffer cmd = beginCmd();

    {
        uint32_t initVal = triCount;
        vkCmdUpdateBuffer(cmd, decimationBufs[DB_COUNTER],
            2 * sizeof(uint32_t), sizeof(uint32_t), &initVal);
        vkCmdUpdateBuffer(cmd, decimationBufs[DB_COUNTER],
            5 * sizeof(uint32_t), sizeof(uint32_t), &initVal);
    }

    vkCmdResetQueryPool(cmd, timestampQueryPool, 0, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 0);

    if (decimationLogEnabled) {
        vkCmdResetQueryPool(cmd, iterTimestampPool, 0, maxDecimationIterations * TS_PER_ITER);
    }

    const uint32_t fullRebuildFreq = decimationFullRebuildFreq;
    const bool debugSync = (std::getenv("DECIM_DEBUG") && std::string(std::getenv("DECIM_DEBUG")) != "0");

    if (debugSync) {
        std::cout << "  [DEBUG] Per-pass sync mode enabled\n";
    }

    for (uint32_t iteration = 0; iteration < maxDecimationIterations; iteration++) {
        bool isLight = (iteration > 0) && (iteration % fullRebuildFreq != 0);

        DecimationPushConstants pc{};
        pc.vertexCount = vertCount;
        pc.triangleCount = triCount;
        pc.edgeCount = maxEdges;
        pc.hashMapSize = hashMapSize;
        pc.costThreshold = decimationCostThreshold;
        pc.iteration = iteration;
        pc.costMode = decimationCostMode;
        pc.costQuantBits = decimationCostQuantBits;
        pc.targetTriCount = std::max(1u, (uint32_t)(triCount * decimationTargetRatio));
        pc.lightIteration = isLight ? 1u : 0u;

        if (debugSync) {
            const char* passNames[] = {
                "P3:adjacency", "P4:edges", "P4b:boundary",
                "P5:quadrics", "P6:cost+scatter", "P7:collapse",
                "P8:mark_degen", "P9:compact", "P10:copyback"
            };
            struct PassInfo { uint32_t pipeIdx; uint32_t wgs; bool skipIfLight; };
            PassInfo passes[] = {
                {2, triDispatchWGs, false},   // P3
                {3, triDispatchWGs, true},    // P4
                {4, edgeDispatchWGs, true},   // P4b
                {5, triDispatchWGs, false},   // P5
                {6, edgeDispatchWGs, false},  // P6
                {7, edgeDispatchWGs, false},  // P7 
                {8, triDispatchWGs, false},   // P8 
                {9, triDispatchWGs, true},    // P9 
                {10, triDispatchWGs, true},   // P10 
            };

            submitAndWait(cmd);
            cmd = beginCmd();
            if (isLight) {
                vkCmdFillBuffer(cmd, decimationBufs[DB_COUNTER], 4, 4, 0);
                vkCmdFillBuffer(cmd, decimationBufs[DB_COUNTER], 12, 8, 0);
            } else {
                vkCmdFillBuffer(cmd, decimationBufs[DB_COUNTER], 0, 8, 0);
                vkCmdFillBuffer(cmd, decimationBufs[DB_COUNTER], 12, 8, 0);
                vkCmdFillBuffer(cmd, decimationBufs[DB_HASHMAP_EDGE], 0, edgeHashMapSize, 0xFFFFFFFF);
            }
            vkCmdFillBuffer(cmd, decimationBufs[DB_COUNTER], 24, 4, 0);
            vkCmdFillBuffer(cmd, decimationBufs[DB_ADJ_HEAD], 0, decimationBufSizes[DB_ADJ_HEAD], 0xFFFFFFFF);
            vkCmdFillBuffer(cmd, decimationBufs[DB_TRI_EDGE], 0, decimationBufSizes[DB_TRI_EDGE], 0);
            vkCmdFillBuffer(cmd, decimationBufs[DB_QUADRIC], 0, decimationBufSizes[DB_QUADRIC], 0);
            if (iteration == 0) {
                uint32_t initVal = triCount;
                vkCmdUpdateBuffer(cmd, decimationBufs[DB_COUNTER],
                    2 * sizeof(uint32_t), sizeof(uint32_t), &initVal);
            }
            submitAndWait(cmd);

            for (int p = 0; p < 9; p++) {
                if (passes[p].skipIfLight && isLight) continue;
                std::cerr << "  [DEBUG] iter=" << iteration << " " << passNames[p] << " ..." << std::flush;
                cmd = beginCmd();
                transferToComputeBarrier(cmd);
                dispatchPass(cmd, passes[p].pipeIdx, pc, passes[p].wgs);
                computeBarrier(cmd);
                submitAndWait(cmd);
                std::cerr << " ok";

                if (decimationUseDeviceLocal) {
                    cmd = beginCmd();
                    VkBufferCopy rgn{}; rgn.size = 5 * sizeof(uint32_t);
                    vkCmdCopyBuffer(cmd, decimationBufs[DB_COUNTER], counterReadbackBuf, 1, &rgn);
                    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
                    submitAndWait(cmd);
                    auto* c = static_cast<uint32_t*>(counterReadbackMapped);
                    std::cerr << " [edges=" << c[0] << " collapse=" << c[1]
                              << " tris=" << c[2] << " elig=" << c[3] << " compact=" << c[4] << "]";
                }
                std::cerr << "\n";
            }

            if (!isLight) {
                cmd = beginCmd();
                vkCmdFillBuffer(cmd, decimationBufs[DB_ALIVE], 0, decimationBufSizes[DB_ALIVE], 1);
                vkCmdFillBuffer(cmd, decimationBufs[DB_COUNTER], 4, 4, 0);
                submitAndWait(cmd);
            }

            cmd = beginCmd();
            continue;
        }

        if (isLight) {
            vkCmdFillBuffer(cmd, decimationBufs[DB_COUNTER], 4, 4, 0);
            vkCmdFillBuffer(cmd, decimationBufs[DB_COUNTER], 12, 8, 0);
        } else {
            vkCmdFillBuffer(cmd, decimationBufs[DB_COUNTER], 0, 8, 0);
            vkCmdFillBuffer(cmd, decimationBufs[DB_COUNTER], 12, 8, 0);
            vkCmdFillBuffer(cmd, decimationBufs[DB_HASHMAP_EDGE], 0, edgeHashMapSize, 0xFFFFFFFF);
        }
        vkCmdFillBuffer(cmd, decimationBufs[DB_COUNTER], 24, 4, 0);  // VERTEX_ACTIVE
        vkCmdFillBuffer(cmd, decimationBufs[DB_ADJ_HEAD], 0, decimationBufSizes[DB_ADJ_HEAD], 0xFFFFFFFF);
        vkCmdFillBuffer(cmd, decimationBufs[DB_TRI_EDGE], 0, decimationBufSizes[DB_TRI_EDGE], 0);  // valence
        if (!isLight)
            vkCmdFillBuffer(cmd, decimationBufs[DB_QUADRIC], 0, decimationBufSizes[DB_QUADRIC], 0);

        if (iteration == 0) {
            uint32_t initVal = triCount;
            vkCmdUpdateBuffer(cmd, decimationBufs[DB_COUNTER],
                2 * sizeof(uint32_t), sizeof(uint32_t), &initVal);   // TRIANGLE_COUNT
            vkCmdUpdateBuffer(cmd, decimationBufs[DB_COUNTER],
                5 * sizeof(uint32_t), sizeof(uint32_t), &initVal);   // ALIVE_ESTIMATE
        }

        transferToComputeBarrier(cmd);

        uint32_t tsBase = iteration * TS_PER_ITER;
        auto tsWrite = [&](uint32_t localIdx) {
            if (decimationLogEnabled)
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    iterTimestampPool, tsBase + localIdx);
        };

        tsWrite(0);  // iteration start (after clears)
        dispatchPass(cmd, 2, pc, triDispatchWGs);    // P3: build adjacency + valence
        computeBarrier(cmd);
        tsWrite(1);

        if (!isLight) {
            dispatchPass(cmd, 3, pc, triDispatchWGs);    // P4: build edges
            computeBarrier(cmd);
        }
        tsWrite(2);

        if (!isLight) {
            dispatchPass(cmd, 4, pc, edgeDispatchWGs);   // P4b: flag boundary
            computeBarrier(cmd);
        }
        tsWrite(3);

        dispatchPass(cmd, 5, pc, triDispatchWGs);    // P5: quadrics + init triDescriptor
        computeBarrier(cmd);
        tsWrite(4);
        dispatchPass(cmd, 6, pc, edgeDispatchWGs);   // P6: cost + scatter (fused)
        computeBarrier(cmd);
        tsWrite(5);
        dispatchPass(cmd, 7, pc, edgeDispatchWGs);   // P9: collapse + mark dirty
        computeBarrier(cmd);
        tsWrite(6);
        dispatchPass(cmd, 8, pc, triDispatchWGs);    // P10: mark degenerate
        computeBarrier(cmd);
        tsWrite(7);

        if (!isLight) {
            dispatchPass(cmd, 9, pc, triDispatchWGs);    // P11: compact
            computeBarrier(cmd);
            tsWrite(8);
            dispatchPass(cmd, 10, pc, triDispatchWGs);   // P12: copyback
            computeBarrier(cmd);
            {
                VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
            }
            vkCmdFillBuffer(cmd, decimationBufs[DB_ALIVE], 0, decimationBufSizes[DB_ALIVE], 1);
            vkCmdFillBuffer(cmd, decimationBufs[DB_COUNTER], 4, 4, 0);
            tsWrite(9);
        } else {
            tsWrite(8);
            tsWrite(9);
        }

        if (decimationLogEnabled) {
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            VkBufferCopy region{};
            region.srcOffset = 0;
            region.dstOffset = (VkDeviceSize)iteration * COUNTERS_PER_ITER * sizeof(uint32_t);
            region.size = COUNTERS_PER_ITER * sizeof(uint32_t);
            vkCmdCopyBuffer(cmd, decimationBufs[DB_COUNTER], iterStatsBuf, 1, &region);

            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
    }

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 1);

    // Copy final counters to host-visible readback buffer
    if (decimationUseDeviceLocal) {
        // Ensure all writes to counter buffer (from compute shaders and transfers)
        // are visible before the copy
        {
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        VkBufferCopy region{};
        region.size = 256;
        vkCmdCopyBuffer(cmd, decimationBufs[DB_COUNTER], counterReadbackBuf, 1, &region);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    submitAndWait(cmd);

    long long totalUs = gpuTimer.getTime();
    uint32_t aliveEstimate = readCounter(5);
    triCount = readCounter(2);

    double totalGpuMs = 0;
    {
        uint64_t ts[2] = {0, 0};
        VkResult qr = vkGetQueryPoolResults(device, timestampQueryPool, 0, 2, sizeof(ts), ts,
            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        if (qr == VK_SUCCESS && ts[1] > ts[0])
            totalGpuMs = (double)(ts[1] - ts[0]) * timestampPeriodNs * 1e-6;
    }

    // Read per-iteration stats (only when logging)
    struct IterStats {
        uint32_t edges, collapses, triangles, eligible, compacted, alive;
        uint32_t vertexActive;
        double gpu_ms;
        double pass_ms[9];
    };
    std::vector<IterStats> iterData;
    if (decimationLogEnabled) {
        VkDeviceSize iterStatsBufSize = (VkDeviceSize)maxDecimationIterations * COUNTERS_PER_ITER * sizeof(uint32_t);
        iterData.resize(maxDecimationIterations);

        void* data;
        vkMapMemory(device, iterStatsMem, 0, iterStatsBufSize, 0, &data);
        uint32_t* p = static_cast<uint32_t*>(data);
        for (uint32_t i = 0; i < maxDecimationIterations; i++) {
            iterData[i].edges     = p[i * COUNTERS_PER_ITER + 0];
            iterData[i].collapses = p[i * COUNTERS_PER_ITER + 1];
            iterData[i].triangles = p[i * COUNTERS_PER_ITER + 2];
            iterData[i].eligible  = p[i * COUNTERS_PER_ITER + 3];
            iterData[i].compacted = p[i * COUNTERS_PER_ITER + 4];
            iterData[i].alive     = p[i * COUNTERS_PER_ITER + 5];
            iterData[i].vertexActive  = p[i * COUNTERS_PER_ITER + 6];
        }
        vkUnmapMemory(device, iterStatsMem);
        vkDestroyBuffer(device, iterStatsBuf, nullptr);
        vkFreeMemory(device, iterStatsMem, nullptr);

        uint32_t totalTsCount = maxDecimationIterations * TS_PER_ITER;
        std::vector<uint64_t> iterTs(totalTsCount);
        vkGetQueryPoolResults(device, iterTimestampPool, 0, totalTsCount,
            iterTs.size() * sizeof(uint64_t), iterTs.data(),
            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

        double toMs = timestampPeriodNs * 1e-6;
        for (uint32_t i = 0; i < maxDecimationIterations; i++) {
            uint32_t base = i * TS_PER_ITER;
            iterData[i].gpu_ms = (double)(iterTs[base + PASSES_PER_ITER] - iterTs[base]) * toMs;
            for (uint32_t p = 0; p < PASSES_PER_ITER; p++) {
                iterData[i].pass_ms[p] = (double)(iterTs[base + p + 1] - iterTs[base + p]) * toMs;
            }
        }
        vkDestroyQueryPool(device, iterTimestampPool, nullptr);

        // Print per-pass average breakdown
        const char* passNames[] = {
            "build_adj", "build_edges", "flag_bndry", "quadrics", "cost+scatter",
            "collapse", "mark_degen", "compact", "copyback"
        };
        double avgPass[PASSES_PER_ITER] = {};
        for (auto& it : iterData)
            for (uint32_t p = 0; p < PASSES_PER_ITER; p++) avgPass[p] += it.pass_ms[p];
        double totalAvg = 0;
        for (uint32_t p = 0; p < PASSES_PER_ITER; p++) {
            avgPass[p] /= maxDecimationIterations;
            totalAvg += avgPass[p];
        }
        std::cout << "  Per-pass avg breakdown:\n";
        for (uint32_t p = 0; p < PASSES_PER_ITER; p++) {
            double pct = (totalAvg > 0) ? (avgPass[p] / totalAvg * 100.0) : 0.0;
            std::cout << "    " << std::setw(14) << std::left << passNames[p]
                      << std::right << std::fixed << std::setprecision(3) << std::setw(8) << avgPass[p] << " ms"
                      << "  (" << std::setprecision(1) << std::setw(5) << pct << "%)\n";
        }
        std::cout << "    " << std::setw(14) << std::left << "TOTAL"
                  << std::right << std::setprecision(3) << std::setw(8) << totalAvg << " ms\n"
                  << std::defaultfloat;
    }

    uint32_t lastEligible = readCounter(3);
    uint32_t lastCollapses = readCounter(1);
    uint32_t lastEdges = readCounter(0);
    uint32_t lastVertActive = readCounter(6);
    double lastAvgValence = (lastVertActive > 0) ? 3.0 * aliveEstimate / lastVertActive : 0.0;
    std::cout << "  " << maxDecimationIterations << " iterations: "
              << triCount << " tris, alive ~" << aliveEstimate << " (was " << originalTriCount << ")"
              << "  gpu=" << std::fixed << std::setprecision(0) << totalGpuMs << "ms"
              << "  total=" << totalUs / 1000 << "ms"
              << "  last iter: " << lastEdges << " edges, " << lastEligible << " eligible, " << lastCollapses << " collapses"
              << ", avg valence " << std::setprecision(2) << lastAvgValence
              << std::defaultfloat << std::endl;

    std::cout << "Decimation complete: " << originalTriCount << " -> " << triCount
              << " triangles (alive estimate: " << aliveEstimate << ")\n";

    {
        auto readVertices = [&](void* data) {
            float* src = static_cast<float*>(data);
            for (uint32_t i = 0; i < vertCount; i++) {
                vertices[i].pos.x      = src[i * 12 + 0];
                vertices[i].pos.y      = src[i * 12 + 1];
                vertices[i].pos.z      = src[i * 12 + 2];
                vertices[i].normal.x   = src[i * 12 + 4];
                vertices[i].normal.y   = src[i * 12 + 5];
                vertices[i].normal.z   = src[i * 12 + 6];
                vertices[i].texCoord.x = src[i * 12 + 8];
                vertices[i].texCoord.y = src[i * 12 + 9];
            }
        };

        VkDeviceSize vertBufSize = decimationBufSizes[DB_VERTEX];
        if (decimationUseDeviceLocal) {
            VkBuffer stagingBuf;
            VkDeviceMemory stagingMem;
            createBuffer(vertBufSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                stagingBuf, stagingMem);
            copyBuffer(decimationBufs[DB_VERTEX], stagingBuf, vertBufSize);
            void* data;
            vkMapMemory(device, stagingMem, 0, vertBufSize, 0, &data);
            readVertices(data);
            vkUnmapMemory(device, stagingMem);
            vkDestroyBuffer(device, stagingBuf, nullptr);
            vkFreeMemory(device, stagingMem, nullptr);
        } else {
            void* data;
            vkMapMemory(device, decimationMem[DB_VERTEX], 0, vertBufSize, 0, &data);
            readVertices(data);
            vkUnmapMemory(device, decimationMem[DB_VERTEX]);
        }
    }

    {
        if (triCount == 0) {
            std::cerr << "WARNING: triCount readback is 0 - counter sync issue?\n";
            triCount = 1;
        }
        VkDeviceSize idxBufSize = (VkDeviceSize)triCount * 3 * sizeof(uint32_t);
        indices.resize(triCount * 3);
        if (decimationUseDeviceLocal) {
            VkBuffer stagingBuf;
            VkDeviceMemory stagingMem;
            createBuffer(idxBufSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                stagingBuf, stagingMem);
            copyBuffer(decimationBufs[DB_POS_INDEX], stagingBuf, idxBufSize);
            void* data;
            vkMapMemory(device, stagingMem, 0, idxBufSize, 0, &data);
            memcpy(indices.data(), data, idxBufSize);
            vkUnmapMemory(device, stagingMem);
            vkDestroyBuffer(device, stagingBuf, nullptr);
            vkFreeMemory(device, stagingMem, nullptr);
        } else {
            void* data;
            vkMapMemory(device, decimationMem[DB_POS_INDEX], 0, idxBufSize, 0, &data);
            memcpy(indices.data(), data, idxBufSize);
            vkUnmapMemory(device, decimationMem[DB_POS_INDEX]);
        }
    }

    for (uint32_t i = 0; i < vertCount; i++) {
        vertices[i].normal = {0.0f, 0.0f, 0.0f};
    }
    for (uint32_t t = 0; t < triCount; t++) {
        uint32_t i0 = indices[t * 3 + 0];
        uint32_t i1 = indices[t * 3 + 1];
        uint32_t i2 = indices[t * 3 + 2];
        if (i0 >= vertCount || i1 >= vertCount || i2 >= vertCount) continue;
        glm::vec3 e1 = vertices[i1].pos - vertices[i0].pos;
        glm::vec3 e2 = vertices[i2].pos - vertices[i0].pos;
        glm::vec3 fn = glm::cross(e1, e2);
        vertices[i0].normal += fn;
        vertices[i1].normal += fn;
        vertices[i2].normal += fn;
    }
    for (uint32_t i = 0; i < vertCount; i++) {
        float len = glm::length(vertices[i].normal);
        if (len > 1e-8f) vertices[i].normal /= len;
    }

    if (decimationLogEnabled) {
        logIterData.resize(iterData.size());
        for (size_t i = 0; i < iterData.size(); i++) {
            logIterData[i].edges     = iterData[i].edges;
            logIterData[i].collapses = iterData[i].collapses;
            logIterData[i].triangles = iterData[i].alive;
            logIterData[i].eligible  = iterData[i].eligible;
            logIterData[i].compacted = iterData[i].compacted;
            logIterData[i].alive     = iterData[i].alive;
            logIterData[i].avgValence = (iterData[i].vertexActive > 0)
                ? 3.0 * iterData[i].alive / iterData[i].vertexActive : 0.0;
            logIterData[i].gpu_ms    = iterData[i].gpu_ms;
            for (int p = 0; p < 8; p++) logIterData[i].pass_ms[p] = iterData[i].pass_ms[p];
        }
        logGpuMs = totalGpuMs;
        logTotalUs = totalUs;
        logOrigTriCount = originalTriCount;
        logFinalTriCount = aliveEstimate;
    }
}

// ============================================================================
// Interactive (step-through) decimation — separate slow path for visualization
// ============================================================================

void App::readbackDecimationState() {
    uint32_t vertCount = interactiveVertCount;

    // Copy counters to readback
    if (decimationUseDeviceLocal) {
        VkCommandBuffer cmd2;
        {
            vkResetCommandBuffer(computeCommandBuffer, 0);
            VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(computeCommandBuffer, &beginInfo);

            VkBufferCopy region{};
            region.size = 256;
            vkCmdCopyBuffer(computeCommandBuffer, decimationBufs[DB_COUNTER], counterReadbackBuf, 1, &region);
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(computeCommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            vkEndCommandBuffer(computeCommandBuffer);
            VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &computeCommandBuffer;
            vkResetFences(device, 1, &computeFence);
            vkQueueSubmit(computeQueue, 1, &submitInfo, computeFence);
            vkWaitForFences(device, 1, &computeFence, VK_TRUE, UINT64_MAX);
        }
    }

    auto readCounter = [&](uint32_t index) -> uint32_t {
        if (decimationUseDeviceLocal) {
            return static_cast<uint32_t*>(counterReadbackMapped)[index];
        } else {
            void* data;
            vkMapMemory(device, decimationMem[DB_COUNTER], 0, decimationBufSizes[DB_COUNTER], 0, &data);
            uint32_t val = static_cast<uint32_t*>(data)[index];
            vkUnmapMemory(device, decimationMem[DB_COUNTER]);
            return val;
        }
    };

    uint32_t aliveEst = readCounter(5);
    interactiveTriCount = readCounter(2);

    // Read back vertices
    {
        auto readVertices = [&](void* data) {
            float* src = static_cast<float*>(data);
            for (uint32_t i = 0; i < vertCount; i++) {
                vertices[i].pos.x      = src[i * 12 + 0];
                vertices[i].pos.y      = src[i * 12 + 1];
                vertices[i].pos.z      = src[i * 12 + 2];
                vertices[i].normal.x   = src[i * 12 + 4];
                vertices[i].normal.y   = src[i * 12 + 5];
                vertices[i].normal.z   = src[i * 12 + 6];
                vertices[i].texCoord.x = src[i * 12 + 8];
                vertices[i].texCoord.y = src[i * 12 + 9];
            }
        };

        VkDeviceSize vertBufSize = decimationBufSizes[DB_VERTEX];
        if (decimationUseDeviceLocal) {
            VkBuffer stagingBuf;
            VkDeviceMemory stagingMem;
            createBuffer(vertBufSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                stagingBuf, stagingMem);
            copyBuffer(decimationBufs[DB_VERTEX], stagingBuf, vertBufSize);
            void* data;
            vkMapMemory(device, stagingMem, 0, vertBufSize, 0, &data);
            readVertices(data);
            vkUnmapMemory(device, stagingMem);
            vkDestroyBuffer(device, stagingBuf, nullptr);
            vkFreeMemory(device, stagingMem, nullptr);
        } else {
            void* data;
            vkMapMemory(device, decimationMem[DB_VERTEX], 0, vertBufSize, 0, &data);
            readVertices(data);
            vkUnmapMemory(device, decimationMem[DB_VERTEX]);
        }
    }

    // Read back from posIndices (position-canonical), same reason as batch path
    uint32_t triCount = interactiveTriCount;
    {
        VkDeviceSize idxBufSize = (VkDeviceSize)triCount * 3 * sizeof(uint32_t);
        indices.resize(triCount * 3);
        if (decimationUseDeviceLocal) {
            VkBuffer stagingBuf;
            VkDeviceMemory stagingMem;
            createBuffer(idxBufSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                stagingBuf, stagingMem);
            copyBuffer(decimationBufs[DB_POS_INDEX], stagingBuf, idxBufSize);
            void* data;
            vkMapMemory(device, stagingMem, 0, idxBufSize, 0, &data);
            memcpy(indices.data(), data, idxBufSize);
            vkUnmapMemory(device, stagingMem);
            vkDestroyBuffer(device, stagingBuf, nullptr);
            vkFreeMemory(device, stagingMem, nullptr);
        } else {
            void* data;
            vkMapMemory(device, decimationMem[DB_POS_INDEX], 0, idxBufSize, 0, &data);
            memcpy(indices.data(), data, idxBufSize);
            vkUnmapMemory(device, decimationMem[DB_POS_INDEX]);
        }
    }

    // Recompute normals
    for (uint32_t i = 0; i < vertCount; i++)
        vertices[i].normal = {0, 0, 0};
    for (uint32_t t = 0; t < triCount; t++) {
        uint32_t i0 = indices[t*3+0], i1 = indices[t*3+1], i2 = indices[t*3+2];
        if (i0 >= vertCount || i1 >= vertCount || i2 >= vertCount) continue;
        glm::vec3 fn = glm::cross(vertices[i1].pos - vertices[i0].pos, vertices[i2].pos - vertices[i0].pos);
        vertices[i0].normal += fn;
        vertices[i1].normal += fn;
        vertices[i2].normal += fn;
    }
    for (uint32_t i = 0; i < vertCount; i++) {
        float len = glm::length(vertices[i].normal);
        if (len > 1e-8f) vertices[i].normal /= len;
    }

    uint32_t lastEdges = readCounter(0);
    uint32_t lastCollapses = readCounter(1);
    uint32_t lastEligible = readCounter(3);
    std::cout << "[step " << interactiveIteration << " COLLAPSE] "
              << triCount << " tris, alive ~" << aliveEst
              << " (was " << interactiveOrigTriCount << ")  "
              << lastEdges << " edges, " << lastEligible << " eligible, "
              << lastCollapses << " collapses" << std::endl;

}

void App::initInteractiveDecimation() {
    // Always start from the original mesh
    auto& orig = meshSnapshots[RENDER_ORIGINAL];
    if (!orig.valid) {
        std::cout << "No original mesh available for interactive decimation\n";
        return;
    }
    vertices = orig.verts;
    indices = orig.inds;

    interactiveVertCount = static_cast<uint32_t>(vertices.size());
    interactiveTriCount  = static_cast<uint32_t>(indices.size() / 3);
    interactiveOrigTriCount = interactiveTriCount;
    interactiveMaxEdges = interactiveTriCount * 3;
    interactiveIteration = 0;

    auto np2 = [](uint32_t v) { v--; v|=v>>1; v|=v>>2; v|=v>>4; v|=v>>8; v|=v>>16; v++; return v; };
    interactiveHashMapSize = np2(std::max(interactiveVertCount, interactiveMaxEdges) * 2);

    std::cout << "Interactive decimation init: " << interactiveVertCount << " verts, "
              << interactiveTriCount << " tris\n";

    allocateDecimationBuffers(interactiveVertCount, interactiveTriCount);
    writeDecimationDescriptorSets();

    const uint32_t WORKGROUP_SIZE = 256;
    uint32_t vertCount = interactiveVertCount;
    uint32_t triCount = interactiveTriCount;

    // Upload vertices
    {
        auto writeVertices = [&](void* data) {
            float* dst = static_cast<float*>(data);
            for (uint32_t i = 0; i < vertCount; i++) {
                dst[i*12+0]  = vertices[i].pos.x;     dst[i*12+1]  = vertices[i].pos.y;
                dst[i*12+2]  = vertices[i].pos.z;     dst[i*12+3]  = 0.0f;
                dst[i*12+4]  = vertices[i].normal.x;  dst[i*12+5]  = vertices[i].normal.y;
                dst[i*12+6]  = vertices[i].normal.z;  dst[i*12+7]  = 0.0f;
                dst[i*12+8]  = vertices[i].texCoord.x; dst[i*12+9] = vertices[i].texCoord.y;
                dst[i*12+10] = 0.0f;                   dst[i*12+11] = 0.0f;
            }
        };
        VkDeviceSize vertBufSize = decimationBufSizes[DB_VERTEX];
        if (decimationUseDeviceLocal) {
            VkBuffer stagingBuf; VkDeviceMemory stagingMem;
            createBuffer(vertBufSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                stagingBuf, stagingMem);
            void* data;
            vkMapMemory(device, stagingMem, 0, vertBufSize, 0, &data);
            writeVertices(data);
            vkUnmapMemory(device, stagingMem);
            copyBuffer(stagingBuf, decimationBufs[DB_VERTEX], vertBufSize);
            vkDestroyBuffer(device, stagingBuf, nullptr);
            vkFreeMemory(device, stagingMem, nullptr);
        } else {
            void* data;
            vkMapMemory(device, decimationMem[DB_VERTEX], 0, vertBufSize, 0, &data);
            writeVertices(data);
            vkUnmapMemory(device, decimationMem[DB_VERTEX]);
        }
    }

    // Upload indices
    {
        VkDeviceSize idxBufSize = (VkDeviceSize)triCount * 3 * sizeof(uint32_t);
        if (decimationUseDeviceLocal) {
            VkBuffer stagingBuf; VkDeviceMemory stagingMem;
            createBuffer(idxBufSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                stagingBuf, stagingMem);
            void* data;
            vkMapMemory(device, stagingMem, 0, idxBufSize, 0, &data);
            memcpy(data, indices.data(), idxBufSize);
            vkUnmapMemory(device, stagingMem);
            copyBuffer(stagingBuf, decimationBufs[DB_INDEX], idxBufSize);
            vkDestroyBuffer(device, stagingBuf, nullptr);
            vkFreeMemory(device, stagingMem, nullptr);
        } else {
            void* data;
            vkMapMemory(device, decimationMem[DB_INDEX], 0, idxBufSize, 0, &data);
            memcpy(data, indices.data(), idxBufSize);
            vkUnmapMemory(device, decimationMem[DB_INDEX]);
        }
    }

    // Run Phase 1: hash + dedup
    {
        vkResetCommandBuffer(computeCommandBuffer, 0);
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(computeCommandBuffer, &beginInfo);

        VkDescriptorSet sets[] = { decimationDescSet0, decimationDescSet1 };
        vkCmdBindDescriptorSets(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            decimationPipelineLayout, 0, 2, sets, 0, nullptr);

        vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_HASHMAP_VERTEX], 0, decimationBufSizes[DB_HASHMAP_VERTEX], 0xFFFFFFFF);
        vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_HASHMAP_POSITION], 0, decimationBufSizes[DB_HASHMAP_POSITION], 0xFFFFFFFF);
        vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_HASHMAP_EDGE], 0, decimationBufSizes[DB_HASHMAP_EDGE], 0xFFFFFFFF);
        vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_VERTEX_FLAGS], 0, decimationBufSizes[DB_VERTEX_FLAGS], 0);
        vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_ALIVE], 0, decimationBufSizes[DB_ALIVE], 1);
        vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_POS_MAP], 0, decimationBufSizes[DB_POS_MAP], 0);
        vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_COUNTER], 0, decimationBufSizes[DB_COUNTER], 0);

        {
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(computeCommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        DecimationPushConstants pc{};
        pc.vertexCount = vertCount;
        pc.triangleCount = triCount;
        pc.hashMapSize = interactiveHashMapSize;
        pc.costThreshold = decimationCostThreshold;
        pc.costMode = decimationCostMode;
        pc.costQuantBits = decimationCostQuantBits;
        pc.targetTriCount = std::max(1u, (uint32_t)(triCount * decimationTargetRatio));
        pc.lightIteration = 0;

        auto dispatchPass = [&](uint32_t passIdx, uint32_t workgroups) {
            vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, decimationPipelines[passIdx]);
            vkCmdPushConstants(computeCommandBuffer, decimationPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(DecimationPushConstants), &pc);
            vkCmdDispatch(computeCommandBuffer, workgroups, 1, 1);
        };
        auto computeBarrier = [&]() {
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(computeCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        };

        dispatchPass(0, divUp(vertCount, WORKGROUP_SIZE));
        computeBarrier();
        dispatchPass(1, divUp(triCount, WORKGROUP_SIZE));
        computeBarrier();

        // Initialize COUNTER_TRIANGLE_COUNT and ALIVE_ESTIMATE
        uint32_t initVal = triCount;
        vkCmdUpdateBuffer(computeCommandBuffer, decimationBufs[DB_COUNTER],
            2 * sizeof(uint32_t), sizeof(uint32_t), &initVal);   // TRIANGLE_COUNT
        vkCmdUpdateBuffer(computeCommandBuffer, decimationBufs[DB_COUNTER],
            5 * sizeof(uint32_t), sizeof(uint32_t), &initVal);   // ALIVE_ESTIMATE
        {
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(computeCommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        vkEndCommandBuffer(computeCommandBuffer);
        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &computeCommandBuffer;
        vkResetFences(device, 1, &computeFence);
        vkQueueSubmit(computeQueue, 1, &submitInfo, computeFence);
        vkWaitForFences(device, 1, &computeFence, VK_TRUE, UINT64_MAX);
    }

    interactiveDecimReady = true;

    // Store original mesh as GPU snapshot and update render buffers
    meshSnapshots[RENDER_GPU].verts = vertices;
    meshSnapshots[RENDER_GPU].inds = indices;
    meshSnapshots[RENDER_GPU].valid = true;
    updateMeshBuffersForMode(RENDER_GPU);
    activeRenderMode = RENDER_GPU;

    std::cout << "Interactive decimation ready. Press [N] to step, [O] for original.\n";
}

void App::stepInteractiveScatter() {
    if (!interactiveDecimReady) return;

    // Save positions before this step for change map
    prevPositions.resize(vertices.size());
    for (size_t i = 0; i < vertices.size(); i++)
        prevPositions[i] = vertices[i].pos;

    const uint32_t WORKGROUP_SIZE = 256;
    uint32_t vertCount = interactiveVertCount;
    uint32_t triCount = interactiveTriCount;

    uint32_t triDispatchWGs = divUp(interactiveOrigTriCount, WORKGROUP_SIZE);
    uint32_t edgeDispatchWGs = divUp(interactiveMaxEdges, WORKGROUP_SIZE);

    vkResetCommandBuffer(computeCommandBuffer, 0);
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(computeCommandBuffer, &beginInfo);

    VkDescriptorSet sets[] = { decimationDescSet0, decimationDescSet1 };
    vkCmdBindDescriptorSets(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        decimationPipelineLayout, 0, 2, sets, 0, nullptr);

    auto dispatchPass = [&](uint32_t passIdx, const DecimationPushConstants& pc, uint32_t workgroups) {
        vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, decimationPipelines[passIdx]);
        vkCmdPushConstants(computeCommandBuffer, decimationPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(DecimationPushConstants), &pc);
        vkCmdDispatch(computeCommandBuffer, workgroups, 1, 1);
    };
    auto computeBarrier = [&]() {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(computeCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    };
    auto transferToComputeBarrier = [&]() {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(computeCommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    };

    bool isLight = (interactiveIteration > 0) && (interactiveIteration % decimationFullRebuildFreq != 0);

    DecimationPushConstants pc{};
    pc.vertexCount = vertCount;
    pc.triangleCount = interactiveOrigTriCount;
    pc.edgeCount = interactiveMaxEdges;
    pc.hashMapSize = interactiveHashMapSize;
    pc.costThreshold = decimationCostThreshold;
    pc.iteration = interactiveIteration;
    pc.costMode = decimationCostMode;
    pc.costQuantBits = decimationCostQuantBits;
    pc.targetTriCount = std::max(1u, (uint32_t)(interactiveOrigTriCount * decimationTargetRatio));
    pc.lightIteration = isLight ? 1u : 0u;

    // Clear per-iteration state
    if (isLight) {
        vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_COUNTER], 4, 4, 0);   // COLLAPSE_COUNT
        vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_COUNTER], 12, 8, 0);
    } else {
        vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_COUNTER], 0, 8, 0);
        vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_COUNTER], 12, 8, 0);
        vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_HASHMAP_EDGE], 0, decimationBufSizes[DB_HASHMAP_EDGE], 0xFFFFFFFF);
    }
    vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_COUNTER], 24, 4, 0);  // VERTEX_ACTIVE
    vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_ADJ_HEAD], 0, decimationBufSizes[DB_ADJ_HEAD], 0xFFFFFFFF);
    vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_TRI_EDGE], 0, decimationBufSizes[DB_TRI_EDGE], 0);
    vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_QUADRIC], 0, decimationBufSizes[DB_QUADRIC], 0);
    transferToComputeBarrier();

    // Passes 2-6: build topology, compute costs, scatter to triDescriptor
    dispatchPass(2, pc, triDispatchWGs);    // build adjacency + valence
    computeBarrier();
    if (!isLight) {
        dispatchPass(3, pc, triDispatchWGs);    // build edges
        computeBarrier();
        dispatchPass(4, pc, edgeDispatchWGs);   // flag boundary
        computeBarrier();
    }
    dispatchPass(5, pc, triDispatchWGs);    // quadrics + init triDescriptor
    computeBarrier();
    dispatchPass(6, pc, edgeDispatchWGs);   // cost + scatter
    computeBarrier();

    vkEndCommandBuffer(computeCommandBuffer);
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &computeCommandBuffer;
    vkResetFences(device, 1, &computeFence);
    vkQueueSubmit(computeQueue, 1, &submitInfo, computeFence);
    vkWaitForFences(device, 1, &computeFence, VK_TRUE, UINT64_MAX);

    // --- Read back scatter-fight state (before collapse) ---
    // Counters
    if (decimationUseDeviceLocal) {
        vkResetCommandBuffer(computeCommandBuffer, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(computeCommandBuffer, &bi);
        VkBufferCopy region{}; region.size = 256;
        vkCmdCopyBuffer(computeCommandBuffer, decimationBufs[DB_COUNTER], counterReadbackBuf, 1, &region);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(computeCommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        vkEndCommandBuffer(computeCommandBuffer);
        VkSubmitInfo si2{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si2.commandBufferCount = 1; si2.pCommandBuffers = &computeCommandBuffer;
        vkResetFences(device, 1, &computeFence);
        vkQueueSubmit(computeQueue, 1, &si2, computeFence);
        vkWaitForFences(device, 1, &computeFence, VK_TRUE, UINT64_MAX);
    }
    auto readCounter = [&](uint32_t index) -> uint32_t {
        if (decimationUseDeviceLocal)
            return static_cast<uint32_t*>(counterReadbackMapped)[index];
        void* data;
        vkMapMemory(device, decimationMem[DB_COUNTER], 0, decimationBufSizes[DB_COUNTER], 0, &data);
        uint32_t val = static_cast<uint32_t*>(data)[index];
        vkUnmapMemory(device, decimationMem[DB_COUNTER]);
        return val;
    };

    uint32_t edgeCount = readCounter(0);
    uint32_t eligible = readCounter(3);
    uint32_t vertActive = readCounter(6);
    stepVis.edgeCount = edgeCount;
    stepVis.triCount = triCount;

    auto readbackBuffer = [&](DecimationBuffer dbIdx, void* dst, VkDeviceSize size) {
        if (size == 0) return;
        if (decimationUseDeviceLocal) {
            VkBuffer stagingBuf; VkDeviceMemory stagingMem;
            createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                stagingBuf, stagingMem);
            copyBuffer(decimationBufs[dbIdx], stagingBuf, size);
            void* mapped;
            vkMapMemory(device, stagingMem, 0, size, 0, &mapped);
            memcpy(dst, mapped, size);
            vkUnmapMemory(device, stagingMem);
            vkDestroyBuffer(device, stagingBuf, nullptr);
            vkFreeMemory(device, stagingMem, nullptr);
        } else {
            void* mapped;
            vkMapMemory(device, decimationMem[dbIdx], 0, size, 0, &mapped);
            memcpy(dst, mapped, size);
            vkUnmapMemory(device, decimationMem[dbIdx]);
        }
    };

    stepVis.edges.resize(edgeCount * 2);
    readbackBuffer(DB_EDGE, stepVis.edges.data(), (VkDeviceSize)edgeCount * 2 * sizeof(uint32_t));

    stepVis.edgeFlags.resize(edgeCount);
    readbackBuffer(DB_EDGE_FLAG, stepVis.edgeFlags.data(), (VkDeviceSize)edgeCount * sizeof(uint32_t));

    stepVis.edgeCost.resize(edgeCount);
    readbackBuffer(DB_EDGE_COST, stepVis.edgeCost.data(), (VkDeviceSize)edgeCount * sizeof(float));

    stepVis.vertexFlags.resize(vertCount);
    readbackBuffer(DB_VERTEX_FLAGS, stepVis.vertexFlags.data(), (VkDeviceSize)vertCount * sizeof(uint32_t));

    stepVis.posIndices.resize(triCount * 3);
    readbackBuffer(DB_POS_INDEX, stepVis.posIndices.data(), (VkDeviceSize)triCount * 3 * sizeof(uint32_t));

    stepVis.triDescriptor.resize(triCount);
    readbackBuffer(DB_TRI_DESCRIPTOR, stepVis.triDescriptor.data(), (VkDeviceSize)triCount * sizeof(uint64_t));

    stepVis.valid = true;
    stepPhase = 1;

    double avgValence = (vertActive > 0) ? 3.0 * triCount / vertActive : 0.0;
    std::cout << "[step " << interactiveIteration << " SCATTER] "
              << triCount << " tris, " << edgeCount << " edges, "
              << eligible << " eligible, avg valence "
              << std::fixed << std::setprecision(2) << avgValence
              << std::defaultfloat << std::endl;

    if (flatDebugMode) {
        buildFlatDebugMesh();
    }
}

void App::stepInteractiveCollapse() {
    if (!interactiveDecimReady) return;

    const uint32_t WORKGROUP_SIZE = 256;
    uint32_t triDispatchWGs = divUp(interactiveOrigTriCount, WORKGROUP_SIZE);
    uint32_t edgeDispatchWGs = divUp(interactiveMaxEdges, WORKGROUP_SIZE);

    bool isLight = (interactiveIteration > 0) && (interactiveIteration % decimationFullRebuildFreq != 0);

    DecimationPushConstants pc{};
    pc.vertexCount = interactiveVertCount;
    pc.triangleCount = interactiveOrigTriCount;
    pc.edgeCount = interactiveMaxEdges;
    pc.hashMapSize = interactiveHashMapSize;
    pc.costThreshold = decimationCostThreshold;
    pc.iteration = interactiveIteration;
    pc.costMode = decimationCostMode;
    pc.costQuantBits = decimationCostQuantBits;
    pc.targetTriCount = std::max(1u, (uint32_t)(interactiveOrigTriCount * decimationTargetRatio));
    pc.lightIteration = isLight ? 1u : 0u;

    vkResetCommandBuffer(computeCommandBuffer, 0);
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(computeCommandBuffer, &beginInfo);

    VkDescriptorSet sets[] = { decimationDescSet0, decimationDescSet1 };
    vkCmdBindDescriptorSets(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        decimationPipelineLayout, 0, 2, sets, 0, nullptr);

    auto dispatchPass = [&](uint32_t passIdx, const DecimationPushConstants& pc2, uint32_t workgroups) {
        vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, decimationPipelines[passIdx]);
        vkCmdPushConstants(computeCommandBuffer, decimationPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(DecimationPushConstants), &pc2);
        vkCmdDispatch(computeCommandBuffer, workgroups, 1, 1);
    };
    auto computeBarrier = [&]() {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(computeCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    };

    // Passes 7-8: collapse, mark degenerate
    dispatchPass(7, pc, edgeDispatchWGs);   // collapse + mark dirty
    computeBarrier();
    dispatchPass(8, pc, triDispatchWGs);    // mark degenerate
    computeBarrier();

    vkEndCommandBuffer(computeCommandBuffer);
    {
        VkSubmitInfo si2{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si2.commandBufferCount = 1; si2.pCommandBuffers = &computeCommandBuffer;
        vkResetFences(device, 1, &computeFence);
        vkQueueSubmit(computeQueue, 1, &si2, computeFence);
        vkWaitForFences(device, 1, &computeFence, VK_TRUE, UINT64_MAX);
    }

    // Build the flat debug mesh from pre-compaction state:
    // triDescriptor[t] still uses the scatter-phase index space,
    // so we read the alive/vertex/index/flag data now (before compaction scrambles order).
    uint32_t oldTriCount = interactiveTriCount;
    if (flatDebugMode && stepVis.valid) {
        auto readbackBuf = [&](DecimationBuffer dbIdx, void* dst, VkDeviceSize size) {
            if (decimationUseDeviceLocal) {
                VkBuffer sb; VkDeviceMemory sm;
                createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, sb, sm);
                copyBuffer(decimationBufs[dbIdx], sb, size);
                void* m; vkMapMemory(device, sm, 0, size, 0, &m);
                memcpy(dst, m, size);
                vkUnmapMemory(device, sm);
                vkDestroyBuffer(device, sb, nullptr); vkFreeMemory(device, sm, nullptr);
            } else {
                void* m; vkMapMemory(device, decimationMem[dbIdx], 0, size, 0, &m);
                memcpy(dst, m, size);
                vkUnmapMemory(device, decimationMem[dbIdx]);
            }
        };

        // Read aliveFlags, vertex positions, indices, vertex flags (pre-compaction)
        std::vector<uint32_t> alive(oldTriCount);
        readbackBuf(DB_ALIVE, alive.data(), (VkDeviceSize)oldTriCount * sizeof(uint32_t));

        stepVis.vertexFlags.resize(interactiveVertCount);
        readbackBuf(DB_VERTEX_FLAGS, stepVis.vertexFlags.data(),
            (VkDeviceSize)interactiveVertCount * sizeof(uint32_t));

        // Read post-collapse vertices for geometry
        std::vector<Vertex> collapseVerts(interactiveVertCount);
        {
            VkDeviceSize vbSize = decimationBufSizes[DB_VERTEX];
            std::vector<float> raw(interactiveVertCount * 12);
            readbackBuf(DB_VERTEX, raw.data(), vbSize);
            for (uint32_t i = 0; i < interactiveVertCount; i++) {
                collapseVerts[i].pos    = {raw[i*12+0], raw[i*12+1], raw[i*12+2]};
                collapseVerts[i].normal = {raw[i*12+4], raw[i*12+5], raw[i*12+6]};
                collapseVerts[i].texCoord = {raw[i*12+8], raw[i*12+9]};
            }
        }

        std::vector<uint32_t> collapseInds(oldTriCount * 3);
        readbackBuf(DB_INDEX, collapseInds.data(), (VkDeviceSize)oldTriCount * 3 * sizeof(uint32_t));

        // Build flat debug mesh from pre-compaction data: same index space as triDescriptor
        const uint32_t FLAG_COLLAPSED_BIT = (1u << 2);
        const uint64_t DESCRIPTOR_MAX = 0xFFFFFFFFFFFFFFFFull;

        auto edgeColor = [](uint32_t edgeIdx) -> glm::vec3 {
            uint32_t h = edgeIdx;
            h = ((h >> 16) ^ h) * 0x45d9f3bu;
            h = ((h >> 16) ^ h) * 0x45d9f3bu;
            h = (h >> 16) ^ h;
            float hue = (h & 0xFFFF) / 65535.0f;
            float sat = 0.75f, val = 0.9f;
            float h6 = hue * 6.0f;
            int hi = (int)h6;
            float f = h6 - hi;
            float p = val * (1.0f - sat);
            float q = val * (1.0f - sat * f);
            float t2 = val * (1.0f - sat * (1.0f - f));
            switch (hi % 6) {
                case 0: return {val, t2, p};
                case 1: return {q, val, p};
                case 2: return {p, val, t2};
                case 3: return {p, q, val};
                case 4: return {t2, p, val};
                default: return {val, p, q};
            }
        };

        // Count alive triangles and build the mesh
        uint32_t aliveCount = 0;
        for (uint32_t t = 0; t < oldTriCount; t++)
            if (alive[t] != 0) aliveCount++;

        std::vector<Vertex> verts(aliveCount * 3);
        std::vector<uint32_t> inds(aliveCount * 3);
        uint32_t wonTris = 0, collapsedTris = 0, outIdx = 0;

        for (uint32_t t = 0; t < oldTriCount; t++) {
            if (alive[t] == 0) continue;

            glm::vec3 col(0.3f, 0.3f, 0.3f);
            if (t < stepVis.triDescriptor.size()) {
                uint64_t desc = stepVis.triDescriptor[t];
                if (desc != DESCRIPTOR_MAX) {
                    uint32_t winnerEdge = (uint32_t)(desc & 0xFFFFFFFFull);
                    col = edgeColor(winnerEdge);
                    wonTris++;

                    if (winnerEdge < stepVis.edgeCount) {
                        uint32_t ev0 = stepVis.edges[winnerEdge * 2 + 0];
                        uint32_t ev1 = stepVis.edges[winnerEdge * 2 + 1];
                        bool collapsed =
                            (ev0 < stepVis.vertexFlags.size() &&
                             (stepVis.vertexFlags[ev0] & FLAG_COLLAPSED_BIT)) ||
                            (ev1 < stepVis.vertexFlags.size() &&
                             (stepVis.vertexFlags[ev1] & FLAG_COLLAPSED_BIT));
                        if (collapsed) {
                            col = glm::mix(col, glm::vec3(1.0f), 0.35f);
                            collapsedTris++;
                        }
                    }
                }
            }

            for (int k = 0; k < 3; k++) {
                uint32_t vi = collapseInds[t * 3 + k];
                Vertex v = (vi < interactiveVertCount) ? collapseVerts[vi] : Vertex{};
                v.normal = col;
                verts[outIdx * 3 + k] = v;
                inds[outIdx * 3 + k] = outIdx * 3 + k;
            }
            outIdx++;
        }

        // Upload flat debug mesh (edge lines from scatter phase are kept)
        vkDeviceWaitIdle(device);
        if (flatDebugSnap.vertBuf != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, flatDebugSnap.vertBuf, nullptr);
            vkFreeMemory(device, flatDebugSnap.vertMem, nullptr);
            flatDebugSnap.vertBuf = VK_NULL_HANDLE;
        }
        if (flatDebugSnap.idxBuf != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, flatDebugSnap.idxBuf, nullptr);
            vkFreeMemory(device, flatDebugSnap.idxMem, nullptr);
            flatDebugSnap.idxBuf = VK_NULL_HANDLE;
        }
        flatDebugSnap.verts = std::move(verts);
        flatDebugSnap.inds = std::move(inds);
        if (!flatDebugSnap.verts.empty()) {
            VkDeviceSize vertSize = flatDebugSnap.verts.size() * sizeof(Vertex);
            createAndCopyBufferLocal(vertSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                flatDebugSnap.verts.data(), flatDebugSnap.vertBuf, flatDebugSnap.vertMem);
            VkDeviceSize idxSize = flatDebugSnap.inds.size() * sizeof(uint32_t);
            createAndCopyBufferLocal(idxSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                flatDebugSnap.inds.data(), flatDebugSnap.idxBuf, flatDebugSnap.idxMem);
        }
        flatDebugSnap.valid = true;

        std::cout << "Flat debug [COLLAPSE]: " << aliveCount << " tris, "
                  << wonTris << " claimed by edges (" << collapsedTris << " collapsed)" << std::endl;
    }

    // Passes 9-10: compact + copyback (full iterations only)
    if (!isLight) {
        vkResetCommandBuffer(computeCommandBuffer, 0);
        VkCommandBufferBeginInfo bi2{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi2.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(computeCommandBuffer, &bi2);
        VkDescriptorSet sets2[] = { decimationDescSet0, decimationDescSet1 };
        vkCmdBindDescriptorSets(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            decimationPipelineLayout, 0, 2, sets2, 0, nullptr);

        dispatchPass(9, pc, triDispatchWGs);    // compact
        computeBarrier();
        dispatchPass(10, pc, triDispatchWGs);   // copyback
        computeBarrier();
        {
            VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(computeCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
        }
        vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_ALIVE], 0, decimationBufSizes[DB_ALIVE], 1);
        vkCmdFillBuffer(computeCommandBuffer, decimationBufs[DB_COUNTER], 4, 4, 0);

        vkEndCommandBuffer(computeCommandBuffer);
        VkSubmitInfo si3{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si3.commandBufferCount = 1; si3.pCommandBuffers = &computeCommandBuffer;
        vkResetFences(device, 1, &computeFence);
        vkQueueSubmit(computeQueue, 1, &si3, computeFence);
        vkWaitForFences(device, 1, &computeFence, VK_TRUE, UINT64_MAX);
    }

    interactiveIteration++;

    readbackDecimationState();

    meshSnapshots[RENDER_GPU].verts = vertices;
    meshSnapshots[RENDER_GPU].inds = indices;
    meshSnapshots[RENDER_GPU].valid = true;

    if (heatMapMode == 1) {
        computeHeatMapColors();
    } else if (heatMapMode == 2) {
        computeChangeMap();
    } else {
        updateMeshBuffersForMode(RENDER_GPU);
    }

    stepPhase = 2;
}

void App::stepInteractiveDecimation() {
    if (!interactiveDecimReady) return;
    if (flatDebugMode) {
        if (stepPhase == 1) {
            stepInteractiveCollapse();
        } else {
            stepInteractiveScatter();
        }
    } else {
        stepInteractiveScatter();
        stepInteractiveCollapse();
    }
}

void App::computeChangeMap() {
    auto& snap = meshSnapshots[activeRenderMode];
    if (!snap.valid || snap.verts.empty() || prevPositions.empty()) return;

    size_t V = snap.verts.size();
    size_t T = snap.inds.size() / 3;

    savedNormals.resize(V);
    for (size_t i = 0; i < V; i++)
        savedNormals[i] = snap.verts[i].normal;

    // Mark vertices that moved
    std::vector<bool> changed(V, false);
    size_t minV = std::min(V, prevPositions.size());
    uint32_t changedCount = 0;
    for (size_t i = 0; i < minV; i++) {
        if (glm::length(snap.verts[i].pos - prevPositions[i]) > 1e-8f) {
            changed[i] = true;
            changedCount++;
        }
    }

    // Mark triangles that touch a changed vertex
    std::vector<bool> triAffected(T, false);
    uint32_t affectedCount = 0;
    for (size_t t = 0; t < T; t++) {
        uint32_t i0 = snap.inds[t*3+0], i1 = snap.inds[t*3+1], i2 = snap.inds[t*3+2];
        if (i0 >= V || i1 >= V || i2 >= V) continue;
        if (changed[i0] || changed[i1] || changed[i2]) {
            triAffected[t] = true;
            affectedCount++;
        }
    }

    // Color per vertex: affected = orange, untouched = dark blue
    std::vector<uint32_t> vertAffected(V, 0);
    for (size_t t = 0; t < T; t++) {
        if (!triAffected[t]) continue;
        uint32_t i0 = snap.inds[t*3+0], i1 = snap.inds[t*3+1], i2 = snap.inds[t*3+2];
        if (i0 < V) vertAffected[i0] = 1;
        if (i1 < V) vertAffected[i1] = 1;
        if (i2 < V) vertAffected[i2] = 1;
    }

    for (size_t i = 0; i < V; i++) {
        if (changed[i])
            snap.verts[i].normal = glm::vec3(1.0f, 0.3f, 0.1f);  // orange: vertex moved
        else if (vertAffected[i])
            snap.verts[i].normal = glm::vec3(1.0f, 0.8f, 0.2f);  // yellow: neighbor moved
        else
            snap.verts[i].normal = glm::vec3(0.25f, 0.25f, 0.25f); // gray: untouched
    }

    updateMeshBuffersForMode(activeRenderMode);
    std::cout << "Change map: " << changedCount << " vertices moved, "
              << affectedCount << "/" << T << " triangles affected" << std::endl;
}

void App::computeHeatMapColors() {
    auto& snap = meshSnapshots[activeRenderMode];
    if (!snap.valid || snap.verts.empty()) return;

    size_t V = snap.verts.size();
    size_t T = snap.inds.size() / 3;

    savedNormals.resize(V);
    for (size_t i = 0; i < V; i++)
        savedNormals[i] = snap.verts[i].normal;

    // Compute per-vertex angle sum for discrete curvature
    std::vector<float> angleSum(V, 0.0f);
    std::vector<uint32_t> valence(V, 0);

    for (size_t t = 0; t < T; t++) {
        uint32_t i0 = snap.inds[t*3+0], i1 = snap.inds[t*3+1], i2 = snap.inds[t*3+2];
        if (i0 >= V || i1 >= V || i2 >= V) continue;
        glm::vec3 p0 = snap.verts[i0].pos, p1 = snap.verts[i1].pos, p2 = snap.verts[i2].pos;

        auto safeAngle = [](glm::vec3 a, glm::vec3 b) -> float {
            float la = glm::length(a), lb = glm::length(b);
            if (la < 1e-10f || lb < 1e-10f) return 0.0f;
            return std::acos(glm::clamp(glm::dot(a, b) / (la * lb), -1.0f, 1.0f));
        };

        angleSum[i0] += safeAngle(p1 - p0, p2 - p0); valence[i0]++;
        angleSum[i1] += safeAngle(p0 - p1, p2 - p1); valence[i1]++;
        angleSum[i2] += safeAngle(p0 - p2, p1 - p2); valence[i2]++;
    }

    // Convert to curvature and find range
    std::vector<float> curvature(V, 0.0f);
    float maxCurv = 0.0f;
    for (size_t i = 0; i < V; i++) {
        if (valence[i] == 0) continue;
        curvature[i] = std::abs(2.0f * 3.14159265f - angleSum[i]);
        maxCurv = std::max(maxCurv, curvature[i]);
    }
    if (maxCurv < 1e-8f) maxCurv = 1.0f;

    // Map to color: blue (flat, low cost) → green (medium) → red (sharp, high cost)
    for (size_t i = 0; i < V; i++) {
        float t = glm::clamp(curvature[i] / (maxCurv * 0.3f), 0.0f, 1.0f);
        glm::vec3 color;
        if (t < 0.5f) {
            float s = t * 2.0f;
            color = glm::mix(glm::vec3(0.1f, 0.2f, 0.9f), glm::vec3(0.1f, 0.9f, 0.2f), s);
        } else {
            float s = (t - 0.5f) * 2.0f;
            color = glm::mix(glm::vec3(0.1f, 0.9f, 0.2f), glm::vec3(0.9f, 0.1f, 0.1f), s);
        }
        snap.verts[i].normal = color;
    }

    updateMeshBuffersForMode(activeRenderMode);
    std::cout << "Heat map computed: max curvature = " << maxCurv
              << ", vertices = " << V << std::endl;
}

void App::buildFlatDebugMesh() {
    auto& src = meshSnapshots[activeRenderMode];
    if (!src.valid || src.verts.empty() || src.inds.empty()) return;

    size_t triCount = src.inds.size() / 3;
    std::vector<Vertex> verts(triCount * 3);
    std::vector<uint32_t> inds(triCount * 3);

    // Deterministic high-contrast color from edge index via hash
    auto edgeColor = [](uint32_t edgeIdx) -> glm::vec3 {
        uint32_t h = edgeIdx;
        h = ((h >> 16) ^ h) * 0x45d9f3bu;
        h = ((h >> 16) ^ h) * 0x45d9f3bu;
        h = (h >> 16) ^ h;
        float hue = (h & 0xFFFF) / 65535.0f;
        float s = 0.75f, v = 0.9f;
        float h6 = hue * 6.0f;
        int hi = (int)h6;
        float f = h6 - hi;
        float p = v * (1.0f - s);
        float q = v * (1.0f - s * f);
        float t = v * (1.0f - s * (1.0f - f));
        switch (hi % 6) {
            case 0: return {v, t, p};
            case 1: return {q, v, p};
            case 2: return {p, v, t};
            case 3: return {p, q, v};
            case 4: return {t, p, v};
            default: return {v, p, q};
        }
    };

    uint32_t wonTris = 0;
    const uint64_t DESCRIPTOR_MAX = 0xFFFFFFFFFFFFFFFFull;

    bool hasDescriptors = stepVis.valid && !stepVis.triDescriptor.empty();

    for (size_t t = 0; t < triCount; t++) {
        glm::vec3 col(0.3f, 0.3f, 0.3f);

        if (hasDescriptors && t < stepVis.triDescriptor.size()) {
            uint64_t desc = stepVis.triDescriptor[t];
            if (desc != DESCRIPTOR_MAX) {
                uint32_t winnerEdge = (uint32_t)(desc & 0xFFFFFFFFull);
                col = edgeColor(winnerEdge);
                wonTris++;
            }
        } else if (!hasDescriptors) {
            uint32_t h = (uint32_t)t;
            h = ((h >> 16) ^ h) * 0x45d9f3bu;
            h = ((h >> 16) ^ h) * 0x45d9f3bu;
            h = (h >> 16) ^ h;
            col.r = ((h >>  0) & 0xFF) / 255.0f * 0.7f + 0.15f;
            col.g = ((h >>  8) & 0xFF) / 255.0f * 0.7f + 0.15f;
            col.b = ((h >> 16) & 0xFF) / 255.0f * 0.7f + 0.15f;
        }

        for (int k = 0; k < 3; k++) {
            uint32_t si = src.inds[t * 3 + k];
            Vertex v = src.verts[si];
            v.normal = col;
            verts[t * 3 + k] = v;
            inds[t * 3 + k] = (uint32_t)(t * 3 + k);
        }
    }

    // Build per-edge line vertices (2 verts per edge, each with the edge's own color)
    std::vector<Vertex> edgeLineVerts;
    if (hasDescriptors && !stepVis.edges.empty()) {
        std::unordered_map<uint32_t, glm::vec3> posMap;
        for (size_t t = 0; t < triCount; t++) {
            for (int k = 0; k < 3; k++) {
                uint32_t posIdx = stepVis.posIndices[t * 3 + k];
                if (posMap.find(posIdx) == posMap.end()) {
                    uint32_t si = src.inds[t * 3 + k];
                    posMap[posIdx] = src.verts[si].pos;
                }
            }
        }

        uint32_t ec = stepVis.edgeCount;
        edgeLineVerts.reserve(ec * 2);
        for (uint32_t e = 0; e < ec; e++) {
            uint32_t p0 = stepVis.edges[e * 2 + 0];
            uint32_t p1 = stepVis.edges[e * 2 + 1];
            auto it0 = posMap.find(p0);
            auto it1 = posMap.find(p1);
            if (it0 == posMap.end() || it1 == posMap.end()) continue;
            glm::vec3 col = edgeColor(e);
            Vertex v0{}, v1{};
            v0.pos = it0->second;
            v0.normal = col;
            v1.pos = it1->second;
            v1.normal = col;
            edgeLineVerts.push_back(v0);
            edgeLineVerts.push_back(v1);
        }
    }

    // Destroy old flat debug buffers
    vkDeviceWaitIdle(device);
    if (flatDebugSnap.vertBuf != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, flatDebugSnap.vertBuf, nullptr);
        vkFreeMemory(device, flatDebugSnap.vertMem, nullptr);
        flatDebugSnap.vertBuf = VK_NULL_HANDLE;
    }
    if (flatDebugSnap.idxBuf != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, flatDebugSnap.idxBuf, nullptr);
        vkFreeMemory(device, flatDebugSnap.idxMem, nullptr);
        flatDebugSnap.idxBuf = VK_NULL_HANDLE;
    }
    if (edgeLineBuf != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, edgeLineBuf, nullptr);
        vkFreeMemory(device, edgeLineMem, nullptr);
        edgeLineBuf = VK_NULL_HANDLE;
    }
    edgeLineVertCount = 0;

    flatDebugSnap.verts = std::move(verts);
    flatDebugSnap.inds = std::move(inds);

    VkDeviceSize vertSize = flatDebugSnap.verts.size() * sizeof(Vertex);
    createAndCopyBufferLocal(vertSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        flatDebugSnap.verts.data(), flatDebugSnap.vertBuf, flatDebugSnap.vertMem);

    VkDeviceSize idxSize = flatDebugSnap.inds.size() * sizeof(uint32_t);
    createAndCopyBufferLocal(idxSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        flatDebugSnap.inds.data(), flatDebugSnap.idxBuf, flatDebugSnap.idxMem);

    if (!edgeLineVerts.empty()) {
        VkDeviceSize elSize = edgeLineVerts.size() * sizeof(Vertex);
        createAndCopyBufferLocal(elSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            edgeLineVerts.data(), edgeLineBuf, edgeLineMem);
        edgeLineVertCount = static_cast<uint32_t>(edgeLineVerts.size());
    }

    flatDebugSnap.valid = true;
    const char* phaseStr = (stepPhase == 1) ? "SCATTER" : "INIT";
    std::cout << "Flat debug [" << phaseStr << "]: " << triCount << " tris, "
              << wonTris << " claimed by edges, "
              << edgeLineVertCount / 2 << " edge lines" << std::endl;
}

void App::updateMeshBuffersForMode(RenderMode mode) {
    auto& snap = meshSnapshots[mode];
    if (!snap.valid || snap.verts.empty() || snap.inds.empty()) return;

    // Destroy old buffers
    if (snap.vertBuf != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
        vkDestroyBuffer(device, snap.vertBuf, nullptr);
        vkFreeMemory(device, snap.vertMem, nullptr);
        snap.vertBuf = VK_NULL_HANDLE;
    }
    if (snap.idxBuf != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, snap.idxBuf, nullptr);
        vkFreeMemory(device, snap.idxMem, nullptr);
        snap.idxBuf = VK_NULL_HANDLE;
    }

    VkDeviceSize vertSize = snap.verts.size() * sizeof(Vertex);
    createAndCopyBufferLocal(vertSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        snap.verts.data(), snap.vertBuf, snap.vertMem);

    VkDeviceSize idxSize = snap.inds.size() * sizeof(uint32_t);
    createAndCopyBufferLocal(idxSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        snap.inds.data(), snap.idxBuf, snap.idxMem);
}

void App::createAndCopyBuffer2(VkDeviceSize bufferSize, VkBufferUsageFlags flags, void* srcData, VkBuffer& dstBuffer, VkDeviceMemory& dstBufferMemory) {
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | flags, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, dstBuffer, dstBufferMemory);

    void* data;
    vkMapMemory(device, dstBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, srcData, (size_t)bufferSize);
    vkUnmapMemory(device, dstBufferMemory);
}

void App::createAndCopyBufferLocal(VkDeviceSize bufferSize, VkBufferUsageFlags flags, void* srcData, VkBuffer& dstBuffer, VkDeviceMemory& dstBufferMemory) {
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, srcData, (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | flags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dstBuffer, dstBufferMemory);

    copyBuffer(stagingBuffer, dstBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}
