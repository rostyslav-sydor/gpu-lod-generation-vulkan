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

    // Reorder mesh for GPU cache locality (spatially nearby triangles become memory-adjacent)
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

// ============================================================================
// CPU Mesh Decimation (meshoptimizer baseline)
// ============================================================================

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

    std::cout << "CPU Decimation (meshoptimizer): " << triCount << " -> " << newTriCount
              << " triangles (" << elapsed << " us)\n";
}

// ============================================================================
// Mesh Quality Metrics
// ============================================================================

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
    glm::vec3 ab = glm::normalize(b - a);
    glm::vec3 ac = glm::normalize(c - a);
    return glm::degrees(std::acos(glm::clamp(glm::dot(ab, ac), -1.0f, 1.0f)));
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
    m.minAngleDeg = 180.0f;
    m.maxAspectRatio = 0.0f;

    for (uint32_t t = 0; t < m.triCount; t++) {
        uint32_t i0 = inds[t*3+0], i1 = inds[t*3+1], i2 = inds[t*3+2];
        if (i0 >= verts.size() || i1 >= verts.size() || i2 >= verts.size()) continue;
        glm::vec3 a = verts[i0].pos, b = verts[i1].pos, c = verts[i2].pos;

        float a0 = triAngle(a, b, c);
        float a1 = triAngle(b, c, a);
        float a2 = triAngle(c, a, b);
        float minA = std::min({a0, a1, a2});
        m.minAngleDeg = std::min(m.minAngleDeg, minA);
        sumMinAngle += minA;

        float ar = triAspectRatio(a, b, c);
        m.maxAspectRatio = std::max(m.maxAspectRatio, ar);
        sumAspect += ar;
    }
    if (m.triCount > 0) {
        m.avgMinAngleDeg = sumMinAngle / m.triCount;
        m.avgAspectRatio = sumAspect / m.triCount;
    }

    if (origVerts && origInds && !origInds->empty()) {
        uint32_t origTriCount = static_cast<uint32_t>(origInds->size() / 3);

        const uint32_t MAX_ORIG_TRIS_FOR_METRICS = 10000;
        uint32_t stride = 1;
        uint32_t sampledOrigTriCount = origTriCount;
        if (origTriCount > MAX_ORIG_TRIS_FOR_METRICS) {
            stride = (origTriCount + MAX_ORIG_TRIS_FOR_METRICS - 1) / MAX_ORIG_TRIS_FOR_METRICS;
            sampledOrigTriCount = (origTriCount + stride - 1) / stride;
        }

        std::unordered_set<uint32_t> usedVerts(inds.begin(), inds.end());
        float maxDist = 0.0f, sumDist = 0.0f;
        uint32_t distCount = 0;
        for (uint32_t vi : usedVerts) {
            if (vi >= verts.size()) continue;
            glm::vec3 p = verts[vi].pos;
            float bestDist = std::numeric_limits<float>::max();
            for (uint32_t t = 0; t < origTriCount; t += stride) {
                glm::vec3 oa = (*origVerts)[(*origInds)[t*3+0]].pos;
                glm::vec3 ob = (*origVerts)[(*origInds)[t*3+1]].pos;
                glm::vec3 oc = (*origVerts)[(*origInds)[t*3+2]].pos;
                bestDist = std::min(bestDist, pointToTriDist(p, oa, ob, oc));
            }
            maxDist = std::max(maxDist, bestDist);
            sumDist += bestDist;
            distCount++;
        }
        m.hausdorffDist = maxDist;
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

            float bestDistSq = std::numeric_limits<float>::max();
            glm::vec3 bestNormal(0,0,1);
            for (uint32_t ot = 0; ot < origTriCount; ot += stride) {
                glm::vec3 oa = (*origVerts)[(*origInds)[ot*3+0]].pos;
                glm::vec3 ob = (*origVerts)[(*origInds)[ot*3+1]].pos;
                glm::vec3 oc = (*origVerts)[(*origInds)[ot*3+2]].pos;
                glm::vec3 oCentroid = (oa + ob + oc) / 3.0f;
                float distSq = glm::dot(oCentroid - centroid, oCentroid - centroid);
                if (distSq < bestDistSq) {
                    bestDistSq = distSq;
                    glm::vec3 ofn = glm::cross(ob - oa, oc - oa);
                    if (glm::length(ofn) > 1e-12f) bestNormal = glm::normalize(ofn);
                }
            }
            float cosA = glm::clamp(glm::dot(fn, bestNormal), -1.0f, 1.0f);
            sumNormalDev += glm::degrees(std::acos(cosA));
            normalCount++;
        }
        m.avgNormalDevDeg = (normalCount > 0) ? sumNormalDev / normalCount : 0.0f;

        if (stride > 1) {
            std::cout << "  (metrics sampled 1/" << stride
                      << " of original triangles for speed)" << std::endl;
        }
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
        printRow("Hausdorff dist",   0.0f,                  gpuM.hausdorffDist,   cpuM.hausdorffDist,   "%12.4f");
        printRow("Avg vertex dist",  0.0f,                  gpuM.avgVertDist,     cpuM.avgVertDist,     "%12.4f");
        printRow("Avg normal dev",   0.0f,                  gpuM.avgNormalDevDeg, cpuM.avgNormalDevDeg, "%12.2f");
    }
    std::cout << "\n";

    if (hasGPU || hasCPU)
        std::cout << "  Keys: [G]PU  [C]PU  [O]riginal\n\n";
}

// ============================================================================
// GPU Mesh Decimation Pipeline
// ============================================================================

static uint32_t divUp(uint32_t n, uint32_t d) { return (n + d - 1) / d; }

void App::runDecimation() {
    const uint32_t WORKGROUP_SIZE = 256;
    uint32_t vertCount = static_cast<uint32_t>(vertices.size());
    uint32_t triCount  = static_cast<uint32_t>(indices.size() / 3);
    uint32_t originalTriCount = triCount;
    uint32_t maxEdges = triCount * 3;

    auto np2 = [](uint32_t v) { v--; v|=v>>1; v|=v>>2; v|=v>>4; v|=v>>8; v|=v>>16; v++; return v; };
    uint32_t hashMapSize = np2(std::max(vertCount, maxEdges) * 2);

    const char* modeNames[] = {"QEM", "Paper (curvature+length+valence)", "Meshopt-like (QEM+borders+reg)"};
    std::cout << "Decimation: " << vertCount << " vertices, " << triCount << " triangles, hashMap=" << hashMapSize
              << ", costMode=" << decimationCostMode << " (" << modeNames[std::min(decimationCostMode, 2u)] << ")"
              << ", rounds=" << decimationInnerRounds << std::endl;

    // --- Allocate buffers and descriptors ---
    std::cout << "  allocating buffers..." << std::flush;
    allocateDecimationBuffers(vertCount, triCount);
    std::cout << " done\n  writing descriptors..." << std::flush;
    writeDecimationDescriptorSets();
    std::cout << " done" << std::endl;

    // --- Upload vertex data (interleaved vec4 pairs) ---
    {
        void* data;
        vkMapMemory(device, decimationMem[DB_VERTEX], 0, decimationBufSizes[DB_VERTEX], 0, &data);
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
        vkUnmapMemory(device, decimationMem[DB_VERTEX]);
    }

    std::cout << "  uploading data..." << std::flush;
    // --- Upload index data ---
    {
        void* data;
        vkMapMemory(device, decimationMem[DB_INDEX], 0, decimationBufSizes[DB_INDEX], 0, &data);
        memcpy(data, indices.data(), triCount * 3 * sizeof(uint32_t));
        vkUnmapMemory(device, decimationMem[DB_INDEX]);
    }
    std::cout << " done" << std::endl;

    // --- Helper: bind both descriptor sets ---
    auto bindDescSets = [&](VkCommandBuffer cmd) {
        VkDescriptorSet sets[] = { decimationDescSet0, decimationDescSet1 };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            decimationPipelineLayout, 0, 2, sets, 0, nullptr);
    };

    // --- Helper: compute->compute barrier ---
    auto computeBarrier = [&](VkCommandBuffer cmd) {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    };

    // --- Helper: transfer->compute barrier ---
    auto transferToComputeBarrier = [&](VkCommandBuffer cmd) {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    };

    // --- Helper: dispatch a pass ---
    auto dispatchPass = [&](VkCommandBuffer cmd, uint32_t passIdx, const DecimationPushConstants& pc, uint32_t workgroups) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, decimationPipelines[passIdx]);
        vkCmdPushConstants(cmd, decimationPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(DecimationPushConstants), &pc);
        vkCmdDispatch(cmd, workgroups, 1, 1);
    };

    // --- Helper: submit command buffer and wait ---
    auto submitAndWait = [&](VkCommandBuffer cmd) {
        vkEndCommandBuffer(cmd);
        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        vkResetFences(device, 1, &computeFence);
        if (vkQueueSubmit(computeQueue, 1, &submitInfo, computeFence) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit decimation command buffer!");
        }
        vkWaitForFences(device, 1, &computeFence, VK_TRUE, UINT64_MAX);
    };

    // --- Helper: begin command buffer ---
    auto beginCmd = [&]() -> VkCommandBuffer {
        vkResetCommandBuffer(computeCommandBuffer, 0);
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(computeCommandBuffer, &beginInfo);
        bindDescSets(computeCommandBuffer);
        return computeCommandBuffer;
    };

    // --- Helper: read uint32_t from counter buffer ---
    auto readCounter = [&](uint32_t index) -> uint32_t {
        void* data;
        vkMapMemory(device, decimationMem[DB_COUNTER], 0, decimationBufSizes[DB_COUNTER], 0, &data);
        uint32_t val = static_cast<uint32_t*>(data)[index];
        vkUnmapMemory(device, decimationMem[DB_COUNTER]);
        return val;
    };

    // ======================================================================
    // Phase 1: One-time setup passes (1-2)
    // ======================================================================
    std::cout << "  passes 1-2 (hash + dedup)..." << std::flush;
    {
        VkCommandBuffer cmd = beginCmd();

        // Clear all 3 hash maps to HASHMAP_EMPTY
        vkCmdFillBuffer(cmd, decimationBufs[DB_HASHMAP_VERTEX], 0, decimationBufSizes[DB_HASHMAP_VERTEX], 0xFFFFFFFF);
        vkCmdFillBuffer(cmd, decimationBufs[DB_HASHMAP_POSITION], 0, decimationBufSizes[DB_HASHMAP_POSITION], 0xFFFFFFFF);
        vkCmdFillBuffer(cmd, decimationBufs[DB_HASHMAP_EDGE], 0, decimationBufSizes[DB_HASHMAP_EDGE], 0xFFFFFFFF);
        // Clear vertex flags
        vkCmdFillBuffer(cmd, decimationBufs[DB_VERTEX_FLAGS], 0, decimationBufSizes[DB_VERTEX_FLAGS], 0);
        // Clear counters
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

        // Pass 1: Hash Vertices
        dispatchPass(cmd, 0, pc, divUp(vertCount, WORKGROUP_SIZE));
        computeBarrier(cmd);

        // Pass 2: Dedup Indices
        dispatchPass(cmd, 1, pc, divUp(triCount, WORKGROUP_SIZE));
        computeBarrier(cmd);

        submitAndWait(cmd);
    }
    std::cout << " done" << std::endl;

    // ======================================================================
    // Phase 2: Iterative decimation loop (passes 3-12)
    // ======================================================================
    VkDeviceSize edgeHashMapSize = decimationBufSizes[DB_HASHMAP_EDGE];

    const char* gpuPassNames[] = {"P3:adj", "P4:edges", "P5:quadrics", "P6:edgecost",
        "P7-9:select+collapse", "(unused)", "(unused)", "P10:degen", "P11:compact", "P12:copyback"};
    double gpuPassTimeMs[10] = {};

    for (uint32_t iteration = 0; iteration < maxDecimationIterations; iteration++) {
        Timer iterTimer;

        DecimationPushConstants pc{};
        pc.vertexCount = vertCount;
        pc.triangleCount = triCount;
        pc.edgeCount = maxEdges;
        pc.hashMapSize = hashMapSize;
        float adaptiveCost = decimationCostThreshold * std::pow(decimationGrowthRate, (float)iteration);
        pc.costThreshold = adaptiveCost;
        pc.iteration = iteration;
        pc.costMode = decimationCostMode;
        pc.costQuantBits = decimationCostQuantBits;

        uint32_t K = decimationInnerRounds;

        VkCommandBuffer cmd = beginCmd();
        vkCmdResetQueryPool(cmd, timestampQueryPool, 0, 16);

        // Clear buffers
        vkCmdFillBuffer(cmd, decimationBufs[DB_ADJ_HEAD], 0, decimationBufSizes[DB_ADJ_HEAD], 0xFFFFFFFF);
        vkCmdFillBuffer(cmd, decimationBufs[DB_HASHMAP_EDGE], 0, edgeHashMapSize, 0xFFFFFFFF);
        vkCmdFillBuffer(cmd, decimationBufs[DB_QUADRIC], 0, decimationBufSizes[DB_QUADRIC], 0);
        vkCmdFillBuffer(cmd, decimationBufs[DB_COUNTER], 0, decimationBufSizes[DB_COUNTER], 0);
        transferToComputeBarrier(cmd);

        // P3: build adjacency
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 0);
        dispatchPass(cmd, 2, pc, divUp(triCount, WORKGROUP_SIZE));
        computeBarrier(cmd);

        // P4: build edges
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 1);
        dispatchPass(cmd, 3, pc, divUp(triCount, WORKGROUP_SIZE));
        computeBarrier(cmd);

        // P5: quadrics
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 2);
        dispatchPass(cmd, 4, pc, divUp(triCount, WORKGROUP_SIZE));
        computeBarrier(cmd);

        // P6: edge cost (over-dispatch, shader reads actual edgeCount from counter)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 3);
        dispatchPass(cmd, 5, pc, divUp(maxEdges, WORKGROUP_SIZE));
        computeBarrier(cmd);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 4);

        // Inner rounds: P7 -> P8 -> P9 repeated K times
        for (uint32_t round = 0; round < K; round++) {
            dispatchPass(cmd, 6, pc, divUp(triCount, WORKGROUP_SIZE));   // P7: init descriptors
            computeBarrier(cmd);
            dispatchPass(cmd, 7, pc, divUp(maxEdges, WORKGROUP_SIZE));   // P8: scatter (over-dispatch)
            computeBarrier(cmd);
            dispatchPass(cmd, 8, pc, divUp(maxEdges, WORKGROUP_SIZE));   // P9: collapse (over-dispatch)
            computeBarrier(cmd);
        }
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 5);

        // P10: mark degenerate
        dispatchPass(cmd, 9, pc, divUp(triCount, WORKGROUP_SIZE));
        computeBarrier(cmd);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 6);

        // P11: compact
        dispatchPass(cmd, 10, pc, divUp(triCount, WORKGROUP_SIZE));
        computeBarrier(cmd);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 7);

        // P12: copyback
        dispatchPass(cmd, 11, pc, divUp(triCount, WORKGROUP_SIZE));
        computeBarrier(cmd);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 8);

        submitAndWait(cmd);

        // Collect timestamps
        {
            uint64_t ts[9];
            vkGetQueryPoolResults(device, timestampQueryPool, 0, 9, sizeof(ts), ts,
                sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            gpuPassTimeMs[0] += (double)(ts[1] - ts[0]) * timestampPeriodNs * 1e-6; // P3
            gpuPassTimeMs[1] += (double)(ts[2] - ts[1]) * timestampPeriodNs * 1e-6; // P4
            gpuPassTimeMs[2] += (double)(ts[3] - ts[2]) * timestampPeriodNs * 1e-6; // P5
            gpuPassTimeMs[3] += (double)(ts[4] - ts[3]) * timestampPeriodNs * 1e-6; // P6
            gpuPassTimeMs[4] += (double)(ts[5] - ts[4]) * timestampPeriodNs * 1e-6; // P7+P8+P9
            gpuPassTimeMs[7] += (double)(ts[6] - ts[5]) * timestampPeriodNs * 1e-6; // P10
            gpuPassTimeMs[8] += (double)(ts[7] - ts[6]) * timestampPeriodNs * 1e-6; // P11
            gpuPassTimeMs[9] += (double)(ts[8] - ts[7]) * timestampPeriodNs * 1e-6; // P12
        }

        uint32_t edgeCount     = readCounter(0);
        uint32_t collapseCount = readCounter(1);
        uint32_t newTriCount   = readCounter(4);
        long long iterUs = iterTimer.getTime();
        uint32_t trisRemoved = triCount - newTriCount;
        double trisPerSec = (iterUs > 0) ? (double)trisRemoved / ((double)iterUs * 1e-6) : 0.0;

        if (iteration < 5 || iteration % 10 == 0 || collapseCount == 0
            || newTriCount <= static_cast<uint32_t>(originalTriCount * decimationTargetRatio)) {
            std::cout << "  iter " << iteration
                      << ": edges=" << edgeCount
                      << " collapsed=" << collapseCount
                      << " tris=" << newTriCount << " (was " << triCount << ")"
                      << " " << iterUs / 1000 << "ms";
            if (trisRemoved > 0) {
                if (trisPerSec >= 1e6)
                    std::cout << " " << std::fixed << std::setprecision(1) << trisPerSec / 1e6 << "M tri/s";
                else
                    std::cout << " " << std::fixed << std::setprecision(1) << trisPerSec / 1e3 << "K tri/s";
            }
            std::cout << std::defaultfloat << std::endl;
        }

        if (edgeCount == 0 || collapseCount == 0) break;
        triCount = newTriCount;
        if (triCount <= static_cast<uint32_t>(originalTriCount * decimationTargetRatio)) {
            std::cout << "  target ratio reached\n";
            break;
        }
    }

    // Print GPU per-pass time breakdown
    {
        double totalGpuMs = 0;
        for (int i = 0; i < 10; i++) totalGpuMs += gpuPassTimeMs[i];
        if (totalGpuMs > 0) {
            std::cout << "GPU pass times (total " << std::fixed << std::setprecision(0)
                      << totalGpuMs << "ms):" << std::setprecision(1);
            for (int i = 0; i < 10; i++) {
                double pct = 100.0 * gpuPassTimeMs[i] / totalGpuMs;
                if (pct >= 0.1)
                    std::cout << "  " << gpuPassNames[i] << " " << pct << "%";
            }
            std::cout << std::defaultfloat << std::endl;
        }
    }

    // ======================================================================
    // Phase 3: Read back results
    // ======================================================================
    std::cout << "Decimation complete: " << originalTriCount << " -> " << triCount << " triangles\n";

    // Read back vertices
    {
        void* data;
        vkMapMemory(device, decimationMem[DB_VERTEX], 0, decimationBufSizes[DB_VERTEX], 0, &data);
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
        vkUnmapMemory(device, decimationMem[DB_VERTEX]);
    }

    // Read back indices
    {
        indices.resize(triCount * 3);
        void* data;
        vkMapMemory(device, decimationMem[DB_INDEX], 0, (VkDeviceSize)triCount * 3 * sizeof(uint32_t), 0, &data);
        memcpy(indices.data(), data, triCount * 3 * sizeof(uint32_t));
        vkUnmapMemory(device, decimationMem[DB_INDEX]);
    }

    // --- Recompute vertex normals from final geometry ---
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

    // --- Validation ---
    uint32_t oobCount = 0, nanCount = 0, bigCount = 0;
    for (uint32_t i = 0; i < indices.size(); i++) {
        if (indices[i] >= vertCount) oobCount++;
    }
    for (uint32_t i = 0; i < vertCount; i++) {
        if (std::isnan(vertices[i].pos.x) || std::isnan(vertices[i].pos.y) || std::isnan(vertices[i].pos.z))
            nanCount++;
        float maxC = std::max({std::abs(vertices[i].pos.x), std::abs(vertices[i].pos.y), std::abs(vertices[i].pos.z)});
        if (maxC > 1e6f) bigCount++;
    }
    std::cout << "Validation: oob_indices=" << oobCount << " nan_verts=" << nanCount << " huge_verts=" << bigCount << std::endl;

    // Print position range of referenced vertices
    float minP = 1e30f, maxP = -1e30f;
    for (uint32_t idx : indices) {
        if (idx < vertCount) {
            auto& p = vertices[idx].pos;
            minP = std::min({minP, p.x, p.y, p.z});
            maxP = std::max({maxP, p.x, p.y, p.z});
        }
    }
    std::cout << "Position range of referenced verts: [" << minP << ", " << maxP << "]" << std::endl;
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
