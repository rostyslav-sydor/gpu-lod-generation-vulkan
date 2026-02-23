#include "renderer.hpp"

void App::loadModel() {
    vertices.clear();
    indices.clear();
    Timer tWhole;
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(MODEL_PATH,
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
    timesLoad.push_back(tWhole.getTime());

    Timer tAlgo;
    if (simplify)
        simplifyMesh();
    else if (useGPUDecimation)
        runDecimation();
    timesAlgo.push_back(tAlgo.getTime());
    
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

    std::cout << "Decimation: " << vertCount << " vertices, " << triCount << " triangles, hashMap=" << hashMapSize << std::endl;

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

        // Clear hashMap (all 3 regions) to HASHMAP_EMPTY
        vkCmdFillBuffer(cmd, decimationBufs[DB_HASHMAP], 0, decimationBufSizes[DB_HASHMAP], 0xFFFFFFFF);
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
    VkDeviceSize hmRegion2Offset = (VkDeviceSize)hashMapSize * 2 * 16;
    VkDeviceSize hmRegion2Size   = (VkDeviceSize)hashMapSize * 16;

    for (uint32_t iteration = 0; iteration < maxDecimationIterations; iteration++) {
        std::cout << "  iter " << iteration << ": passes 3-4..." << std::flush;
        // --- Phase A: Passes 3-4 (topology + edge discovery) ---
        {
            VkCommandBuffer cmd = beginCmd();

            // Clear buffers for this iteration
            vkCmdFillBuffer(cmd, decimationBufs[DB_ADJ_HEAD], 0, decimationBufSizes[DB_ADJ_HEAD], 0xFFFFFFFF);
            vkCmdFillBuffer(cmd, decimationBufs[DB_HASHMAP], hmRegion2Offset, hmRegion2Size, 0xFFFFFFFF);
            vkCmdFillBuffer(cmd, decimationBufs[DB_QUADRIC], 0, decimationBufSizes[DB_QUADRIC], 0);
            vkCmdFillBuffer(cmd, decimationBufs[DB_COUNTER], 0, decimationBufSizes[DB_COUNTER], 0);

            transferToComputeBarrier(cmd);

            DecimationPushConstants pc{};
            pc.vertexCount = vertCount;
            pc.triangleCount = triCount;
            pc.edgeCount = 0;
            pc.hashMapSize = hashMapSize;
            pc.costThreshold = decimationCostThreshold;
            pc.iteration = iteration;

            // Pass 3: Build Adjacency
            dispatchPass(cmd, 2, pc, divUp(triCount, WORKGROUP_SIZE));
            computeBarrier(cmd);

            // Pass 4: Build Edges
            dispatchPass(cmd, 3, pc, divUp(triCount, WORKGROUP_SIZE));
            computeBarrier(cmd);

            submitAndWait(cmd);
        }
        std::cout << " ok, readback..." << std::flush;

        // Read back edge count
        uint32_t edgeCount = readCounter(0);
        std::cout << " edges=" << edgeCount << std::flush;
        if (edgeCount == 0) { std::cout << " (done)\n"; break; }

        {
            DecimationPushConstants pc{};
            pc.vertexCount = vertCount;
            pc.triangleCount = triCount;
            pc.edgeCount = edgeCount;
            pc.hashMapSize = hashMapSize;
            pc.costThreshold = decimationCostThreshold;
            pc.iteration = iteration;

            struct { int idx; const char* name; uint32_t count; } passes[] = {
                {4, "P5:quadrics",  divUp(triCount, WORKGROUP_SIZE)},
                {5, "P6:edgecost",  divUp(edgeCount, WORKGROUP_SIZE)},
                {6, "P7:initdesc",  divUp(triCount, WORKGROUP_SIZE)},
                {7, "P8:scatter",   divUp(edgeCount, WORKGROUP_SIZE)},
                {8, "P9:collapse",  divUp(edgeCount, WORKGROUP_SIZE)},
                {9, "P10:degen",    divUp(triCount, WORKGROUP_SIZE)},
                {10,"P11:compact",  divUp(triCount, WORKGROUP_SIZE)},
                {11,"P12:copyback", divUp(triCount, WORKGROUP_SIZE)},
            };

            for (auto& p : passes) {
                std::cout << " " << p.name << std::flush;
                VkCommandBuffer cmd = beginCmd();
                dispatchPass(cmd, p.idx, pc, p.count);
                computeBarrier(cmd);
                submitAndWait(cmd);
                std::cout << "+" << std::flush;
            }
        }

        uint32_t collapseCount = readCounter(1);
        uint32_t newTriCount   = readCounter(4);

        std::cout << " collapsed=" << collapseCount
                  << " tris=" << newTriCount << " (was " << triCount << ")" << std::endl;

        if (collapseCount == 0) break;
        triCount = newTriCount;
        if (triCount <= static_cast<uint32_t>(originalTriCount * decimationTargetRatio)) {
            std::cout << "  target ratio reached\n";
            break;
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
