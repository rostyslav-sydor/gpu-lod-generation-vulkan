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
