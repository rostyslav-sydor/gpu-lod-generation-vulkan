#include "renderer.hpp"

void App::copyComputeBuffersLocal(std::vector<meshopt_Meshlet>& meshlets,
    std::vector<uint32_t>& meshletVertices,
    std::vector<Triangle>& meshletTriangles) {
    VkDeviceSize bufferSize = std::max(compVertexBufferSize, compMeshletsBufferSize);
    bufferSize = std::max(bufferSize, compMeshletVerticesBufferSize);                                   
    bufferSize = std::max(bufferSize, compMeshletTrianglesBufferSize);

    void* data;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    auto copyToBuf = [&](VkDeviceSize bufSize, void* srcData, VkBuffer& dstBuf){
        vkMapMemory(device, stagingBufferMemory, 0, bufSize, 0, &data);
        memcpy(data, srcData, (size_t)bufSize);
        vkUnmapMemory(device, stagingBufferMemory);
        copyBuffer(stagingBuffer, dstBuf, bufSize);
    };

    auto copyToBuf2 = [&](VkCommandBuffer cmdBuf, VkDeviceSize bufSize, void* srcData, VkBuffer& dstBuf, VkDeviceSize dstOff){
        vkMapMemory(device, stagingBufferMemory, 0, bufSize, 0, &data);
        memcpy(data, srcData, (size_t)bufSize);
        vkUnmapMemory(device, stagingBufferMemory);
        copyBuffer2(cmdBuf, stagingBuffer, dstBuf, bufSize, 0, dstOff);
    };

    if(singleBuffer) {
        VkCommandBuffer cmdBuf = beginSingleTimeCommands();
        copyToBuf2(cmdBuf, compVertexBufferSize, vertices.data(), totalBuffer, 0);
        copyToBuf2(cmdBuf, compMeshletsBufferSize, meshlets.data(), totalBuffer, meshletsOffset);
        copyToBuf2(cmdBuf, compMeshletVerticesBufferSize, meshletVertices.data(), totalBuffer, meshletVerticesOffset);
        copyToBuf2(cmdBuf, compMeshletTrianglesBufferSize, meshletTriangles.data(), totalBuffer, meshletTrianglesOffset);
        endSingleTimeCommands(cmdBuf);
    } else {
        copyToBuf(compVertexBufferSize, vertices.data(), compVertexBuffer);
        copyToBuf(compMeshletsBufferSize, meshlets.data(), compMeshletsBuffer);
        copyToBuf(compMeshletVerticesBufferSize, meshletVertices.data(), compMeshletVerticesBuffer);
        copyToBuf(compMeshletTrianglesBufferSize, meshletTriangles.data(), compMeshletTrianglesBuffer);
    }

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void App::copyComputeBuffers(std::vector<meshopt_Meshlet>& meshlets,
    std::vector<uint32_t>& meshletVertices,
    std::vector<Triangle>& meshletTriangles) {
    
    void* data;
    auto copyToBuf = [&](VkDeviceSize bufSize, void* srcData, VkDeviceMemory& dstMem, VkDeviceSize offset = 0){
        vkMapMemory(device, dstMem, offset, bufSize, 0, &data);
        memcpy(data, srcData, (size_t)bufSize);
        vkUnmapMemory(device, dstMem);
    };
    
    if(singleBuffer) {
        copyToBuf(compVertexBufferSize, vertices.data(), totalBufferMemory, 0);
        copyToBuf(compMeshletsBufferSize, meshlets.data(), totalBufferMemory, meshletsOffset);
        copyToBuf(compMeshletVerticesBufferSize, meshletVertices.data(), totalBufferMemory, meshletVerticesOffset);
        copyToBuf(compMeshletTrianglesBufferSize, meshletTriangles.data(), totalBufferMemory, meshletTrianglesOffset);
    } else {
        copyToBuf(compVertexBufferSize, vertices.data(), compVertexBufferMemory);
        copyToBuf(compMeshletsBufferSize, meshlets.data(), compMeshletsMemory);
        copyToBuf(compMeshletVerticesBufferSize, meshletVertices.data(), compMeshletVerticesMemory);
        copyToBuf(compMeshletTrianglesBufferSize, meshletTriangles.data(), compMeshletTrianglesMemory);
    }
}

void App::createComputeBuffersLocal(std::vector<meshopt_Meshlet>& meshlets,
    std::vector<uint32_t>& meshletVertices,
    std::vector<Triangle>& meshletTriangles) {

    if (singleBuffer) {
        createBuffer(totalBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, totalBuffer, totalBufferMemory);
    } else {
        createBuffer(compVertexBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT| VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, compVertexBuffer, compVertexBufferMemory);
        
        createBuffer(compMeshletsBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT| VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, compMeshletsBuffer, compMeshletsMemory);
        
        createBuffer(compMeshletVerticesBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT| VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, compMeshletVerticesBuffer, compMeshletVerticesMemory);

        createBuffer(compMeshletTrianglesBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT| VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, compMeshletTrianglesBuffer, compMeshletTrianglesMemory);
    }
}

void App::createComputeBuffers(std::vector<meshopt_Meshlet>& meshlets,
    std::vector<uint32_t>& meshletVertices,
    std::vector<Triangle>& meshletTriangles) {

    if(singleBuffer) {
        createBuffer(totalBufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, totalBuffer, totalBufferMemory);
    } else {
        createAndCopyBuffer2(compVertexBufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vertices.data(),compVertexBuffer, compVertexBufferMemory);
        
        createAndCopyBuffer2(compMeshletsBufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            meshlets.data(), compMeshletsBuffer, compMeshletsMemory);

        createAndCopyBuffer2(compMeshletVerticesBufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
            meshletVertices.data(), compMeshletVerticesBuffer, compMeshletVerticesMemory);

        createAndCopyBuffer2(compMeshletTrianglesBufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
            meshletTriangles.data(), compMeshletTrianglesBuffer, compMeshletTrianglesMemory);
    }
}

void App::createComputeDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 4> layoutBindings = {{
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        }
    }};

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = layoutBindings.size();
    layoutInfo.pBindings = layoutBindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create compute descriptor set layout!");
    }
}

void App::createComputeDescriptorSet() {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &computeDescriptorSetLayout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &computeDescriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate compute descriptor set!");
    }
    std::array<VkDescriptorBufferInfo, 4> descriptorInfos;
    if(singleBuffer) {
        descriptorInfos = {{
            {
                .buffer = totalBuffer,
                .offset = 0,
                .range = compVertexBufferSize
            }, 
            {
                .buffer = totalBuffer,
                .offset = meshletsOffset,
                .range = compMeshletsBufferSize
            }, 
            {
                .buffer = totalBuffer,
                .offset = meshletVerticesOffset,
                .range = compMeshletVerticesBufferSize
            },
            {
                .buffer = totalBuffer,
                .offset = meshletTrianglesOffset,
                .range = compMeshletTrianglesBufferSize
            }
        }};
    } else {
        descriptorInfos = {{
            {
                .buffer = compVertexBuffer,
                .offset = 0,
                .range = VK_WHOLE_SIZE
            }, 
            {
                .buffer = compMeshletsBuffer,
                .offset = 0,
                .range = VK_WHOLE_SIZE
            }, 
            {
                .buffer = compMeshletVerticesBuffer,
                .offset = 0,
                .range = VK_WHOLE_SIZE
            },
            {
                .buffer = compMeshletTrianglesBuffer,
                .offset = 0,
                .range = VK_WHOLE_SIZE
            }
        }};
    }
    VkWriteDescriptorSet genericSet = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = computeDescriptorSet,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
    };

    std::array<VkWriteDescriptorSet, 4> descriptorWrites{};
    descriptorWrites.fill(genericSet);

    for (int i = 0; i < descriptorWrites.size(); ++i) {
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].pBufferInfo = &descriptorInfos[i];
    }

    vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}

void App::createComputePipeline() { 
    auto computeShaderCode = readFile("shaders/comp.spv");

    VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);
    
    VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
    computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeShaderStageInfo.module = computeShaderModule;
    computeShaderStageInfo.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create compute pipeline layout!");
    }

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = computePipelineLayout;
    pipelineInfo.stage = computeShaderStageInfo;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create compute pipeline!");
    }

    vkDestroyShaderModule(device, computeShaderModule, nullptr);
}

void App::createComputeCommandBuffer() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = computeCommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device, &allocInfo, &computeCommandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

// ============================================================================
// Decimation pipeline infrastructure
// ============================================================================

void App::createDecimationDescriptorSetLayouts() {
    // Set 0: 16 storage buffer bindings (bindings 0-15)
    std::array<VkDescriptorSetLayoutBinding, 16> set0Bindings{};
    for (uint32_t i = 0; i < 16; i++) {
        set0Bindings[i].binding = i;
        set0Bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        set0Bindings[i].descriptorCount = 1;
        set0Bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo0{};
    layoutInfo0.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo0.bindingCount = set0Bindings.size();
    layoutInfo0.pBindings = set0Bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo0, nullptr, &decimationDescSetLayout0) != VK_SUCCESS) {
        throw std::runtime_error("failed to create decimation descriptor set layout 0!");
    }

    // Set 1: 6 storage buffer bindings (bindings 0-5)
    std::array<VkDescriptorSetLayoutBinding, 6> set1Bindings{};
    for (uint32_t i = 0; i < 6; i++) {
        set1Bindings[i].binding = i;
        set1Bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        set1Bindings[i].descriptorCount = 1;
        set1Bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo1{};
    layoutInfo1.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo1.bindingCount = set1Bindings.size();
    layoutInfo1.pBindings = set1Bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo1, nullptr, &decimationDescSetLayout1) != VK_SUCCESS) {
        throw std::runtime_error("failed to create decimation descriptor set layout 1!");
    }
}

void App::createDecimationPipelineLayout() {
    VkDescriptorSetLayout setLayouts[] = { decimationDescSetLayout0, decimationDescSetLayout1 };

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(DecimationPushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 2;
    layoutInfo.pSetLayouts = setLayouts;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &decimationPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create decimation pipeline layout!");
    }
}

void App::createDecimationPipelines() {
    const std::string shaderDir = "shaders2/mesh_decimation/";
    const std::string shaderNames[DECIMATION_PASS_COUNT] = {
        "01_hash_vertices", "02_dedup_indices", "03_build_adjacency",
        "04_build_edges", "04b_flag_boundary",
        "05_compute_quadrics", "06_compute_cost_and_scatter",
        "09_collapse_edges",
        "10_mark_degenerate", "11_compact", "12_copy_back",
        "13_gate"
    };

    for (uint32_t i = 0; i < DECIMATION_PASS_COUNT; i++) {
        auto code = readFile(shaderDir + shaderNames[i] + ".spv");
        VkShaderModule shaderModule = createShaderModule(code);

        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = shaderModule;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = decimationPipelineLayout;
        pipelineInfo.stage = stageInfo;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &decimationPipelines[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create decimation pipeline " + shaderNames[i]);
        }

        vkDestroyShaderModule(device, shaderModule, nullptr);
    }
}

static uint32_t nextPowerOf2(uint32_t v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    v++;
    return v;
}

void App::allocateDecimationBuffers(uint32_t vertCount, uint32_t triCount) {
    uint32_t maxEdges = triCount * 3;
    uint32_t hashMapSize = nextPowerOf2(std::max(vertCount, maxEdges) * 2);

    decimationBufSizes[DB_VERTEX]         = (VkDeviceSize)vertCount * 3 * sizeof(float) * 4;
    decimationBufSizes[DB_INDEX]          = (VkDeviceSize)triCount * 3 * sizeof(uint32_t);
    decimationBufSizes[DB_POS_INDEX]      = (VkDeviceSize)triCount * 3 * sizeof(uint32_t);
    decimationBufSizes[DB_VERTEX_FLAGS]   = (VkDeviceSize)vertCount * sizeof(uint32_t);
    decimationBufSizes[DB_ADJ_HEAD]       = (VkDeviceSize)vertCount * sizeof(uint32_t);
    decimationBufSizes[DB_TRI_ADJ_NEXT]   = (VkDeviceSize)triCount * 3 * sizeof(uint32_t);
    decimationBufSizes[DB_EDGE]           = (VkDeviceSize)maxEdges * 2 * sizeof(uint32_t);
    decimationBufSizes[DB_EDGE_TRI]       = (VkDeviceSize)maxEdges * 2 * sizeof(uint32_t);
    decimationBufSizes[DB_TRI_EDGE]       = (VkDeviceSize)vertCount * sizeof(uint32_t);
    decimationBufSizes[DB_QUADRIC]        = (VkDeviceSize)vertCount * 11 * sizeof(int32_t);
    decimationBufSizes[DB_EDGE_COST]      = (VkDeviceSize)maxEdges * sizeof(float);
    decimationBufSizes[DB_EDGE_FLAG]      = (VkDeviceSize)maxEdges * sizeof(uint32_t);
    decimationBufSizes[DB_EDGE_TARGET]    = (VkDeviceSize)maxEdges * 3 * sizeof(float) * 4;
    decimationBufSizes[DB_TRI_DESCRIPTOR] = (VkDeviceSize)triCount * sizeof(uint64_t);
    decimationBufSizes[DB_HASHMAP_EDGE]   = (VkDeviceSize)hashMapSize * 4 * sizeof(uint32_t);
    decimationBufSizes[DB_HASHMAP_VERTEX] = (VkDeviceSize)hashMapSize * 4 * sizeof(uint32_t);
    decimationBufSizes[DB_HASHMAP_POSITION] = (VkDeviceSize)hashMapSize * 4 * sizeof(uint32_t);
    decimationBufSizes[DB_COUNTER]        = 256;
    decimationBufSizes[DB_VERTEX_MAP]     = (VkDeviceSize)vertCount * sizeof(uint32_t);
    decimationBufSizes[DB_POS_MAP]        = (VkDeviceSize)vertCount * sizeof(uint32_t);
    decimationBufSizes[DB_SCAN]           = (VkDeviceSize)triCount * 6 * sizeof(uint32_t);
    decimationBufSizes[DB_ALIVE]          = (VkDeviceSize)triCount * sizeof(uint32_t);

    VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    decimationUseDeviceLocal = true;
    for (int i = 0; i < DB_COUNT; i++) {
        if (decimationBufSizes[i] == 0) decimationBufSizes[i] = 4;
    }

    // Try device-local first; if any allocation fails, fall back to host-visible for all
    for (int i = 0; i < DB_COUNT; i++) {
        VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufInfo.size = decimationBufSizes[i];
        bufInfo.usage = storageUsage;
        if (i == DB_COUNTER) bufInfo.usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &bufInfo, nullptr, &decimationBufs[i]) != VK_SUCCESS) {
            decimationUseDeviceLocal = false;
            break;
        }
        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device, decimationBufs[i], &memReq);
        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &allocInfo, nullptr, &decimationMem[i]) != VK_SUCCESS) {
            vkDestroyBuffer(device, decimationBufs[i], nullptr);
            decimationBufs[i] = VK_NULL_HANDLE;
            decimationUseDeviceLocal = false;
            break;
        }
        vkBindBufferMemory(device, decimationBufs[i], decimationMem[i], 0);
    }

    if (!decimationUseDeviceLocal) {
        for (int i = 0; i < DB_COUNT; i++) {
            if (decimationBufs[i] != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, decimationBufs[i], nullptr);
                decimationBufs[i] = VK_NULL_HANDLE;
            }
            if (decimationMem[i] != VK_NULL_HANDLE) {
                vkFreeMemory(device, decimationMem[i], nullptr);
                decimationMem[i] = VK_NULL_HANDLE;
            }
        }
        std::cout << "  [WARNING] DEVICE_LOCAL alloc failed, falling back to HOST_VISIBLE (slower)\n";
        for (int i = 0; i < DB_COUNT; i++) {
            VkBufferUsageFlags usage = storageUsage;
            if (i == DB_COUNTER) usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
            createBuffer(decimationBufSizes[i], usage,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                decimationBufs[i], decimationMem[i]);
        }
    }

    createBuffer(256,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        counterReadbackBuf, counterReadbackMem);
    vkMapMemory(device, counterReadbackMem, 0, 256, 0, &counterReadbackMapped);
}

void App::writeDecimationDescriptorSets() {
    if (decimationDescSet0 == VK_NULL_HANDLE) {
        VkDescriptorSetLayout layouts[] = { decimationDescSetLayout0, decimationDescSetLayout1 };
        VkDescriptorSet sets[2];

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 2;
        allocInfo.pSetLayouts = layouts;

        if (vkAllocateDescriptorSets(device, &allocInfo, sets) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate decimation descriptor sets!");
        }
        decimationDescSet0 = sets[0];
        decimationDescSet1 = sets[1];
    }

    // Set 0: bindings 0-15 map to DB_VERTEX..DB_COUNTER
    // (binding 14 = DB_HASHMAP_EDGE, which replaced the old combined DB_HASHMAP)
    std::array<VkWriteDescriptorSet, 22> writes{};
    std::array<VkDescriptorBufferInfo, 22> bufInfos{};

    for (int i = 0; i < 16; i++) {
        bufInfos[i] = { decimationBufs[i], 0, VK_WHOLE_SIZE };
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = decimationDescSet0;
        writes[i].dstBinding = i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    // Set 1: bindings 0-3 map to DB_VERTEX_MAP..DB_ALIVE
    //         bindings 4-5 map to DB_HASHMAP_VERTEX, DB_HASHMAP_POSITION
    for (int i = 0; i < 4; i++) {
        int dbIdx = DB_VERTEX_MAP + i;
        bufInfos[16 + i] = { decimationBufs[dbIdx], 0, VK_WHOLE_SIZE };
        writes[16 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[16 + i].dstSet = decimationDescSet1;
        writes[16 + i].dstBinding = i;
        writes[16 + i].dstArrayElement = 0;
        writes[16 + i].descriptorCount = 1;
        writes[16 + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[16 + i].pBufferInfo = &bufInfos[16 + i];
    }

    bufInfos[20] = { decimationBufs[DB_HASHMAP_VERTEX], 0, VK_WHOLE_SIZE };
    writes[20].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[20].dstSet = decimationDescSet1;
    writes[20].dstBinding = 4;
    writes[20].dstArrayElement = 0;
    writes[20].descriptorCount = 1;
    writes[20].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[20].pBufferInfo = &bufInfos[20];

    bufInfos[21] = { decimationBufs[DB_HASHMAP_POSITION], 0, VK_WHOLE_SIZE };
    writes[21].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[21].dstSet = decimationDescSet1;
    writes[21].dstBinding = 5;
    writes[21].dstArrayElement = 0;
    writes[21].descriptorCount = 1;
    writes[21].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[21].pBufferInfo = &bufInfos[21];

    vkUpdateDescriptorSets(device, 22, writes.data(), 0, nullptr);
}

void App::cleanupDecimation() {
    for (uint32_t i = 0; i < DECIMATION_PASS_COUNT; i++) {
        if (decimationPipelines[i] != VK_NULL_HANDLE)
            vkDestroyPipeline(device, decimationPipelines[i], nullptr);
    }
    if (decimationPipelineLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, decimationPipelineLayout, nullptr);
    if (decimationDescSetLayout0 != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, decimationDescSetLayout0, nullptr);
    if (decimationDescSetLayout1 != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, decimationDescSetLayout1, nullptr);

    for (int i = 0; i < DB_COUNT; i++) {
        if (decimationBufs[i] != VK_NULL_HANDLE)
            vkDestroyBuffer(device, decimationBufs[i], nullptr);
        if (decimationMem[i] != VK_NULL_HANDLE)
            vkFreeMemory(device, decimationMem[i], nullptr);
    }

    if (counterReadbackMapped) {
        vkUnmapMemory(device, counterReadbackMem);
        counterReadbackMapped = nullptr;
    }
    if (counterReadbackBuf != VK_NULL_HANDLE)
        vkDestroyBuffer(device, counterReadbackBuf, nullptr);
    if (counterReadbackMem != VK_NULL_HANDLE)
        vkFreeMemory(device, counterReadbackMem, nullptr);
    counterReadbackBuf = VK_NULL_HANDLE;
    counterReadbackMem = VK_NULL_HANDLE;

    if (stagingReadbackBuf != VK_NULL_HANDLE)
        vkDestroyBuffer(device, stagingReadbackBuf, nullptr);
    if (stagingReadbackMem != VK_NULL_HANDLE)
        vkFreeMemory(device, stagingReadbackMem, nullptr);
    stagingReadbackBuf = VK_NULL_HANDLE;
    stagingReadbackMem = VK_NULL_HANDLE;
}

void App::recordComputeCommandBuffer(VkCommandBuffer commandBuffer, int workgroupsCount) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording compute command buffer!");
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSet, 0, nullptr);

    vkCmdDispatch(commandBuffer, workgroupsCount, 1, 1);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record compute command buffer!");
    }

}