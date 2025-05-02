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