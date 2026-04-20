// Most of the bare renderer implementation is based on code by https://vulkan-tutorial.com

// Define WSL_COMPAT to relax Vulkan requirements for WSL2/software drivers:
//  - Lowers API version from 1.4 to 1.1
//  - Makes VK_KHR_shader_non_semantic_info optional
//  - Falls back to graphics queue for compute if no dedicated compute queue
// Undefine this when targeting real hardware.
#define WSL_COMPAT

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtc/matrix_transform.hpp>


#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "meshoptimizer.h"

#include "vertex.hpp"
#include "utils.hpp"
#include "timer.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <bitset>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <limits>
#include <array>
#include <optional>
#include <set>


const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

inline std::string modelPath = std::getenv("MODEL_PATH") ? std::getenv("MODEL_PATH") : "scene.gltf";
inline std::string texturePath = std::getenv("TEXTURE_PATH") ? std::getenv("TEXTURE_PATH") : "bunny.png";

const bool simplify = false;
inline bool useGPUDecimation = true;
inline float decimationCostThreshold = std::getenv("DECIM_COST") ? std::stof(std::getenv("DECIM_COST")) : 0.01f;
inline float decimationTargetRatio = std::getenv("DECIM_RATIO") ? std::stof(std::getenv("DECIM_RATIO")) : 0.1f;
inline uint32_t maxDecimationIterations = std::getenv("DECIM_NUM") ? std::stoi(std::getenv("DECIM_NUM")) : 100;
inline float decimationGrowthRate = std::getenv("DECIM_GROW") ? std::stof(std::getenv("DECIM_GROW")) : 1.0f;
inline bool useCPUDecimation = (std::getenv("CPU_DECIM") ? std::string(std::getenv("CPU_DECIM")) != "0" : false);
inline uint32_t decimationCostMode = std::getenv("DECIM_MODE") ? std::stoi(std::getenv("DECIM_MODE")) : 0;
inline uint32_t decimationCostQuantBits = std::getenv("DECIM_QUANT") ? std::stoi(std::getenv("DECIM_QUANT")) : 23;
inline uint32_t decimationInnerRounds = std::getenv("DECIM_ROUNDS") ? std::stoi(std::getenv("DECIM_ROUNDS")) : 1;

const bool deviceLocalBuffer = false;
const bool singleBuffer = false;
const bool separateQueueFamily = false;

const bool enableValidationLayers = false;

struct DecimationPushConstants {
    uint32_t vertexCount;
    uint32_t triangleCount;
    uint32_t edgeCount;
    uint32_t hashMapSize;
    float costThreshold;
    uint32_t iteration;
    uint32_t costMode;
    uint32_t costQuantBits;
};

enum DecimationBuffer {
    DB_VERTEX = 0,
    DB_INDEX,
    DB_POS_INDEX,
    DB_VERTEX_FLAGS,
    DB_ADJ_HEAD,
    DB_TRI_ADJ_NEXT,
    DB_EDGE,
    DB_EDGE_TRI,
    DB_TRI_EDGE,
    DB_QUADRIC,
    DB_EDGE_COST,
    DB_EDGE_FLAG,
    DB_EDGE_TARGET,
    DB_TRI_DESCRIPTOR,
    DB_HASHMAP_EDGE,
    DB_COUNTER,
    DB_VERTEX_MAP,
    DB_POS_MAP,
    DB_SCAN,
    DB_ALIVE,
    DB_HASHMAP_VERTEX,
    DB_HASHMAP_POSITION,
    DB_COUNT
};

enum RenderMode { RENDER_GPU = 0, RENDER_CPU, RENDER_ORIGINAL, RENDER_MODE_COUNT };

struct MeshSnapshot {
    std::vector<Vertex> verts;
    std::vector<uint32_t> inds;
    VkBuffer vertBuf = VK_NULL_HANDLE;
    VkDeviceMemory vertMem = VK_NULL_HANDLE;
    VkBuffer idxBuf = VK_NULL_HANDLE;
    VkDeviceMemory idxMem = VK_NULL_HANDLE;
    bool valid = false;
};

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef WSL_COMPAT
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME,
};
const std::vector<const char*> optionalDeviceExtensions = {
    VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME
};
#else
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
    VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME,
};
const std::vector<const char*> optionalDeviceExtensions = {};
#endif

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> computeFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value() && computeFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class App {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    
    GLFWwindow* window;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkQueue computeQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkRenderPass computeRenderPass;
    VkDescriptorSetLayout computeDescriptorSetLayout;
    VkDescriptorSet computeDescriptorSet;
    VkPipelineLayout computePipelineLayout;
    VkPipeline computePipeline;

    VkCommandPool computeCommandPool;
    VkCommandPool commandPool;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    VkBuffer compVertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory compVertexBufferMemory  = VK_NULL_HANDLE;
    VkDeviceSize compVertexBufferSize;
    VkBuffer compMeshletsBuffer = VK_NULL_HANDLE;
    VkDeviceMemory compMeshletsMemory = VK_NULL_HANDLE;
    VkDeviceSize compMeshletsBufferSize;
    VkBuffer compMeshletVerticesBuffer = VK_NULL_HANDLE;
    VkDeviceMemory compMeshletVerticesMemory = VK_NULL_HANDLE;
    VkDeviceSize compMeshletVerticesBufferSize;
    VkBuffer compMeshletTrianglesBuffer = VK_NULL_HANDLE;
    VkDeviceMemory compMeshletTrianglesMemory = VK_NULL_HANDLE;
    VkDeviceSize compMeshletTrianglesBufferSize;

    VkBuffer totalBuffer = VK_NULL_HANDLE;
    VkDeviceMemory totalBufferMemory = VK_NULL_HANDLE;
    VkDeviceSize totalBufferSize;
    VkDeviceSize meshletsOffset;
    VkDeviceSize meshletVerticesOffset;
    VkDeviceSize meshletTrianglesOffset;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    VkCommandBuffer computeCommandBuffer;
    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;

    VkSemaphore computeFinishedSemaphore;
    VkFence computeFence;

    VkQueryPool timestampQueryPool = VK_NULL_HANDLE;
    float timestampPeriodNs = 1.0f;

    static constexpr uint32_t DECIMATION_PASS_COUNT = 12;
    VkPipeline decimationPipelines[DECIMATION_PASS_COUNT] = {};
    VkPipelineLayout decimationPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout decimationDescSetLayout0 = VK_NULL_HANDLE;
    VkDescriptorSetLayout decimationDescSetLayout1 = VK_NULL_HANDLE;
    VkDescriptorSet decimationDescSet0 = VK_NULL_HANDLE;
    VkDescriptorSet decimationDescSet1 = VK_NULL_HANDLE;

    VkBuffer decimationBufs[DB_COUNT] = {};
    VkDeviceMemory decimationMem[DB_COUNT] = {};
    VkDeviceSize decimationBufSizes[DB_COUNT] = {};

    bool framebufferResized = false;

    MeshSnapshot meshSnapshots[RENDER_MODE_COUNT];
    RenderMode activeRenderMode = RENDER_GPU;

    // Free camera state
    glm::vec3 cameraPos   = glm::vec3(2.0f, 2.0f, 2.0f);
    glm::vec3 cameraFront = glm::normalize(glm::vec3(-1.0f, -1.0f, -1.0f));
    glm::vec3 cameraUp    = glm::vec3(0.0f, 0.0f, 1.0f);
    float cameraYaw   = -135.0f;
    float cameraPitch = -35.26f;
    float cameraSpeed = 2.0f;
    float mouseSensitivity = 0.1f;
    bool mouseCapture = false;
    double lastMouseX = 0.0, lastMouseY = 0.0;
    bool firstMouse = true;
    float lastFrameTime = 0.0f;

    struct alignas(16) Triangle {
        glm::uvec3 v;
        Triangle(uint8_t v1, uint8_t v2, uint8_t v3) {v[0] = v1; v[1]=v2; v[2]=v3;}
    };

    struct UniformBufferObject {
        alignas(16) glm::mat4 model;
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
    };    

    void initWindow();
    void initVulkan();
    void mainLoop() {
        lastFrameTime = (float)glfwGetTime();
        while (!glfwWindowShouldClose(window)) {
            float currentTime = (float)glfwGetTime();
            float deltaTime = currentTime - lastFrameTime;
            lastFrameTime = currentTime;

            glfwPollEvents();
            processInput(deltaTime);
            drawFrame();
        }

        vkDeviceWaitIdle(device);
    }

    void printTimes() {
        auto avg = [](const std::vector<long long>& v) {return std::accumulate(v.begin(), v.end(), 0) / v.size();};
        std::cout << "total avg: "  << avg(timesWhole) << '\n';
        std::cout << "load avg: "   << avg(timesLoad) << '\n';
        std::cout << "algo avg: "   << avg(timesAlgo) << '\n';
        std::cout << "shader avg: " << avg(timesShader) << '\n';
    }

    void cleanupSwapChain();
    void cleanup();
    void recreateSwapChain();
    
    void createInstance();

    void setupDebugMessenger();

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void pickPhysicalDevice();

    void createLogicalDevice();

    void createSwapChain();

    void createImageViews();

    void createRenderPass();

    void createDescriptorSetLayout();

    void copyComputeBuffersLocal(std::vector<meshopt_Meshlet>& meshlets,
        std::vector<uint32_t>& meshletVertices,
        std::vector<Triangle>& meshletTriangles);

    void copyComputeBuffers(std::vector<meshopt_Meshlet>& meshlets,
        std::vector<uint32_t>& meshletVertices,
        std::vector<Triangle>& meshletTriangles);

    void createComputeBuffersLocal(std::vector<meshopt_Meshlet>& meshlets,
        std::vector<uint32_t>& meshletVertices,
        std::vector<Triangle>& meshletTriangles);

    void createComputeBuffers(std::vector<meshopt_Meshlet>& meshlets,
        std::vector<uint32_t>& meshletVertices,
        std::vector<Triangle>& meshletTriangles);

    void createComputeDescriptorSetLayout();

    void createComputeDescriptorSet();

    void createComputePipeline();

    void createGraphicsPipeline();

    void createFramebuffers();

    void createCommandPools();

    void createDepthResources();

    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

    VkFormat findDepthFormat();

    bool hasStencilComponent(VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    void createTextureImage();

    void createTextureImageView();

    void createTextureSampler();

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);

    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

    void createComputeCommandBuffer();

    std::vector<long long> timesWhole;
    std::vector<long long> timesLoad;
    std::vector<long long> timesAlgo;
    std::vector<long long> timesShader;

    void loadModel();
    void simplifyMesh();

    void createDecimationDescriptorSetLayouts();
    void createDecimationPipelineLayout();
    void createDecimationPipelines();
    void allocateDecimationBuffers(uint32_t vertexCount, uint32_t triangleCount);
    void writeDecimationDescriptorSets();
    void runDecimation();
    void runCPUDecimation();
    void createMeshBuffers();
    void cleanupMeshBuffers();
    void printDecimationMetrics();
    void cleanupDecimation();

    void splitMesh(std::vector<meshopt_Meshlet>& meshlets,
        std::vector<uint32_t>& meshletVertices, std::vector<Triangle>& meshletTriangles);
    
    void retrieveDataLocal(std::vector<meshopt_Meshlet>& meshlets,
        std::vector<uint32_t>& meshletVertices,
        std::vector<Triangle>& meshletTriangles);
    
    void retrieveData(std::vector<meshopt_Meshlet>& meshlets,
        std::vector<uint32_t>& meshletVertices,
        std::vector<Triangle>& meshletTriangles);

    void createAndCopyBufferLocal(VkDeviceSize bufferSize, VkBufferUsageFlags flags, void* srcData, VkBuffer& dstBuffer, VkDeviceMemory& dstBufferMemory);
    void createAndCopyBuffer2(VkDeviceSize bufferSize, VkBufferUsageFlags flags, void* srcData, VkBuffer& dstBuffer, VkDeviceMemory& dstBufferMemory);

    void createVertexBuffer();

    void createIndexBuffer();

    void createUniformBuffers();

    void createDescriptorPool();

    void createDescriptorSets();

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);

    VkCommandBuffer beginSingleTimeCommands();

    void endSingleTimeCommands(VkCommandBuffer commandBuffer);

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

    void copyBuffer2(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkDeviceSize srcOff, VkDeviceSize dstOff);

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    void createCommandBuffers();

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);

    void recordComputeCommandBuffer(VkCommandBuffer commandBuffer, int workgroupsCount);

    void createSyncObjects();

    void updateUniformBuffer(uint32_t currentImage);

    void drawFrame();

    VkShaderModule createShaderModule(const std::vector<char>& code);

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

    bool isDeviceSuitable(VkPhysicalDevice device);

    bool checkDeviceExtensionSupport(VkPhysicalDevice device);

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    
    std::vector<const char*> getRequiredExtensions();

    bool checkValidationLayerSupport();

    void processInput(float deltaTime);

    static std::vector<char> readFile(const std::string& filename);
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
};
