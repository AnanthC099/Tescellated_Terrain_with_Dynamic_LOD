#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <math.h>

#include "VK.h"
#define LOG_FILE (char*)"Log.txt"

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "Quaternion.h"

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

#pragma comment(lib, "vulkan-1.lib")

#include <cuda_runtime.h>
#include <curand_kernel.h>

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// Forward declarations for functions defined later
cudaError_t initialize_curand_noise_system(void);
VkResult CreateNoiseStorageBuffers(void);

const char* gpszAppName = "ARTR";

HWND ghwnd = NULL;
BOOL gbActive = FALSE;
DWORD dwStyle = 0;

WINDOWPLACEMENT wpPrev;
BOOL gbFullscreen = FALSE;
BOOL bWindowMinimize = FALSE;

// ============================================================================
// FREE CAMERA SYSTEM
// ============================================================================

// Input state for tracking keyboard and mouse
struct InputState {
    // Movement keys (WASD)
    BOOL keyW = FALSE;
    BOOL keyA = FALSE;
    BOOL keyS = FALSE;
    BOOL keyD = FALSE;
    // Vertical movement
    BOOL keySpace = FALSE;      // Move up
    BOOL keyCtrl = FALSE;       // Move down
    // Speed modifier
    BOOL keyShift = FALSE;      // Move faster
    // Mouse state
    int mouseX = 0;
    int mouseY = 0;
    int lastMouseX = 0;
    int lastMouseY = 0;
    int mouseDeltaX = 0;
    int mouseDeltaY = 0;
    BOOL mouseRightButton = FALSE;  // Right mouse button for look-around
    BOOL mouseCaptured = FALSE;     // Is mouse captured for camera control
};
InputState gInputState;

// Free camera structure
struct FreeCamera {
    glm::vec3 position;
    float yaw;              // Horizontal rotation (radians)
    float pitch;            // Vertical rotation (radians)
    float moveSpeed;        // Units per second
    float fastMoveSpeed;    // Speed when holding shift
    float mouseSensitivity; // Mouse look sensitivity

    // Computed vectors (updated each frame)
    glm::vec3 forward;
    glm::vec3 right;
    glm::vec3 up;
    glm::mat4 viewMatrix;

    FreeCamera() {
        // Start at the same position as the original static camera
        position = glm::vec3(0.0f, 30.0f, 150.0f);
        yaw = glm::radians(180.0f);  // Looking toward origin (negative Z)
        pitch = 0.0f;
        moveSpeed = 50.0f;
        fastMoveSpeed = 150.0f;
        mouseSensitivity = 0.002f;
        forward = glm::vec3(0.0f, 0.0f, -1.0f);
        right = glm::vec3(1.0f, 0.0f, 0.0f);
        up = glm::vec3(0.0f, 1.0f, 0.0f);
        viewMatrix = glm::mat4(1.0f);
    }

    void updateVectors() {
        // Calculate forward vector from yaw and pitch
        forward.x = cos(pitch) * sin(yaw);
        forward.y = sin(pitch);
        forward.z = cos(pitch) * cos(yaw);
        forward = glm::normalize(forward);

        // Right vector is perpendicular to forward and world up
        right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));

        // Camera up is perpendicular to forward and right
        up = glm::normalize(glm::cross(right, forward));

        // Build view matrix
        viewMatrix = glm::lookAt(position, position + forward, up);
    }
};
FreeCamera gCamera;

// Delta time for frame-rate independent movement
LARGE_INTEGER gPerfFrequency;
LARGE_INTEGER gLastFrameTime;
float gDeltaTime = 0.016f;  // Default to ~60fps

uint32_t enabledInstanceExtensionsCount = 0;

const char* enabledInstanceExtensionNames_array[3];

VkInstance vkInstance = VK_NULL_HANDLE;

VkSurfaceKHR vkSurfaceKHR = VK_NULL_HANDLE;

VkPhysicalDevice vkPhysicalDevice_selected = VK_NULL_HANDLE;
uint32_t graphicsQuequeFamilyIndex_selected = UINT32_MAX;
VkPhysicalDeviceMemoryProperties vkPhysicalDeviceMemoryProperties;

uint32_t physicalDeviceCount = 0;
VkPhysicalDevice *vkPhysicalDevice_array = NULL;

uint32_t enabledDeviceExtensionsCount = 0;

const char* enabledDeviceExtensionNames_array[2];

VkDevice vkDevice = VK_NULL_HANDLE;

VkQueue vkQueue =  VK_NULL_HANDLE;

VkFormat vkFormat_color = VK_FORMAT_UNDEFINED;
VkColorSpaceKHR vkColorSpaceKHR = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

VkPresentModeKHR vkPresentModeKHR = VK_PRESENT_MODE_FIFO_KHR;

int winWidth = WIN_WIDTH;
int winHeight = WIN_HEIGHT;

VkSwapchainKHR vkSwapchainKHR =  VK_NULL_HANDLE;

VkExtent2D vkExtent2D_SwapChain;

uint32_t swapchainImageCount = UINT32_MAX;

VkImage *swapChainImage_array = NULL;

VkImageView *swapChainImageView_array = NULL;

VkCommandPool vkCommandPool = VK_NULL_HANDLE;

VkRenderPass vkRenderPass = VK_NULL_HANDLE;

VkFramebuffer *vkFramebuffer_array = NULL;

VkSemaphore vkSemaphore_BackBuffer = VK_NULL_HANDLE;
VkSemaphore vkSemaphore_RenderComplete = VK_NULL_HANDLE;

VkFence *vkFence_array = NULL;

VkClearColorValue vkClearColorValue;

VkClearDepthStencilValue vkClearDepthStencilValue;

BOOL bInitialized = FALSE;
uint32_t currentImageIndex = UINT32_MAX;

BOOL bValidation = TRUE;
uint32_t enabledValidationLayerCount = 0;
const char* enabledValidationlayerNames_array[1];
VkDebugReportCallbackEXT vkDebugReportCallbackEXT = VK_NULL_HANDLE;

PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT_fnptr = NULL;

typedef struct
{
	VkBuffer vkBuffer;
	VkDeviceMemory vkDeviceMemory;
}VertexData;

struct MyUniformData
{
	glm::mat4 modelMatrix;
	glm::mat4 viewMatrix;
	glm::mat4 projectionMatrix;
	glm::vec4 color;
};

struct UniformData
{
	VkBuffer vkBuffer;
	VkDeviceMemory vkDeviceMemory;
};

struct UniformData uniformData;

VkShaderModule vkShaderMoudule_vertex_shader = VK_NULL_HANDLE;
VkShaderModule vkShaderMoudule_fragment_shader = VK_NULL_HANDLE;
VkShaderModule vkShaderModule_tess_control = VK_NULL_HANDLE;
VkShaderModule vkShaderModule_tess_eval = VK_NULL_HANDLE;

VkDescriptorSetLayout vkDescriptorSetLayout = VK_NULL_HANDLE;

VkPipelineLayout vkPipelineLayout = VK_NULL_HANDLE;

VkDescriptorPool vkDescriptorPool = VK_NULL_HANDLE;

VkDescriptorSet vkDescriptorSet = VK_NULL_HANDLE;

VkViewport vkViewPort;

VkRect2D vkRect2D_scissor;

VkPipeline vkPipeline = VK_NULL_HANDLE;

VkFormat vkFormat_depth = VK_FORMAT_UNDEFINED;
VkImage vkImage_depth = VK_NULL_HANDLE;
VkDeviceMemory vkDeviceMemory_depth = VK_NULL_HANDLE;
VkImageView vkImageView_depth = VK_NULL_HANDLE;

VkCommandBuffer *vkCommandBuffer_for_1024_x_1024_graphics_array = NULL;

float animationTime = 0.0f;

cudaError_t cudaResult;

VkExternalMemoryHandleTypeFlagBits vkExternalMemoryHandleTypeFlagBits;

cudaExternalMemory_t cuExternalMemory_t;

void* pos_CUDA = NULL;

VertexData vertexData_external;

VertexData vertexdata_indirect_buffer;
BOOL bIndirectBufferMemoryCoherent = TRUE;

int iResult = 0;

// ============================================================================
// CUDA TERRAIN GENERATION SYSTEM
// Complete terrain heightmap generation using cuRAND and CUDA streams
// All noise computation moved from shader to CUDA for maximum performance
// ============================================================================

#define HEIGHTMAP_SIZE 2048          // 2048x2048 heightmap resolution
#define GRADIENT_TABLE_SIZE 512      // Gradient vectors for Perlin noise
#define PERMUTATION_TABLE_SIZE 512   // Permutation table for noise indexing
#define NUM_CUDA_STREAMS 4           // Number of CUDA streams for async ops

// CUDA streams for parallel/async operations
cudaStream_t cudaStreams[NUM_CUDA_STREAMS];

// Heightmap data - stores complete terrain height computed by CUDA
struct HeightmapData {
    // Vulkan resources
    VkImage vkImage;
    VkDeviceMemory vkDeviceMemory;
    VkImageView vkImageView;
    VkSampler vkSampler;
    // CUDA interop
    cudaExternalMemory_t cuExternalMemory;
    cudaMipmappedArray_t cuMipmappedArray;
    cudaSurfaceObject_t cuSurfaceObject;
    // Direct CUDA buffer (for non-interop path)
    float* d_heightmap;
};
HeightmapData heightmapData;

// Gradient and permutation tables for noise generation
float4* d_gradientTable = NULL;            // GPU gradient vectors (cuRAND generated)
int* d_permutationTable = NULL;            // GPU permutation table (cuRAND shuffled)

// cuRAND state array for parallel random generation
curandState* d_curandStates = NULL;
int numCurandStates = 0;

// Fixed position for distance-based LOD
float3 h_fixedPos = make_float3(0.0f, 30.0f, 150.0f);
__constant__ float3 d_fixedPos;

// Terrain generation parameters
__constant__ float d_terrainSize;
__constant__ float d_heightScale;
__constant__ int d_heightmapSize;

BOOL bCudaTerrainSystemInitialized = FALSE;

// Noise texture configuration (used for cuRAND state initialization)
#define NOISE_TEXTURE_SIZE 512  // Size for cuRAND state grid

// Flag to track noise system initialization
BOOL bNoiseTexturesInitialized = FALSE;

// Storage buffers for passing cuRAND-generated noise tables to shaders
struct NoiseStorageBuffers {
    VkBuffer gradientBuffer;
    VkDeviceMemory gradientMemory;
    VkBuffer permutationBuffer;
    VkDeviceMemory permutationMemory;
    // Heightmap storage buffer for shader access
    VkBuffer heightmapBuffer;
    VkDeviceMemory heightmapMemory;
    // Normal map buffer (computed from heightmap gradients)
    VkBuffer normalBuffer;
    VkDeviceMemory normalMemory;
    // Tangent space buffer (tangent + bitangent)
    VkBuffer tangentBuffer;
    VkDeviceMemory tangentMemory;
    // Tessellation factors buffer (adaptive LOD)
    VkBuffer tessFactorBuffer;
    VkDeviceMemory tessFactorMemory;
    // Visibility buffer for frustum culling
    VkBuffer visibilityBuffer;
    VkDeviceMemory visibilityMemory;
};
struct NoiseStorageBuffers noiseStorageBuffers;

// ============================================================================
// CUDA ADVANCED TERRAIN PROCESSING
// Normal maps, tangent spaces, frustum culling, adaptive tessellation, LOD
// ============================================================================

// LOD mipmap levels for heightmap
#define NUM_LOD_LEVELS 4
float* d_heightmapLOD[NUM_LOD_LEVELS] = {NULL, NULL, NULL, NULL};
int heightmapLODSizes[NUM_LOD_LEVELS] = {2048, 1024, 512, 256};

// Normal map buffer (vec3 normals packed as float4 for alignment)
float4* d_normalMap = NULL;

// Tangent space buffer (tangent.xyz + bitangent sign in w)
float4* d_tangentMap = NULL;

// Tessellation factors (one per patch)
float* d_tessellationFactors = NULL;

// Visibility mask for frustum culling (one int per patch, 0=culled, 1=visible)
int* d_visibilityMask = NULL;

// Visible patch count (atomic counter for indirect draw)
int* d_visiblePatchCount = NULL;

// Indirect draw command buffer (GPU-driven rendering)
struct IndirectDrawCommand {
    unsigned int vertexCount;
    unsigned int instanceCount;
    unsigned int firstVertex;
    unsigned int firstInstance;
};
IndirectDrawCommand* d_indirectDrawCmd = NULL;

// Frustum planes (6 planes: left, right, bottom, top, near, far)
float4* d_frustumPlanes = NULL;

// Force heightmap update on first frame
BOOL bForceHeightmapUpdate = TRUE;

// Vulkan descriptor resources for heightmap texture
VkDescriptorSetLayout vkDescriptorSetLayout_heightmap = VK_NULL_HANDLE;
VkDescriptorPool vkDescriptorPool_heightmap = VK_NULL_HANDLE;
VkDescriptorSet vkDescriptorSet_heightmap = VK_NULL_HANDLE;

void FileIO(const char* format, ...)
{
	FILE* file = fopen(LOG_FILE, "a");
	if (file)
	{
		va_list args;
		va_start(args, format);
		vfprintf(file, format, args);
		va_end(args);
		fclose(file);
	}
}

#define PATCH_GRID_SIZE 64
#define TERRAIN_SIZE 800.0f
#define HEIGHT_SCALE 35.0f

// ============================================================================
// CUDA TERRAIN GENERATION - COMPLETE NOISE SYSTEM
// All terrain computation moved from shader to CUDA for maximum performance
// Uses cuRAND for high-quality random number generation
// Uses CUDA streams for parallel/async execution
// ============================================================================

// Patch generation kernel (generates flat terrain base grid)
__global__ void generateFlatTerrainPatches(float4 *pos, unsigned int patchGridSize,
                                           float terrainSize) {
    unsigned int patchX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int patchY = blockIdx.y * blockDim.y + threadIdx.y;

    if (patchX >= patchGridSize || patchY >= patchGridSize) return;

    unsigned int baseIdx = (patchY * patchGridSize + patchX) * 4;

    float patchSize = terrainSize / (float)patchGridSize;
    float x0 = patchX * patchSize - (terrainSize * 0.5f);
    float z0 = patchY * patchSize - (terrainSize * 0.5f);
    float x1 = x0 + patchSize;
    float z1 = z0 + patchSize;

    pos[baseIdx + 0] = make_float4(x0, 0.0f, z0, 1.0f);
    pos[baseIdx + 1] = make_float4(x1, 0.0f, z0, 1.0f);
    pos[baseIdx + 2] = make_float4(x1, 0.0f, z1, 1.0f);
    pos[baseIdx + 3] = make_float4(x0, 0.0f, z1, 1.0f);
}

// ============================================================================
// cuRAND INITIALIZATION KERNELS
// ============================================================================

__global__ void initCurandStates(curandState* states, unsigned long long seed, int numStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numStates) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void generateGradientTable(float4* gradients, curandState* states, int tableSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tableSize) {
        curandState localState = states[idx % tableSize];
        float u1 = curand_uniform(&localState);
        float u2 = curand_uniform(&localState);
        float theta = 2.0f * 3.14159265359f * u1;
        float phi = acosf(2.0f * u2 - 1.0f);
        gradients[idx] = make_float4(sinf(phi) * cosf(theta), sinf(phi) * sinf(theta),
                                      cosf(phi), curand_uniform(&localState) * 2.0f - 1.0f);
        states[idx % tableSize] = localState;
    }
}

__global__ void generatePermutationTable(int* permutation, curandState* states, int tableSize) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        curandState localState = states[0];
        for (int i = 0; i < tableSize; i++) permutation[i] = i % 256;
        for (int i = tableSize - 1; i > 0; i--) {
            int j = (int)(curand_uniform(&localState) * (i + 1));
            if (j > i) j = i;
            int temp = permutation[i];
            permutation[i] = permutation[j];
            permutation[j] = temp;
        }
        states[0] = localState;
    }
}

// ============================================================================
// CUDA DEVICE NOISE FUNCTIONS - Complete terrain noise library
// ============================================================================

// Quintic interpolation for smooth noise
__device__ __forceinline__ float quintic(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// Smoothstep function
__device__ __forceinline__ float smoothstep_cuda(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Linear interpolation
__device__ __forceinline__ float mix_cuda(float a, float b, float t) {
    return a + t * (b - a);
}

// Permutation table lookup
__device__ __forceinline__ int perm(const int* __restrict__ permTable, int idx) {
    return permTable[idx & (PERMUTATION_TABLE_SIZE - 1)];
}

// Gradient lookup
__device__ __forceinline__ float2 grad2D(const float4* __restrict__ gradTable, int idx) {
    float4 g = gradTable[idx & (GRADIENT_TABLE_SIZE - 1)];
    return make_float2(g.x, g.y);
}

// ============================================================================
// 2D PERLIN NOISE using cuRAND tables
// ============================================================================
__device__ float noise2D(float2 p, const float4* __restrict__ gradTable,
                         const int* __restrict__ permTable) {
    int ix = (int)floorf(p.x) & 255;
    int iy = (int)floorf(p.y) & 255;
    float fx = p.x - floorf(p.x);
    float fy = p.y - floorf(p.y);

    float u = quintic(fx);
    float v = quintic(fy);

    int p00 = perm(permTable, perm(permTable, ix) + iy);
    int p10 = perm(permTable, perm(permTable, ix + 1) + iy);
    int p01 = perm(permTable, perm(permTable, ix) + iy + 1);
    int p11 = perm(permTable, perm(permTable, ix + 1) + iy + 1);

    float2 g00 = grad2D(gradTable, p00);
    float2 g10 = grad2D(gradTable, p10);
    float2 g01 = grad2D(gradTable, p01);
    float2 g11 = grad2D(gradTable, p11);

    float n00 = g00.x * fx + g00.y * fy;
    float n10 = g10.x * (fx - 1.0f) + g10.y * fy;
    float n01 = g01.x * fx + g01.y * (fy - 1.0f);
    float n11 = g11.x * (fx - 1.0f) + g11.y * (fy - 1.0f);

    return mix_cuda(mix_cuda(n00, n10, u), mix_cuda(n01, n11, u), v);
}

// Seeded noise with offset
__device__ float noise2DSeeded(float2 p, float seed, const float4* __restrict__ gradTable,
                                const int* __restrict__ permTable) {
    int seedOffset = (int)(seed * 17.31f) & 255;
    p.x += seed * 7.31f;
    p.y += seed * 11.17f;

    int ix = ((int)floorf(p.x) + seedOffset) & 255;
    int iy = ((int)floorf(p.y) + (int)(seed * 23.57f)) & 255;
    float fx = p.x - floorf(p.x);
    float fy = p.y - floorf(p.y);

    float u = quintic(fx);
    float v = quintic(fy);

    int gradOffset = (int)(seed * 7.13f) & 255;
    int p00 = perm(permTable, perm(permTable, ix) + iy) + gradOffset;
    int p10 = perm(permTable, perm(permTable, ix + 1) + iy) + gradOffset;
    int p01 = perm(permTable, perm(permTable, ix) + iy + 1) + gradOffset;
    int p11 = perm(permTable, perm(permTable, ix + 1) + iy + 1) + gradOffset;

    float2 g00 = grad2D(gradTable, p00);
    float2 g10 = grad2D(gradTable, p10);
    float2 g01 = grad2D(gradTable, p01);
    float2 g11 = grad2D(gradTable, p11);

    float n00 = g00.x * fx + g00.y * fy;
    float n10 = g10.x * (fx - 1.0f) + g10.y * fy;
    float n01 = g01.x * fx + g01.y * (fy - 1.0f);
    float n11 = g11.x * (fx - 1.0f) + g11.y * (fy - 1.0f);

    return mix_cuda(mix_cuda(n00, n10, u), mix_cuda(n01, n11, u), v);
}

// ============================================================================
// FBM (Fractional Brownian Motion) VARIANTS
// ============================================================================

// Rotation matrix constants for domain rotation
__device__ __constant__ float ROT_COS = 0.80f;
__device__ __constant__ float ROT_SIN = 0.60f;

__device__ float2 rotateVec(float2 p) {
    return make_float2(p.x * ROT_COS - p.y * ROT_SIN, p.x * ROT_SIN + p.y * ROT_COS);
}

// Standard FBM with rotation
__device__ float fbmRotated(float2 p, int octaves, const float4* gradTable, const int* permTable) {
    float value = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise2D(make_float2(p.x * frequency, p.y * frequency), gradTable, permTable);
        amplitude *= 0.5f;
        frequency *= 2.0f;
        p = rotateVec(p);
    }
    return value;
}

// Seeded FBM
__device__ float fbmRotatedSeeded(float2 p, int octaves, float seed,
                                   const float4* gradTable, const int* permTable) {
    float value = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise2DSeeded(make_float2(p.x * frequency, p.y * frequency),
                                            seed + (float)i * 7.31f, gradTable, permTable);
        amplitude *= 0.5f;
        frequency *= 2.0f;
        p = rotateVec(p);
    }
    return value;
}

// Billowy FBM (squared noise for organic shapes)
__device__ float fbmBillowySeeded(float2 p, int octaves, float seed,
                                   const float4* gradTable, const int* permTable) {
    float value = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;

    for (int i = 0; i < octaves; i++) {
        float n = noise2DSeeded(make_float2(p.x * frequency, p.y * frequency),
                                 seed + (float)i * 11.17f, gradTable, permTable);
        n = n * n;  // Square for billowy effect
        value += amplitude * n;
        amplitude *= 0.45f;
        frequency *= 2.0f;
        p = rotateVec(p);
    }
    return value;
}

// Ridged FBM (for subtle ridges)
__device__ float fbmRidgedSeeded(float2 p, int octaves, float seed,
                                  const float4* gradTable, const int* permTable) {
    float value = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;
    float weight = 1.0f;

    for (int i = 0; i < octaves; i++) {
        float signal = 1.0f - fabsf(noise2DSeeded(make_float2(p.x * frequency, p.y * frequency),
                                                   seed + (float)i * 13.37f, gradTable, permTable));
        signal = signal * signal * weight;
        weight = fminf(fmaxf(signal * 1.5f, 0.0f), 1.0f);
        value += amplitude * signal;
        amplitude *= 0.4f;
        frequency *= 2.2f;
        p = rotateVec(p);
    }
    return value * 0.7f;
}

// ============================================================================
// DOMAIN WARPING
// ============================================================================

__device__ float2 domainWarpLight(float2 p, float strength,
                                   const float4* gradTable, const int* permTable) {
    float warpX = fbmRotatedSeeded(p, 3, 3.0f, gradTable, permTable);
    float warpY = fbmRotatedSeeded(p, 3, 7.0f, gradTable, permTable);
    return make_float2(p.x + warpX * strength, p.y + warpY * strength);
}

__device__ float2 domainWarpStrong(float2 p, float strength,
                                    const float4* gradTable, const int* permTable) {
    float2 q = make_float2(fbmRotatedSeeded(p, 3, 11.0f, gradTable, permTable),
                            fbmRotatedSeeded(p, 3, 13.0f, gradTable, permTable));
    float2 r = make_float2(
        fbmRotatedSeeded(make_float2(p.x + 3.0f * q.x, p.y + 3.0f * q.y), 3, 17.0f, gradTable, permTable),
        fbmRotatedSeeded(make_float2(p.x + 3.0f * q.x, p.y + 3.0f * q.y), 3, 19.0f, gradTable, permTable));
    return make_float2(p.x + r.x * strength, p.y + r.y * strength);
}

// ============================================================================
// CONTROL MASKS
// ============================================================================

__device__ float getRoughnessMask(float2 p, const float4* gradTable, const int* permTable) {
    float n = fbmRotatedSeeded(make_float2(p.x * 0.003f, p.y * 0.003f), 4, 23.0f, gradTable, permTable);
    return smoothstep_cuda(-0.3f, 0.5f, n);
}

__device__ float getMoundMask(float2 p, const float4* gradTable, const int* permTable) {
    float n = fbmBillowySeeded(make_float2(p.x * 0.004f, p.y * 0.004f), 3, 29.0f, gradTable, permTable);
    return smoothstep_cuda(0.1f, 0.5f, n);
}

__device__ float getBasinMask(float2 p, const float4* gradTable, const int* permTable) {
    float n = fbmRotatedSeeded(make_float2(p.x * 0.005f, p.y * 0.005f), 3, 31.0f, gradTable, permTable);
    return smoothstep_cuda(0.4f, 0.1f, n);
}

__device__ float getPathMask(float2 p, const float4* gradTable, const int* permTable) {
    float2 pathCoord = make_float2(p.x * 0.006f, p.y * 0.006f);
    float pathNoise = fbmRotatedSeeded(pathCoord, 3, 37.0f, gradTable, permTable);
    float pathCurve = sinf(p.x * 0.015f + pathNoise * 3.0f) * 0.5f + 0.5f;
    float pathWidth = smoothstep_cuda(0.35f, 0.45f, pathCurve) * smoothstep_cuda(0.65f, 0.55f, pathCurve);
    return pathWidth * 0.8f;
}

__device__ float getLawnMask(float2 p, const float4* gradTable, const int* permTable) {
    float n = fbmBillowySeeded(make_float2(p.x * 0.003f, p.y * 0.003f), 3, 41.0f, gradTable, permTable);
    return smoothstep_cuda(0.3f, 0.6f, n) * 0.6f;
}

// ============================================================================
// TERRAIN LAYER COMPUTATIONS
// ============================================================================

__device__ float computeMacroHeight(float2 worldPos, float* overallSlope,
                                     const float4* gradTable, const int* permTable) {
    float2 macroCoord = make_float2(worldPos.x * 0.0015f, worldPos.y * 0.0015f);
    float2 warpedCoord = domainWarpStrong(macroCoord, 0.3f, gradTable, permTable);

    *overallSlope = worldPos.x * 0.0003f + worldPos.y * 0.0002f;
    *overallSlope += sinf(worldPos.x * 0.002f) * 0.01f;

    float broadMounds = fbmBillowySeeded(warpedCoord, 4, 131.0f, gradTable, permTable) * 0.6f;
    broadMounds = fmaxf(broadMounds, 0.0f);

    float basins = fbmRotatedSeeded(make_float2(warpedCoord.x * 0.8f, warpedCoord.y * 0.8f),
                                     3, 137.0f, gradTable, permTable);
    basins = fminf(basins, 0.0f) * 0.4f;

    return *overallSlope + broadMounds + basins;
}

__device__ float computeMesoHeight(float2 worldPos, float roughnessMask,
                                    const float4* gradTable, const int* permTable) {
    float2 mesoCoord = make_float2(worldPos.x * 0.008f, worldPos.y * 0.008f);
    float2 warpedCoord = domainWarpLight(mesoCoord, 0.5f, gradTable, permTable);

    float irregularLumps = fbmRotatedSeeded(warpedCoord, 5, 139.0f, gradTable, permTable) * 0.2f;
    float softRidges = fbmRidgedSeeded(make_float2(warpedCoord.x * 1.5f, warpedCoord.y * 1.5f),
                                        4, 149.0f, gradTable, permTable) * 0.12f;

    return (irregularLumps + softRidges) * roughnessMask;
}

__device__ float computeMicroHeight(float2 worldPos, float roughnessMask, float distanceFade,
                                     const float4* gradTable, const int* permTable) {
    float2 microCoord = make_float2(worldPos.x * 0.03f, worldPos.y * 0.03f);

    float detail1 = noise2DSeeded(microCoord, 151.0f, gradTable, permTable) * 0.03f;
    float detail2 = noise2DSeeded(make_float2(microCoord.x * 2.5f, microCoord.y * 2.5f),
                                   157.0f, gradTable, permTable) * 0.015f;
    float detail3 = noise2DSeeded(make_float2(microCoord.x * 4.0f, microCoord.y * 4.0f),
                                   163.0f, gradTable, permTable) * 0.008f;

    float maskSquared = roughnessMask * roughnessMask;
    return (detail1 + detail2 + detail3) * maskSquared * distanceFade;
}

// ============================================================================
// NATURAL IMPERFECTIONS
// ============================================================================

__device__ float getHighFrequencyBumps(float2 worldPos, float heterogeneityMask,
                                        const float4* gradTable, const int* permTable) {
    float freq1 = noise2DSeeded(make_float2(worldPos.x * 0.12f, worldPos.y * 0.12f),
                                 79.0f, gradTable, permTable) * 0.008f;
    float freq2 = noise2DSeeded(make_float2(worldPos.x * 0.25f, worldPos.y * 0.25f),
                                 83.0f, gradTable, permTable) * 0.005f;
    float freq3 = noise2DSeeded(make_float2(worldPos.x * 0.4f, worldPos.y * 0.4f),
                                 89.0f, gradTable, permTable) * 0.003f;
    float freq4 = noise2DSeeded(make_float2(worldPos.x * 0.6f, worldPos.y * 0.6f),
                                 97.0f, gradTable, permTable) * 0.002f;

    float bumps = freq1 + freq2 + freq3 + freq4;
    bumps *= (0.5f + heterogeneityMask * 0.8f);

    float clumpNoise = fbmRotatedSeeded(make_float2(worldPos.x * 0.02f, worldPos.y * 0.02f),
                                         2, 101.0f, gradTable, permTable);
    float clumps = smoothstep_cuda(0.3f, 0.5f, clumpNoise) *
                   noise2DSeeded(make_float2(worldPos.x * 0.08f, worldPos.y * 0.08f),
                                  103.0f, gradTable, permTable) * 0.012f;
    bumps += clumps * heterogeneityMask;

    return bumps;
}

__device__ float getRootBumps(float2 worldPos, const float4* gradTable, const int* permTable) {
    float2 rootCoord = make_float2(worldPos.x * 0.02f, worldPos.y * 0.02f);

    float rootPattern1 = sinf(rootCoord.x * 5.0f +
                              noise2DSeeded(make_float2(rootCoord.x * 2.0f, rootCoord.y * 2.0f),
                                             107.0f, gradTable, permTable) * 4.0f);
    float rootPattern2 = sinf(rootCoord.y * 4.5f +
                              noise2DSeeded(make_float2(rootCoord.x * 2.5f, rootCoord.y * 2.5f),
                                             109.0f, gradTable, permTable) * 3.5f);

    float ridge1 = 1.0f - fabsf(rootPattern1);
    float ridge2 = 1.0f - fabsf(rootPattern2);
    ridge1 = powf(ridge1, 4.0f) * 0.015f;
    ridge2 = powf(ridge2, 4.0f) * 0.012f;

    float rootMask = smoothstep_cuda(0.2f, 0.5f,
                                      fbmRotatedSeeded(make_float2(worldPos.x * 0.008f, worldPos.y * 0.008f),
                                                        2, 113.0f, gradTable, permTable));
    return (ridge1 + ridge2) * rootMask;
}

__device__ float computeNaturalImperfections(float2 worldPos, float distanceFade, float pathMask,
                                              const float4* gradTable, const int* permTable) {
    float soilMask = 1.0f - pathMask * 0.8f;
    float surfaceBumps = getHighFrequencyBumps(worldPos, soilMask, gradTable, permTable);
    float rootBumps = getRootBumps(worldPos, gradTable, permTable);
    return (surfaceBumps + rootBumps * 0.5f) * distanceFade;
}

// ============================================================================
// FEATURE COMPUTATIONS
// ============================================================================

__device__ float computeFeatures(float2 worldPos, float baseHeight, float roughnessMask,
                                  const float4* gradTable, const int* permTable) {
    float pathMask = getPathMask(worldPos, gradTable, permTable);
    float moundMask = getMoundMask(worldPos, gradTable, permTable);

    float bedHeight = moundMask * 0.15f;
    float pathDepression = pathMask * -0.08f;

    return bedHeight + pathDepression;
}

__device__ float applyFlatRegions(float height, float2 worldPos,
                                   const float4* gradTable, const int* permTable) {
    float lawnMask = getLawnMask(worldPos, gradTable, permTable);
    float targetHeight = 0.1f;
    float flattenAlpha = lawnMask * 0.5f;
    float flattenedHeight = mix_cuda(height, targetHeight, flattenAlpha);

    float pathMask = getPathMask(worldPos, gradTable, permTable);
    float pathTargetHeight = -0.05f;
    flattenedHeight = mix_cuda(flattenedHeight, pathTargetHeight, pathMask * 0.6f);

    return flattenedHeight;
}

// ============================================================================
// MAIN TERRAIN HEIGHT CALCULATION (runs on GPU)
// ============================================================================

__device__ float calculateTerrainHeight(float2 worldPos, float3 cameraPos,
                                         const float4* gradTable, const int* permTable) {
    // Control masks
    float roughnessMask = getRoughnessMask(worldPos, gradTable, permTable);
    float dist = sqrtf((cameraPos.x - worldPos.x) * (cameraPos.x - worldPos.x) +
                       (cameraPos.z - worldPos.y) * (cameraPos.z - worldPos.y));
    float distanceFade = 1.0f - smoothstep_cuda(50.0f, 400.0f, dist);
    float pathMask = getPathMask(worldPos, gradTable, permTable);

    // Macro layer
    float overallSlope;
    float macroHeight = computeMacroHeight(worldPos, &overallSlope, gradTable, permTable);

    // Meso layer
    float mesoHeight = computeMesoHeight(worldPos, roughnessMask, gradTable, permTable);

    // Micro layer
    float microHeight = computeMicroHeight(worldPos, roughnessMask, distanceFade, gradTable, permTable);

    // Features
    float baseHeight = macroHeight + mesoHeight;
    float features = computeFeatures(worldPos, baseHeight, roughnessMask, gradTable, permTable);

    // Natural imperfections
    float naturalImperfections = computeNaturalImperfections(worldPos, distanceFade, pathMask,
                                                              gradTable, permTable);

    // Combine layers
    float height = macroHeight + mesoHeight + microHeight + features + naturalImperfections;

    // Apply flat regions
    height = applyFlatRegions(height, worldPos, gradTable, permTable);

    // Distance-based detail reduction
    float detailContrib = (mesoHeight + microHeight) * (1.0f - distanceFade) * 0.3f;
    height -= detailContrib;

    return height;
}

// ============================================================================
// MAIN HEIGHTMAP GENERATION KERNEL
// Generates complete terrain heightmap using all noise layers
// ============================================================================

__global__ void generateTerrainHeightmap(
    float* __restrict__ heightmap,
    const float4* __restrict__ gradTable,
    const int* __restrict__ permTable,
    int heightmapSize,
    float terrainSize,
    float3 cameraPos
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= heightmapSize || y >= heightmapSize) return;

    // Convert heightmap coordinates to world coordinates
    float worldX = ((float)x / (float)(heightmapSize - 1) - 0.5f) * terrainSize;
    float worldZ = ((float)y / (float)(heightmapSize - 1) - 0.5f) * terrainSize;
    float2 worldPos = make_float2(worldX, worldZ);

    // Calculate terrain height using complete noise system
    float height = calculateTerrainHeight(worldPos, cameraPos, gradTable, permTable);

    // Store in heightmap (normalized height will be scaled in shader)
    heightmap[y * heightmapSize + x] = height;
}

// ============================================================================
// ASYNC HEIGHTMAP GENERATION WITH STREAMS
// Divides heightmap into quadrants for parallel generation
// ============================================================================

__global__ void generateTerrainHeightmapQuadrant(
    float* __restrict__ heightmap,
    const float4* __restrict__ gradTable,
    const int* __restrict__ permTable,
    int heightmapSize,
    float terrainSize,
    float3 cameraPos,
    int quadrantX,
    int quadrantY,
    int quadrantSize
) {
    int localX = blockIdx.x * blockDim.x + threadIdx.x;
    int localY = blockIdx.y * blockDim.y + threadIdx.y;

    if (localX >= quadrantSize || localY >= quadrantSize) return;

    int x = quadrantX * quadrantSize + localX;
    int y = quadrantY * quadrantSize + localY;

    if (x >= heightmapSize || y >= heightmapSize) return;

    float worldX = ((float)x / (float)(heightmapSize - 1) - 0.5f) * terrainSize;
    float worldZ = ((float)y / (float)(heightmapSize - 1) - 0.5f) * terrainSize;
    float2 worldPos = make_float2(worldX, worldZ);

    float height = calculateTerrainHeight(worldPos, cameraPos, gradTable, permTable);
    heightmap[y * heightmapSize + x] = height;
}

// ============================================================================
// CUDA ADVANCED TERRAIN PROCESSING KERNELS
// Normal maps, tangent spaces, frustum culling, adaptive tessellation, LOD
// ============================================================================

// ----------------------------------------------------------------------------
// NORMAL MAP COMPUTATION
// Computes surface normals from heightmap using central differences
// ----------------------------------------------------------------------------
__global__ void computeNormalMap(
    float4* __restrict__ normalMap,
    const float* __restrict__ heightmap,
    int heightmapSize,
    float terrainSize,
    float heightScale
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= heightmapSize || y >= heightmapSize) return;

    // Sample neighboring heights with boundary clamping
    int xm = max(x - 1, 0);
    int xp = min(x + 1, heightmapSize - 1);
    int ym = max(y - 1, 0);
    int yp = min(y + 1, heightmapSize - 1);

    float hL = heightmap[y * heightmapSize + xm] * heightScale;
    float hR = heightmap[y * heightmapSize + xp] * heightScale;
    float hD = heightmap[ym * heightmapSize + x] * heightScale;
    float hU = heightmap[yp * heightmapSize + x] * heightScale;

    // Compute world-space step size
    float stepSize = terrainSize / (float)(heightmapSize - 1);

    // Central difference gradients
    float dhdx = (hR - hL) / (2.0f * stepSize);
    float dhdz = (hU - hD) / (2.0f * stepSize);

    // Normal from cross product of tangent vectors
    // tangentX = (1, dhdx, 0), tangentZ = (0, dhdz, 1)
    // normal = tangentZ × tangentX = (-dhdx, 1, -dhdz)
    float3 normal = make_float3(-dhdx, 1.0f, -dhdz);

    // Normalize
    float len = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    normal.x /= len;
    normal.y /= len;
    normal.z /= len;

    // Store as float4 (w = slope magnitude for detail mapping)
    float slope = sqrtf(dhdx * dhdx + dhdz * dhdz);
    normalMap[y * heightmapSize + x] = make_float4(normal.x, normal.y, normal.z, slope);
}

// ----------------------------------------------------------------------------
// TANGENT SPACE COMPUTATION
// Computes tangent and bitangent for normal mapping
// ----------------------------------------------------------------------------
__global__ void computeTangentSpace(
    float4* __restrict__ tangentMap,
    const float4* __restrict__ normalMap,
    int heightmapSize
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= heightmapSize || y >= heightmapSize) return;

    float4 normalData = normalMap[y * heightmapSize + x];
    float3 N = make_float3(normalData.x, normalData.y, normalData.z);

    // Choose reference vector that's not parallel to normal
    float3 up = make_float3(0.0f, 1.0f, 0.0f);
    float3 ref = (fabsf(N.y) < 0.999f) ? up : make_float3(1.0f, 0.0f, 0.0f);

    // Compute tangent using Gram-Schmidt orthogonalization
    // T = normalize(ref - N * dot(ref, N))
    float dot_ref_N = ref.x * N.x + ref.y * N.y + ref.z * N.z;
    float3 T = make_float3(
        ref.x - N.x * dot_ref_N,
        ref.y - N.y * dot_ref_N,
        ref.z - N.z * dot_ref_N
    );

    float lenT = sqrtf(T.x * T.x + T.y * T.y + T.z * T.z);
    if (lenT > 0.0001f) {
        T.x /= lenT;
        T.y /= lenT;
        T.z /= lenT;
    }

    // Compute bitangent: B = N × T
    float3 B = make_float3(
        N.y * T.z - N.z * T.y,
        N.z * T.x - N.x * T.z,
        N.x * T.y - N.y * T.x
    );

    // Store tangent with bitangent sign in w
    // (bitangent can be reconstructed: B = cross(N, T) * sign)
    float sign = (B.x * (N.y * T.z - N.z * T.y) +
                  B.y * (N.z * T.x - N.x * T.z) +
                  B.z * (N.x * T.y - N.y * T.x)) >= 0.0f ? 1.0f : -1.0f;

    tangentMap[y * heightmapSize + x] = make_float4(T.x, T.y, T.z, sign);
}

// ----------------------------------------------------------------------------
// LOD MIPMAP GENERATION
// Downsamples heightmap to lower resolution levels
// ----------------------------------------------------------------------------
__global__ void generateHeightmapMipmap(
    float* __restrict__ dstMip,
    const float* __restrict__ srcMip,
    int dstSize,
    int srcSize
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstSize || y >= dstSize) return;

    // 2x2 box filter from source
    int srcX = x * 2;
    int srcY = y * 2;

    float h00 = srcMip[srcY * srcSize + srcX];
    float h10 = srcMip[srcY * srcSize + min(srcX + 1, srcSize - 1)];
    float h01 = srcMip[min(srcY + 1, srcSize - 1) * srcSize + srcX];
    float h11 = srcMip[min(srcY + 1, srcSize - 1) * srcSize + min(srcX + 1, srcSize - 1)];

    dstMip[y * dstSize + x] = (h00 + h10 + h01 + h11) * 0.25f;
}

// ----------------------------------------------------------------------------
// FRUSTUM CULLING
// Tests terrain patches against camera frustum planes
// ----------------------------------------------------------------------------
__device__ bool isPointInFrustum(float3 point, const float4* frustumPlanes) {
    for (int i = 0; i < 6; i++) {
        float4 plane = frustumPlanes[i];
        float dist = plane.x * point.x + plane.y * point.y + plane.z * point.z + plane.w;
        if (dist < 0.0f) return false;
    }
    return true;
}

__device__ bool isAABBInFrustum(float3 minBound, float3 maxBound, const float4* frustumPlanes) {
    // Test AABB against each frustum plane
    for (int i = 0; i < 6; i++) {
        float4 plane = frustumPlanes[i];

        // Find the positive vertex (furthest along plane normal)
        float3 pVertex = make_float3(
            plane.x >= 0.0f ? maxBound.x : minBound.x,
            plane.y >= 0.0f ? maxBound.y : minBound.y,
            plane.z >= 0.0f ? maxBound.z : minBound.z
        );

        // If positive vertex is outside, AABB is completely outside
        float dist = plane.x * pVertex.x + plane.y * pVertex.y + plane.z * pVertex.z + plane.w;
        if (dist < 0.0f) return false;
    }
    return true;
}

__global__ void performFrustumCulling(
    int* __restrict__ visibilityMask,
    int* __restrict__ visibleCount,
    const float* __restrict__ heightmap,
    const float4* __restrict__ frustumPlanes,
    int patchGridSize,
    float terrainSize,
    int heightmapSize,
    float heightScale
) {
    int patchX = blockIdx.x * blockDim.x + threadIdx.x;
    int patchY = blockIdx.y * blockDim.y + threadIdx.y;

    if (patchX >= patchGridSize || patchY >= patchGridSize) return;

    int patchIdx = patchY * patchGridSize + patchX;

    // Compute patch bounds in world space
    float patchSize = terrainSize / (float)patchGridSize;
    float x0 = patchX * patchSize - (terrainSize * 0.5f);
    float z0 = patchY * patchSize - (terrainSize * 0.5f);
    float x1 = x0 + patchSize;
    float z1 = z0 + patchSize;

    // Sample heightmap to get min/max height for this patch
    int hmX0 = (int)((x0 / terrainSize + 0.5f) * (heightmapSize - 1));
    int hmZ0 = (int)((z0 / terrainSize + 0.5f) * (heightmapSize - 1));
    int hmX1 = (int)((x1 / terrainSize + 0.5f) * (heightmapSize - 1));
    int hmZ1 = (int)((z1 / terrainSize + 0.5f) * (heightmapSize - 1));

    hmX0 = max(0, min(hmX0, heightmapSize - 1));
    hmZ0 = max(0, min(hmZ0, heightmapSize - 1));
    hmX1 = max(0, min(hmX1, heightmapSize - 1));
    hmZ1 = max(0, min(hmZ1, heightmapSize - 1));

    float minH = 1e10f, maxH = -1e10f;
    for (int hz = hmZ0; hz <= hmZ1; hz++) {
        for (int hx = hmX0; hx <= hmX1; hx++) {
            float h = heightmap[hz * heightmapSize + hx] * heightScale;
            minH = fminf(minH, h);
            maxH = fmaxf(maxH, h);
        }
    }

    // Safeguard: if no samples were found, use full height range
    if (minH > maxH) {
        minH = 0.0f;
        maxH = heightScale;
    }

    // Add large margin for safety - prevents incorrect culling at grazing angles near horizon
    // Use full height scale as margin to ensure no patches are incorrectly culled
    // This is more conservative but eliminates the black line artifact at horizon
    float heightMargin = heightScale * 1.0f;  // 100% of height range for maximum safety
    minH -= heightMargin;
    maxH += heightMargin;

    // Ensure minimum AABB height extent to avoid degenerate cases at flat areas
    // Use larger extent to be more conservative with culling
    float minExtent = heightScale * 0.5f;
    if (maxH - minH < minExtent) {
        float center = (maxH + minH) * 0.5f;
        minH = center - minExtent * 0.5f;
        maxH = center + minExtent * 0.5f;
    }

    // Create AABB for this patch
    float3 minBound = make_float3(x0, minH, z0);
    float3 maxBound = make_float3(x1, maxH, z1);

    // Perform frustum test
    int visible = isAABBInFrustum(minBound, maxBound, frustumPlanes) ? 1 : 0;
    visibilityMask[patchIdx] = visible;

    // Atomic increment of visible count
    if (visible) {
        atomicAdd(visibleCount, 1);
    }
}

// ----------------------------------------------------------------------------
// ADAPTIVE TESSELLATION FACTORS
// Computes tessellation level based on distance and terrain complexity
// ----------------------------------------------------------------------------
__global__ void computeTessellationFactors(
    float* __restrict__ tessFactors,
    const float* __restrict__ heightmap,
    const float4* __restrict__ normalMap,
    const int* __restrict__ visibilityMask,
    float3 cameraPos,
    int patchGridSize,
    float terrainSize,
    int heightmapSize,
    float heightScale,
    float maxTessLevel,
    float minTessLevel
) {
    int patchX = blockIdx.x * blockDim.x + threadIdx.x;
    int patchY = blockIdx.y * blockDim.y + threadIdx.y;

    if (patchX >= patchGridSize || patchY >= patchGridSize) return;

    int patchIdx = patchY * patchGridSize + patchX;

    // Skip culled patches
    if (visibilityMask[patchIdx] == 0) {
        tessFactors[patchIdx] = 0.0f;
        return;
    }

    // Compute patch center in world space
    float patchSize = terrainSize / (float)patchGridSize;
    float centerX = (patchX + 0.5f) * patchSize - (terrainSize * 0.5f);
    float centerZ = (patchY + 0.5f) * patchSize - (terrainSize * 0.5f);

    // Sample height at center
    int hmX = (int)((centerX / terrainSize + 0.5f) * (heightmapSize - 1));
    int hmZ = (int)((centerZ / terrainSize + 0.5f) * (heightmapSize - 1));
    hmX = max(0, min(hmX, heightmapSize - 1));
    hmZ = max(0, min(hmZ, heightmapSize - 1));

    float centerY = heightmap[hmZ * heightmapSize + hmX] * heightScale;

    // Distance from camera
    float dx = centerX - cameraPos.x;
    float dy = centerY - cameraPos.y;
    float dz = centerZ - cameraPos.z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);

    // Distance-based tessellation factor
    float distFactor = 1.0f - fminf(dist / 500.0f, 1.0f);  // Max detail within 500 units

    // Slope/complexity factor from normal map
    float4 normalData = normalMap[hmZ * heightmapSize + hmX];
    float slope = normalData.w;  // Stored slope magnitude
    float slopeFactor = fminf(slope * 2.0f, 1.0f);  // Scale slope influence

    // Combine factors
    float factor = distFactor * 0.7f + slopeFactor * 0.3f;

    // Map to tessellation level range
    float tessLevel = minTessLevel + factor * (maxTessLevel - minTessLevel);

    tessFactors[patchIdx] = tessLevel;
}

// ----------------------------------------------------------------------------
// COMPACT VISIBLE PATCHES (for GPU-driven rendering)
// Generates indirect draw commands for visible patches only
// ----------------------------------------------------------------------------
__global__ void compactVisiblePatches(
    IndirectDrawCommand* __restrict__ indirectCmd,
    const int* __restrict__ visibilityMask,
    int patchGridSize,
    int verticesPerPatch
) {
    // Single thread kernel - could be optimized with parallel prefix sum
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int visibleCount = 0;
    int totalPatches = patchGridSize * patchGridSize;

    for (int i = 0; i < totalPatches; i++) {
        if (visibilityMask[i] != 0) {
            visibleCount++;
        }
    }

    // Update indirect draw command
    indirectCmd->vertexCount = visibleCount * verticesPerPatch;
    indirectCmd->instanceCount = 1;
    indirectCmd->firstVertex = 0;
    indirectCmd->firstInstance = 0;
}

// ----------------------------------------------------------------------------
// EXTRACT FRUSTUM PLANES FROM VIEW-PROJECTION MATRIX
// CPU helper function to compute frustum planes
// ----------------------------------------------------------------------------
void extractFrustumPlanes(float4* planes, const glm::mat4& viewProj) {
    // Left plane
    planes[0] = make_float4(
        viewProj[0][3] + viewProj[0][0],
        viewProj[1][3] + viewProj[1][0],
        viewProj[2][3] + viewProj[2][0],
        viewProj[3][3] + viewProj[3][0]
    );

    // Right plane
    planes[1] = make_float4(
        viewProj[0][3] - viewProj[0][0],
        viewProj[1][3] - viewProj[1][0],
        viewProj[2][3] - viewProj[2][0],
        viewProj[3][3] - viewProj[3][0]
    );

    // Bottom plane
    planes[2] = make_float4(
        viewProj[0][3] + viewProj[0][1],
        viewProj[1][3] + viewProj[1][1],
        viewProj[2][3] + viewProj[2][1],
        viewProj[3][3] + viewProj[3][1]
    );

    // Top plane
    planes[3] = make_float4(
        viewProj[0][3] - viewProj[0][1],
        viewProj[1][3] - viewProj[1][1],
        viewProj[2][3] - viewProj[2][1],
        viewProj[3][3] - viewProj[3][1]
    );

    // Near plane
    planes[4] = make_float4(
        viewProj[0][3] + viewProj[0][2],
        viewProj[1][3] + viewProj[1][2],
        viewProj[2][3] + viewProj[2][2],
        viewProj[3][3] + viewProj[3][2]
    );

    // Far plane
    planes[5] = make_float4(
        viewProj[0][3] - viewProj[0][2],
        viewProj[1][3] - viewProj[1][2],
        viewProj[2][3] - viewProj[2][2],
        viewProj[3][3] - viewProj[3][2]
    );

    // Normalize all planes
    for (int i = 0; i < 6; i++) {
        float len = sqrtf(planes[i].x * planes[i].x +
                         planes[i].y * planes[i].y +
                         planes[i].z * planes[i].z);
        planes[i].x /= len;
        planes[i].y /= len;
        planes[i].z /= len;
        planes[i].w /= len;
    }
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{

	VkResult initialize(void);
	void uninitialize(void);
	VkResult display(void);
	void update(void);

	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szAppName[256];

	int SW = GetSystemMetrics(SM_CXSCREEN);
	int SH = GetSystemMetrics(SM_CYSCREEN);
	int xCoordinate = ((SW / 2) - (WIN_WIDTH / 2));
	int yCoordinate = ((SH / 2) - (WIN_HEIGHT / 2));

	BOOL bDone = FALSE;
	VkResult vkResult = VK_SUCCESS;

	FILE* file = fopen(LOG_FILE, "w");
	if (!file)
	{
		MessageBox(NULL, TEXT("Program cannot open log file!"), TEXT("Error"), MB_OK | MB_ICONERROR);
		exit(0);
	}
	fclose(file);
	FileIO("WinMain()-> Program started successfully\n");

	wsprintf(szAppName, TEXT("%s"), gpszAppName);

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hInstance = hInstance;
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON));
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.lpszClassName = szAppName;
	wndclass.lpszMenuName = NULL;
	wndclass.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON));

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
						szAppName,
						TEXT("05_PhysicalDevice"),
						WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
						xCoordinate,
						yCoordinate,
						WIN_WIDTH,
						WIN_HEIGHT,
						NULL,
						NULL,
						hInstance,
						NULL);

	ghwnd = hwnd;

	vkResult = initialize();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("WinMain(): initialize()  function failed\n");
		DestroyWindow(hwnd);
		hwnd = NULL;
	}
	else
	{
		FileIO("WinMain(): initialize() succedded\n");
	}

	ShowWindow(hwnd, iCmdShow);
	UpdateWindow(hwnd);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	// Initialize performance counter for delta time calculation
	QueryPerformanceFrequency(&gPerfFrequency);
	QueryPerformanceCounter(&gLastFrameTime);

	// Initialize camera vectors
	gCamera.updateVectors();

	FileIO("WinMain(): Free camera initialized. Controls:\n");
	FileIO("  WASD - Move forward/left/back/right\n");
	FileIO("  Space/Ctrl - Move up/down\n");
	FileIO("  Shift - Move faster\n");
	FileIO("  Right Mouse Button - Look around (hold)\n");
	FileIO("  Left Click - Toggle mouse capture for continuous look\n");
	FileIO("  Arrow Keys - Rotate camera\n");
	FileIO("  R - Reset camera position\n");
	FileIO("  F - Toggle fullscreen\n");
	FileIO("  ESC - Exit\n");

	while (bDone == FALSE)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
				bDone = TRUE;
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			if (gbActive == TRUE)
			{
				if(bWindowMinimize == FALSE)
				{
					vkResult = display();
					if ((vkResult != VK_FALSE) && (vkResult != VK_SUCCESS) && (vkResult != VK_ERROR_OUT_OF_DATE_KHR) && ((vkResult != VK_SUBOPTIMAL_KHR)))
					{
						FileIO("WinMain(): display() function failed\n");
						bDone = TRUE;
					}
					update();
				}
			}
		}
	}

	uninitialize();

	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{

	void ToggleFullscreen( void );
	VkResult resize(int, int);
	void uninitialize(void);

	VkResult vkResult;

	switch (iMsg)
	{
		case WM_CREATE:
			memset((void*)&wpPrev, 0 , sizeof(WINDOWPLACEMENT));
			wpPrev.length = sizeof(WINDOWPLACEMENT);
		break;

		case WM_SETFOCUS:
			gbActive = TRUE;
			break;

		case WM_KILLFOCUS:
			gbActive = FALSE;
			break;

		case WM_SIZE:
			if(wParam == SIZE_MINIMIZED)
			{
				bWindowMinimize = TRUE;
			}
			else
			{
				bWindowMinimize = FALSE;
				vkResult = resize(LOWORD(lParam), HIWORD(lParam));
				if (vkResult != VK_SUCCESS)
				{
					FileIO("WndProc(): resize() function failed with error code %d\n", vkResult);
					return vkResult;
				}
				else
				{
					FileIO("WndProc(): resize() succedded\n");
				}
			}
			break;

		case WM_KEYDOWN:
			switch (LOWORD(wParam))
			{
			case VK_ESCAPE:
				FileIO("WndProc() VK_ESCAPE-> Program ended successfully.\n");
				DestroyWindow(hwnd);
				break;
			// Camera movement keys
			case 'W':
				gInputState.keyW = TRUE;
				break;
			case 'A':
				gInputState.keyA = TRUE;
				break;
			case 'S':
				gInputState.keyS = TRUE;
				break;
			case 'D':
				gInputState.keyD = TRUE;
				break;
			case VK_SPACE:
				gInputState.keySpace = TRUE;
				break;
			case VK_CONTROL:
				gInputState.keyCtrl = TRUE;
				break;
			case VK_SHIFT:
				gInputState.keyShift = TRUE;
				break;
			// Arrow keys for camera rotation
			case VK_UP:
				gCamera.pitch += 0.05f;
				if (gCamera.pitch > 1.5f) gCamera.pitch = 1.5f;
				break;
			case VK_DOWN:
				gCamera.pitch -= 0.05f;
				if (gCamera.pitch < -1.5f) gCamera.pitch = -1.5f;
				break;
			case VK_LEFT:
				gCamera.yaw -= 0.05f;
				break;
			case VK_RIGHT:
				gCamera.yaw += 0.05f;
				break;
			}
			break;

		case WM_KEYUP:
			switch (LOWORD(wParam))
			{
			case 'W':
				gInputState.keyW = FALSE;
				break;
			case 'A':
				gInputState.keyA = FALSE;
				break;
			case 'S':
				gInputState.keyS = FALSE;
				break;
			case 'D':
				gInputState.keyD = FALSE;
				break;
			case VK_SPACE:
				gInputState.keySpace = FALSE;
				break;
			case VK_CONTROL:
				gInputState.keyCtrl = FALSE;
				break;
			case VK_SHIFT:
				gInputState.keyShift = FALSE;
				break;
			}
			break;

		case WM_MOUSEMOVE:
			{
				int newMouseX = LOWORD(lParam);
				int newMouseY = HIWORD(lParam);

				if (gInputState.mouseRightButton) {
					// Calculate mouse delta
					gInputState.mouseDeltaX = newMouseX - gInputState.lastMouseX;
					gInputState.mouseDeltaY = newMouseY - gInputState.lastMouseY;

					// Update camera rotation based on mouse movement
					gCamera.yaw += gInputState.mouseDeltaX * gCamera.mouseSensitivity;
					gCamera.pitch -= gInputState.mouseDeltaY * gCamera.mouseSensitivity;

					// Clamp pitch to avoid gimbal lock
					if (gCamera.pitch > 1.5f) gCamera.pitch = 1.5f;
					if (gCamera.pitch < -1.5f) gCamera.pitch = -1.5f;
				}

				gInputState.lastMouseX = newMouseX;
				gInputState.lastMouseY = newMouseY;
				gInputState.mouseX = newMouseX;
				gInputState.mouseY = newMouseY;
			}
			break;

		case WM_RBUTTONDOWN:
			gInputState.mouseRightButton = TRUE;
			gInputState.lastMouseX = LOWORD(lParam);
			gInputState.lastMouseY = HIWORD(lParam);
			SetCapture(hwnd);
			break;

		case WM_RBUTTONUP:
			gInputState.mouseRightButton = FALSE;
			ReleaseCapture();
			break;

		case WM_LBUTTONDOWN:
			// Left click to toggle mouse capture for continuous camera control
			if (!gInputState.mouseCaptured) {
				gInputState.mouseCaptured = TRUE;
				SetCapture(hwnd);
				ShowCursor(FALSE);
				// Center cursor
				RECT rect;
				GetClientRect(hwnd, &rect);
				POINT center = { (rect.right - rect.left) / 2, (rect.bottom - rect.top) / 2 };
				ClientToScreen(hwnd, &center);
				SetCursorPos(center.x, center.y);
				gInputState.lastMouseX = (rect.right - rect.left) / 2;
				gInputState.lastMouseY = (rect.bottom - rect.top) / 2;
			} else {
				gInputState.mouseCaptured = FALSE;
				ReleaseCapture();
				ShowCursor(TRUE);
			}
			break;

		case WM_CHAR:
			switch (LOWORD(wParam))
			{
			case 'F':
			case 'f':
				if (gbFullscreen == FALSE)
				{
					ToggleFullscreen();
					gbFullscreen = TRUE;
					FileIO("WndProc() WM_CHAR(F key)-> Program entered Fullscreen.\n");
				}
				else
				{
					ToggleFullscreen();
					gbFullscreen = FALSE;
					FileIO("WndProc() WM_CHAR(F key)-> Program ended Fullscreen.\n");
				}
				break;
			// Reset camera to starting position
			case 'R':
			case 'r':
				gCamera.position = glm::vec3(0.0f, 30.0f, 150.0f);
				gCamera.yaw = glm::radians(180.0f);
				gCamera.pitch = 0.0f;
				break;

			}
			break;

		case WM_CLOSE:
			uninitialize();
			DestroyWindow(hwnd);
			break;

		case WM_DESTROY:
			PostQuitMessage(0);
			break;

		default:
			break;
	}

	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullscreen(void)
{

	MONITORINFO mi = { sizeof(MONITORINFO) };

	if (gbFullscreen == FALSE)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);

		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);

				SetWindowPos(ghwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);

			}
		}

		ShowCursor(FALSE);
	}
	else {
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
	}
}

VkResult initialize(void)
{

	VkResult CreateVulkanInstance(void);
	VkResult GetSupportedSurface(void);
	VkResult GetPhysicalDevice(void);
	VkResult PrintVulkanInfo(void);
	VkResult CreateVulKanDevice(void);
	void GetDeviceQueque(void);
	VkResult CreateSwapChain(VkBool32);
	VkResult CreateImagesAndImageViews(void);
	VkResult CreateCommandPool(void);
	VkResult CreateCommandBuffers(VkCommandBuffer**);
	VkResult CreateIndirectBuffer(void);

	VkResult CreateUniformBuffer(void);

	VkResult CreateShaders(void);

	VkResult CreateDescriptorSetLayout(void);

	VkResult CreatePipelineLayout(void);

	VkResult CreateDescriptorPool(void);
	VkResult CreateDescriptorSet(void);

	VkResult CreateRenderPass(void);

	VkResult CreatePipeline(void);

	VkResult CreateFramebuffers(void);
	VkResult CreateSemaphores(void);
	VkResult CreateFences(void);
	VkResult buildCommandBuffers(void);

	cudaError_t initialize_cuda(void);
	VkResult CreateExternalVertexBuffer(unsigned int, unsigned int, VertexData*);

	VkResult vkResult = VK_SUCCESS;

	vkResult = CreateVulkanInstance();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateVulkanInstance() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateVulkanInstance() succedded\n");
	}

	vkResult = GetSupportedSurface();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): GetSupportedSurface() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): GetSupportedSurface() succedded\n");
	}

	vkResult = GetPhysicalDevice();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): GetPhysicalDevice() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): GetPhysicalDevice() succedded\n");
	}

	vkResult = PrintVulkanInfo();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): PrintVulkanInfo() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): PrintVulkanInfo() succedded\n");
	}

	vkResult = CreateVulKanDevice();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateVulKanDevice() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateVulKanDevice() succedded\n");
	}

	GetDeviceQueque();

	cudaResult = initialize_cuda();
	if (cudaResult != cudaSuccess)
	{
	  vkResult = VK_ERROR_INITIALIZATION_FAILED;
	  FileIO("initialize(): initialize_cuda() function failed with error code %d\n", vkResult);
	  return vkResult;
    }
	else
	{
	   FileIO("initialize(): initialize_cuda() succedded\n");
	}

	// Initialize cuRAND-based noise generation system
	cudaResult = initialize_curand_noise_system();
	if (cudaResult != cudaSuccess)
	{
	  vkResult = VK_ERROR_INITIALIZATION_FAILED;
	  FileIO("initialize(): initialize_curand_noise_system() failed with error code %d\n", cudaResult);
	  return vkResult;
	}
	else
	{
	   FileIO("initialize(): initialize_curand_noise_system() succeeded\n");
	}

	vkResult = CreateSwapChain(VK_FALSE);
	if (vkResult != VK_SUCCESS)
	{

		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		FileIO("initialize(): CreateSwapChain() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateSwapChain() succedded\n");
	}

	vkResult =  CreateImagesAndImageViews();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateImagesAndImageViews() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateImagesAndImageViews() succedded with SwapChain Image count as %d\n", swapchainImageCount);
	}

	vkResult = CreateCommandPool();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateCommandPool() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateCommandPool() succedded\n");
	}

	vkResult  = CreateCommandBuffers(&vkCommandBuffer_for_1024_x_1024_graphics_array);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateCommandBuffers() function failed with error code %d for 1024 x 1024 command buffer\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateCommandBuffers() succedded  for 1024 x 1024 command buffer\n");
	}

	memset((void*)&vertexData_external, 0, sizeof(VertexData));

	unsigned int totalVertices = PATCH_GRID_SIZE * PATCH_GRID_SIZE * 4;
	vkResult = CreateExternalVertexBuffer(totalVertices, 1, &vertexData_external);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateExternalVertexBuffer() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateExternalVertexBuffer() succedded\n");
	}

	vkResult = CreateIndirectBuffer();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateIndirectBuffer() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateIndirectBuffer() succedded\n");
	}

	vkResult  = CreateUniformBuffer();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateUniformBuffer() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateUniformBuffer() succedded\n");
	}

	// Create storage buffers for cuRAND-generated noise tables
	vkResult = CreateNoiseStorageBuffers();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateNoiseStorageBuffers() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateNoiseStorageBuffers() succedded\n");
	}

	vkResult = CreateShaders();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateShaders() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateShaders() succedded\n");
	}

	vkResult = CreateDescriptorSetLayout();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateDescriptorSetLayout() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateDescriptorSetLayout() succedded\n");
	}

	vkResult = CreatePipelineLayout();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreatePipelineLayout() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreatePipelineLayout() succedded\n");
	}

	vkResult = CreateDescriptorPool();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateDescriptorPool() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateDescriptorPool() succedded\n");
	}

	vkResult = CreateDescriptorSet();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateDescriptorSet() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateDescriptorSet() succedded\n");
	}

	vkResult =  CreateRenderPass();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateRenderPass() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateRenderPass() succedded\n");
	}

	vkResult = CreatePipeline();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreatePipeline() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreatePipeline() succedded\n");
	}

	vkResult = CreateFramebuffers();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateFramebuffers() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateFramebuffers() succedded\n");
	}

	vkResult = CreateSemaphores();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateSemaphores() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateSemaphores() succedded\n");
	}

	vkResult = CreateFences();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateFences() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateFences() succedded\n");
	}

	memset((void*)&vkClearColorValue, 0, sizeof(VkClearColorValue));

	vkClearColorValue.float32[0] = 0.0f;
	vkClearColorValue.float32[1] = 0.0f;
	vkClearColorValue.float32[2] = 0.0f;
	vkClearColorValue.float32[3] = 1.0f;

	memset((void*)&vkClearDepthStencilValue, 0, sizeof(VkClearDepthStencilValue));

	vkClearDepthStencilValue.depth = 1.0f;

	vkClearDepthStencilValue.stencil = 0;

	vkResult = buildCommandBuffers();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): buildCommandBuffers() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): buildCommandBuffers() succedded\n");
	}

	bInitialized = TRUE;

	FileIO("initialize(): initialize() completed sucessfully");

	return vkResult;
}

cudaError_t initialize_cuda(void)
{

	int devCount = 0;
    cudaResult = cudaGetDeviceCount(&devCount);
    if (cudaResult != cudaSuccess)
	{
		FileIO("initialize_cuda(): cudaGetDeviceCount failed..\n");
		return cudaResult;
	}
	else if (devCount == 0)
	{
		FileIO("initialize_cuda(): No CUDA device detected..\n");
		return cudaResult;
	}

	VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties;
	memset((void*)&vkPhysicalDeviceIDProperties, 0, sizeof(VkPhysicalDeviceIDProperties));

	vkPhysicalDeviceIDProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
	vkPhysicalDeviceIDProperties.pNext = NULL;

	VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2;
	memset((void*)&vkPhysicalDeviceProperties2, 0, sizeof(VkPhysicalDeviceProperties2));
	vkPhysicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;

	vkGetPhysicalDeviceProperties2(vkPhysicalDevice_selected, &vkPhysicalDeviceProperties2);
	uint8_t vulkanDeviceUUID[VK_UUID_SIZE];
	memcpy((void*)&vulkanDeviceUUID, vkPhysicalDeviceIDProperties.deviceUUID, VK_UUID_SIZE);

	int iVulkanCUDAInterOpDeviceFound = -1;
	for(int i=0; i < devCount; i++)
	{

		int compute_Mode;

		cudaResult = cudaDeviceGetAttribute(&compute_Mode, cudaDevAttrComputeMode, i);
		if(cudaResult != CUDA_SUCCESS)
		{
			continue;
		}

		if(compute_Mode == cudaComputeModeProhibited)
		{
			continue;
		}

		cudaDeviceProp devProp;
		memset((void*)&devProp, 0, sizeof(cudaDeviceProp));
		cudaResult = cudaGetDeviceProperties(&devProp, i);
		if(cudaResult != CUDA_SUCCESS)
		{
			continue;
		}

		int iResult = memcmp((void*)&devProp.uuid, (void*)&vulkanDeviceUUID, VK_UUID_SIZE);
		if(iResult != 0)
		{
			continue;
		}

		cudaResult = cudaSetDevice(i);
		if(cudaResult != CUDA_SUCCESS)
		{
			continue;
		}

		FileIO("initialize_cuda(): selected device is %s\n", devProp.name);
		iVulkanCUDAInterOpDeviceFound = 1;
		break;
	}

	if(iVulkanCUDAInterOpDeviceFound == -1)
	{
		FileIO("initialize_cuda(): no device found\n");
		return cudaErrorUnknown;
	}

	vkExternalMemoryHandleTypeFlagBits = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

	return cudaSuccess;
}

// ============================================================================
// cuRAND NOISE TEXTURE INITIALIZATION
// Creates Vulkan 3D textures, imports them to CUDA, and generates noise
// ============================================================================

cudaError_t initialize_curand_noise_system(void)
{
    FileIO("initialize_curand_noise_system(): Starting cuRAND noise and CUDA terrain system...\n");

    cudaError_t cudaErr = cudaSuccess;
    int numStates = NOISE_TEXTURE_SIZE * NOISE_TEXTURE_SIZE;

    // Step 1: Create CUDA streams for parallel/async operations
    FileIO("initialize_curand_noise_system(): Creating %d CUDA streams...\n", NUM_CUDA_STREAMS);
    for (int i = 0; i < NUM_CUDA_STREAMS; i++) {
        cudaErr = cudaStreamCreate(&cudaStreams[i]);
        if (cudaErr != cudaSuccess) {
            FileIO("initialize_curand_noise_system(): Failed to create CUDA stream %d: %s\n",
                   i, cudaGetErrorString(cudaErr));
            return cudaErr;
        }
    }
    FileIO("initialize_curand_noise_system(): Created %d CUDA streams for async terrain generation\n",
           NUM_CUDA_STREAMS);

    // Step 2: Allocate cuRAND states
    cudaErr = cudaMalloc((void**)&d_curandStates, numStates * sizeof(curandState));
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): Failed to allocate cuRAND states: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }
    FileIO("initialize_curand_noise_system(): Allocated %d cuRAND states\n", numStates);

    // Step 3: Initialize cuRAND states with high-quality seed
    unsigned long long seed = 42ULL;  // Reproducible seed for consistent terrain
    int blockSize = 256;
    int numBlocks = (numStates + blockSize - 1) / blockSize;

    initCurandStates<<<numBlocks, blockSize>>>(d_curandStates, seed, numStates);
    cudaErr = cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): cuRAND state initialization failed: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }
    FileIO("initialize_curand_noise_system(): cuRAND states initialized with seed %llu\n", seed);

    // Step 4: Allocate and generate gradient table
    cudaErr = cudaMalloc((void**)&d_gradientTable, GRADIENT_TABLE_SIZE * sizeof(float4));
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): Failed to allocate gradient table: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }

    numBlocks = (GRADIENT_TABLE_SIZE + blockSize - 1) / blockSize;
    generateGradientTable<<<numBlocks, blockSize>>>(d_gradientTable, d_curandStates, GRADIENT_TABLE_SIZE);
    cudaErr = cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): Gradient table generation failed: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }
    FileIO("initialize_curand_noise_system(): Generated %d gradient vectors using cuRAND\n",
           GRADIENT_TABLE_SIZE);

    // Step 5: Allocate and generate permutation table
    cudaErr = cudaMalloc((void**)&d_permutationTable, PERMUTATION_TABLE_SIZE * sizeof(int));
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): Failed to allocate permutation table: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }

    generatePermutationTable<<<1, 1>>>(d_permutationTable, d_curandStates, PERMUTATION_TABLE_SIZE);
    cudaErr = cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): Permutation table generation failed: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }
    FileIO("initialize_curand_noise_system(): Generated Fisher-Yates shuffled permutation table\n");

    // Step 6: Allocate heightmap buffer for complete terrain generation
    size_t heightmapBytes = HEIGHTMAP_SIZE * HEIGHTMAP_SIZE * sizeof(float);
    cudaErr = cudaMalloc((void**)&heightmapData.d_heightmap, heightmapBytes);
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): Failed to allocate heightmap buffer: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }
    FileIO("initialize_curand_noise_system(): Allocated %dx%d heightmap buffer (%.2f MB)\n",
           HEIGHTMAP_SIZE, HEIGHTMAP_SIZE, (float)heightmapBytes / (1024.0f * 1024.0f));

    // Step 7: Generate initial heightmap using all 4 CUDA streams in parallel
    FileIO("initialize_curand_noise_system(): Generating initial terrain heightmap...\n");
    float3 initialCameraPos = make_float3(0.0f, 50.0f, 0.0f);
    int quadrantSize = HEIGHTMAP_SIZE / 2;
    dim3 blockDim(16, 16);
    dim3 gridDim((quadrantSize + blockDim.x - 1) / blockDim.x,
                 (quadrantSize + blockDim.y - 1) / blockDim.y);

    // Launch heightmap generation for each quadrant on separate streams
    for (int qy = 0; qy < 2; qy++) {
        for (int qx = 0; qx < 2; qx++) {
            int streamIdx = qy * 2 + qx;
            generateTerrainHeightmapQuadrant<<<gridDim, blockDim, 0, cudaStreams[streamIdx]>>>(
                heightmapData.d_heightmap,
                d_gradientTable,
                d_permutationTable,
                HEIGHTMAP_SIZE,
                TERRAIN_SIZE,
                initialCameraPos,
                qx, qy, quadrantSize
            );
        }
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_CUDA_STREAMS; i++) {
        cudaStreamSynchronize(cudaStreams[i]);
    }
    cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): Initial heightmap generation failed: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }
    FileIO("initialize_curand_noise_system(): Initial heightmap generated using %d parallel streams\n",
           NUM_CUDA_STREAMS);

    // ============================================================================
    // ADVANCED CUDA TERRAIN PROCESSING BUFFERS
    // ============================================================================

    // Step 8: Allocate normal map buffer
    cudaErr = cudaMalloc((void**)&d_normalMap, HEIGHTMAP_SIZE * HEIGHTMAP_SIZE * sizeof(float4));
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): Failed to allocate normal map: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }
    FileIO("initialize_curand_noise_system(): Allocated normal map buffer (%.2f MB)\n",
           (float)(HEIGHTMAP_SIZE * HEIGHTMAP_SIZE * sizeof(float4)) / (1024.0f * 1024.0f));

    // Step 9: Allocate tangent space buffer
    cudaErr = cudaMalloc((void**)&d_tangentMap, HEIGHTMAP_SIZE * HEIGHTMAP_SIZE * sizeof(float4));
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): Failed to allocate tangent map: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }
    FileIO("initialize_curand_noise_system(): Allocated tangent space buffer\n");

    // Step 10: Allocate LOD mipmap buffers
    d_heightmapLOD[0] = heightmapData.d_heightmap;  // Level 0 is the main heightmap
    for (int i = 1; i < NUM_LOD_LEVELS; i++) {
        int mipSize = heightmapLODSizes[i];
        cudaErr = cudaMalloc((void**)&d_heightmapLOD[i], mipSize * mipSize * sizeof(float));
        if (cudaErr != cudaSuccess) {
            FileIO("initialize_curand_noise_system(): Failed to allocate LOD level %d: %s\n",
                   i, cudaGetErrorString(cudaErr));
            return cudaErr;
        }
    }
    FileIO("initialize_curand_noise_system(): Allocated %d LOD mipmap levels\n", NUM_LOD_LEVELS);

    // Step 11: Allocate tessellation factors buffer (one per patch)
    int numPatches = PATCH_GRID_SIZE * PATCH_GRID_SIZE;
    cudaErr = cudaMalloc((void**)&d_tessellationFactors, numPatches * sizeof(float));
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): Failed to allocate tessellation factors: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }
    FileIO("initialize_curand_noise_system(): Allocated tessellation factors for %d patches\n", numPatches);

    // Step 12: Allocate visibility mask for frustum culling
    cudaErr = cudaMalloc((void**)&d_visibilityMask, numPatches * sizeof(int));
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): Failed to allocate visibility mask: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }
    // Initialize all patches as visible
    cudaMemset(d_visibilityMask, 1, numPatches * sizeof(int));

    // Step 13: Allocate visible patch counter
    cudaErr = cudaMalloc((void**)&d_visiblePatchCount, sizeof(int));
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): Failed to allocate visible patch counter: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }

    // Step 14: Allocate frustum planes buffer
    cudaErr = cudaMalloc((void**)&d_frustumPlanes, 6 * sizeof(float4));
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): Failed to allocate frustum planes: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }

    // Step 15: Allocate indirect draw command buffer
    cudaErr = cudaMalloc((void**)&d_indirectDrawCmd, sizeof(IndirectDrawCommand));
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): Failed to allocate indirect draw command: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }
    FileIO("initialize_curand_noise_system(): Allocated frustum culling and indirect draw buffers\n");

    // Step 16: Compute initial normal map from heightmap
    dim3 normalBlockDim(16, 16);
    dim3 normalGridDim((HEIGHTMAP_SIZE + normalBlockDim.x - 1) / normalBlockDim.x,
                       (HEIGHTMAP_SIZE + normalBlockDim.y - 1) / normalBlockDim.y);

    computeNormalMap<<<normalGridDim, normalBlockDim>>>(
        d_normalMap,
        heightmapData.d_heightmap,
        HEIGHTMAP_SIZE,
        TERRAIN_SIZE,
        HEIGHT_SCALE
    );
    cudaErr = cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): Normal map computation failed: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }
    FileIO("initialize_curand_noise_system(): Computed initial normal map\n");

    // Step 17: Compute initial tangent space from normal map
    computeTangentSpace<<<normalGridDim, normalBlockDim>>>(
        d_tangentMap,
        d_normalMap,
        HEIGHTMAP_SIZE
    );
    cudaErr = cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): Tangent space computation failed: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }
    FileIO("initialize_curand_noise_system(): Computed initial tangent space\n");

    // Step 18: Generate LOD mipmaps
    for (int i = 1; i < NUM_LOD_LEVELS; i++) {
        int srcSize = heightmapLODSizes[i - 1];
        int dstSize = heightmapLODSizes[i];
        dim3 mipBlockDim(16, 16);
        dim3 mipGridDim((dstSize + mipBlockDim.x - 1) / mipBlockDim.x,
                        (dstSize + mipBlockDim.y - 1) / mipBlockDim.y);

        generateHeightmapMipmap<<<mipGridDim, mipBlockDim>>>(
            d_heightmapLOD[i],
            d_heightmapLOD[i - 1],
            dstSize,
            srcSize
        );
    }
    cudaErr = cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess) {
        FileIO("initialize_curand_noise_system(): LOD mipmap generation failed: %s\n",
               cudaGetErrorString(cudaErr));
        return cudaErr;
    }
    FileIO("initialize_curand_noise_system(): Generated %d LOD mipmap levels\n", NUM_LOD_LEVELS);

    bNoiseTexturesInitialized = TRUE;
    bCudaTerrainSystemInitialized = TRUE;
    FileIO("initialize_curand_noise_system(): CUDA terrain system with advanced processing initialized successfully\n");

    return cudaSuccess;
}

void cleanup_curand_noise_system(void)
{
    FileIO("cleanup_curand_noise_system(): Cleaning up CUDA terrain system resources...\n");

    // Clean up advanced processing buffers
    if (d_normalMap) {
        cudaFree(d_normalMap);
        d_normalMap = NULL;
        FileIO("cleanup_curand_noise_system(): Freed normal map\n");
    }

    if (d_tangentMap) {
        cudaFree(d_tangentMap);
        d_tangentMap = NULL;
        FileIO("cleanup_curand_noise_system(): Freed tangent map\n");
    }

    // Free LOD mipmap buffers (skip level 0, which is the main heightmap)
    for (int i = 1; i < NUM_LOD_LEVELS; i++) {
        if (d_heightmapLOD[i]) {
            cudaFree(d_heightmapLOD[i]);
            d_heightmapLOD[i] = NULL;
        }
    }
    FileIO("cleanup_curand_noise_system(): Freed LOD mipmap buffers\n");

    if (d_tessellationFactors) {
        cudaFree(d_tessellationFactors);
        d_tessellationFactors = NULL;
        FileIO("cleanup_curand_noise_system(): Freed tessellation factors\n");
    }

    if (d_visibilityMask) {
        cudaFree(d_visibilityMask);
        d_visibilityMask = NULL;
    }

    if (d_visiblePatchCount) {
        cudaFree(d_visiblePatchCount);
        d_visiblePatchCount = NULL;
    }

    if (d_frustumPlanes) {
        cudaFree(d_frustumPlanes);
        d_frustumPlanes = NULL;
    }

    if (d_indirectDrawCmd) {
        cudaFree(d_indirectDrawCmd);
        d_indirectDrawCmd = NULL;
    }
    FileIO("cleanup_curand_noise_system(): Freed frustum culling buffers\n");

    // Clean up heightmap buffer
    if (heightmapData.d_heightmap) {
        cudaFree(heightmapData.d_heightmap);
        heightmapData.d_heightmap = NULL;
        d_heightmapLOD[0] = NULL;  // Clear LOD level 0 reference
        FileIO("cleanup_curand_noise_system(): Freed heightmap buffer\n");
    }

    // Destroy CUDA streams
    for (int i = 0; i < NUM_CUDA_STREAMS; i++) {
        if (cudaStreams[i]) {
            cudaStreamDestroy(cudaStreams[i]);
            cudaStreams[i] = NULL;
        }
    }
    FileIO("cleanup_curand_noise_system(): Destroyed %d CUDA streams\n", NUM_CUDA_STREAMS);

    if (d_curandStates) {
        cudaFree(d_curandStates);
        d_curandStates = NULL;
        FileIO("cleanup_curand_noise_system(): Freed cuRAND states\n");
    }

    if (d_gradientTable) {
        cudaFree(d_gradientTable);
        d_gradientTable = NULL;
        FileIO("cleanup_curand_noise_system(): Freed gradient table\n");
    }

    if (d_permutationTable) {
        cudaFree(d_permutationTable);
        d_permutationTable = NULL;
        FileIO("cleanup_curand_noise_system(): Freed permutation table\n");
    }

    bNoiseTexturesInitialized = FALSE;
    bCudaTerrainSystemInitialized = FALSE;
    FileIO("cleanup_curand_noise_system(): CUDA terrain system cleanup complete\n");
}

// Create Vulkan storage buffers and copy cuRAND-generated data
VkResult CreateNoiseStorageBuffers(void)
{
    VkResult vkResult = VK_SUCCESS;
    cudaError_t cudaErr = cudaSuccess;

    FileIO("CreateNoiseStorageBuffers(): Creating storage buffers for cuRAND noise tables...\n");

    // Calculate buffer sizes
    VkDeviceSize gradientBufferSize = GRADIENT_TABLE_SIZE * sizeof(float) * 4;  // vec4 per gradient
    VkDeviceSize permutationBufferSize = PERMUTATION_TABLE_SIZE * sizeof(int);

    // ================== GRADIENT BUFFER ==================
    // Create Vulkan buffer
    VkBufferCreateInfo gradientBufferInfo;
    memset(&gradientBufferInfo, 0, sizeof(VkBufferCreateInfo));
    gradientBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    gradientBufferInfo.size = gradientBufferSize;
    gradientBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    gradientBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkResult = vkCreateBuffer(vkDevice, &gradientBufferInfo, NULL, &noiseStorageBuffers.gradientBuffer);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to create gradient buffer: %d\n", vkResult);
        return vkResult;
    }

    // Get memory requirements and allocate
    VkMemoryRequirements gradientMemReqs;
    vkGetBufferMemoryRequirements(vkDevice, noiseStorageBuffers.gradientBuffer, &gradientMemReqs);

    VkMemoryAllocateInfo gradientAllocInfo;
    memset(&gradientAllocInfo, 0, sizeof(VkMemoryAllocateInfo));
    gradientAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    gradientAllocInfo.allocationSize = gradientMemReqs.size;

    // Find host-visible memory type for data upload
    BOOL bFoundMemoryType = FALSE;
    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++) {
        if ((gradientMemReqs.memoryTypeBits & (1 << i)) &&
            (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            gradientAllocInfo.memoryTypeIndex = i;
            bFoundMemoryType = TRUE;
            break;
        }
    }

    if (!bFoundMemoryType) {
        FileIO("CreateNoiseStorageBuffers(): Failed to find suitable memory type for gradient buffer\n");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    vkResult = vkAllocateMemory(vkDevice, &gradientAllocInfo, NULL, &noiseStorageBuffers.gradientMemory);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to allocate gradient buffer memory: %d\n", vkResult);
        return vkResult;
    }

    vkResult = vkBindBufferMemory(vkDevice, noiseStorageBuffers.gradientBuffer, noiseStorageBuffers.gradientMemory, 0);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to bind gradient buffer memory: %d\n", vkResult);
        return vkResult;
    }

    // Copy cuRAND gradient data from CUDA to Vulkan buffer
    void* gradientData = NULL;
    vkResult = vkMapMemory(vkDevice, noiseStorageBuffers.gradientMemory, 0, gradientBufferSize, 0, &gradientData);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to map gradient buffer memory: %d\n", vkResult);
        return vkResult;
    }

    cudaErr = cudaMemcpy(gradientData, d_gradientTable, gradientBufferSize, cudaMemcpyDeviceToHost);
    vkUnmapMemory(vkDevice, noiseStorageBuffers.gradientMemory);

    if (cudaErr != cudaSuccess) {
        FileIO("CreateNoiseStorageBuffers(): Failed to copy gradient data from CUDA: %s\n", cudaGetErrorString(cudaErr));
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    FileIO("CreateNoiseStorageBuffers(): Gradient buffer created and populated (%u bytes)\n", (unsigned int)gradientBufferSize);

    // ================== PERMUTATION BUFFER ==================
    VkBufferCreateInfo permutationBufferInfo;
    memset(&permutationBufferInfo, 0, sizeof(VkBufferCreateInfo));
    permutationBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    permutationBufferInfo.size = permutationBufferSize;
    permutationBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    permutationBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkResult = vkCreateBuffer(vkDevice, &permutationBufferInfo, NULL, &noiseStorageBuffers.permutationBuffer);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to create permutation buffer: %d\n", vkResult);
        return vkResult;
    }

    VkMemoryRequirements permutationMemReqs;
    vkGetBufferMemoryRequirements(vkDevice, noiseStorageBuffers.permutationBuffer, &permutationMemReqs);

    VkMemoryAllocateInfo permutationAllocInfo;
    memset(&permutationAllocInfo, 0, sizeof(VkMemoryAllocateInfo));
    permutationAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    permutationAllocInfo.allocationSize = permutationMemReqs.size;

    bFoundMemoryType = FALSE;
    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++) {
        if ((permutationMemReqs.memoryTypeBits & (1 << i)) &&
            (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            permutationAllocInfo.memoryTypeIndex = i;
            bFoundMemoryType = TRUE;
            break;
        }
    }

    if (!bFoundMemoryType) {
        FileIO("CreateNoiseStorageBuffers(): Failed to find suitable memory type for permutation buffer\n");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    vkResult = vkAllocateMemory(vkDevice, &permutationAllocInfo, NULL, &noiseStorageBuffers.permutationMemory);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to allocate permutation buffer memory: %d\n", vkResult);
        return vkResult;
    }

    vkResult = vkBindBufferMemory(vkDevice, noiseStorageBuffers.permutationBuffer, noiseStorageBuffers.permutationMemory, 0);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to bind permutation buffer memory: %d\n", vkResult);
        return vkResult;
    }

    // Copy cuRAND permutation data from CUDA to Vulkan buffer
    void* permutationData = NULL;
    vkResult = vkMapMemory(vkDevice, noiseStorageBuffers.permutationMemory, 0, permutationBufferSize, 0, &permutationData);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to map permutation buffer memory: %d\n", vkResult);
        return vkResult;
    }

    cudaErr = cudaMemcpy(permutationData, d_permutationTable, permutationBufferSize, cudaMemcpyDeviceToHost);
    vkUnmapMemory(vkDevice, noiseStorageBuffers.permutationMemory);

    if (cudaErr != cudaSuccess) {
        FileIO("CreateNoiseStorageBuffers(): Failed to copy permutation data from CUDA: %s\n", cudaGetErrorString(cudaErr));
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    FileIO("CreateNoiseStorageBuffers(): Permutation buffer created and populated (%u bytes)\n", (unsigned int)permutationBufferSize);

    // ================== HEIGHTMAP BUFFER ==================
    // Create storage buffer for CUDA-generated heightmap data
    VkDeviceSize heightmapBufferSize = HEIGHTMAP_SIZE * HEIGHTMAP_SIZE * sizeof(float);

    VkBufferCreateInfo heightmapBufferInfo;
    memset(&heightmapBufferInfo, 0, sizeof(VkBufferCreateInfo));
    heightmapBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    heightmapBufferInfo.size = heightmapBufferSize;
    heightmapBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    heightmapBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkResult = vkCreateBuffer(vkDevice, &heightmapBufferInfo, NULL, &noiseStorageBuffers.heightmapBuffer);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to create heightmap buffer: %d\n", vkResult);
        return vkResult;
    }

    VkMemoryRequirements heightmapMemReqs;
    vkGetBufferMemoryRequirements(vkDevice, noiseStorageBuffers.heightmapBuffer, &heightmapMemReqs);

    VkMemoryAllocateInfo heightmapAllocInfo;
    memset(&heightmapAllocInfo, 0, sizeof(VkMemoryAllocateInfo));
    heightmapAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    heightmapAllocInfo.allocationSize = heightmapMemReqs.size;

    bFoundMemoryType = FALSE;
    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++) {
        if ((heightmapMemReqs.memoryTypeBits & (1 << i)) &&
            (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            heightmapAllocInfo.memoryTypeIndex = i;
            bFoundMemoryType = TRUE;
            break;
        }
    }

    if (!bFoundMemoryType) {
        FileIO("CreateNoiseStorageBuffers(): Failed to find suitable memory type for heightmap buffer\n");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    vkResult = vkAllocateMemory(vkDevice, &heightmapAllocInfo, NULL, &noiseStorageBuffers.heightmapMemory);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to allocate heightmap buffer memory: %d\n", vkResult);
        return vkResult;
    }

    vkResult = vkBindBufferMemory(vkDevice, noiseStorageBuffers.heightmapBuffer, noiseStorageBuffers.heightmapMemory, 0);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to bind heightmap buffer memory: %d\n", vkResult);
        return vkResult;
    }

    // Initialize heightmap buffer with data from CUDA if available
    if (heightmapData.d_heightmap) {
        void* heightmapMapData = NULL;
        vkResult = vkMapMemory(vkDevice, noiseStorageBuffers.heightmapMemory, 0, heightmapBufferSize, 0, &heightmapMapData);
        if (vkResult == VK_SUCCESS) {
            cudaErr = cudaMemcpy(heightmapMapData, heightmapData.d_heightmap, heightmapBufferSize, cudaMemcpyDeviceToHost);
            vkUnmapMemory(vkDevice, noiseStorageBuffers.heightmapMemory);
            if (cudaErr != cudaSuccess) {
                FileIO("CreateNoiseStorageBuffers(): Warning - Failed to copy initial heightmap data: %s\n",
                       cudaGetErrorString(cudaErr));
            }
        }
    }

    FileIO("CreateNoiseStorageBuffers(): Heightmap buffer created (%dx%d, %.2f MB)\n",
           HEIGHTMAP_SIZE, HEIGHTMAP_SIZE, (float)heightmapBufferSize / (1024.0f * 1024.0f));

    // ================== NORMAL MAP BUFFER ==================
    // Create storage buffer for CUDA-generated normal map data (float4 per texel)
    VkDeviceSize normalBufferSize = HEIGHTMAP_SIZE * HEIGHTMAP_SIZE * sizeof(float) * 4;

    VkBufferCreateInfo normalBufferInfo;
    memset(&normalBufferInfo, 0, sizeof(VkBufferCreateInfo));
    normalBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    normalBufferInfo.size = normalBufferSize;
    normalBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    normalBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkResult = vkCreateBuffer(vkDevice, &normalBufferInfo, NULL, &noiseStorageBuffers.normalBuffer);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to create normal buffer: %d\n", vkResult);
        return vkResult;
    }

    VkMemoryRequirements normalMemReqs;
    vkGetBufferMemoryRequirements(vkDevice, noiseStorageBuffers.normalBuffer, &normalMemReqs);

    VkMemoryAllocateInfo normalAllocInfo;
    memset(&normalAllocInfo, 0, sizeof(VkMemoryAllocateInfo));
    normalAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    normalAllocInfo.allocationSize = normalMemReqs.size;

    bFoundMemoryType = FALSE;
    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++) {
        if ((normalMemReqs.memoryTypeBits & (1 << i)) &&
            (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            normalAllocInfo.memoryTypeIndex = i;
            bFoundMemoryType = TRUE;
            break;
        }
    }

    if (!bFoundMemoryType) {
        FileIO("CreateNoiseStorageBuffers(): Failed to find suitable memory type for normal buffer\n");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    vkResult = vkAllocateMemory(vkDevice, &normalAllocInfo, NULL, &noiseStorageBuffers.normalMemory);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to allocate normal buffer memory: %d\n", vkResult);
        return vkResult;
    }

    vkResult = vkBindBufferMemory(vkDevice, noiseStorageBuffers.normalBuffer, noiseStorageBuffers.normalMemory, 0);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to bind normal buffer memory: %d\n", vkResult);
        return vkResult;
    }

    // Initialize normal buffer with data from CUDA if available
    if (d_normalMap) {
        void* normalMapData = NULL;
        vkResult = vkMapMemory(vkDevice, noiseStorageBuffers.normalMemory, 0, normalBufferSize, 0, &normalMapData);
        if (vkResult == VK_SUCCESS) {
            cudaErr = cudaMemcpy(normalMapData, d_normalMap, normalBufferSize, cudaMemcpyDeviceToHost);
            vkUnmapMemory(vkDevice, noiseStorageBuffers.normalMemory);
            if (cudaErr != cudaSuccess) {
                FileIO("CreateNoiseStorageBuffers(): Warning - Failed to copy initial normal map data: %s\n",
                       cudaGetErrorString(cudaErr));
            }
        }
    }

    FileIO("CreateNoiseStorageBuffers(): Normal map buffer created (%dx%d, %.2f MB)\n",
           HEIGHTMAP_SIZE, HEIGHTMAP_SIZE, (float)normalBufferSize / (1024.0f * 1024.0f));

    // ================== TANGENT MAP BUFFER ==================
    // Create storage buffer for CUDA-generated tangent space data (float4 per texel)
    VkDeviceSize tangentBufferSize = HEIGHTMAP_SIZE * HEIGHTMAP_SIZE * sizeof(float) * 4;

    VkBufferCreateInfo tangentBufferInfo;
    memset(&tangentBufferInfo, 0, sizeof(VkBufferCreateInfo));
    tangentBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    tangentBufferInfo.size = tangentBufferSize;
    tangentBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    tangentBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkResult = vkCreateBuffer(vkDevice, &tangentBufferInfo, NULL, &noiseStorageBuffers.tangentBuffer);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to create tangent buffer: %d\n", vkResult);
        return vkResult;
    }

    VkMemoryRequirements tangentMemReqs;
    vkGetBufferMemoryRequirements(vkDevice, noiseStorageBuffers.tangentBuffer, &tangentMemReqs);

    VkMemoryAllocateInfo tangentAllocInfo;
    memset(&tangentAllocInfo, 0, sizeof(VkMemoryAllocateInfo));
    tangentAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    tangentAllocInfo.allocationSize = tangentMemReqs.size;

    bFoundMemoryType = FALSE;
    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++) {
        if ((tangentMemReqs.memoryTypeBits & (1 << i)) &&
            (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            tangentAllocInfo.memoryTypeIndex = i;
            bFoundMemoryType = TRUE;
            break;
        }
    }

    if (!bFoundMemoryType) {
        FileIO("CreateNoiseStorageBuffers(): Failed to find suitable memory type for tangent buffer\n");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    vkResult = vkAllocateMemory(vkDevice, &tangentAllocInfo, NULL, &noiseStorageBuffers.tangentMemory);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to allocate tangent buffer memory: %d\n", vkResult);
        return vkResult;
    }

    vkResult = vkBindBufferMemory(vkDevice, noiseStorageBuffers.tangentBuffer, noiseStorageBuffers.tangentMemory, 0);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to bind tangent buffer memory: %d\n", vkResult);
        return vkResult;
    }

    // Initialize tangent buffer with data from CUDA if available
    if (d_tangentMap) {
        void* tangentMapData = NULL;
        vkResult = vkMapMemory(vkDevice, noiseStorageBuffers.tangentMemory, 0, tangentBufferSize, 0, &tangentMapData);
        if (vkResult == VK_SUCCESS) {
            cudaErr = cudaMemcpy(tangentMapData, d_tangentMap, tangentBufferSize, cudaMemcpyDeviceToHost);
            vkUnmapMemory(vkDevice, noiseStorageBuffers.tangentMemory);
            if (cudaErr != cudaSuccess) {
                FileIO("CreateNoiseStorageBuffers(): Warning - Failed to copy initial tangent map data: %s\n",
                       cudaGetErrorString(cudaErr));
            }
        }
    }

    FileIO("CreateNoiseStorageBuffers(): Tangent map buffer created (%dx%d, %.2f MB)\n",
           HEIGHTMAP_SIZE, HEIGHTMAP_SIZE, (float)tangentBufferSize / (1024.0f * 1024.0f));

    // ================== TESSELLATION FACTOR BUFFER ==================
    // Create storage buffer for CUDA-computed tessellation factors (one float per patch)
    VkDeviceSize tessFactorBufferSize = PATCH_GRID_SIZE * PATCH_GRID_SIZE * sizeof(float);

    VkBufferCreateInfo tessFactorBufferInfo;
    memset(&tessFactorBufferInfo, 0, sizeof(VkBufferCreateInfo));
    tessFactorBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    tessFactorBufferInfo.size = tessFactorBufferSize;
    tessFactorBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    tessFactorBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkResult = vkCreateBuffer(vkDevice, &tessFactorBufferInfo, NULL, &noiseStorageBuffers.tessFactorBuffer);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to create tessellation factor buffer: %d\n", vkResult);
        return vkResult;
    }

    VkMemoryRequirements tessFactorMemReqs;
    vkGetBufferMemoryRequirements(vkDevice, noiseStorageBuffers.tessFactorBuffer, &tessFactorMemReqs);

    VkMemoryAllocateInfo tessFactorAllocInfo;
    memset(&tessFactorAllocInfo, 0, sizeof(VkMemoryAllocateInfo));
    tessFactorAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    tessFactorAllocInfo.allocationSize = tessFactorMemReqs.size;

    bFoundMemoryType = FALSE;
    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++) {
        if ((tessFactorMemReqs.memoryTypeBits & (1 << i)) &&
            (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            tessFactorAllocInfo.memoryTypeIndex = i;
            bFoundMemoryType = TRUE;
            break;
        }
    }

    if (!bFoundMemoryType) {
        FileIO("CreateNoiseStorageBuffers(): Failed to find suitable memory type for tessellation factor buffer\n");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    vkResult = vkAllocateMemory(vkDevice, &tessFactorAllocInfo, NULL, &noiseStorageBuffers.tessFactorMemory);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to allocate tessellation factor buffer memory: %d\n", vkResult);
        return vkResult;
    }

    vkResult = vkBindBufferMemory(vkDevice, noiseStorageBuffers.tessFactorBuffer, noiseStorageBuffers.tessFactorMemory, 0);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to bind tessellation factor buffer memory: %d\n", vkResult);
        return vkResult;
    }

    // Initialize with default tessellation factors if CUDA data available
    if (d_tessellationFactors) {
        void* tessFactorData = NULL;
        vkResult = vkMapMemory(vkDevice, noiseStorageBuffers.tessFactorMemory, 0, tessFactorBufferSize, 0, &tessFactorData);
        if (vkResult == VK_SUCCESS) {
            cudaErr = cudaMemcpy(tessFactorData, d_tessellationFactors, tessFactorBufferSize, cudaMemcpyDeviceToHost);
            vkUnmapMemory(vkDevice, noiseStorageBuffers.tessFactorMemory);
            if (cudaErr != cudaSuccess) {
                FileIO("CreateNoiseStorageBuffers(): Warning - Failed to copy initial tessellation factor data: %s\n",
                       cudaGetErrorString(cudaErr));
            }
        }
    }

    FileIO("CreateNoiseStorageBuffers(): Tessellation factor buffer created (%dx%d patches, %u bytes)\n",
           PATCH_GRID_SIZE, PATCH_GRID_SIZE, (unsigned int)tessFactorBufferSize);

    // ================== VISIBILITY BUFFER ==================
    // Create storage buffer for frustum culling visibility mask
    VkDeviceSize visibilityBufferSize = PATCH_GRID_SIZE * PATCH_GRID_SIZE * sizeof(int);

    VkBufferCreateInfo visibilityBufferInfo;
    memset(&visibilityBufferInfo, 0, sizeof(VkBufferCreateInfo));
    visibilityBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    visibilityBufferInfo.size = visibilityBufferSize;
    visibilityBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    visibilityBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkResult = vkCreateBuffer(vkDevice, &visibilityBufferInfo, NULL, &noiseStorageBuffers.visibilityBuffer);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to create visibility buffer: %d\n", vkResult);
        return vkResult;
    }

    VkMemoryRequirements visibilityMemReqs;
    vkGetBufferMemoryRequirements(vkDevice, noiseStorageBuffers.visibilityBuffer, &visibilityMemReqs);

    VkMemoryAllocateInfo visibilityAllocInfo;
    memset(&visibilityAllocInfo, 0, sizeof(VkMemoryAllocateInfo));
    visibilityAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    visibilityAllocInfo.allocationSize = visibilityMemReqs.size;

    bFoundMemoryType = FALSE;
    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++) {
        if ((visibilityMemReqs.memoryTypeBits & (1 << i)) &&
            (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            visibilityAllocInfo.memoryTypeIndex = i;
            bFoundMemoryType = TRUE;
            break;
        }
    }

    if (!bFoundMemoryType) {
        FileIO("CreateNoiseStorageBuffers(): Failed to find suitable memory type for visibility buffer\n");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    vkResult = vkAllocateMemory(vkDevice, &visibilityAllocInfo, NULL, &noiseStorageBuffers.visibilityMemory);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to allocate visibility buffer memory: %d\n", vkResult);
        return vkResult;
    }

    vkResult = vkBindBufferMemory(vkDevice, noiseStorageBuffers.visibilityBuffer, noiseStorageBuffers.visibilityMemory, 0);
    if (vkResult != VK_SUCCESS) {
        FileIO("CreateNoiseStorageBuffers(): Failed to bind visibility buffer memory: %d\n", vkResult);
        return vkResult;
    }

    FileIO("CreateNoiseStorageBuffers(): Visibility buffer created (%dx%d patches, %u bytes)\n",
           PATCH_GRID_SIZE, PATCH_GRID_SIZE, (unsigned int)visibilityBufferSize);

    FileIO("CreateNoiseStorageBuffers(): All noise storage buffers initialized successfully\n");

    return VK_SUCCESS;
}

void CleanupNoiseStorageBuffers(void)
{
    FileIO("CleanupNoiseStorageBuffers(): Cleaning up noise storage buffers...\n");

    if (noiseStorageBuffers.heightmapBuffer) {
        vkDestroyBuffer(vkDevice, noiseStorageBuffers.heightmapBuffer, NULL);
        noiseStorageBuffers.heightmapBuffer = VK_NULL_HANDLE;
    }
    if (noiseStorageBuffers.heightmapMemory) {
        vkFreeMemory(vkDevice, noiseStorageBuffers.heightmapMemory, NULL);
        noiseStorageBuffers.heightmapMemory = VK_NULL_HANDLE;
    }
    if (noiseStorageBuffers.gradientBuffer) {
        vkDestroyBuffer(vkDevice, noiseStorageBuffers.gradientBuffer, NULL);
        noiseStorageBuffers.gradientBuffer = VK_NULL_HANDLE;
    }
    if (noiseStorageBuffers.gradientMemory) {
        vkFreeMemory(vkDevice, noiseStorageBuffers.gradientMemory, NULL);
        noiseStorageBuffers.gradientMemory = VK_NULL_HANDLE;
    }
    if (noiseStorageBuffers.permutationBuffer) {
        vkDestroyBuffer(vkDevice, noiseStorageBuffers.permutationBuffer, NULL);
        noiseStorageBuffers.permutationBuffer = VK_NULL_HANDLE;
    }
    if (noiseStorageBuffers.permutationMemory) {
        vkFreeMemory(vkDevice, noiseStorageBuffers.permutationMemory, NULL);
        noiseStorageBuffers.permutationMemory = VK_NULL_HANDLE;
    }
    if (noiseStorageBuffers.normalBuffer) {
        vkDestroyBuffer(vkDevice, noiseStorageBuffers.normalBuffer, NULL);
        noiseStorageBuffers.normalBuffer = VK_NULL_HANDLE;
    }
    if (noiseStorageBuffers.normalMemory) {
        vkFreeMemory(vkDevice, noiseStorageBuffers.normalMemory, NULL);
        noiseStorageBuffers.normalMemory = VK_NULL_HANDLE;
    }
    if (noiseStorageBuffers.tangentBuffer) {
        vkDestroyBuffer(vkDevice, noiseStorageBuffers.tangentBuffer, NULL);
        noiseStorageBuffers.tangentBuffer = VK_NULL_HANDLE;
    }
    if (noiseStorageBuffers.tangentMemory) {
        vkFreeMemory(vkDevice, noiseStorageBuffers.tangentMemory, NULL);
        noiseStorageBuffers.tangentMemory = VK_NULL_HANDLE;
    }
    if (noiseStorageBuffers.tessFactorBuffer) {
        vkDestroyBuffer(vkDevice, noiseStorageBuffers.tessFactorBuffer, NULL);
        noiseStorageBuffers.tessFactorBuffer = VK_NULL_HANDLE;
    }
    if (noiseStorageBuffers.tessFactorMemory) {
        vkFreeMemory(vkDevice, noiseStorageBuffers.tessFactorMemory, NULL);
        noiseStorageBuffers.tessFactorMemory = VK_NULL_HANDLE;
    }
    if (noiseStorageBuffers.visibilityBuffer) {
        vkDestroyBuffer(vkDevice, noiseStorageBuffers.visibilityBuffer, NULL);
        noiseStorageBuffers.visibilityBuffer = VK_NULL_HANDLE;
    }
    if (noiseStorageBuffers.visibilityMemory) {
        vkFreeMemory(vkDevice, noiseStorageBuffers.visibilityMemory, NULL);
        noiseStorageBuffers.visibilityMemory = VK_NULL_HANDLE;
    }

    FileIO("CleanupNoiseStorageBuffers(): Noise storage buffers cleaned up\n");
}

VkResult resize(int width, int height)
{

	VkResult CreateSwapChain(VkBool32);
	VkResult CreateImagesAndImageViews(void);
	VkResult CreateRenderPass(void);
	VkResult CreatePipelineLayout(void);
	VkResult CreatePipeline(void);
	VkResult CreateFramebuffers(void);
	VkResult CreateCommandBuffers(VkCommandBuffer**);
	VkResult buildCommandBuffers(void);

	VkResult vkResult = VK_SUCCESS;

	if(height <= 0)
	{
		height = 1;
	}

	if(bInitialized == FALSE)
	{

		FileIO("resize(): initialization yet not completed or failed\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}

	bInitialized = FALSE;

	winWidth = width;
	winHeight = height;

	if(vkDevice)
	{
		vkDeviceWaitIdle(vkDevice);
		FileIO("resize(): vkDeviceWaitIdle() is done\n");
	}

	if(vkSwapchainKHR == VK_NULL_HANDLE)
	{
		FileIO("resize(): vkSwapchainKHR is already NULL, cannot proceed\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}

	for(uint32_t i =0; i < swapchainImageCount; i++)
	{
		vkDestroyFramebuffer(vkDevice, vkFramebuffer_array[i], NULL);
		vkFramebuffer_array[i] = NULL;
		FileIO("resize(): vkDestroyFramebuffer() is done\n");
	}

	if(vkFramebuffer_array)
	{
		free(vkFramebuffer_array);
		vkFramebuffer_array = NULL;
		FileIO("resize(): vkFramebuffer_array is freed\n");
	}

	for(uint32_t i =0; i < swapchainImageCount; i++)
	{
		vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_for_1024_x_1024_graphics_array[i]);
		FileIO("resize(): vkFreeCommandBuffers() is done \n");
	}

	if(vkCommandBuffer_for_1024_x_1024_graphics_array)
	{
		free(vkCommandBuffer_for_1024_x_1024_graphics_array);
		vkCommandBuffer_for_1024_x_1024_graphics_array = NULL;
		FileIO("resize(): vkCommandBuffer_for_1024_x_1024_graphics_array is freed\n");
	}

	if(vkPipeline)
	{
		vkDestroyPipeline(vkDevice, vkPipeline, NULL);
		vkPipeline = VK_NULL_HANDLE;
		FileIO("resize(): vkPipeline is freed\n");
	}

	if(vkPipelineLayout)
	{
		vkDestroyPipelineLayout(vkDevice, vkPipelineLayout, NULL);
		vkPipelineLayout = VK_NULL_HANDLE;
		FileIO("resize(): vkPipelineLayout is freed\n");
	}

	if(vkRenderPass)
	{
		vkDestroyRenderPass(vkDevice, vkRenderPass, NULL);
		vkRenderPass = VK_NULL_HANDLE;
		FileIO("resize(): vkDestroyRenderPass() is done\n");
	}

	if(vkImageView_depth)
	{
		vkDestroyImageView(vkDevice, vkImageView_depth, NULL);
		vkImageView_depth = VK_NULL_HANDLE;
	}

	if(vkDeviceMemory_depth)
	{
		vkFreeMemory(vkDevice, vkDeviceMemory_depth, NULL);
		vkDeviceMemory_depth = VK_NULL_HANDLE;
	}

	if(vkImage_depth)
	{

		vkDestroyImage(vkDevice, vkImage_depth, NULL);
		vkImage_depth = VK_NULL_HANDLE;
	}

	for(uint32_t i =0; i < swapchainImageCount; i++)
	{
		vkDestroyImageView(vkDevice, swapChainImageView_array[i], NULL);
		FileIO("resize(): vkDestroyImageView() is done\n");
	}

	if(swapChainImageView_array)
	{
		free(swapChainImageView_array);
		swapChainImageView_array = NULL;
		FileIO("resize(): swapChainImageView_array is freed\n");
	}

	if(swapChainImage_array)
	{
		free(swapChainImage_array);
		swapChainImage_array = NULL;
		FileIO("resize(): swapChainImage_array is freed\n");
	}

	vkDestroySwapchainKHR(vkDevice, vkSwapchainKHR, NULL);
	vkSwapchainKHR = VK_NULL_HANDLE;
	FileIO("resize(): vkDestroySwapchainKHR() is done\n");

	vkResult = CreateSwapChain(VK_FALSE);
	if (vkResult != VK_SUCCESS)
	{
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		FileIO("resize(): CreateSwapChain() function failed with error code %d\n", vkResult);
		return vkResult;
	}

	vkResult =  CreateImagesAndImageViews();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("resize(): CreateImagesAndImageViews() function failed with error code %d\n", vkResult);
		return vkResult;
	}

	vkResult =  CreateRenderPass();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("resize(): CreateRenderPass() function failed with error code %d\n", vkResult);
		return vkResult;
	}

	vkResult = CreatePipelineLayout();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("resize(): CreatePipelineLayout() function failed with error code %d\n", vkResult);
		return vkResult;
	}

	vkResult = CreatePipeline();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("resize(): CreatePipeline() function failed with error code %d\n", vkResult);
		return vkResult;
	}

	vkResult = CreateFramebuffers();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("resize(): CreateFramebuffers() function failed with error code %d\n", vkResult);
		return vkResult;
	}

	vkResult  = CreateCommandBuffers(&vkCommandBuffer_for_1024_x_1024_graphics_array);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("resize(): CreateCommandBuffers() function failed with error code %d for 1024 x 1024\n", vkResult);
		return vkResult;
	}

	vkResult = buildCommandBuffers();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("resize(): buildCommandBuffers() function failed with error code %d\n", vkResult);
		return vkResult;
	}

	bInitialized = TRUE;

	return vkResult;
}

VkResult UpdateUniformBuffer(void)
{

	VkResult vkResult = VK_SUCCESS;

	MyUniformData myUniformData;
	memset((void*)&myUniformData, 0, sizeof(struct MyUniformData));

	myUniformData.modelMatrix = glm::mat4(1.0f);

	// Use the dynamic free camera view matrix
	myUniformData.viewMatrix = gCamera.viewMatrix;

	// Static projection matrix (Y-flipped for Vulkan coordinate system)
	float aspectRatio = (float)winWidth / (float)winHeight;
	glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspectRatio, 0.1f, 2000.0f);
	proj[1][1] *= -1.0f;  // Flip Y for Vulkan
	myUniformData.projectionMatrix = proj;

	// White base color for terrain
	myUniformData.color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

	void* data = NULL;
	vkResult = vkMapMemory(vkDevice, uniformData.vkDeviceMemory, 0, sizeof(struct MyUniformData), 0, &data);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("UpdateUniformBuffer(): vkMapMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}

	memcpy(data, &myUniformData, sizeof(struct MyUniformData));

	vkUnmapMemory(vkDevice, uniformData.vkDeviceMemory);

	return vkResult;
}

VkResult display(void)
{

	VkResult resize(int, int);

	VkResult UpdateUniformBuffer(void);
	VkResult UpdateIndirectBuffer(void);
	VkResult buildCommandBuffers(void);

	VkResult vkResult = VK_SUCCESS;

	VkCommandBuffer *vkCommandBuffer_array = NULL;

	if(bInitialized == FALSE)
	{
		FileIO("display(): initialization not completed yet\n");
		return (VkResult)VK_FALSE;
	}

	vkCommandBuffer_array = vkCommandBuffer_for_1024_x_1024_graphics_array;

	// ============================================================================
	// ADVANCED CUDA TERRAIN PROCESSING
	// Heightmap generation, normal maps, frustum culling, adaptive tessellation
	// ============================================================================

	// Static position for LOD calculations
	float3 fixedPos = h_fixedPos;

	if (bCudaTerrainSystemInitialized && heightmapData.d_heightmap && bForceHeightmapUpdate) {
		// Launch heightmap generation on all 4 streams in parallel (quadrant-based)
		int quadrantSize = HEIGHTMAP_SIZE / 2;
		dim3 hmBlockDim(16, 16);
		dim3 hmGridDim((quadrantSize + hmBlockDim.x - 1) / hmBlockDim.x,
		               (quadrantSize + hmBlockDim.y - 1) / hmBlockDim.y);

		// Launch all 4 quadrants on separate CUDA streams for maximum parallelism
		for (int qy = 0; qy < 2; qy++) {
			for (int qx = 0; qx < 2; qx++) {
				int streamIdx = qy * 2 + qx;
				generateTerrainHeightmapQuadrant<<<hmGridDim, hmBlockDim, 0, cudaStreams[streamIdx]>>>(
					heightmapData.d_heightmap,
					d_gradientTable,
					d_permutationTable,
					HEIGHTMAP_SIZE,
					TERRAIN_SIZE,
					fixedPos,
					qx, qy, quadrantSize
				);
			}
		}

		// Synchronize heightmap streams
		for (int i = 0; i < NUM_CUDA_STREAMS; i++) {
			cudaStreamSynchronize(cudaStreams[i]);
		}

		// Compute normal map from heightmap
		dim3 normalBlockDim(16, 16);
		dim3 normalGridDim((HEIGHTMAP_SIZE + normalBlockDim.x - 1) / normalBlockDim.x,
		                   (HEIGHTMAP_SIZE + normalBlockDim.y - 1) / normalBlockDim.y);

		computeNormalMap<<<normalGridDim, normalBlockDim>>>(
			d_normalMap,
			heightmapData.d_heightmap,
			HEIGHTMAP_SIZE,
			TERRAIN_SIZE,
			HEIGHT_SCALE
		);

		// Compute tangent space from normal map
		computeTangentSpace<<<normalGridDim, normalBlockDim>>>(
			d_tangentMap,
			d_normalMap,
			HEIGHTMAP_SIZE
		);

		// Generate LOD mipmaps
		for (int i = 1; i < NUM_LOD_LEVELS; i++) {
			int srcSize = heightmapLODSizes[i - 1];
			int dstSize = heightmapLODSizes[i];
			dim3 mipBlockDim(16, 16);
			dim3 mipGridDim((dstSize + mipBlockDim.x - 1) / mipBlockDim.x,
			                (dstSize + mipBlockDim.y - 1) / mipBlockDim.y);

			generateHeightmapMipmap<<<mipGridDim, mipBlockDim>>>(
				d_heightmapLOD[i],
				d_heightmapLOD[i - 1],
				dstSize,
				srcSize
			);
		}

		bForceHeightmapUpdate = FALSE;
	}

	// ============================================================================
	// FRUSTUM CULLING AND ADAPTIVE TESSELLATION
	// ============================================================================

	if (bCudaTerrainSystemInitialized && d_frustumPlanes && d_visibilityMask) {
		// Extract frustum planes from view-projection matrix
		// Use dynamic free camera view matrix
		float aspectRatio = (float)winWidth / (float)winHeight;
		glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspectRatio, 0.1f, 2000.0f);
		glm::mat4 viewProj = proj * gCamera.viewMatrix;
		float4 h_frustumPlanes[6];
		extractFrustumPlanes(h_frustumPlanes, viewProj);

		// Copy frustum planes to GPU
		cudaMemcpy(d_frustumPlanes, h_frustumPlanes, 6 * sizeof(float4), cudaMemcpyHostToDevice);

		// Reset visible patch counter
		int zero = 0;
		cudaMemcpy(d_visiblePatchCount, &zero, sizeof(int), cudaMemcpyHostToDevice);

		// Perform frustum culling
		dim3 cullBlockDim(8, 8);
		dim3 cullGridDim((PATCH_GRID_SIZE + cullBlockDim.x - 1) / cullBlockDim.x,
		                 (PATCH_GRID_SIZE + cullBlockDim.y - 1) / cullBlockDim.y);

		performFrustumCulling<<<cullGridDim, cullBlockDim>>>(
			d_visibilityMask,
			d_visiblePatchCount,
			heightmapData.d_heightmap,
			d_frustumPlanes,
			PATCH_GRID_SIZE,
			TERRAIN_SIZE,
			HEIGHTMAP_SIZE,
			HEIGHT_SCALE
		);

		// Use dynamic camera position for tessellation LOD calculation
		float3 cameraPos = make_float3(gCamera.position.x, gCamera.position.y, gCamera.position.z);

		// Compute adaptive tessellation factors
		computeTessellationFactors<<<cullGridDim, cullBlockDim>>>(
			d_tessellationFactors,
			heightmapData.d_heightmap,
			d_normalMap,
			d_visibilityMask,
			cameraPos,
			PATCH_GRID_SIZE,
			TERRAIN_SIZE,
			HEIGHTMAP_SIZE,
			HEIGHT_SCALE,
			64.0f,  // maxTessLevel
			4.0f    // minTessLevel
		);
	}

	// Generate flat terrain patches (base grid)
	dim3 block(8, 8, 1);
	dim3 grid((PATCH_GRID_SIZE + block.x - 1) / block.x, (PATCH_GRID_SIZE + block.y - 1) / block.y, 1);
	generateFlatTerrainPatches<<<grid, block>>>((float4*)pos_CUDA, PATCH_GRID_SIZE, TERRAIN_SIZE);

	cudaResult = cudaDeviceSynchronize();
	if(cudaResult != CUDA_SUCCESS)
	{
		FileIO("display(): cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaResult));
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}

	// Copy heightmap data to Vulkan storage buffer for shader access
	if (bCudaTerrainSystemInitialized && noiseStorageBuffers.heightmapBuffer && heightmapData.d_heightmap) {
		void* heightmapPtr = NULL;
		VkDeviceSize heightmapSize = HEIGHTMAP_SIZE * HEIGHTMAP_SIZE * sizeof(float);
		vkResult = vkMapMemory(vkDevice, noiseStorageBuffers.heightmapMemory, 0, heightmapSize, 0, &heightmapPtr);
		if (vkResult == VK_SUCCESS) {
			cudaResult = cudaMemcpy(heightmapPtr, heightmapData.d_heightmap, heightmapSize, cudaMemcpyDeviceToHost);
			vkUnmapMemory(vkDevice, noiseStorageBuffers.heightmapMemory);
		}
	}

	// Copy normal map data to Vulkan storage buffer
	if (bCudaTerrainSystemInitialized && noiseStorageBuffers.normalBuffer && d_normalMap) {
		void* normalPtr = NULL;
		VkDeviceSize normalSize = HEIGHTMAP_SIZE * HEIGHTMAP_SIZE * sizeof(float) * 4;
		vkResult = vkMapMemory(vkDevice, noiseStorageBuffers.normalMemory, 0, normalSize, 0, &normalPtr);
		if (vkResult == VK_SUCCESS) {
			cudaResult = cudaMemcpy(normalPtr, d_normalMap, normalSize, cudaMemcpyDeviceToHost);
			vkUnmapMemory(vkDevice, noiseStorageBuffers.normalMemory);
		}
	}

	// Copy tessellation factors to Vulkan storage buffer
	if (bCudaTerrainSystemInitialized && noiseStorageBuffers.tessFactorBuffer && d_tessellationFactors) {
		void* tessPtr = NULL;
		VkDeviceSize tessSize = PATCH_GRID_SIZE * PATCH_GRID_SIZE * sizeof(float);
		vkResult = vkMapMemory(vkDevice, noiseStorageBuffers.tessFactorMemory, 0, tessSize, 0, &tessPtr);
		if (vkResult == VK_SUCCESS) {
			cudaResult = cudaMemcpy(tessPtr, d_tessellationFactors, tessSize, cudaMemcpyDeviceToHost);
			vkUnmapMemory(vkDevice, noiseStorageBuffers.tessFactorMemory);
		}
	}

	// Copy visibility mask to Vulkan storage buffer
	if (bCudaTerrainSystemInitialized && noiseStorageBuffers.visibilityBuffer && d_visibilityMask) {
		void* visibilityPtr = NULL;
		VkDeviceSize visibilitySize = PATCH_GRID_SIZE * PATCH_GRID_SIZE * sizeof(int);
		vkResult = vkMapMemory(vkDevice, noiseStorageBuffers.visibilityMemory, 0, visibilitySize, 0, &visibilityPtr);
		if (vkResult == VK_SUCCESS) {
			cudaResult = cudaMemcpy(visibilityPtr, d_visibilityMask, visibilitySize, cudaMemcpyDeviceToHost);
			vkUnmapMemory(vkDevice, noiseStorageBuffers.visibilityMemory);
		}
	}

	vkResult = UpdateIndirectBuffer();
	if(vkResult != VK_SUCCESS)
	{
		FileIO("display(): UpdateIndirectBuffer() failed\n");
		return vkResult;
	}

	vkResult = buildCommandBuffers();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("display(): buildCommandBuffers() function failed with error code %d\n", vkResult);
		return vkResult;
	}

	vkResult = vkAcquireNextImageKHR(vkDevice, vkSwapchainKHR, UINT64_MAX, vkSemaphore_BackBuffer, VK_NULL_HANDLE, &currentImageIndex);
	if(vkResult != VK_SUCCESS)
	{
		if((vkResult == VK_ERROR_OUT_OF_DATE_KHR) || (vkResult == VK_SUBOPTIMAL_KHR))
		{
			resize(winWidth, winHeight);
		}
		else
		{
			FileIO("display(): vkAcquireNextImageKHR() failed\n");
			return vkResult;
		}
	}

	vkResult = vkWaitForFences(vkDevice, 1, &vkFence_array[currentImageIndex], VK_TRUE, UINT64_MAX);
	if(vkResult != VK_SUCCESS)
	{
		FileIO("display(): vkWaitForFences() failed\n");
		return vkResult;
	}

	vkResult = vkResetFences(vkDevice, 1, &vkFence_array[currentImageIndex]);
	if(vkResult != VK_SUCCESS)
	{
		FileIO("display(): vkResetFences() failed\n");
		return vkResult;
	}

	const VkPipelineStageFlags waitDstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	VkSubmitInfo vkSubmitInfo;
	memset((void*)&vkSubmitInfo, 0, sizeof(VkSubmitInfo));
	vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	vkSubmitInfo.pNext = NULL;
	vkSubmitInfo.pWaitDstStageMask = &waitDstStageMask;
	vkSubmitInfo.waitSemaphoreCount = 1;
	vkSubmitInfo.pWaitSemaphores = &vkSemaphore_BackBuffer;
	vkSubmitInfo.commandBufferCount = 1;
	vkSubmitInfo.pCommandBuffers = &vkCommandBuffer_array[currentImageIndex];
	vkSubmitInfo.signalSemaphoreCount = 1;
	vkSubmitInfo.pSignalSemaphores = &vkSemaphore_RenderComplete;

	vkResult = vkQueueSubmit(vkQueue, 1, &vkSubmitInfo, vkFence_array[currentImageIndex]);
	if(vkResult != VK_SUCCESS)
	{
		FileIO("display(): vkQueueSubmit() failed\n");
		return vkResult;
	}

	VkPresentInfoKHR  vkPresentInfoKHR;
	memset((void*)&vkPresentInfoKHR, 0, sizeof(VkPresentInfoKHR));
	vkPresentInfoKHR.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	vkPresentInfoKHR.pNext = NULL;
	vkPresentInfoKHR.swapchainCount = 1;
	vkPresentInfoKHR.pSwapchains = &vkSwapchainKHR;
	vkPresentInfoKHR.pImageIndices = &currentImageIndex;
	vkPresentInfoKHR.waitSemaphoreCount = 1;
    vkPresentInfoKHR.pWaitSemaphores = &vkSemaphore_RenderComplete;
	vkPresentInfoKHR.pResults = NULL;

	vkResult =  vkQueuePresentKHR(vkQueue, &vkPresentInfoKHR);
	if(vkResult != VK_SUCCESS)
	{
		if((vkResult == VK_ERROR_OUT_OF_DATE_KHR) || (vkResult == VK_SUBOPTIMAL_KHR))
		{
			resize(winWidth, winHeight);
		}
		else
		{
			FileIO("display(): vkQueuePresentKHR() failed\n");
			return vkResult;
		}
	}

	vkResult = UpdateUniformBuffer();
	if(vkResult != VK_SUCCESS)
	{
		FileIO("display(): updateUniformBuffer() function failed with error code %d\n", vkResult);
		return vkResult;
	}

	vkDeviceWaitIdle(vkDevice);
	return vkResult;
}

void update(void)
{
    // Calculate delta time for frame-rate independent movement
    LARGE_INTEGER currentTime;
    QueryPerformanceCounter(&currentTime);

    if (gLastFrameTime.QuadPart != 0) {
        gDeltaTime = (float)(currentTime.QuadPart - gLastFrameTime.QuadPart) / (float)gPerfFrequency.QuadPart;
        // Clamp delta time to avoid huge jumps
        if (gDeltaTime > 0.1f) gDeltaTime = 0.1f;
    }
    gLastFrameTime = currentTime;

    // Choose movement speed based on shift key
    float currentSpeed = gInputState.keyShift ? gCamera.fastMoveSpeed : gCamera.moveSpeed;

    // Update camera vectors based on current yaw and pitch
    gCamera.updateVectors();

    // Handle mouse capture mode continuous rotation
    if (gInputState.mouseCaptured) {
        RECT rect;
        GetClientRect(ghwnd, &rect);
        int centerX = (rect.right - rect.left) / 2;
        int centerY = (rect.bottom - rect.top) / 2;

        // Calculate delta from center
        int deltaX = gInputState.mouseX - centerX;
        int deltaY = gInputState.mouseY - centerY;

        if (deltaX != 0 || deltaY != 0) {
            gCamera.yaw += deltaX * gCamera.mouseSensitivity;
            gCamera.pitch -= deltaY * gCamera.mouseSensitivity;

            // Clamp pitch
            if (gCamera.pitch > 1.5f) gCamera.pitch = 1.5f;
            if (gCamera.pitch < -1.5f) gCamera.pitch = -1.5f;

            // Re-center cursor
            POINT center = { centerX, centerY };
            ClientToScreen(ghwnd, &center);
            SetCursorPos(center.x, center.y);
            gInputState.mouseX = centerX;
            gInputState.mouseY = centerY;
        }
    }

    // Calculate movement direction
    glm::vec3 moveDirection(0.0f);

    // Forward/Backward (W/S)
    if (gInputState.keyW) {
        moveDirection += gCamera.forward;
    }
    if (gInputState.keyS) {
        moveDirection -= gCamera.forward;
    }

    // Left/Right (A/D)
    if (gInputState.keyA) {
        moveDirection -= gCamera.right;
    }
    if (gInputState.keyD) {
        moveDirection += gCamera.right;
    }

    // Up/Down (Space/Ctrl)
    if (gInputState.keySpace) {
        moveDirection += glm::vec3(0.0f, 1.0f, 0.0f);
    }
    if (gInputState.keyCtrl) {
        moveDirection -= glm::vec3(0.0f, 1.0f, 0.0f);
    }

    // Normalize movement direction to avoid faster diagonal movement
    if (glm::length(moveDirection) > 0.0f) {
        moveDirection = glm::normalize(moveDirection);
        gCamera.position += moveDirection * currentSpeed * gDeltaTime;
    }

    // Update camera view matrix after movement
    gCamera.updateVectors();
}

void uninitialize(void)
{

		void ToggleFullScreen(void);

		cudaError_t uninitialize_cuda(void);

		if (gbFullscreen == TRUE)
		{
			ToggleFullscreen();
			gbFullscreen = FALSE;
		}

		if (ghwnd)
		{
			DestroyWindow(ghwnd);
			ghwnd = NULL;
		}

		if(vkDevice)
		{
			vkDeviceWaitIdle(vkDevice);
			FileIO("uninitialize(): vkDeviceWaitIdle() is done\n");

			for(uint32_t i = 0; i< swapchainImageCount; i++)
			{
				vkDestroyFence(vkDevice, vkFence_array[i], NULL);
				FileIO("uninitialize(): vkFence_array[%d] is freed\n", i);
			}

			if(vkFence_array)
			{
				free(vkFence_array);
				vkFence_array = NULL;
				FileIO("uninitialize(): vkFence_array is freed\n");
			}

			if(vkSemaphore_RenderComplete)
			{
				vkDestroySemaphore(vkDevice, vkSemaphore_RenderComplete, NULL);
				vkSemaphore_RenderComplete = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkSemaphore_RenderComplete is freed\n");
			}

			if(vkSemaphore_BackBuffer)
			{
				vkDestroySemaphore(vkDevice, vkSemaphore_BackBuffer, NULL);
				vkSemaphore_RenderComplete = VK_NULL_HANDLE;
					FileIO("uninitialize(): vkSemaphore_BackBuffer is freed\n");
			}

			for(uint32_t i =0; i < swapchainImageCount; i++)
			{
				vkDestroyFramebuffer(vkDevice, vkFramebuffer_array[i], NULL);
				vkFramebuffer_array[i] = NULL;
				FileIO("uninitialize(): vkDestroyFramebuffer() is done\n");
			}

			if(vkFramebuffer_array)
			{
				free(vkFramebuffer_array);
				vkFramebuffer_array = NULL;
				FileIO("uninitialize(): vkFramebuffer_array is freed\n");
			}

			if(vkPipeline)
			{
				vkDestroyPipeline(vkDevice, vkPipeline, NULL);
				vkPipeline = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkPipeline is freed\n");
			}

			if(vkDescriptorSetLayout)
			{
				vkDestroyDescriptorSetLayout(vkDevice, vkDescriptorSetLayout, NULL);
				vkDescriptorSetLayout = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkDescriptorSetLayout is freed\n");
			}

			if(vkPipelineLayout)
			{
				vkDestroyPipelineLayout(vkDevice, vkPipelineLayout, NULL);
				vkPipelineLayout = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkPipelineLayout is freed\n");
			}

			if(vkRenderPass)
			{
				vkDestroyRenderPass(vkDevice, vkRenderPass, NULL);
				vkRenderPass = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkDestroyRenderPass() is done\n");
			}

			if(vkDescriptorPool)
			{

				vkDestroyDescriptorPool(vkDevice, vkDescriptorPool, NULL);
				vkDescriptorPool = VK_NULL_HANDLE;
				vkDescriptorSet = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkDestroyDescriptorPool() is done for vkDescriptorPool and vkDescriptorSet both\n");
			}

			if(vkShaderModule_tess_eval)
			{
				vkDestroyShaderModule(vkDevice, vkShaderModule_tess_eval, NULL);
				vkShaderModule_tess_eval = VK_NULL_HANDLE;
				FileIO("uninitialize(): VkShaderModule for tessellation evaluation shader is done\n");
			}

			if(vkShaderModule_tess_control)
			{
				vkDestroyShaderModule(vkDevice, vkShaderModule_tess_control, NULL);
				vkShaderModule_tess_control = VK_NULL_HANDLE;
				FileIO("uninitialize(): VkShaderModule for tessellation control shader is done\n");
			}

			if(vkShaderMoudule_fragment_shader)
			{
				vkDestroyShaderModule(vkDevice, vkShaderMoudule_fragment_shader, NULL);
				vkShaderMoudule_fragment_shader = VK_NULL_HANDLE;
				FileIO("uninitialize(): VkShaderMoudule for fragment shader is done\n");
			}

			if(vkShaderMoudule_vertex_shader)
			{
				vkDestroyShaderModule(vkDevice, vkShaderMoudule_vertex_shader, NULL);
				vkShaderMoudule_vertex_shader = VK_NULL_HANDLE;
				FileIO("uninitialize(): VkShaderMoudule for vertex shader is done\n");
			}

			if(uniformData.vkBuffer)
			{
				vkDestroyBuffer(vkDevice, uniformData.vkBuffer, NULL);
				uniformData.vkBuffer = VK_NULL_HANDLE;
				FileIO("uninitialize(): uniformData.vkBuffer is freed\n");
			}

			if(uniformData.vkDeviceMemory)
			{
				vkFreeMemory(vkDevice, uniformData.vkDeviceMemory, NULL);
				uniformData.vkDeviceMemory = VK_NULL_HANDLE;
				FileIO("uninitialize(): uniformData.vkDeviceMemory is freed\n");
			}

			// Cleanup cuRAND noise storage buffers (Vulkan side)
			CleanupNoiseStorageBuffers();

			// Cleanup cuRAND noise system (CUDA side) before CUDA cleanup
			cleanup_curand_noise_system();

			cudaResult = uninitialize_cuda();
			if(cudaResult != CUDA_SUCCESS)
			{
				FileIO("uninitialize(): uninitialize_cuda() failed\n");
			}
			else
			{
				FileIO("uninitialize(): uninitialize_cuda() suceeded\n");
			}

			if(vertexData_external.vkDeviceMemory)
			{
				vkFreeMemory(vkDevice, vertexData_external.vkDeviceMemory, NULL);
				vertexData_external.vkDeviceMemory = VK_NULL_HANDLE;
				FileIO("uninitialize(): vertexData_external.vkDeviceMemory is freed\n");
			}

			if(vertexData_external.vkBuffer)
			{
				vkDestroyBuffer(vkDevice, vertexData_external.vkBuffer, NULL);
				vertexData_external.vkBuffer = VK_NULL_HANDLE;
				FileIO("uninitialize(): vertexData_external.vkBuffer is freed\n");
			}

			if(vertexdata_indirect_buffer.vkDeviceMemory)
			{
				vkFreeMemory(vkDevice, vertexdata_indirect_buffer.vkDeviceMemory, NULL);
				vertexdata_indirect_buffer.vkDeviceMemory = VK_NULL_HANDLE;
				FileIO("uninitialize(): vertexdata_indirect_buffer.vkDeviceMemory is freed\n");
			}

			if(vertexdata_indirect_buffer.vkBuffer)
			{
				vkDestroyBuffer(vkDevice, vertexdata_indirect_buffer.vkBuffer, NULL);
				vertexdata_indirect_buffer.vkBuffer = VK_NULL_HANDLE;
				FileIO("uninitialize(): vertexdata_indirect_buffer.vkBuffer is freed\n");
			}

			for(uint32_t i =0; i < swapchainImageCount; i++)
			{
				vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_for_1024_x_1024_graphics_array[i]);
				FileIO("uninitialize(): vkFreeCommandBuffers() is done\n");
			}

			if(vkCommandBuffer_for_1024_x_1024_graphics_array)
			{
				free(vkCommandBuffer_for_1024_x_1024_graphics_array);
				vkCommandBuffer_for_1024_x_1024_graphics_array = NULL;
				FileIO("uninitialize(): vkCommandBuffer_for_1024_x_1024_graphics_array is freed\n");
			}

			if(vkCommandPool)
			{
				vkDestroyCommandPool(vkDevice, vkCommandPool, NULL);
				vkCommandPool = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkDestroyCommandPool() is done\n");
			}

			if(vkImageView_depth)
			{
				vkDestroyImageView(vkDevice, vkImageView_depth, NULL);
				vkImageView_depth = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkImageView_depth is done\n");
			}

			if(vkDeviceMemory_depth)
			{
				vkFreeMemory(vkDevice, vkDeviceMemory_depth, NULL);
				vkDeviceMemory_depth = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkDeviceMemory_depth is done\n");
			}

			if(vkImage_depth)
			{

				vkDestroyImage(vkDevice, vkImage_depth, NULL);
				vkImage_depth = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkImage_depth is done\n");
			}

			for(uint32_t i =0; i < swapchainImageCount; i++)
			{
				vkDestroyImageView(vkDevice, swapChainImageView_array[i], NULL);
				FileIO("uninitialize(): vkDestroyImageView() is done\n");
			}

			if(swapChainImageView_array)
			{
				free(swapChainImageView_array);
				swapChainImageView_array = NULL;
				FileIO("uninitialize():swapChainImageView_array is freed\n");
			}

			if(swapChainImage_array)
			{
				free(swapChainImage_array);
				swapChainImage_array = NULL;
				FileIO("uninitialize():swapChainImage_array is freed\n");
			}

			vkDestroySwapchainKHR(vkDevice, vkSwapchainKHR, NULL);
			vkSwapchainKHR = VK_NULL_HANDLE;
			FileIO("uninitialize(): vkDestroySwapchainKHR() is done\n");

			vkDestroyDevice(vkDevice, NULL);
			vkDevice = VK_NULL_HANDLE;
			FileIO("uninitialize(): vkDestroyDevice() is done\n");
		}

		if(vkSurfaceKHR)
		{

			vkDestroySurfaceKHR(vkInstance, vkSurfaceKHR, NULL);
			vkSurfaceKHR = VK_NULL_HANDLE;
			FileIO("uninitialize(): vkDestroySurfaceKHR() sucedded\n");
		}

		if(vkDebugReportCallbackEXT && vkDestroyDebugReportCallbackEXT_fnptr)
		{
			vkDestroyDebugReportCallbackEXT_fnptr(vkInstance, vkDebugReportCallbackEXT, NULL);
			vkDebugReportCallbackEXT = VK_NULL_HANDLE;
			vkDestroyDebugReportCallbackEXT_fnptr = NULL;
		}

		if(vkInstance)
		{
			vkDestroyInstance(vkInstance, NULL);
			vkInstance = VK_NULL_HANDLE;
			FileIO("uninitialize(): vkDestroyInstance() sucedded\n");
		}

		FileIO("uninitialize()-> Program ended successfully.\n");
}

cudaError_t uninitialize_cuda(void)
{

	if(pos_CUDA)
	{
		cudaResult = cudaFree(pos_CUDA);
		if(cudaResult != CUDA_SUCCESS)
		{
			FileIO("uninitialize_cuda()-> cudaFree() failed\n");
			return cudaResult;
		}
		else
		{
			pos_CUDA = NULL;
			return cudaResult;
		}
	}

	if(cuExternalMemory_t)
	{
		cudaResult =  cudaDestroyExternalMemory(cuExternalMemory_t);
		if(cudaResult != CUDA_SUCCESS)
		{
			FileIO("uninitialize_cuda()-> cudaDestroyExternalMemory() failed\n");
			return cudaResult;
		}
		else
		{
			cuExternalMemory_t = NULL;
			return cudaResult;
		}
	}
	return cudaSuccess;
}

VkResult CreateVulkanInstance(void)
{

	VkResult FillInstanceExtensionNames(void);

	VkResult FillValidationLayerNames(void);
	VkResult CreateValidationCallbackFunction(void);

	VkResult vkResult = VK_SUCCESS;

	vkResult = FillInstanceExtensionNames();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateVulkanInstance(): FillInstanceExtensionNames()  function failed\n");
		return vkResult;
	}
	else
	{
		FileIO("CreateVulkanInstance(): FillInstanceExtensionNames() succedded\n");
	}

	if(bValidation == TRUE)
	{

		vkResult = FillValidationLayerNames();
		if (vkResult != VK_SUCCESS)
		{
			FileIO("CreateVulkanInstance(): FillValidationLayerNames()  function failed\n");
			return vkResult;
		}
		else
		{
			FileIO("CreateVulkanInstance(): FillValidationLayerNames() succedded\n");
		}
	}

	struct VkApplicationInfo vkApplicationInfo;
	memset((void*)&vkApplicationInfo, 0, sizeof(struct VkApplicationInfo));

	vkApplicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	vkApplicationInfo.pNext = NULL;
	vkApplicationInfo.pApplicationName = gpszAppName;
	vkApplicationInfo.applicationVersion = 1;
	vkApplicationInfo.pEngineName = gpszAppName;
	vkApplicationInfo.engineVersion = 1;

	vkApplicationInfo.apiVersion = VK_API_VERSION_1_4;

	struct VkInstanceCreateInfo vkInstanceCreateInfo;
	memset((void*)&vkInstanceCreateInfo, 0, sizeof(struct VkInstanceCreateInfo));

	vkInstanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	vkInstanceCreateInfo.pNext = NULL;
	vkInstanceCreateInfo.pApplicationInfo = &vkApplicationInfo;

	vkInstanceCreateInfo.enabledExtensionCount = enabledInstanceExtensionsCount;
	vkInstanceCreateInfo.ppEnabledExtensionNames = enabledInstanceExtensionNames_array;

	if(bValidation == TRUE)
	{
		vkInstanceCreateInfo.enabledLayerCount = enabledValidationLayerCount;
		vkInstanceCreateInfo.ppEnabledLayerNames = enabledValidationlayerNames_array;
	}
	else
	{
		vkInstanceCreateInfo.enabledLayerCount = 0;
		vkInstanceCreateInfo.ppEnabledLayerNames = NULL;
	}

	vkResult = vkCreateInstance(&vkInstanceCreateInfo, NULL, &vkInstance);
	if (vkResult == VK_ERROR_INCOMPATIBLE_DRIVER)
	{
		FileIO("CreateVulkanInstance(): vkCreateInstance() function failed due to incompatible driver with error code %d\n", vkResult);
		return vkResult;
	}
	else if (vkResult == VK_ERROR_EXTENSION_NOT_PRESENT)
	{
		FileIO("CreateVulkanInstance(): vkCreateInstance() function failed due to required extension not present with error code %d\n", vkResult);
		return vkResult;
	}
	else if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateVulkanInstance(): vkCreateInstance() function failed due to unknown reason with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateVulkanInstance(): vkCreateInstance() succedded\n");
	}

	if(bValidation)
	{

		vkResult = CreateValidationCallbackFunction();
		if (vkResult != VK_SUCCESS)
		{
			FileIO("CreateVulkanInstance(): CreateValidationCallbackFunction()  function failed\n");
			return vkResult;
		}
		else
		{
			FileIO("CreateVulkanInstance(): CreateValidationCallbackFunction() succedded\n");
		}
	}

	return vkResult;
}

VkResult FillInstanceExtensionNames(void)
{

	VkResult vkResult = VK_SUCCESS;

	uint32_t instanceExtensionCount = 0;

	vkResult = vkEnumerateInstanceExtensionProperties(NULL, &instanceExtensionCount, NULL);

	if (vkResult != VK_SUCCESS)
	{
		FileIO("FillInstanceExtensionNames(): First call to vkEnumerateInstanceExtensionProperties()  function failed\n");
		return vkResult;
	}
	else
	{
		FileIO("FillInstanceExtensionNames(): First call to vkEnumerateInstanceExtensionProperties() succedded\n");
	}

	VkExtensionProperties* vkExtensionProperties_array = NULL;
	vkExtensionProperties_array = (VkExtensionProperties*)malloc(sizeof(VkExtensionProperties) * instanceExtensionCount);
	if (vkExtensionProperties_array != NULL)
	{

	}
	else
	{

	}

	vkResult = vkEnumerateInstanceExtensionProperties(NULL, &instanceExtensionCount, vkExtensionProperties_array);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("FillInstanceExtensionNames(): Second call to vkEnumerateInstanceExtensionProperties()  function failed\n");
		return vkResult;
	}
	else
	{
		FileIO("FillInstanceExtensionNames(): Second call to vkEnumerateInstanceExtensionProperties() succedded\n");
	}

	char** instanceExtensionNames_array = NULL;
	instanceExtensionNames_array = (char**)malloc(sizeof(char*) * instanceExtensionCount);
	if (instanceExtensionNames_array != NULL)
	{

	}
	else
	{

	}

	for (uint32_t i =0; i < instanceExtensionCount; i++)
	{

		instanceExtensionNames_array[i] = (char*)malloc( sizeof(char) * (strlen(vkExtensionProperties_array[i].extensionName) + 1));
		memcpy(instanceExtensionNames_array[i], vkExtensionProperties_array[i].extensionName, (strlen(vkExtensionProperties_array[i].extensionName) + 1));
		FileIO("FillInstanceExtensionNames(): Vulkan Instance Extension Name = %s\n", instanceExtensionNames_array[i]);
	}

	if (vkExtensionProperties_array)
	{
		free(vkExtensionProperties_array);
		vkExtensionProperties_array = NULL;
	}

	VkBool32 vulkanSurfaceExtensionFound = VK_FALSE;
	VkBool32 vulkanWin32SurfaceExtensionFound = VK_FALSE;

	VkBool32 debugReportExtensionFound = VK_FALSE;

	for (uint32_t i = 0; i < instanceExtensionCount; i++)
	{
		if (strcmp(instanceExtensionNames_array[i], VK_KHR_SURFACE_EXTENSION_NAME) == 0)
		{
			vulkanSurfaceExtensionFound = VK_TRUE;
			enabledInstanceExtensionNames_array[enabledInstanceExtensionsCount++] = VK_KHR_SURFACE_EXTENSION_NAME;
		}

		if (strcmp(instanceExtensionNames_array[i], VK_KHR_WIN32_SURFACE_EXTENSION_NAME) == 0)
		{
			vulkanWin32SurfaceExtensionFound = VK_TRUE;
			enabledInstanceExtensionNames_array[enabledInstanceExtensionsCount++] = VK_KHR_WIN32_SURFACE_EXTENSION_NAME;
		}

		if (strcmp(instanceExtensionNames_array[i], VK_EXT_DEBUG_REPORT_EXTENSION_NAME) == 0)
		{
			debugReportExtensionFound = VK_TRUE;
			if(bValidation == TRUE)
			{
				enabledInstanceExtensionNames_array[enabledInstanceExtensionsCount++] = VK_EXT_DEBUG_REPORT_EXTENSION_NAME;
			}
			else
			{

			}
		}
	}

	for (uint32_t i =0 ; i < instanceExtensionCount; i++)
	{
		free(instanceExtensionNames_array[i]);
	}
	free(instanceExtensionNames_array);

	if (vulkanSurfaceExtensionFound == VK_FALSE)
	{

		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		FileIO("FillInstanceExtensionNames(): VK_KHR_SURFACE_EXTENSION_NAME not found\n");
		return vkResult;
	}
	else
	{
		FileIO("FillInstanceExtensionNames(): VK_KHR_SURFACE_EXTENSION_NAME is found\n");
	}

	if (vulkanWin32SurfaceExtensionFound == VK_FALSE)
	{

		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		FileIO("FillInstanceExtensionNames(): VK_KHR_WIN32_SURFACE_EXTENSION_NAME not found\n");
		return vkResult;
	}
	else
	{
		FileIO("FillInstanceExtensionNames(): VK_KHR_WIN32_SURFACE_EXTENSION_NAME is found\n");
	}

	if (debugReportExtensionFound == VK_FALSE)
	{
		if(bValidation == TRUE)
		{

			vkResult = VK_ERROR_INITIALIZATION_FAILED;
			FileIO("FillInstanceExtensionNames(): Validation is ON, but required VK_EXT_DEBUG_REPORT_EXTENSION_NAME is not supported\n");
			return vkResult;
		}
		else
		{
			FileIO("FillInstanceExtensionNames(): Validation is OFF, but VK_EXT_DEBUG_REPORT_EXTENSION_NAME is not supported\n");
		}
	}
	else
	{
		if(bValidation == TRUE)
		{

			FileIO("FillInstanceExtensionNames(): Validation is ON, but required VK_EXT_DEBUG_REPORT_EXTENSION_NAME is also supported\n");

		}
		else
		{
			FileIO("FillInstanceExtensionNames(): Validation is OFF, but VK_EXT_DEBUG_REPORT_EXTENSION_NAME is also supported\n");
		}
	}

	for (uint32_t i = 0; i < enabledInstanceExtensionsCount; i++)
	{
		FileIO("FillInstanceExtensionNames(): Enabled Vulkan Instance Extension Name = %s\n", enabledInstanceExtensionNames_array[i]);
	}

	return vkResult;
}

VkResult FillValidationLayerNames(void)
{

	VkResult vkResult = VK_SUCCESS;

	uint32_t validationLayerCount = 0;

	vkResult = vkEnumerateInstanceLayerProperties(&validationLayerCount, NULL);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("FillValidationLayerNames(): First call to vkEnumerateInstanceLayerProperties()  function failed\n");
		return vkResult;
	}
	else
	{
		FileIO("FillValidationLayerNames(): First call to vkEnumerateInstanceLayerProperties() succedded\n");
	}

	VkLayerProperties* vkLayerProperties_array = NULL;
	vkLayerProperties_array = (VkLayerProperties*)malloc(sizeof(VkLayerProperties) * validationLayerCount);
	if (vkLayerProperties_array != NULL)
	{

	}
	else
	{

	}

	vkResult = vkEnumerateInstanceLayerProperties(&validationLayerCount, vkLayerProperties_array);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("FillValidationLayerNames(): Second call to vkEnumerateInstanceLayerProperties()  function failed\n");
		return vkResult;
	}
	else
	{
		FileIO("FillValidationLayerNames(): Second call to vkEnumerateInstanceLayerProperties() succedded\n");
	}

	char** validationLayerNames_array = NULL;
	validationLayerNames_array = (char**)malloc(sizeof(char*) * validationLayerCount);
	if (validationLayerNames_array != NULL)
	{

	}
	else
	{

	}

	for (uint32_t i =0; i < validationLayerCount; i++)
	{

		validationLayerNames_array[i] = (char*)malloc( sizeof(char) * (strlen(vkLayerProperties_array[i].layerName) + 1));
		memcpy(validationLayerNames_array[i], vkLayerProperties_array[i].layerName, (strlen(vkLayerProperties_array[i].layerName) + 1));
		FileIO("FillValidationLayerNames(): Vulkan Instance Layer Name = %s\n", validationLayerNames_array[i]);
	}

	if (vkLayerProperties_array)
	{
		free(vkLayerProperties_array);
		vkLayerProperties_array = NULL;
	}

	VkBool32 validationLayerFound = VK_FALSE;

	for (uint32_t i = 0; i < validationLayerCount; i++)
	{
		if (strcmp(validationLayerNames_array[i], "VK_LAYER_KHRONOS_validation") == 0)
		{
			validationLayerFound = VK_TRUE;
			enabledValidationlayerNames_array[enabledValidationLayerCount++] = "VK_LAYER_KHRONOS_validation";
		}
	}

	for (uint32_t i =0 ; i < validationLayerCount; i++)
	{
		free(validationLayerNames_array[i]);
	}
	free(validationLayerNames_array);

	if(validationLayerFound == VK_FALSE)
	{

		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		FileIO("FillValidationLayerNames(): VK_LAYER_KHRONOS_validation not supported\n");
		return vkResult;
	}
	else
	{
		FileIO("FillValidationLayerNames(): VK_LAYER_KHRONOS_validation is supported\n");
	}

	for (uint32_t i = 0; i < enabledValidationLayerCount; i++)
	{
		FileIO("FillValidationLayerNames(): Enabled Vulkan Validation Layer Name = %s\n", enabledValidationlayerNames_array[i]);
	}

	return vkResult;
}

VkResult CreateValidationCallbackFunction(void)
{

	VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallback(VkDebugReportFlagsEXT, VkDebugReportObjectTypeEXT, uint64_t, size_t, int32_t, const char*, const char*, void*);

	PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT_fnptr = NULL;

	VkResult vkResult = VK_SUCCESS;

	vkCreateDebugReportCallbackEXT_fnptr = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(vkInstance, "vkCreateDebugReportCallbackEXT");
	if(vkCreateDebugReportCallbackEXT_fnptr == NULL)
	{
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		FileIO("CreateValidationCallbackFunction(): vkGetInstanceProcAddr() failed to get function pointer for vkCreateDebugReportCallbackEXT\n");
		return vkResult;
	}
	else
	{
		FileIO("CreateValidationCallbackFunction(): vkGetInstanceProcAddr() suceeded getting function pointer for vkCreateDebugReportCallbackEXT\n");
	}

	vkDestroyDebugReportCallbackEXT_fnptr = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(vkInstance, "vkDestroyDebugReportCallbackEXT");
	if(vkDestroyDebugReportCallbackEXT_fnptr == NULL)
	{
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		FileIO("CreateValidationCallbackFunction(): vkGetInstanceProcAddr() failed to get function pointer for vkDestroyDebugReportCallbackEXT\n");
		return vkResult;
	}
	else
	{
		FileIO("CreateValidationCallbackFunction(): vkGetInstanceProcAddr() suceeded getting function pointer for vkDestroyDebugReportCallbackEXT\n");
	}

	VkDebugReportCallbackCreateInfoEXT vkDebugReportCallbackCreateInfoEXT ;
	memset((void*)&vkDebugReportCallbackCreateInfoEXT, 0, sizeof(VkDebugReportCallbackCreateInfoEXT));

	vkDebugReportCallbackCreateInfoEXT.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
	vkDebugReportCallbackCreateInfoEXT.pNext = NULL;
	vkDebugReportCallbackCreateInfoEXT.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT|VK_DEBUG_REPORT_WARNING_BIT_EXT|VK_DEBUG_REPORT_INFORMATION_BIT_EXT|VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT|VK_DEBUG_REPORT_DEBUG_BIT_EXT;
	vkDebugReportCallbackCreateInfoEXT.pfnCallback = debugReportCallback;
	vkDebugReportCallbackCreateInfoEXT.pUserData = NULL;

	vkResult = vkCreateDebugReportCallbackEXT_fnptr(vkInstance, &vkDebugReportCallbackCreateInfoEXT, NULL, &vkDebugReportCallbackEXT);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateValidationCallbackFunction(): vkCreateDebugReportCallbackEXT_fnptr()  function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateValidationCallbackFunction(): vkCreateDebugReportCallbackEXT_fnptr() succedded\n");
	}

	return vkResult;
}

VkResult GetSupportedSurface(void)
{

	VkResult vkResult = VK_SUCCESS;

	VkWin32SurfaceCreateInfoKHR vkWin32SurfaceCreateInfoKHR;
	memset((void*)&vkWin32SurfaceCreateInfoKHR, 0 , sizeof(struct VkWin32SurfaceCreateInfoKHR));

	vkWin32SurfaceCreateInfoKHR.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
	vkWin32SurfaceCreateInfoKHR.pNext = NULL;
	vkWin32SurfaceCreateInfoKHR.flags = 0;
	vkWin32SurfaceCreateInfoKHR.hinstance = (HINSTANCE)GetWindowLongPtr(ghwnd, GWLP_HINSTANCE);
	vkWin32SurfaceCreateInfoKHR.hwnd = ghwnd;

	vkResult = vkCreateWin32SurfaceKHR(vkInstance, &vkWin32SurfaceCreateInfoKHR, NULL, &vkSurfaceKHR);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("GetSupportedSurface(): vkCreateWin32SurfaceKHR()  function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("GetSupportedSurface(): vkCreateWin32SurfaceKHR() succedded\n");
	}

	return vkResult;
}

VkResult GetPhysicalDevice(void)
{

	VkResult vkResult = VK_SUCCESS;

	vkResult = vkEnumeratePhysicalDevices(vkInstance, &physicalDeviceCount, NULL);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("GetPhysicalDevice(): vkEnumeratePhysicalDevices() first call failed with error code %d\n", vkResult);
		return vkResult;
	}
	else if (physicalDeviceCount == 0)
	{
		FileIO("GetPhysicalDevice(): vkEnumeratePhysicalDevices() first call resulted in 0 physical devices\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	else
	{
		FileIO("GetPhysicalDevice(): vkEnumeratePhysicalDevices() first call succedded\n");
	}

	vkPhysicalDevice_array = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * physicalDeviceCount);

	vkResult = vkEnumeratePhysicalDevices(vkInstance, &physicalDeviceCount, vkPhysicalDevice_array);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("GetPhysicalDevice(): vkEnumeratePhysicalDevices() second call failed with error code %d\n", vkResult);
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	else
	{
		FileIO("GetPhysicalDevice(): vkEnumeratePhysicalDevices() second call succedded\n");
	}

	VkBool32 bFound = VK_FALSE;
	for(uint32_t i = 0; i < physicalDeviceCount; i++)
	{

		uint32_t quequeCount = UINT32_MAX;

		vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice_array[i], &quequeCount, NULL);

		struct VkQueueFamilyProperties *vkQueueFamilyProperties_array = NULL;
		vkQueueFamilyProperties_array = (struct VkQueueFamilyProperties*) malloc(sizeof(struct VkQueueFamilyProperties) * quequeCount);

		vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice_array[i], &quequeCount, vkQueueFamilyProperties_array);

		VkBool32 *isQuequeSurfaceSupported_array = NULL;
		isQuequeSurfaceSupported_array = (VkBool32*) malloc(sizeof(VkBool32) * quequeCount);

		for(uint32_t j =0; j < quequeCount ; j++)
		{

			vkResult = vkGetPhysicalDeviceSurfaceSupportKHR(vkPhysicalDevice_array[i], j, vkSurfaceKHR, &isQuequeSurfaceSupported_array[j]);
		}

		for(uint32_t j =0; j < quequeCount ; j++)
		{

			if(vkQueueFamilyProperties_array[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{

				if(isQuequeSurfaceSupported_array[j] == VK_TRUE)
				{
					vkPhysicalDevice_selected = vkPhysicalDevice_array[i];
					graphicsQuequeFamilyIndex_selected = j;
					bFound = VK_TRUE;
					break;
				}
			}
		}

		if(isQuequeSurfaceSupported_array)
		{
			free(isQuequeSurfaceSupported_array);
			isQuequeSurfaceSupported_array = NULL;
			FileIO("GetPhysicalDevice(): succedded to free isQuequeSurfaceSupported_array\n");
		}

		if(vkQueueFamilyProperties_array)
		{
			free(vkQueueFamilyProperties_array);
			vkQueueFamilyProperties_array = NULL;
			FileIO("GetPhysicalDevice(): succedded to free vkQueueFamilyProperties_array\n");
		}

		if(bFound == VK_TRUE)
		{
			break;
		}
	}

	if(bFound == VK_TRUE)
	{
		FileIO("GetPhysicalDevice(): GetPhysicalDevice() suceeded to select required physical device with graphics enabled\n");

	}
	else
	{
		FileIO("GetPhysicalDevice(): GetPhysicalDevice() failed to obtain graphics supported physical device\n");

		if(vkPhysicalDevice_array)
		{
			free(vkPhysicalDevice_array);
			vkPhysicalDevice_array = NULL;
			FileIO("GetPhysicalDevice(): succedded to free vkPhysicalDevice_array\n");
		}

		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}

	memset((void*)&vkPhysicalDeviceMemoryProperties, 0, sizeof(struct VkPhysicalDeviceMemoryProperties));

	vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice_selected, &vkPhysicalDeviceMemoryProperties);

	VkPhysicalDeviceFeatures vkPhysicalDeviceFeatures;
	memset((void*)&vkPhysicalDeviceFeatures, 0, sizeof(VkPhysicalDeviceFeatures));
	vkGetPhysicalDeviceFeatures(vkPhysicalDevice_selected, &vkPhysicalDeviceFeatures);

	if(vkPhysicalDeviceFeatures.tessellationShader)
	{
		FileIO("GetPhysicalDevice(): Supported physical device supports tessellation shader\n");
	}
	else
	{
		FileIO("GetPhysicalDevice(): Supported physical device does not support tessellation shader\n");
	}

	if(vkPhysicalDeviceFeatures.geometryShader)
	{
		FileIO("GetPhysicalDevice(): Supported physical device supports geometry shader\n");
	}
	else
	{
		FileIO("GetPhysicalDevice(): Supported physical device does not support geometry shader\n");
	}

	return vkResult;
}

VkResult PrintVulkanInfo(void)
{
	VkResult vkResult = VK_SUCCESS;

	FileIO("************************* Shree Ganesha******************************\n");

	for(uint32_t i = 0; i < physicalDeviceCount; i++)
	{

		VkPhysicalDeviceProperties vkPhysicalDeviceProperties;
		memset((void*)&vkPhysicalDeviceProperties, 0, sizeof(VkPhysicalDeviceProperties));
		vkGetPhysicalDeviceProperties(vkPhysicalDevice_array[i], &vkPhysicalDeviceProperties );

		uint32_t majorVersion = VK_API_VERSION_MAJOR(vkPhysicalDeviceProperties.apiVersion);
		uint32_t minorVersion = VK_API_VERSION_MINOR(vkPhysicalDeviceProperties.apiVersion);
		uint32_t patchVersion = VK_API_VERSION_PATCH(vkPhysicalDeviceProperties.apiVersion);

		FileIO("apiVersion = %d.%d.%d\n", majorVersion, minorVersion, patchVersion);

		FileIO("deviceName = %s\n", vkPhysicalDeviceProperties.deviceName);

		switch(vkPhysicalDeviceProperties.deviceType)
		{
			case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
				FileIO("deviceType = Integrated GPU (iGPU)\n");
			break;

			case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
				FileIO("deviceType = Discrete GPU (dGPU)\n");
			break;

			case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
				FileIO("deviceType = Virtual GPU (vGPU)\n");
			break;

			case VK_PHYSICAL_DEVICE_TYPE_CPU:
				FileIO("deviceType = CPU\n");
			break;

			case VK_PHYSICAL_DEVICE_TYPE_OTHER:
				FileIO("deviceType = Other\n");
			break;

			default:
				FileIO("deviceType = UNKNOWN\n");
			break;
		}

		FileIO("vendorID = 0x%04x\n", vkPhysicalDeviceProperties.vendorID);

		FileIO("deviceID = 0x%04x\n", vkPhysicalDeviceProperties.deviceID);
	}

	if(vkPhysicalDevice_array)
	{
		free(vkPhysicalDevice_array);
		vkPhysicalDevice_array = NULL;
		FileIO("PrintVkInfo(): succedded to free vkPhysicalDevice_array\n");
	}

	return vkResult;
}

VkResult FillDeviceExtensionNames(void)
{

	VkResult vkResult = VK_SUCCESS;

	uint32_t deviceExtensionCount = 0;

	vkResult = vkEnumerateDeviceExtensionProperties(vkPhysicalDevice_selected, NULL, &deviceExtensionCount, NULL );
	if (vkResult != VK_SUCCESS)
	{
		FileIO("FillDeviceExtensionNames(): First call to vkEnumerateDeviceExtensionProperties()  function failed\n");
		return vkResult;
	}
	else
	{
		FileIO("FillDeviceExtensionNames(): First call to vkEnumerateDeviceExtensionProperties() succedded and returned %u count\n", deviceExtensionCount);
	}

	VkExtensionProperties* vkExtensionProperties_array = NULL;
	vkExtensionProperties_array = (VkExtensionProperties*)malloc(sizeof(VkExtensionProperties) * deviceExtensionCount);
	if (vkExtensionProperties_array != NULL)
	{

	}
	else
	{

	}

	vkResult = vkEnumerateDeviceExtensionProperties(vkPhysicalDevice_selected, NULL, &deviceExtensionCount, vkExtensionProperties_array);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("FillDeviceExtensionNames(): Second call to vkEnumerateDeviceExtensionProperties()  function failed\n");
		return vkResult;
	}
	else
	{
		FileIO("FillDeviceExtensionNames(): Second call to vkEnumerateDeviceExtensionProperties() succedded\n");
	}

	char** deviceExtensionNames_array = NULL;
	deviceExtensionNames_array = (char**)malloc(sizeof(char*) * deviceExtensionCount);
	if (deviceExtensionNames_array != NULL)
	{

	}
	else
	{

	}

	for (uint32_t i =0; i < deviceExtensionCount; i++)
	{

		deviceExtensionNames_array[i] = (char*)malloc( sizeof(char) * (strlen(vkExtensionProperties_array[i].extensionName) + 1));
		memcpy(deviceExtensionNames_array[i], vkExtensionProperties_array[i].extensionName, (strlen(vkExtensionProperties_array[i].extensionName) + 1));
		FileIO("FillDeviceExtensionNames(): Vulkan Device Extension Name = %s\n", deviceExtensionNames_array[i]);
	}

	if (vkExtensionProperties_array)
	{
		free(vkExtensionProperties_array);
		vkExtensionProperties_array = NULL;
	}

	VkBool32 vulkanSwapchainExtensionFound = VK_FALSE;
	VkBool32 vulkanExternalMemoryWin32ExtensionFound = VK_FALSE;
	for (uint32_t i = 0; i < deviceExtensionCount; i++)
	{
		if (strcmp(deviceExtensionNames_array[i], VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0)
		{
			vulkanSwapchainExtensionFound = VK_TRUE;
			enabledDeviceExtensionNames_array[enabledDeviceExtensionsCount++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
		}

		if (strcmp(deviceExtensionNames_array[i], VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME) == 0)
		{
			vulkanExternalMemoryWin32ExtensionFound = VK_TRUE;
			enabledDeviceExtensionNames_array[enabledDeviceExtensionsCount++] = VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME;
		}
	}

	for (uint32_t i =0 ; i < deviceExtensionCount; i++)
	{
		free(deviceExtensionNames_array[i]);
	}
	free(deviceExtensionNames_array);

	if (vulkanSwapchainExtensionFound == VK_FALSE)
	{

		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		FileIO("FillDeviceExtensionNames(): VK_KHR_SWAPCHAIN_EXTENSION_NAME not found\n");
		return vkResult;
	}
	else
	{
		FileIO("FillDeviceExtensionNames(): VK_KHR_SWAPCHAIN_EXTENSION_NAME is found\n");
	}

	if (vulkanExternalMemoryWin32ExtensionFound == VK_FALSE)
	{

		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		FileIO("FillDeviceExtensionNames(): VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME not found\n");
		return vkResult;
	}
	else
	{
		FileIO("FillDeviceExtensionNames(): VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME is found\n");
	}

	for (uint32_t i = 0; i < enabledDeviceExtensionsCount; i++)
	{
		FileIO("FillDeviceExtensionNames(): Enabled Vulkan Device Extension Name = %s\n", enabledDeviceExtensionNames_array[i]);
	}

	return vkResult;
}

VkResult CreateVulKanDevice(void)
{

	VkResult FillDeviceExtensionNames(void);

	VkResult vkResult = VK_SUCCESS;

	vkResult = FillDeviceExtensionNames();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateVulKanDevice(): FillDeviceExtensionNames()  function failed\n");
		return vkResult;
	}
	else
	{
		FileIO("CreateVulKanDevice(): FillDeviceExtensionNames() succedded\n");
	}

	float queuePriorities[1];
	queuePriorities[0] = 1.0f;
	VkDeviceQueueCreateInfo vkDeviceQueueCreateInfo;
	memset(&vkDeviceQueueCreateInfo, 0, sizeof(VkDeviceQueueCreateInfo));

	vkDeviceQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	vkDeviceQueueCreateInfo.pNext = NULL;
	vkDeviceQueueCreateInfo.flags = 0;
	vkDeviceQueueCreateInfo.queueFamilyIndex = graphicsQuequeFamilyIndex_selected;
	vkDeviceQueueCreateInfo.queueCount = 1;
	vkDeviceQueueCreateInfo.pQueuePriorities = queuePriorities;

	VkDeviceCreateInfo vkDeviceCreateInfo;
	memset(&vkDeviceCreateInfo, 0, sizeof(VkDeviceCreateInfo));

	VkPhysicalDeviceFeatures enabledFeatures;
	memset(&enabledFeatures, 0, sizeof(VkPhysicalDeviceFeatures));
	enabledFeatures.tessellationShader = VK_TRUE;
	enabledFeatures.fillModeNonSolid = VK_TRUE;

	vkDeviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	vkDeviceCreateInfo.pNext = NULL;
	vkDeviceCreateInfo.flags = 0;
	vkDeviceCreateInfo.enabledExtensionCount = enabledDeviceExtensionsCount;
	vkDeviceCreateInfo.ppEnabledExtensionNames = enabledDeviceExtensionNames_array;
	vkDeviceCreateInfo.enabledLayerCount = 0;
	vkDeviceCreateInfo.ppEnabledLayerNames = NULL;
	vkDeviceCreateInfo.pEnabledFeatures = &enabledFeatures;
	vkDeviceCreateInfo.queueCreateInfoCount = 1;
	vkDeviceCreateInfo.pQueueCreateInfos = &vkDeviceQueueCreateInfo;

	vkResult = vkCreateDevice(vkPhysicalDevice_selected, &vkDeviceCreateInfo, NULL, &vkDevice);
	if(vkResult != VK_SUCCESS)
	{
		FileIO("CreateVulKanDevice(): vkCreateDevice()  function failed\n");
		return vkResult;
	}
	else
	{
		FileIO("CreateVulKanDevice(): vkCreateDevice() succedded\n");
	}

	return vkResult;
}

void GetDeviceQueque(void)
{

	vkGetDeviceQueue(vkDevice, graphicsQuequeFamilyIndex_selected, 0, &vkQueue);
	if(vkQueue == VK_NULL_HANDLE)
	{
		FileIO("GetDeviceQueque(): vkGetDeviceQueue() returned NULL for vkQueue\n");
		return;
	}
	else
	{
		FileIO("GetDeviceQueque(): vkGetDeviceQueue() succedded\n");
	}
}

VkResult getPhysicalDeviceSurfaceFormatAndColorSpace(void)
{

	VkResult vkResult = VK_SUCCESS;

	uint32_t FormatCount = 0;

	vkResult = vkGetPhysicalDeviceSurfaceFormatsKHR(vkPhysicalDevice_selected, vkSurfaceKHR, &FormatCount, NULL);
	if(vkResult != VK_SUCCESS)
	{
		FileIO("getPhysicalDeviceSurfaceFormatAndColorSpace(): First call to vkGetPhysicalDeviceSurfaceFormatsKHR() failed\n");
		return vkResult;
	}
	else if(FormatCount == 0)
	{
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		FileIO("vkGetPhysicalDeviceSurfaceFormatsKHR():: First call to vkGetPhysicalDeviceSurfaceFormatsKHR() returned FormatCount as 0\n");
		return vkResult;
	}
	else
	{
		FileIO("getPhysicalDeviceSurfaceFormatAndColorSpace(): First call to vkGetPhysicalDeviceSurfaceFormatsKHR() succedded\n");
	}

	VkSurfaceFormatKHR *vkSurfaceFormatKHR_array = (VkSurfaceFormatKHR*)malloc(FormatCount * sizeof(VkSurfaceFormatKHR));

	vkResult = vkGetPhysicalDeviceSurfaceFormatsKHR(vkPhysicalDevice_selected, vkSurfaceKHR, &FormatCount, vkSurfaceFormatKHR_array);
	if(vkResult != VK_SUCCESS)
	{
		FileIO("getPhysicalDeviceSurfaceFormatAndColorSpace(): Second call to vkGetPhysicalDeviceSurfaceFormatsKHR()  function failed\n");
		return vkResult;
	}
	else
	{
		FileIO("getPhysicalDeviceSurfaceFormatAndColorSpace():  Second call to vkGetPhysicalDeviceSurfaceFormatsKHR() succedded\n");
	}

	if( (1 == FormatCount) && (vkSurfaceFormatKHR_array[0].format == VK_FORMAT_UNDEFINED) )
	{
		vkFormat_color = VK_FORMAT_B8G8R8A8_UNORM;
	}
	else
	{
		vkFormat_color = vkSurfaceFormatKHR_array[0].format;
	}

	vkColorSpaceKHR = vkSurfaceFormatKHR_array[0].colorSpace;

	if(vkSurfaceFormatKHR_array)
	{
		FileIO("getPhysicalDeviceSurfaceFormatAndColorSpace(): vkSurfaceFormatKHR_array is freed\n");
		free(vkSurfaceFormatKHR_array);
		vkSurfaceFormatKHR_array = NULL;
	}

	return vkResult;
}

VkResult getPhysicalDevicePresentMode(void)
{

	VkResult vkResult = VK_SUCCESS;

	uint32_t presentModeCount = 0;

	vkResult = vkGetPhysicalDeviceSurfacePresentModesKHR(vkPhysicalDevice_selected, vkSurfaceKHR, &presentModeCount, NULL);
	if(vkResult != VK_SUCCESS)
	{
		FileIO("getPhysicalDevicePresentMode(): First call to vkGetPhysicalDeviceSurfaceFormatsKHR() failed\n");
		return vkResult;
	}
	else if(presentModeCount == 0)
	{
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		FileIO("getPhysicalDevicePresentMode():: First call to vkGetPhysicalDeviceSurfaceFormatsKHR() returned presentModeCount as 0\n");
		return vkResult;
	}
	else
	{
		FileIO("getPhysicalDevicePresentMode(): First call to vkGetPhysicalDeviceSurfaceFormatsKHR() succedded\n");
	}

	VkPresentModeKHR  *vkPresentModeKHR_array = (VkPresentModeKHR*)malloc(presentModeCount * sizeof(VkPresentModeKHR));

	vkResult = vkGetPhysicalDeviceSurfacePresentModesKHR(vkPhysicalDevice_selected, vkSurfaceKHR, &presentModeCount, vkPresentModeKHR_array);
	if(vkResult != VK_SUCCESS)
	{
		FileIO("getPhysicalDevicePresentMode(): Second call to vkGetPhysicalDeviceSurfacePresentModesKHR()  function failed\n");
		return vkResult;
	}
	else
	{
		FileIO("getPhysicalDevicePresentMode():  Second call to vkGetPhysicalDeviceSurfacePresentModesKHR() succedded\n");
	}

	for(uint32_t i=0; i < presentModeCount; i++)
	{
		if(vkPresentModeKHR_array[i] == VK_PRESENT_MODE_MAILBOX_KHR)
		{
			vkPresentModeKHR = VK_PRESENT_MODE_MAILBOX_KHR;
			break;
		}
	}

	if(vkPresentModeKHR != VK_PRESENT_MODE_MAILBOX_KHR)
	{
		vkPresentModeKHR = VK_PRESENT_MODE_FIFO_KHR;
	}

	FileIO("getPhysicalDevicePresentMode(): vkPresentModeKHR is %d\n", vkPresentModeKHR);

	if(vkPresentModeKHR_array)
	{
		FileIO("getPhysicalDevicePresentMode(): vkPresentModeKHR_array is freed\n");
		free(vkPresentModeKHR_array);
		vkPresentModeKHR_array = NULL;
	}

	return vkResult;
}

VkResult CreateSwapChain(VkBool32 vsync)
{

	VkResult getPhysicalDeviceSurfaceFormatAndColorSpace(void);
	VkResult getPhysicalDevicePresentMode(void);

	VkResult vkResult = VK_SUCCESS;

	vkResult = getPhysicalDeviceSurfaceFormatAndColorSpace();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateSwapChain(): getPhysicalDeviceSurfaceFormatAndColorSpace() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateSwapChain(): getPhysicalDeviceSurfaceFormatAndColorSpace() succedded\n");
	}

	VkSurfaceCapabilitiesKHR vkSurfaceCapabilitiesKHR;
	memset((void*)&vkSurfaceCapabilitiesKHR, 0, sizeof(VkSurfaceCapabilitiesKHR));
	vkResult = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vkPhysicalDevice_selected, vkSurfaceKHR, &vkSurfaceCapabilitiesKHR);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateSwapChain(): vkGetPhysicalDeviceSurfaceCapabilitiesKHR() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateSwapChain(): vkGetPhysicalDeviceSurfaceCapabilitiesKHR() succedded\n");
	}

	uint32_t testingNumerOfSwapChainImages = vkSurfaceCapabilitiesKHR.minImageCount + 1;
	uint32_t desiredNumerOfSwapChainImages = 0;
	if( (vkSurfaceCapabilitiesKHR.maxImageCount > 0) && (vkSurfaceCapabilitiesKHR.maxImageCount < testingNumerOfSwapChainImages) )
	{
		desiredNumerOfSwapChainImages = vkSurfaceCapabilitiesKHR.maxImageCount;
	}
	else
	{
		desiredNumerOfSwapChainImages = vkSurfaceCapabilitiesKHR.minImageCount;
	}

	memset((void*)&vkExtent2D_SwapChain, 0 , sizeof(VkExtent2D));
	if(vkSurfaceCapabilitiesKHR.currentExtent.width != UINT32_MAX)
	{
		vkExtent2D_SwapChain.width = vkSurfaceCapabilitiesKHR.currentExtent.width;
		vkExtent2D_SwapChain.height = vkSurfaceCapabilitiesKHR.currentExtent.height;
		FileIO("CreateSwapChain(): Swapchain Image Width x SwapChain  Image Height = %d X %d\n", vkExtent2D_SwapChain.width, vkExtent2D_SwapChain.height);
	}
	else
	{
		vkExtent2D_SwapChain.width = vkSurfaceCapabilitiesKHR.currentExtent.width;
		vkExtent2D_SwapChain.height = vkSurfaceCapabilitiesKHR.currentExtent.height;
		FileIO("CreateSwapChain(): Swapchain Image Width x SwapChain  Image Height = %d X %d\n", vkExtent2D_SwapChain.width, vkExtent2D_SwapChain.height);

		VkExtent2D vkExtent2D;
		memset((void*)&vkExtent2D, 0, sizeof(VkExtent2D));
		vkExtent2D.width = (uint32_t)winWidth;
		vkExtent2D.height = (uint32_t)winHeight;

		vkExtent2D_SwapChain.width = glm::max(vkSurfaceCapabilitiesKHR.minImageExtent.width, glm::min(vkSurfaceCapabilitiesKHR.maxImageExtent.width, vkExtent2D.width));
		vkExtent2D_SwapChain.height = glm::max(vkSurfaceCapabilitiesKHR.minImageExtent.height, glm::min(vkSurfaceCapabilitiesKHR.maxImageExtent.height, vkExtent2D.height));
		FileIO("CreateSwapChain(): Swapchain Image Width x SwapChain  Image Height = %d X %d\n", vkExtent2D_SwapChain.width, vkExtent2D_SwapChain.height);
	}

	VkImageUsageFlagBits vkImageUsageFlagBits = (VkImageUsageFlagBits) (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

	VkSurfaceTransformFlagBitsKHR vkSurfaceTransformFlagBitsKHR;
	if(vkSurfaceCapabilitiesKHR.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
	{
		vkSurfaceTransformFlagBitsKHR = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	}
	else
	{
		vkSurfaceTransformFlagBitsKHR = vkSurfaceCapabilitiesKHR.currentTransform;
	}

	vkResult = getPhysicalDevicePresentMode();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateSwapChain(): getPhysicalDevicePresentMode() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateSwapChain(): getPhysicalDevicePresentMode() succedded\n");
	}

	struct VkSwapchainCreateInfoKHR vkSwapchainCreateInfoKHR;
	memset((void*)&vkSwapchainCreateInfoKHR, 0, sizeof(struct VkSwapchainCreateInfoKHR));
	vkSwapchainCreateInfoKHR.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	vkSwapchainCreateInfoKHR.pNext = NULL;
	vkSwapchainCreateInfoKHR.flags = 0;
	vkSwapchainCreateInfoKHR.surface = vkSurfaceKHR;
	vkSwapchainCreateInfoKHR.minImageCount = desiredNumerOfSwapChainImages;
	vkSwapchainCreateInfoKHR.imageFormat = vkFormat_color;
	vkSwapchainCreateInfoKHR.imageColorSpace = vkColorSpaceKHR;
	vkSwapchainCreateInfoKHR.imageExtent.width = vkExtent2D_SwapChain.width;
	vkSwapchainCreateInfoKHR.imageExtent.height = vkExtent2D_SwapChain.height;
	vkSwapchainCreateInfoKHR.imageUsage = vkImageUsageFlagBits;
	vkSwapchainCreateInfoKHR.preTransform = vkSurfaceTransformFlagBitsKHR;
	vkSwapchainCreateInfoKHR.imageArrayLayers = 1;
	vkSwapchainCreateInfoKHR.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	vkSwapchainCreateInfoKHR.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	vkSwapchainCreateInfoKHR.presentMode = vkPresentModeKHR;
	vkSwapchainCreateInfoKHR.clipped = VK_TRUE;

	vkResult = vkCreateSwapchainKHR(vkDevice, &vkSwapchainCreateInfoKHR, NULL, &vkSwapchainKHR);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateSwapChain(): vkCreateSwapchainKHR() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateSwapChain(): vkCreateSwapchainKHR() succedded\n");
	}

	return vkResult;
}

VkResult CreateImagesAndImageViews(void)
{

	VkResult GetSupportedDepthFormat(void);

	VkResult vkResult = VK_SUCCESS;

	vkResult = vkGetSwapchainImagesKHR(vkDevice, vkSwapchainKHR, &swapchainImageCount, NULL);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateImagesAndImageViews(): first call to vkGetSwapchainImagesKHR() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else if(swapchainImageCount == 0)
	{
		vkResult = vkResult = VK_ERROR_INITIALIZATION_FAILED;
		FileIO("CreateImagesAndImageViews(): first call to vkGetSwapchainImagesKHR() function returned swapchain Image Count as 0\n");
		return vkResult;
	}
	else
	{
		FileIO("CreateImagesAndImageViews(): first call to vkGetSwapchainImagesKHR() succedded with swapchainImageCount as %d\n", swapchainImageCount);
	}

	swapChainImage_array = (VkImage*)malloc(sizeof(VkImage) * swapchainImageCount);
	if(swapChainImage_array == NULL)
	{
			FileIO("CreateImagesAndImageViews(): swapChainImage_array is NULL. malloc() failed\n");
	}

	vkResult = vkGetSwapchainImagesKHR(vkDevice, vkSwapchainKHR, &swapchainImageCount, swapChainImage_array);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateImagesAndImageViews(): second call to vkGetSwapchainImagesKHR() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateImagesAndImageViews(): second call to vkGetSwapchainImagesKHR() succedded with swapchainImageCount as %d\n", swapchainImageCount);
	}

	swapChainImageView_array = (VkImageView*)malloc(sizeof(VkImageView) * swapchainImageCount);
	if(swapChainImageView_array == NULL)
	{
			FileIO("CreateImagesAndImageViews(): swapChainImageView_array is NULL. malloc() failed\n");
	}

	VkImageViewCreateInfo vkImageViewCreateInfo;
	memset((void*)&vkImageViewCreateInfo, 0, sizeof(VkImageViewCreateInfo));

	vkImageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	vkImageViewCreateInfo.pNext = NULL;
	vkImageViewCreateInfo.flags = 0;

	vkImageViewCreateInfo.format = vkFormat_color;

	vkImageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_R;
	vkImageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_G;
	vkImageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_B;
	vkImageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_A;

	vkImageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	vkImageViewCreateInfo.subresourceRange.baseMipLevel = 0;
	vkImageViewCreateInfo.subresourceRange.levelCount = 1;
	vkImageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
	vkImageViewCreateInfo.subresourceRange.layerCount = 1;

	vkImageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;

	for(uint32_t i = 0; i < swapchainImageCount; i++)
	{
		vkImageViewCreateInfo.image = swapChainImage_array[i];

		vkResult = vkCreateImageView(vkDevice, &vkImageViewCreateInfo, NULL, &swapChainImageView_array[i]);
		if (vkResult != VK_SUCCESS)
		{
			FileIO("CreateImagesAndImageViews(): vkCreateImageView() function failed with error code %d at iteration %d\n", vkResult, i);
			return vkResult;
		}
		else
		{
			FileIO("CreateImagesAndImageViews(): vkCreateImageView() succedded for iteration %d\n", i);
		}
	}

	vkResult = GetSupportedDepthFormat();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateImagesAndImageViews(): GetSupportedDepthFormat() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateImagesAndImageViews(): GetSupportedDepthFormat() succedded\n");
	}

	VkImageCreateInfo vkImageCreateInfo;
	memset((void*)&vkImageCreateInfo, 0, sizeof(VkImageCreateInfo));
	vkImageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	vkImageCreateInfo.pNext = NULL;
	vkImageCreateInfo.flags = 0;
	vkImageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
	vkImageCreateInfo.format = vkFormat_depth;

	vkImageCreateInfo.extent.width = (uint32_t)winWidth;
	vkImageCreateInfo.extent.height = (uint32_t)winHeight;
	vkImageCreateInfo.extent.depth = 1;

	vkImageCreateInfo.mipLevels = 1;
	vkImageCreateInfo.arrayLayers = 1;
	vkImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	vkImageCreateInfo.tiling =  VK_IMAGE_TILING_OPTIMAL;
	vkImageCreateInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

	vkResult = vkCreateImage(vkDevice, &vkImageCreateInfo, NULL, &vkImage_depth);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateImagesAndImageViews(): vkCreateImage() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateImagesAndImageViews(): vkCreateImage() succedded\n");
	}

	VkMemoryRequirements vkMemoryRequirements;
	memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));

	vkGetImageMemoryRequirements(vkDevice, vkImage_depth, &vkMemoryRequirements);

	VkMemoryAllocateInfo vkMemoryAllocateInfo;
	memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
	vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	vkMemoryAllocateInfo.pNext = NULL;
	vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;

	vkMemoryAllocateInfo.memoryTypeIndex = 0;
	for(uint32_t i =0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if((vkMemoryRequirements.memoryTypeBits & 1) == 1)
		{

			if(vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
			{
				vkMemoryAllocateInfo.memoryTypeIndex = i;
				break;
			}
		}
		vkMemoryRequirements.memoryTypeBits >>= 1;
	}

	vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vkDeviceMemory_depth);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateImagesAndImageViews(): vkAllocateMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateImagesAndImageViews(): vkAllocateMemory() succedded\n");
	}

	vkResult = vkBindImageMemory(vkDevice, vkImage_depth, vkDeviceMemory_depth, 0);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateImagesAndImageViews(): vkBindBufferMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateImagesAndImageViews(): vkBindBufferMemory() succedded\n");
	}

	memset((void*)&vkImageViewCreateInfo, 0, sizeof(VkImageViewCreateInfo));

	vkImageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	vkImageViewCreateInfo.pNext = NULL;
	vkImageViewCreateInfo.flags = 0;

	vkImageViewCreateInfo.format = vkFormat_depth;

	vkImageViewCreateInfo.subresourceRange.aspectMask =  VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT;
	vkImageViewCreateInfo.subresourceRange.baseMipLevel = 0;
	vkImageViewCreateInfo.subresourceRange.levelCount = 1;
	vkImageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
	vkImageViewCreateInfo.subresourceRange.layerCount = 1;

	vkImageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	vkImageViewCreateInfo.image = vkImage_depth;

	vkResult = vkCreateImageView(vkDevice, &vkImageViewCreateInfo, NULL, &vkImageView_depth);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateImagesAndImageViews(): vkCreateImageView() function failed with error code %d for depth image\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateImagesAndImageViews(): vkCreateImageView() succedded for depth image\n");
	}

	return vkResult;
}

VkResult GetSupportedDepthFormat(void)
{

	VkResult vkResult = VK_SUCCESS;

	VkFormat vkFormat_depth_array[] =
	{
		VK_FORMAT_D32_SFLOAT_S8_UINT,
		VK_FORMAT_D32_SFLOAT,
		VK_FORMAT_D24_UNORM_S8_UINT,
		VK_FORMAT_D16_UNORM_S8_UINT,
		VK_FORMAT_D16_UNORM
	};

	for(uint32_t i =0;i < (sizeof(vkFormat_depth_array)/sizeof(vkFormat_depth_array[0])); i++)
	{

		VkFormatProperties vkFormatProperties;
		memset((void*)&vkFormatProperties, 0, sizeof(vkFormatProperties));

		vkGetPhysicalDeviceFormatProperties(vkPhysicalDevice_selected, vkFormat_depth_array[i], &vkFormatProperties);
		if(vkFormatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
		{
			vkFormat_depth = vkFormat_depth_array[i];
			vkResult = VK_SUCCESS;
			break;
		}
	}

	return vkResult;
}

VkResult CreateCommandPool()
{

	VkResult vkResult = VK_SUCCESS;

	VkCommandPoolCreateInfo vkCommandPoolCreateInfo;
	memset((void*)&vkCommandPoolCreateInfo, 0, sizeof(VkCommandPoolCreateInfo));

	vkCommandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	vkCommandPoolCreateInfo.pNext = NULL;

	vkCommandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	vkCommandPoolCreateInfo.queueFamilyIndex = graphicsQuequeFamilyIndex_selected;

	vkResult = vkCreateCommandPool(vkDevice, &vkCommandPoolCreateInfo, NULL, &vkCommandPool);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateCommandPool(): vkCreateCommandPool() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateCommandPool(): vkCreateCommandPool() succedded\n");
	}

	return vkResult;
}

VkResult CreateCommandBuffers(VkCommandBuffer** ppvkCommandBuffer_array)
{

	VkResult vkResult = VK_SUCCESS;

	VkCommandBuffer *vkCommandBuffer_array = NULL;

	VkCommandBufferAllocateInfo vkCommandBufferAllocateInfo;
	memset((void*)&vkCommandBufferAllocateInfo, 0, sizeof(VkCommandBufferAllocateInfo));
	vkCommandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	vkCommandBufferAllocateInfo.pNext = NULL;

	vkCommandBufferAllocateInfo.commandPool = vkCommandPool;
	vkCommandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	vkCommandBufferAllocateInfo.commandBufferCount = 1;

	vkCommandBuffer_array = (VkCommandBuffer*)malloc(sizeof(VkCommandBuffer) * swapchainImageCount);

	for(uint32_t i = 0; i < swapchainImageCount; i++)
	{

		vkResult = vkAllocateCommandBuffers(vkDevice, &vkCommandBufferAllocateInfo, &vkCommandBuffer_array[i]);
		if (vkResult != VK_SUCCESS)
		{
			FileIO("CreateCommandBuffers(): vkAllocateCommandBuffers() function failed with error code %d at iteration %d\n", vkResult, i);
			return vkResult;
		}
		else
		{
			FileIO("CreateCommandBuffers(): vkAllocateCommandBuffers() succedded for iteration %d\n", i);
		}
	}

	*ppvkCommandBuffer_array = vkCommandBuffer_array;
	return vkResult;
}

VkResult CreateExternalVertexBuffer(unsigned int mesh_width, unsigned int mesh_height, VertexData *pVertexData)
{

	VkResult vkResult = VK_SUCCESS;

	VertexData vertexdata_position;

	unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);

	VkExternalMemoryBufferCreateInfo vkExternalMemoryBufferCreateInfo;
	memset((void*)&vkExternalMemoryBufferCreateInfo, 0, sizeof(VkExternalMemoryBufferCreateInfo));
	vkExternalMemoryBufferCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
	vkExternalMemoryBufferCreateInfo.pNext = NULL;
	vkExternalMemoryBufferCreateInfo.handleTypes = vkExternalMemoryHandleTypeFlagBits;

	memset((void*)&vertexdata_position, 0, sizeof(VertexData));

	VkBufferCreateInfo vkBufferCreateInfo;
	memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));

	vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vkBufferCreateInfo.pNext = &vkExternalMemoryBufferCreateInfo;
	vkBufferCreateInfo.flags = 0;
	vkBufferCreateInfo.size = size;
	vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexdata_position.vkBuffer);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateExternalVertexBuffer(): vkCreateBuffer() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateExternalVertexBuffer(): vkCreateBuffer() succedded\n");
	}

	VkMemoryRequirements vkMemoryRequirements;
	memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));

	vkGetBufferMemoryRequirements(vkDevice, vertexdata_position.vkBuffer, &vkMemoryRequirements);

	VkExportMemoryAllocateInfo vkExportMemoryAllocateInfo;
	memset((void*)&vkExportMemoryAllocateInfo, 0, sizeof(VkExportMemoryAllocateInfo));
	vkExportMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
	vkExportMemoryAllocateInfo.pNext = NULL;
	vkExportMemoryAllocateInfo.handleTypes = vkExternalMemoryHandleTypeFlagBits;

	VkMemoryAllocateInfo vkMemoryAllocateInfo;
	memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
	vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	vkMemoryAllocateInfo.pNext = &vkExportMemoryAllocateInfo;
	vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;

	vkMemoryAllocateInfo.memoryTypeIndex = 0;
	for(uint32_t i =0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if((vkMemoryRequirements.memoryTypeBits & 1) == 1)
		{

			if(vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
			{
				vkMemoryAllocateInfo.memoryTypeIndex = i;
				break;
			}
		}
		vkMemoryRequirements.memoryTypeBits >>= 1;
	}

	vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vertexdata_position.vkDeviceMemory);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateExternalVertexBuffer(): vkAllocateMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateExternalVertexBuffer(): vkAllocateMemory() succedded\n");
	}

	vkResult = vkBindBufferMemory(vkDevice, vertexdata_position.vkBuffer, vertexdata_position.vkDeviceMemory, 0);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateExternalVertexBuffer(): vkBindBufferMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateExternalVertexBuffer(): vkBindBufferMemory() succedded\n");
	}

	HANDLE hMemoryWin32Handle = NULL;

	VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR;
	memset((void*)&vkMemoryGetWin32HandleInfoKHR, 0, sizeof(VkMemoryGetWin32HandleInfoKHR));
	vkMemoryGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
	vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
	vkMemoryGetWin32HandleInfoKHR.memory = vertexdata_position.vkDeviceMemory;
	vkMemoryGetWin32HandleInfoKHR.handleType = vkExternalMemoryHandleTypeFlagBits;

	PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(vkDevice, "vkGetMemoryWin32HandleKHR");
	if (vkGetMemoryWin32HandleKHR == NULL)
	{
		FileIO("CreateExternalVertexBuffer(): vkGetMemoryWin32HandleKHR() api not obtained from vkGetDeviceProcAddr\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	else
	{
		FileIO("CreateExternalVertexBuffer(): vkGetMemoryWin32HandleKHR() api obtained from vkGetDeviceProcAddr\n");
	}

	vkResult = vkGetMemoryWin32HandleKHR(vkDevice, &vkMemoryGetWin32HandleInfoKHR, &hMemoryWin32Handle);
	if(vkResult != VK_SUCCESS)
	{
		FileIO("CreateExternalVertexBuffer(): vkGetMemoryWin32HandleKHR() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateExternalVertexBuffer(): vkGetMemoryWin32HandleKHR() succedded\n");
	}

	cudaExternalMemoryHandleDesc cuExternalMemoryHandleDesc;
	memset((void*)&cuExternalMemoryHandleDesc, 0, sizeof(cudaExternalMemoryHandleDesc));
	cuExternalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	cuExternalMemoryHandleDesc.handle.win32.handle = hMemoryWin32Handle;
	cuExternalMemoryHandleDesc.size = mesh_width * mesh_height * 4 * sizeof(float);
	cuExternalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;

	cudaResult = cudaImportExternalMemory(&cuExternalMemory_t, &cuExternalMemoryHandleDesc);
	if(cudaResult != CUDA_SUCCESS)
	{
		FileIO("CreateExternalVertexBuffer(): cudaImportExternalMemory() failed\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	else
	{
		FileIO("CreateExternalVertexBuffer(): cudaImportExternalMemory() suceeded\n");
	}

	CloseHandle(hMemoryWin32Handle);
	hMemoryWin32Handle = NULL;

	cudaExternalMemoryBufferDesc cuExternalMemoryBufferDesc;
	memset((void*)&cuExternalMemoryBufferDesc, 0, sizeof(cudaExternalMemoryBufferDesc));
	cuExternalMemoryBufferDesc.offset = 0;
	cuExternalMemoryBufferDesc.size = mesh_width * mesh_height * 4 * sizeof(float);
	cuExternalMemoryBufferDesc.flags = 0;

	cudaResult = cudaExternalMemoryGetMappedBuffer(&pos_CUDA, cuExternalMemory_t, &cuExternalMemoryBufferDesc);
	if(cudaResult != CUDA_SUCCESS)
	{
		FileIO("CreateExternalVertexBuffer(): cudaExternalMemoryGetMappedBuffer() failed\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	else
	{
		FileIO("CreateExternalVertexBuffer(): cudaExternalMemoryGetMappedBuffer() suceeded\n");
	}

	*pVertexData = vertexdata_position;

	return vkResult;
}

VkResult CreateIndirectBuffer(void)
{
	VkResult UpdateIndirectBuffer(void);

	VkResult vkResult = VK_SUCCESS;

	memset((void*)&vertexdata_indirect_buffer, 0, sizeof(VertexData));

	VkBufferCreateInfo vkBufferCreateInfo;
	memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));

	vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vkBufferCreateInfo.pNext = NULL;
	vkBufferCreateInfo.flags = 0;
	vkBufferCreateInfo.size = sizeof(VkDrawIndirectCommand);
	vkBufferCreateInfo.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexdata_indirect_buffer.vkBuffer);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateIndirectBuffer(): vkCreateBuffer() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateIndirectBuffer(): vkCreateBuffer() succedded\n");
	}

	VkMemoryRequirements vkMemoryRequirements;
	memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));

	vkGetBufferMemoryRequirements(vkDevice, vertexdata_indirect_buffer.vkBuffer, &vkMemoryRequirements);

	VkMemoryAllocateInfo vkMemoryAllocateInfo;
	memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
	vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	vkMemoryAllocateInfo.pNext = NULL;
	vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;

	BOOL bFoundMemoryType = FALSE;
	uint32_t memoryTypeBits = vkMemoryRequirements.memoryTypeBits;

	for(uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if((memoryTypeBits & (1 << i)) != 0)
		{
			if((vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
			{
				vkMemoryAllocateInfo.memoryTypeIndex = i;
				bIndirectBufferMemoryCoherent = TRUE;
				bFoundMemoryType = TRUE;
				FileIO("CreateIndirectBuffer(): Found HOST_VISIBLE + HOST_COHERENT memory type at index %u\n", i);
				break;
			}
		}
	}

	if(!bFoundMemoryType)
	{
		for(uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
		{
			if((memoryTypeBits & (1 << i)) != 0)
			{
				if(vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
				{
					vkMemoryAllocateInfo.memoryTypeIndex = i;
					bIndirectBufferMemoryCoherent = FALSE;
					bFoundMemoryType = TRUE;
					FileIO("CreateIndirectBuffer(): No HOST_COHERENT memory available, using HOST_VISIBLE only at index %u (will use vkFlushMappedMemoryRanges)\n", i);
					break;
				}
			}
		}
	}

	if(!bFoundMemoryType)
	{
		FileIO("CreateIndirectBuffer(): Failed to find suitable HOST_VISIBLE memory type\n");
		return VK_ERROR_FEATURE_NOT_PRESENT;
	}

	vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vertexdata_indirect_buffer.vkDeviceMemory);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateIndirectBuffer(): vkAllocateMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateIndirectBuffer(): vkAllocateMemory() succedded\n");
	}

	vkResult = vkBindBufferMemory(vkDevice, vertexdata_indirect_buffer.vkBuffer, vertexdata_indirect_buffer.vkDeviceMemory, 0);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateIndirectBuffer(): vkBindBufferMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateIndirectBuffer(): vkBindBufferMemory() succedded\n");
	}

	vkResult = UpdateIndirectBuffer();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateIndirectBuffer(): UpdateIndirectBuffer() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateIndirectBuffer(): UpdateIndirectBuffer() succedded\n");
	}

	return vkResult;
}

VkResult UpdateIndirectBuffer(void)
{

	VkResult vkResult = VK_SUCCESS;

	VkDrawIndirectCommand vkDrawIndirectCommand;
	memset((void*)&vkDrawIndirectCommand, 0, sizeof(VkDrawIndirectCommand));

	vkDrawIndirectCommand.vertexCount = PATCH_GRID_SIZE * PATCH_GRID_SIZE * 4;
	vkDrawIndirectCommand.instanceCount = 1;
	vkDrawIndirectCommand.firstVertex = 0;
	vkDrawIndirectCommand.firstInstance = 0;

	void* data = NULL;
	vkResult = vkMapMemory(vkDevice, vertexdata_indirect_buffer.vkDeviceMemory, 0, VK_WHOLE_SIZE, 0, &data);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("UpdateIndirectBuffer(): vkMapMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}

	memcpy(data, &vkDrawIndirectCommand, sizeof(VkDrawIndirectCommand));

	if(!bIndirectBufferMemoryCoherent)
	{
		VkMappedMemoryRange mappedMemoryRange;
		memset((void*)&mappedMemoryRange, 0, sizeof(VkMappedMemoryRange));
		mappedMemoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
		mappedMemoryRange.pNext = NULL;
		mappedMemoryRange.memory = vertexdata_indirect_buffer.vkDeviceMemory;
		mappedMemoryRange.offset = 0;
		mappedMemoryRange.size = VK_WHOLE_SIZE;

		vkResult = vkFlushMappedMemoryRanges(vkDevice, 1, &mappedMemoryRange);
		if(vkResult != VK_SUCCESS)
		{
			FileIO("UpdateIndirectBuffer(): vkFlushMappedMemoryRanges() function failed with error code %d\n", vkResult);
			vkUnmapMemory(vkDevice, vertexdata_indirect_buffer.vkDeviceMemory);
			return vkResult;
		}
		FileIO("UpdateIndirectBuffer(): Flushed non-coherent memory\n");
	}

	vkUnmapMemory(vkDevice, vertexdata_indirect_buffer.vkDeviceMemory);

	return vkResult;
}

VkResult CreateUniformBuffer()
{

	VkResult UpdateUniformBuffer(void);

	VkResult vkResult = VK_SUCCESS;

	VkBufferCreateInfo vkBufferCreateInfo;
	memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));

	vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vkBufferCreateInfo.pNext = NULL;
	vkBufferCreateInfo.flags = 0;
	vkBufferCreateInfo.size = sizeof(struct MyUniformData);
	vkBufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

	memset((void*)&uniformData, 0, sizeof(struct UniformData));

	vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &uniformData.vkBuffer);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateUniformBuffer(): vkCreateBuffer() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateUniformBuffer(): vkCreateBuffer() succedded\n");
	}

	VkMemoryRequirements vkMemoryRequirements;
	memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));

	vkGetBufferMemoryRequirements(vkDevice, uniformData.vkBuffer, &vkMemoryRequirements);

	VkMemoryAllocateInfo vkMemoryAllocateInfo;
	memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
	vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	vkMemoryAllocateInfo.pNext = NULL;
	vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;

	vkMemoryAllocateInfo.memoryTypeIndex = 0;
	for(uint32_t i =0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if((vkMemoryRequirements.memoryTypeBits & 1) == 1)
		{

			if(vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
			{
				vkMemoryAllocateInfo.memoryTypeIndex = i;
				break;
			}
		}
		vkMemoryRequirements.memoryTypeBits >>= 1;
	}

	vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &uniformData.vkDeviceMemory);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateUniformBuffer(): vkAllocateMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateUniformBuffer(): vkAllocateMemory() succedded\n");
	}

	vkResult = vkBindBufferMemory(vkDevice, uniformData.vkBuffer, uniformData.vkDeviceMemory, 0);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateUniformBuffer(): vkBindBufferMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateUniformBuffer(): vkBindBufferMemory() succedded\n");
	}

	vkResult = UpdateUniformBuffer();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateUniformBuffer(): updateUniformBuffer() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateUniformBuffer(): updateUniformBuffer() succedded\n");
	}

	return vkResult;
}

VkResult CreateShaders(void)
{

	VkResult vkResult = VK_SUCCESS;

	const char* szFileName = "Shader.vert.spv";
	FILE* fp = NULL;
	size_t size;

	fp = fopen(szFileName, "rb");
	if(fp == NULL)
	{
		FileIO("CreateShaders(): failed to open Vertex Shader SPIRV file\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): sucedded to open Vertex Shader SPIRV file\n");
	}

	fseek(fp, 0L, SEEK_END);

	size = ftell(fp);
	if(size == 0)
	{
		FileIO("CreateShaders(): Vertex Shader SPIRV file size is 0\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}

	fseek(fp, 0L, SEEK_SET);

	char* shaderData = (char*)malloc(sizeof(char) * size);
	if(shaderData == NULL)
	{
		FileIO("CreateShaders(): malloc for Vertex Shader SPIRV file failed\n");
	}
	else
	{
		FileIO("CreateShaders(): malloc for Vertex Shader SPIRV file done\n");
	}

	size_t retVal = fread(shaderData, size, 1, fp);
	if(retVal != 1)
	{
		FileIO("CreateShaders(): failed to read Vertex Shader SPIRV file\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): sucedded to read Vertex Shader SPIRV file\n");
	}

	fclose(fp);

	VkShaderModuleCreateInfo vkShaderModuleCreateInfo;
	memset((void*)&vkShaderModuleCreateInfo, 0, sizeof(VkShaderModuleCreateInfo));
	vkShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	vkShaderModuleCreateInfo.pNext = NULL;
	vkShaderModuleCreateInfo.flags = 0;
	vkShaderModuleCreateInfo.codeSize = size;
	vkShaderModuleCreateInfo.pCode = (uint32_t*)shaderData;

	vkResult = vkCreateShaderModule(vkDevice, &vkShaderModuleCreateInfo, NULL, &vkShaderMoudule_vertex_shader);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateShaders(): vkCreateShaderModule() function for vertex SPIRV shader file failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): vkCreateShaderModule() for vertex SPIRV shader file succedded\n");
	}

	if(shaderData)
	{
		free(shaderData);
		shaderData = NULL;
	}
	FileIO("CreateShaders(): vertex Shader module successfully created\n");

	szFileName = "Shader.frag.spv";
	size = 0;
	fp = NULL;

	fp = fopen(szFileName, "rb");
	if(fp == NULL)
	{
		FileIO("CreateShaders(): failed to open Fragment Shader SPIRV file\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): sucedded to open Fragment Shader SPIRV file\n");
	}

	fseek(fp, 0L, SEEK_END);

	size = ftell(fp);
	if(size == 0)
	{
		FileIO("CreateShaders(): Fragment Shader SPIRV file size is 0\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}

	fseek(fp, 0L, SEEK_SET);

	shaderData = (char*)malloc(sizeof(char) * size);
	if(shaderData == NULL)
	{
		FileIO("CreateShaders(): malloc for Fragment Shader SPIRV file failed\n");
	}
	else
	{
		FileIO("CreateShaders(): malloc for Fragment Shader SPIRV file done\n");
	}

	retVal = fread(shaderData, size, 1, fp);
	if(retVal != 1)
	{
		FileIO("CreateShaders(): failed to read Fragment Shader SPIRV file\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): sucedded to read Fragment Shader SPIRV file\n");
	}

	fclose(fp);

	memset((void*)&vkShaderModuleCreateInfo, 0, sizeof(VkShaderModuleCreateInfo));
	vkShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	vkShaderModuleCreateInfo.pNext = NULL;
	vkShaderModuleCreateInfo.flags = 0;
	vkShaderModuleCreateInfo.codeSize = size;
	vkShaderModuleCreateInfo.pCode = (uint32_t*)shaderData;

	vkResult = vkCreateShaderModule(vkDevice, &vkShaderModuleCreateInfo, NULL, &vkShaderMoudule_fragment_shader);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateShaders(): vkCreateShaderModule() function for fragment SPIRV shader file failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): vkCreateShaderModule() function for fragment SPIRV shader file succedded\n");
	}

	if(shaderData)
	{
		free(shaderData);
		shaderData = NULL;
	}
	FileIO("CreateShaders(): fragment Shader module successfully created\n");

	szFileName = "Shader.tesc.spv";
	size = 0;
	fp = NULL;

	fp = fopen(szFileName, "rb");
	if(fp == NULL)
	{
		FileIO("CreateShaders(): failed to open Tessellation Control Shader SPIRV file\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): sucedded to open Tessellation Control Shader SPIRV file\n");
	}

	fseek(fp, 0L, SEEK_END);

	size = ftell(fp);
	if(size == 0)
	{
		FileIO("CreateShaders(): Tessellation Control Shader SPIRV file size is 0\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}

	fseek(fp, 0L, SEEK_SET);

	shaderData = (char*)malloc(sizeof(char) * size);
	if(shaderData == NULL)
	{
		FileIO("CreateShaders(): malloc for Tessellation Control Shader SPIRV file failed\n");
	}
	else
	{
		FileIO("CreateShaders(): malloc for Tessellation Control Shader SPIRV file done\n");
	}

	retVal = fread(shaderData, size, 1, fp);
	if(retVal != 1)
	{
		FileIO("CreateShaders(): failed to read Tessellation Control Shader SPIRV file\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): sucedded to read Tessellation Control Shader SPIRV file\n");
	}

	fclose(fp);

	memset((void*)&vkShaderModuleCreateInfo, 0, sizeof(VkShaderModuleCreateInfo));
	vkShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	vkShaderModuleCreateInfo.pNext = NULL;
	vkShaderModuleCreateInfo.flags = 0;
	vkShaderModuleCreateInfo.codeSize = size;
	vkShaderModuleCreateInfo.pCode = (uint32_t*)shaderData;

	vkResult = vkCreateShaderModule(vkDevice, &vkShaderModuleCreateInfo, NULL, &vkShaderModule_tess_control);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateShaders(): vkCreateShaderModule() function for tessellation control SPIRV shader file failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): vkCreateShaderModule() function for tessellation control SPIRV shader file succedded\n");
	}

	if(shaderData)
	{
		free(shaderData);
		shaderData = NULL;
	}
	FileIO("CreateShaders(): tessellation control Shader module successfully created\n");

	szFileName = "Shader.tese.spv";
	size = 0;
	fp = NULL;

	fp = fopen(szFileName, "rb");
	if(fp == NULL)
	{
		FileIO("CreateShaders(): failed to open Tessellation Evaluation Shader SPIRV file\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): sucedded to open Tessellation Evaluation Shader SPIRV file\n");
	}

	fseek(fp, 0L, SEEK_END);

	size = ftell(fp);
	if(size == 0)
	{
		FileIO("CreateShaders(): Tessellation Evaluation Shader SPIRV file size is 0\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}

	fseek(fp, 0L, SEEK_SET);

	shaderData = (char*)malloc(sizeof(char) * size);
	if(shaderData == NULL)
	{
		FileIO("CreateShaders(): malloc for Tessellation Evaluation Shader SPIRV file failed\n");
	}
	else
	{
		FileIO("CreateShaders(): malloc for Tessellation Evaluation Shader SPIRV file done\n");
	}

	retVal = fread(shaderData, size, 1, fp);
	if(retVal != 1)
	{
		FileIO("CreateShaders(): failed to read Tessellation Evaluation Shader SPIRV file\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): sucedded to read Tessellation Evaluation Shader SPIRV file\n");
	}

	fclose(fp);

	memset((void*)&vkShaderModuleCreateInfo, 0, sizeof(VkShaderModuleCreateInfo));
	vkShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	vkShaderModuleCreateInfo.pNext = NULL;
	vkShaderModuleCreateInfo.flags = 0;
	vkShaderModuleCreateInfo.codeSize = size;
	vkShaderModuleCreateInfo.pCode = (uint32_t*)shaderData;

	vkResult = vkCreateShaderModule(vkDevice, &vkShaderModuleCreateInfo, NULL, &vkShaderModule_tess_eval);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateShaders(): vkCreateShaderModule() function for tessellation evaluation SPIRV shader file failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): vkCreateShaderModule() function for tessellation evaluation SPIRV shader file succedded\n");
	}

	if(shaderData)
	{
		free(shaderData);
		shaderData = NULL;
	}
	FileIO("CreateShaders(): tessellation evaluation Shader module successfully created\n");

	return vkResult;
}

VkResult CreateDescriptorSetLayout()
{

	VkResult vkResult = VK_SUCCESS;

	// Array of descriptor set layout bindings:
	// Binding 0: MVP Uniform buffer
	// Binding 1: cuRAND Gradient table (storage buffer)
	// Binding 2: cuRAND Permutation table (storage buffer)
	// Binding 3: CUDA-generated Heightmap (storage buffer)
	// Binding 4: CUDA-generated Normal map (storage buffer)
	// Binding 5: CUDA-generated Tangent map (storage buffer)
	// Binding 6: CUDA-computed Tessellation factors (storage buffer)
	// Binding 7: Frustum culling Visibility mask (storage buffer)
	VkDescriptorSetLayoutBinding vkDescriptorSetLayoutBindings[8];
	memset((void*)&vkDescriptorSetLayoutBindings, 0, sizeof(vkDescriptorSetLayoutBindings));

	// Binding 0: Uniform buffer for MVP matrices
	vkDescriptorSetLayoutBindings[0].binding = 0;
	vkDescriptorSetLayoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	vkDescriptorSetLayoutBindings[0].descriptorCount = 1;
	vkDescriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT|VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
	vkDescriptorSetLayoutBindings[0].pImmutableSamplers = NULL;

	// Binding 1: Storage buffer for cuRAND gradient table (vec4 * GRADIENT_TABLE_SIZE)
	vkDescriptorSetLayoutBindings[1].binding = 1;
	vkDescriptorSetLayoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	vkDescriptorSetLayoutBindings[1].descriptorCount = 1;
	vkDescriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
	vkDescriptorSetLayoutBindings[1].pImmutableSamplers = NULL;

	// Binding 2: Storage buffer for cuRAND permutation table (int * PERMUTATION_TABLE_SIZE)
	vkDescriptorSetLayoutBindings[2].binding = 2;
	vkDescriptorSetLayoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	vkDescriptorSetLayoutBindings[2].descriptorCount = 1;
	vkDescriptorSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
	vkDescriptorSetLayoutBindings[2].pImmutableSamplers = NULL;

	// Binding 3: Storage buffer for CUDA-generated heightmap (float * HEIGHTMAP_SIZE^2)
	vkDescriptorSetLayoutBindings[3].binding = 3;
	vkDescriptorSetLayoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	vkDescriptorSetLayoutBindings[3].descriptorCount = 1;
	vkDescriptorSetLayoutBindings[3].stageFlags = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	vkDescriptorSetLayoutBindings[3].pImmutableSamplers = NULL;

	// Binding 4: Storage buffer for CUDA-generated normal map (vec4 * HEIGHTMAP_SIZE^2)
	vkDescriptorSetLayoutBindings[4].binding = 4;
	vkDescriptorSetLayoutBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	vkDescriptorSetLayoutBindings[4].descriptorCount = 1;
	vkDescriptorSetLayoutBindings[4].stageFlags = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	vkDescriptorSetLayoutBindings[4].pImmutableSamplers = NULL;

	// Binding 5: Storage buffer for CUDA-generated tangent space (vec4 * HEIGHTMAP_SIZE^2)
	vkDescriptorSetLayoutBindings[5].binding = 5;
	vkDescriptorSetLayoutBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	vkDescriptorSetLayoutBindings[5].descriptorCount = 1;
	vkDescriptorSetLayoutBindings[5].stageFlags = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	vkDescriptorSetLayoutBindings[5].pImmutableSamplers = NULL;

	// Binding 6: Storage buffer for CUDA-computed tessellation factors (float * PATCH_GRID_SIZE^2)
	vkDescriptorSetLayoutBindings[6].binding = 6;
	vkDescriptorSetLayoutBindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	vkDescriptorSetLayoutBindings[6].descriptorCount = 1;
	vkDescriptorSetLayoutBindings[6].stageFlags = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
	vkDescriptorSetLayoutBindings[6].pImmutableSamplers = NULL;

	// Binding 7: Storage buffer for frustum culling visibility mask (int * PATCH_GRID_SIZE^2)
	vkDescriptorSetLayoutBindings[7].binding = 7;
	vkDescriptorSetLayoutBindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	vkDescriptorSetLayoutBindings[7].descriptorCount = 1;
	vkDescriptorSetLayoutBindings[7].stageFlags = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
	vkDescriptorSetLayoutBindings[7].pImmutableSamplers = NULL;

	VkDescriptorSetLayoutCreateInfo vkDescriptorSetLayoutCreateInfo;
	memset((void*)&vkDescriptorSetLayoutCreateInfo, 0, sizeof(VkDescriptorSetLayoutCreateInfo));
	vkDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	vkDescriptorSetLayoutCreateInfo.pNext = NULL;
	vkDescriptorSetLayoutCreateInfo.flags = 0;

	vkDescriptorSetLayoutCreateInfo.bindingCount = 8;
	vkDescriptorSetLayoutCreateInfo.pBindings = vkDescriptorSetLayoutBindings;

	vkResult = vkCreateDescriptorSetLayout(vkDevice, &vkDescriptorSetLayoutCreateInfo, NULL, &vkDescriptorSetLayout);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateDescriptorSetLayout(): vkCreateDescriptorSetLayout() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateDescriptorSetLayout(): vkCreateDescriptorSetLayout() function succedded with 8 bindings (UBO + 7 SSBOs)\n");
	}

	return vkResult;
}

VkResult CreatePipelineLayout(void)
{

	VkResult vkResult = VK_SUCCESS;

	VkPipelineLayoutCreateInfo vkPipelineLayoutCreateInfo;
	memset((void*)&vkPipelineLayoutCreateInfo, 0, sizeof(VkPipelineLayoutCreateInfo));
	vkPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	vkPipelineLayoutCreateInfo.pNext = NULL;
	vkPipelineLayoutCreateInfo.flags = 0;
	vkPipelineLayoutCreateInfo.setLayoutCount = 1;
	vkPipelineLayoutCreateInfo.pSetLayouts = &vkDescriptorSetLayout;
	vkPipelineLayoutCreateInfo.pushConstantRangeCount = 0;
	vkPipelineLayoutCreateInfo.pPushConstantRanges = NULL;

	vkResult = vkCreatePipelineLayout(vkDevice, &vkPipelineLayoutCreateInfo, NULL, &vkPipelineLayout);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreatePipelineLayout(): vkCreatePipelineLayout() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreatePipelineLayout(): vkCreatePipelineLayout() function succedded\n");
	}

	return vkResult;
}

VkResult CreateDescriptorPool(void)
{

	VkResult vkResult = VK_SUCCESS;

	// Pool sizes for: 1 uniform buffer + 7 storage buffers
	// SSBOs: gradient, permutation, heightmap, normal, tangent, tessellation factors, visibility
	VkDescriptorPoolSize vkDescriptorPoolSizes[2];
	memset((void*)&vkDescriptorPoolSizes, 0, sizeof(vkDescriptorPoolSizes));

	// Uniform buffer pool size
	vkDescriptorPoolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	vkDescriptorPoolSizes[0].descriptorCount = 1;

	// Storage buffer pool size for all CUDA data buffers
	vkDescriptorPoolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	vkDescriptorPoolSizes[1].descriptorCount = 7;  // gradient + permutation + heightmap + normal + tangent + tessFactors + visibility

	VkDescriptorPoolCreateInfo vkDescriptorPoolCreateInfo;
	memset((void*)&vkDescriptorPoolCreateInfo, 0, sizeof(VkDescriptorPoolCreateInfo));
	vkDescriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	vkDescriptorPoolCreateInfo.pNext = NULL;
	vkDescriptorPoolCreateInfo.flags = 0;
	vkDescriptorPoolCreateInfo.maxSets = 1;
	vkDescriptorPoolCreateInfo.poolSizeCount = 2;
	vkDescriptorPoolCreateInfo.pPoolSizes = vkDescriptorPoolSizes;

	vkResult = vkCreateDescriptorPool(vkDevice, &vkDescriptorPoolCreateInfo, NULL, &vkDescriptorPool);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateDescriptorPool(): vkCreateDescriptorPool() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateDescriptorPool(): vkCreateDescriptorPool() succedded (UBO + 7 SSBOs)\n");
	}

	return vkResult;
}

VkResult CreateDescriptorSet(void)
{

	VkResult vkResult = VK_SUCCESS;

	VkDescriptorSetAllocateInfo vkDescriptorSetAllocateInfo;
	memset((void*)&vkDescriptorSetAllocateInfo, 0, sizeof(VkDescriptorSetAllocateInfo));
	vkDescriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	vkDescriptorSetAllocateInfo.pNext = NULL;
	vkDescriptorSetAllocateInfo.descriptorPool = vkDescriptorPool;

	vkDescriptorSetAllocateInfo.descriptorSetCount = 1;

	vkDescriptorSetAllocateInfo.pSetLayouts = &vkDescriptorSetLayout;

	vkResult = vkAllocateDescriptorSets(vkDevice, &vkDescriptorSetAllocateInfo, &vkDescriptorSet);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateDescriptorSet(): vkAllocateDescriptorSets() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateDescriptorSet(): vkAllocateDescriptorSets() succedded\n");
	}

	// Buffer info for all descriptors
	VkDescriptorBufferInfo uniformBufferInfo;
	memset((void*)&uniformBufferInfo, 0, sizeof(VkDescriptorBufferInfo));
	uniformBufferInfo.buffer = uniformData.vkBuffer;
	uniformBufferInfo.offset = 0;
	uniformBufferInfo.range = sizeof(struct MyUniformData);

	VkDescriptorBufferInfo gradientBufferInfo;
	memset((void*)&gradientBufferInfo, 0, sizeof(VkDescriptorBufferInfo));
	gradientBufferInfo.buffer = noiseStorageBuffers.gradientBuffer;
	gradientBufferInfo.offset = 0;
	gradientBufferInfo.range = GRADIENT_TABLE_SIZE * sizeof(float) * 4;

	VkDescriptorBufferInfo permutationBufferInfo;
	memset((void*)&permutationBufferInfo, 0, sizeof(VkDescriptorBufferInfo));
	permutationBufferInfo.buffer = noiseStorageBuffers.permutationBuffer;
	permutationBufferInfo.offset = 0;
	permutationBufferInfo.range = PERMUTATION_TABLE_SIZE * sizeof(int);

	VkDescriptorBufferInfo heightmapBufferInfo;
	memset((void*)&heightmapBufferInfo, 0, sizeof(VkDescriptorBufferInfo));
	heightmapBufferInfo.buffer = noiseStorageBuffers.heightmapBuffer;
	heightmapBufferInfo.offset = 0;
	heightmapBufferInfo.range = HEIGHTMAP_SIZE * HEIGHTMAP_SIZE * sizeof(float);

	VkDescriptorBufferInfo normalBufferInfo;
	memset((void*)&normalBufferInfo, 0, sizeof(VkDescriptorBufferInfo));
	normalBufferInfo.buffer = noiseStorageBuffers.normalBuffer;
	normalBufferInfo.offset = 0;
	normalBufferInfo.range = HEIGHTMAP_SIZE * HEIGHTMAP_SIZE * sizeof(float) * 4;

	VkDescriptorBufferInfo tangentBufferInfo;
	memset((void*)&tangentBufferInfo, 0, sizeof(VkDescriptorBufferInfo));
	tangentBufferInfo.buffer = noiseStorageBuffers.tangentBuffer;
	tangentBufferInfo.offset = 0;
	tangentBufferInfo.range = HEIGHTMAP_SIZE * HEIGHTMAP_SIZE * sizeof(float) * 4;

	VkDescriptorBufferInfo tessFactorBufferInfo;
	memset((void*)&tessFactorBufferInfo, 0, sizeof(VkDescriptorBufferInfo));
	tessFactorBufferInfo.buffer = noiseStorageBuffers.tessFactorBuffer;
	tessFactorBufferInfo.offset = 0;
	tessFactorBufferInfo.range = PATCH_GRID_SIZE * PATCH_GRID_SIZE * sizeof(float);

	VkDescriptorBufferInfo visibilityBufferInfo;
	memset((void*)&visibilityBufferInfo, 0, sizeof(VkDescriptorBufferInfo));
	visibilityBufferInfo.buffer = noiseStorageBuffers.visibilityBuffer;
	visibilityBufferInfo.offset = 0;
	visibilityBufferInfo.range = PATCH_GRID_SIZE * PATCH_GRID_SIZE * sizeof(int);

	// Array of write descriptor sets for all bindings
	VkWriteDescriptorSet vkWriteDescriptorSets[8];
	memset((void*)&vkWriteDescriptorSets, 0, sizeof(vkWriteDescriptorSets));

	// Binding 0: Uniform buffer (MVP matrices)
	vkWriteDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	vkWriteDescriptorSets[0].pNext = NULL;
	vkWriteDescriptorSets[0].dstSet = vkDescriptorSet;
	vkWriteDescriptorSets[0].dstBinding = 0;
	vkWriteDescriptorSets[0].dstArrayElement = 0;
	vkWriteDescriptorSets[0].descriptorCount = 1;
	vkWriteDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	vkWriteDescriptorSets[0].pImageInfo = NULL;
	vkWriteDescriptorSets[0].pBufferInfo = &uniformBufferInfo;
	vkWriteDescriptorSets[0].pTexelBufferView = NULL;

	// Binding 1: Storage buffer (cuRAND gradient table)
	vkWriteDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	vkWriteDescriptorSets[1].pNext = NULL;
	vkWriteDescriptorSets[1].dstSet = vkDescriptorSet;
	vkWriteDescriptorSets[1].dstBinding = 1;
	vkWriteDescriptorSets[1].dstArrayElement = 0;
	vkWriteDescriptorSets[1].descriptorCount = 1;
	vkWriteDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	vkWriteDescriptorSets[1].pImageInfo = NULL;
	vkWriteDescriptorSets[1].pBufferInfo = &gradientBufferInfo;
	vkWriteDescriptorSets[1].pTexelBufferView = NULL;

	// Binding 2: Storage buffer (cuRAND permutation table)
	vkWriteDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	vkWriteDescriptorSets[2].pNext = NULL;
	vkWriteDescriptorSets[2].dstSet = vkDescriptorSet;
	vkWriteDescriptorSets[2].dstBinding = 2;
	vkWriteDescriptorSets[2].dstArrayElement = 0;
	vkWriteDescriptorSets[2].descriptorCount = 1;
	vkWriteDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	vkWriteDescriptorSets[2].pImageInfo = NULL;
	vkWriteDescriptorSets[2].pBufferInfo = &permutationBufferInfo;
	vkWriteDescriptorSets[2].pTexelBufferView = NULL;

	// Binding 3: Storage buffer (CUDA-generated heightmap)
	vkWriteDescriptorSets[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	vkWriteDescriptorSets[3].pNext = NULL;
	vkWriteDescriptorSets[3].dstSet = vkDescriptorSet;
	vkWriteDescriptorSets[3].dstBinding = 3;
	vkWriteDescriptorSets[3].dstArrayElement = 0;
	vkWriteDescriptorSets[3].descriptorCount = 1;
	vkWriteDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	vkWriteDescriptorSets[3].pImageInfo = NULL;
	vkWriteDescriptorSets[3].pBufferInfo = &heightmapBufferInfo;
	vkWriteDescriptorSets[3].pTexelBufferView = NULL;

	// Binding 4: Storage buffer (CUDA-generated normal map)
	vkWriteDescriptorSets[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	vkWriteDescriptorSets[4].pNext = NULL;
	vkWriteDescriptorSets[4].dstSet = vkDescriptorSet;
	vkWriteDescriptorSets[4].dstBinding = 4;
	vkWriteDescriptorSets[4].dstArrayElement = 0;
	vkWriteDescriptorSets[4].descriptorCount = 1;
	vkWriteDescriptorSets[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	vkWriteDescriptorSets[4].pImageInfo = NULL;
	vkWriteDescriptorSets[4].pBufferInfo = &normalBufferInfo;
	vkWriteDescriptorSets[4].pTexelBufferView = NULL;

	// Binding 5: Storage buffer (CUDA-generated tangent space)
	vkWriteDescriptorSets[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	vkWriteDescriptorSets[5].pNext = NULL;
	vkWriteDescriptorSets[5].dstSet = vkDescriptorSet;
	vkWriteDescriptorSets[5].dstBinding = 5;
	vkWriteDescriptorSets[5].dstArrayElement = 0;
	vkWriteDescriptorSets[5].descriptorCount = 1;
	vkWriteDescriptorSets[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	vkWriteDescriptorSets[5].pImageInfo = NULL;
	vkWriteDescriptorSets[5].pBufferInfo = &tangentBufferInfo;
	vkWriteDescriptorSets[5].pTexelBufferView = NULL;

	// Binding 6: Storage buffer (CUDA-computed tessellation factors)
	vkWriteDescriptorSets[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	vkWriteDescriptorSets[6].pNext = NULL;
	vkWriteDescriptorSets[6].dstSet = vkDescriptorSet;
	vkWriteDescriptorSets[6].dstBinding = 6;
	vkWriteDescriptorSets[6].dstArrayElement = 0;
	vkWriteDescriptorSets[6].descriptorCount = 1;
	vkWriteDescriptorSets[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	vkWriteDescriptorSets[6].pImageInfo = NULL;
	vkWriteDescriptorSets[6].pBufferInfo = &tessFactorBufferInfo;
	vkWriteDescriptorSets[6].pTexelBufferView = NULL;

	// Binding 7: Storage buffer (frustum culling visibility mask)
	vkWriteDescriptorSets[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	vkWriteDescriptorSets[7].pNext = NULL;
	vkWriteDescriptorSets[7].dstSet = vkDescriptorSet;
	vkWriteDescriptorSets[7].dstBinding = 7;
	vkWriteDescriptorSets[7].dstArrayElement = 0;
	vkWriteDescriptorSets[7].descriptorCount = 1;
	vkWriteDescriptorSets[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	vkWriteDescriptorSets[7].pImageInfo = NULL;
	vkWriteDescriptorSets[7].pBufferInfo = &visibilityBufferInfo;
	vkWriteDescriptorSets[7].pTexelBufferView = NULL;

	vkUpdateDescriptorSets(vkDevice, 8, vkWriteDescriptorSets, 0, NULL);

	FileIO("CreateDescriptorSet(): vkUpdateDescriptorSets() succedded (UBO + 7 SSBOs)\n");

	return vkResult;
}

VkResult CreateRenderPass(void)
{

	VkResult vkResult = VK_SUCCESS;

	VkAttachmentDescription  vkAttachmentDescription_array[2];
	memset((void*)vkAttachmentDescription_array, 0, sizeof(VkAttachmentDescription) * _ARRAYSIZE(vkAttachmentDescription_array));

	vkAttachmentDescription_array[0].flags = 0;

	vkAttachmentDescription_array[0].format = vkFormat_color;

	vkAttachmentDescription_array[0].samples = VK_SAMPLE_COUNT_1_BIT;

	vkAttachmentDescription_array[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;

	vkAttachmentDescription_array[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;

	vkAttachmentDescription_array[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	vkAttachmentDescription_array[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

	vkAttachmentDescription_array[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	vkAttachmentDescription_array[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	vkAttachmentDescription_array[1].flags = 0;

	vkAttachmentDescription_array[1].format = vkFormat_depth;

	vkAttachmentDescription_array[1].samples = VK_SAMPLE_COUNT_1_BIT;

	vkAttachmentDescription_array[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;

	vkAttachmentDescription_array[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;

	vkAttachmentDescription_array[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	vkAttachmentDescription_array[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

	vkAttachmentDescription_array[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	vkAttachmentDescription_array[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference vkAttachmentReference_color;
	memset((void*)&vkAttachmentReference_color, 0, sizeof(VkAttachmentReference));
	vkAttachmentReference_color.attachment = 0;

	vkAttachmentReference_color.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference vkAttachmentReference_depth;
	memset((void*)&vkAttachmentReference_depth, 0, sizeof(VkAttachmentReference));
	vkAttachmentReference_depth.attachment = 1;

	vkAttachmentReference_depth.layout =  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription vkSubpassDescription;
	memset((void*)&vkSubpassDescription, 0, sizeof(VkSubpassDescription));

	vkSubpassDescription.flags = 0;
	vkSubpassDescription.pipelineBindPoint =  VK_PIPELINE_BIND_POINT_GRAPHICS;
	vkSubpassDescription.inputAttachmentCount = 0;
	vkSubpassDescription.pInputAttachments = NULL;
	vkSubpassDescription.colorAttachmentCount = 1;
	vkSubpassDescription.pColorAttachments = (const VkAttachmentReference*)&vkAttachmentReference_color;
	vkSubpassDescription.pResolveAttachments = NULL;
	vkSubpassDescription.pDepthStencilAttachment = (const VkAttachmentReference*)&vkAttachmentReference_depth;
	vkSubpassDescription.preserveAttachmentCount = 0;
	vkSubpassDescription.pPreserveAttachments = NULL;

	VkRenderPassCreateInfo vkRenderPassCreateInfo;
	memset((void*)&vkRenderPassCreateInfo, 0, sizeof(VkRenderPassCreateInfo));
	vkRenderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	vkRenderPassCreateInfo.pNext = NULL;
	vkRenderPassCreateInfo.flags = 0;
	vkRenderPassCreateInfo.attachmentCount = _ARRAYSIZE(vkAttachmentDescription_array);
	vkRenderPassCreateInfo.pAttachments = vkAttachmentDescription_array;
	vkRenderPassCreateInfo.subpassCount = 1;
	vkRenderPassCreateInfo.pSubpasses = &vkSubpassDescription;
	vkRenderPassCreateInfo.dependencyCount = 0;
	vkRenderPassCreateInfo.pDependencies = NULL;

	vkResult = vkCreateRenderPass(vkDevice, &vkRenderPassCreateInfo, NULL, &vkRenderPass);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateRenderPass(): vkCreateRenderPass() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateRenderPass(): vkCreateRenderPass() succedded\n");
	}

	return vkResult;
}

VkResult CreatePipeline(void)
{

	VkResult vkResult = VK_SUCCESS;

	VkVertexInputBindingDescription vkVertexInputBindingDescription_array[1];
	memset((void*)vkVertexInputBindingDescription_array, 0,  sizeof(VkVertexInputBindingDescription) * _ARRAYSIZE(vkVertexInputBindingDescription_array));
	vkVertexInputBindingDescription_array[0].binding = 0;
	vkVertexInputBindingDescription_array[0].stride = sizeof(float) * 4;
	vkVertexInputBindingDescription_array[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription vkVertexInputAttributeDescription_array[1];
	memset((void*)vkVertexInputAttributeDescription_array, 0,  sizeof(VkVertexInputAttributeDescription) * _ARRAYSIZE(vkVertexInputAttributeDescription_array));
	vkVertexInputAttributeDescription_array[0].location = 0;
	vkVertexInputAttributeDescription_array[0].binding = 0;
	vkVertexInputAttributeDescription_array[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
	vkVertexInputAttributeDescription_array[0].offset = 0;

	VkPipelineVertexInputStateCreateInfo vkPipelineVertexInputStateCreateInfo;
	memset((void*)&vkPipelineVertexInputStateCreateInfo, 0,  sizeof(VkPipelineVertexInputStateCreateInfo));
	vkPipelineVertexInputStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vkPipelineVertexInputStateCreateInfo.pNext = NULL;
	vkPipelineVertexInputStateCreateInfo.flags = 0;
	vkPipelineVertexInputStateCreateInfo.vertexBindingDescriptionCount = _ARRAYSIZE(vkVertexInputBindingDescription_array);
	vkPipelineVertexInputStateCreateInfo.pVertexBindingDescriptions = vkVertexInputBindingDescription_array;
	vkPipelineVertexInputStateCreateInfo.vertexAttributeDescriptionCount = _ARRAYSIZE(vkVertexInputAttributeDescription_array);
	vkPipelineVertexInputStateCreateInfo.pVertexAttributeDescriptions = vkVertexInputAttributeDescription_array;

	VkPipelineInputAssemblyStateCreateInfo vkPipelineInputAssemblyStateCreateInfo;
	memset((void*)&vkPipelineInputAssemblyStateCreateInfo, 0,  sizeof(VkPipelineInputAssemblyStateCreateInfo));
	vkPipelineInputAssemblyStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	vkPipelineInputAssemblyStateCreateInfo.pNext = NULL;
	vkPipelineInputAssemblyStateCreateInfo.flags = 0;
	vkPipelineInputAssemblyStateCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
	vkPipelineInputAssemblyStateCreateInfo.primitiveRestartEnable = VK_FALSE;

	VkPipelineRasterizationStateCreateInfo vkPipelineRasterizationStateCreateInfo;
	memset((void*)&vkPipelineRasterizationStateCreateInfo, 0,  sizeof(VkPipelineRasterizationStateCreateInfo));
	vkPipelineRasterizationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	vkPipelineRasterizationStateCreateInfo.pNext = NULL;
	vkPipelineRasterizationStateCreateInfo.flags = 0;

	vkPipelineRasterizationStateCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
	vkPipelineRasterizationStateCreateInfo.cullMode = VK_CULL_MODE_NONE;
	vkPipelineRasterizationStateCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

	vkPipelineRasterizationStateCreateInfo.lineWidth = 1.0f;

	VkPipelineColorBlendAttachmentState vkPipelineColorBlendAttachmentState_array[1];
	memset((void*)vkPipelineColorBlendAttachmentState_array, 0, sizeof(VkPipelineColorBlendAttachmentState) * _ARRAYSIZE(vkPipelineColorBlendAttachmentState_array));
	vkPipelineColorBlendAttachmentState_array[0].blendEnable = VK_FALSE;

	vkPipelineColorBlendAttachmentState_array[0].colorWriteMask = 0xF;

	VkPipelineColorBlendStateCreateInfo vkPipelineColorBlendStateCreateInfo;
	memset((void*)&vkPipelineColorBlendStateCreateInfo, 0, sizeof(VkPipelineColorBlendStateCreateInfo));
	vkPipelineColorBlendStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	vkPipelineColorBlendStateCreateInfo.pNext = NULL;
	vkPipelineColorBlendStateCreateInfo.flags = 0;

	vkPipelineColorBlendStateCreateInfo.attachmentCount = _ARRAYSIZE(vkPipelineColorBlendAttachmentState_array);
	vkPipelineColorBlendStateCreateInfo.pAttachments = vkPipelineColorBlendAttachmentState_array;

	VkPipelineViewportStateCreateInfo vkPipelineViewportStateCreateInfo;
	memset((void*)&vkPipelineViewportStateCreateInfo, 0, sizeof(VkPipelineViewportStateCreateInfo));
	vkPipelineViewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	vkPipelineViewportStateCreateInfo.pNext = NULL;
	vkPipelineViewportStateCreateInfo.flags = 0;

	vkPipelineViewportStateCreateInfo.viewportCount = 1;
	memset((void*)&vkViewPort, 0 , sizeof(VkViewport));
	vkViewPort.x = 0;
	vkViewPort.y = 0;
	vkViewPort.width = (float)vkExtent2D_SwapChain.width;
	vkViewPort.height = (float)vkExtent2D_SwapChain.height;

	vkViewPort.minDepth = 0.0f;
	vkViewPort.maxDepth = 1.0f;

	vkPipelineViewportStateCreateInfo.pViewports = &vkViewPort;

	vkPipelineViewportStateCreateInfo.scissorCount = 1;
	memset((void*)&vkRect2D_scissor, 0 , sizeof(VkRect2D));
	vkRect2D_scissor.offset.x = 0;
	vkRect2D_scissor.offset.y = 0;
	vkRect2D_scissor.extent.width = vkExtent2D_SwapChain.width;
	vkRect2D_scissor.extent.height = vkExtent2D_SwapChain.height;

	vkPipelineViewportStateCreateInfo.pScissors = &vkRect2D_scissor;

	VkPipelineMultisampleStateCreateInfo vkPipelineMultisampleStateCreateInfo;
	memset((void*)&vkPipelineMultisampleStateCreateInfo, 0, sizeof(VkPipelineMultisampleStateCreateInfo));
	vkPipelineMultisampleStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	vkPipelineMultisampleStateCreateInfo.pNext = NULL;
	vkPipelineMultisampleStateCreateInfo.flags = 0;
	vkPipelineMultisampleStateCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineShaderStageCreateInfo vkPipelineShaderStageCreateInfo_array[4];
	memset((void*)vkPipelineShaderStageCreateInfo_array, 0, sizeof(VkPipelineShaderStageCreateInfo) * _ARRAYSIZE(vkPipelineShaderStageCreateInfo_array));

	vkPipelineShaderStageCreateInfo_array[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vkPipelineShaderStageCreateInfo_array[0].pNext = NULL;
	vkPipelineShaderStageCreateInfo_array[0].flags = 0;
	vkPipelineShaderStageCreateInfo_array[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	vkPipelineShaderStageCreateInfo_array[0].module = vkShaderMoudule_vertex_shader;
	vkPipelineShaderStageCreateInfo_array[0].pName = "main";
	vkPipelineShaderStageCreateInfo_array[0].pSpecializationInfo = NULL;

	vkPipelineShaderStageCreateInfo_array[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vkPipelineShaderStageCreateInfo_array[1].pNext = NULL;
	vkPipelineShaderStageCreateInfo_array[1].flags = 0;
	vkPipelineShaderStageCreateInfo_array[1].stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
	vkPipelineShaderStageCreateInfo_array[1].module = vkShaderModule_tess_control;
	vkPipelineShaderStageCreateInfo_array[1].pName = "main";
	vkPipelineShaderStageCreateInfo_array[1].pSpecializationInfo = NULL;

	vkPipelineShaderStageCreateInfo_array[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vkPipelineShaderStageCreateInfo_array[2].pNext = NULL;
	vkPipelineShaderStageCreateInfo_array[2].flags = 0;
	vkPipelineShaderStageCreateInfo_array[2].stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
	vkPipelineShaderStageCreateInfo_array[2].module = vkShaderModule_tess_eval;
	vkPipelineShaderStageCreateInfo_array[2].pName = "main";
	vkPipelineShaderStageCreateInfo_array[2].pSpecializationInfo = NULL;

	vkPipelineShaderStageCreateInfo_array[3].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vkPipelineShaderStageCreateInfo_array[3].pNext = NULL;
	vkPipelineShaderStageCreateInfo_array[3].flags = 0;
	vkPipelineShaderStageCreateInfo_array[3].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	vkPipelineShaderStageCreateInfo_array[3].module = vkShaderMoudule_fragment_shader;
	vkPipelineShaderStageCreateInfo_array[3].pName = "main";
	vkPipelineShaderStageCreateInfo_array[3].pSpecializationInfo = NULL;

	VkPipelineTessellationStateCreateInfo vkPipelineTessellationStateCreateInfo;
	memset((void*)&vkPipelineTessellationStateCreateInfo, 0, sizeof(VkPipelineTessellationStateCreateInfo));
	vkPipelineTessellationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
	vkPipelineTessellationStateCreateInfo.pNext = NULL;
	vkPipelineTessellationStateCreateInfo.flags = 0;
	vkPipelineTessellationStateCreateInfo.patchControlPoints = 4;

	VkPipelineCacheCreateInfo vkPipelineCacheCreateInfo;
	memset((void*)&vkPipelineCacheCreateInfo, 0, sizeof(VkPipelineCacheCreateInfo));
	vkPipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	vkPipelineCacheCreateInfo.pNext = NULL;
	vkPipelineCacheCreateInfo.flags = 0;

	VkPipelineCache vkPipelineCache = VK_NULL_HANDLE;
	vkResult = vkCreatePipelineCache(vkDevice, &vkPipelineCacheCreateInfo, NULL, &vkPipelineCache);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreatePipeline(): vkCreatePipelineCache() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreatePipeline(): vkCreatePipelineCache() succedded\n");
	}

	VkGraphicsPipelineCreateInfo vkGraphicsPipelineCreateInfo;
	memset((void*)&vkGraphicsPipelineCreateInfo, 0, sizeof(VkGraphicsPipelineCreateInfo));
	vkGraphicsPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	vkGraphicsPipelineCreateInfo.pNext = NULL;
	vkGraphicsPipelineCreateInfo.flags = 0;
	vkGraphicsPipelineCreateInfo.stageCount = _ARRAYSIZE(vkPipelineShaderStageCreateInfo_array);
	vkGraphicsPipelineCreateInfo.pStages = vkPipelineShaderStageCreateInfo_array;
	vkGraphicsPipelineCreateInfo.pVertexInputState = &vkPipelineVertexInputStateCreateInfo;
	vkGraphicsPipelineCreateInfo.pInputAssemblyState = &vkPipelineInputAssemblyStateCreateInfo;
	vkGraphicsPipelineCreateInfo.pTessellationState = &vkPipelineTessellationStateCreateInfo;
	vkGraphicsPipelineCreateInfo.pViewportState = &vkPipelineViewportStateCreateInfo;
	vkGraphicsPipelineCreateInfo.pRasterizationState = &vkPipelineRasterizationStateCreateInfo;
	vkGraphicsPipelineCreateInfo.pMultisampleState = &vkPipelineMultisampleStateCreateInfo;

	VkPipelineDepthStencilStateCreateInfo vkPipelineDepthStencilStateCreateInfo;
	memset((void*)&vkPipelineDepthStencilStateCreateInfo, 0, sizeof(VkPipelineDepthStencilStateCreateInfo));
	vkPipelineDepthStencilStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	vkPipelineDepthStencilStateCreateInfo.pNext = NULL;
	vkPipelineDepthStencilStateCreateInfo.flags = 0;
	vkPipelineDepthStencilStateCreateInfo.depthTestEnable = VK_TRUE;
	vkPipelineDepthStencilStateCreateInfo.depthWriteEnable= VK_TRUE;
	vkPipelineDepthStencilStateCreateInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
	vkPipelineDepthStencilStateCreateInfo.depthBoundsTestEnable= VK_FALSE;
	vkPipelineDepthStencilStateCreateInfo.stencilTestEnable = VK_FALSE;

	vkPipelineDepthStencilStateCreateInfo.back.failOp = VK_STENCIL_OP_KEEP;
	vkPipelineDepthStencilStateCreateInfo.back.passOp = VK_STENCIL_OP_KEEP;
	vkPipelineDepthStencilStateCreateInfo.back.compareOp = VK_COMPARE_OP_ALWAYS;

	vkPipelineDepthStencilStateCreateInfo.front = vkPipelineDepthStencilStateCreateInfo.back;

	vkGraphicsPipelineCreateInfo.pDepthStencilState = &vkPipelineDepthStencilStateCreateInfo;

	vkGraphicsPipelineCreateInfo.pColorBlendState = &vkPipelineColorBlendStateCreateInfo;
	vkGraphicsPipelineCreateInfo.pDynamicState = NULL;
	vkGraphicsPipelineCreateInfo.layout = vkPipelineLayout;
	vkGraphicsPipelineCreateInfo.renderPass = vkRenderPass;
	vkGraphicsPipelineCreateInfo.subpass = 0;
	vkGraphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
	vkGraphicsPipelineCreateInfo.basePipelineIndex = 0;

	vkResult = vkCreateGraphicsPipelines(vkDevice, vkPipelineCache, 1, &vkGraphicsPipelineCreateInfo, NULL, &vkPipeline);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("vkCreateGraphicsPipelines(): vkCreatePipelineCache() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("vkCreateGraphicsPipelines(): vkCreatePipelineCache() succedded\n");
	}

	if(vkPipelineCache != VK_NULL_HANDLE)
	{
		vkDestroyPipelineCache(vkDevice, vkPipelineCache, NULL);
		vkPipelineCache = VK_NULL_HANDLE;
		FileIO("vkCreateGraphicsPipelines(): vkPipelineCache is freed\n");
	}

	return vkResult;
}

VkResult CreateFramebuffers(void)
{

	VkResult vkResult = VK_SUCCESS;

	vkFramebuffer_array = (VkFramebuffer*)malloc(sizeof(VkFramebuffer) * swapchainImageCount);

	for(uint32_t i = 0 ; i < swapchainImageCount; i++)
	{

		VkImageView vkImageView_attachment_array[2];
		memset((void*)vkImageView_attachment_array, 0, sizeof(VkImageView) * _ARRAYSIZE(vkImageView_attachment_array));

		VkFramebufferCreateInfo vkFramebufferCreateInfo;
		memset((void*)&vkFramebufferCreateInfo, 0, sizeof(VkFramebufferCreateInfo));

		vkFramebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		vkFramebufferCreateInfo.pNext = NULL;
		vkFramebufferCreateInfo.flags = 0;
		vkFramebufferCreateInfo.renderPass = vkRenderPass;
		vkFramebufferCreateInfo.attachmentCount = _ARRAYSIZE(vkImageView_attachment_array);
		vkFramebufferCreateInfo.pAttachments = vkImageView_attachment_array;
		vkFramebufferCreateInfo.width = vkExtent2D_SwapChain.width;
		vkFramebufferCreateInfo.height = vkExtent2D_SwapChain.height;
		vkFramebufferCreateInfo.layers = 1;

		vkImageView_attachment_array[0] = swapChainImageView_array[i];
		vkImageView_attachment_array[1] = vkImageView_depth;

		vkResult = vkCreateFramebuffer(vkDevice, &vkFramebufferCreateInfo, NULL, &vkFramebuffer_array[i]);
		if (vkResult != VK_SUCCESS)
		{
			FileIO("CreateFramebuffers(): vkCreateFramebuffer() function failed with error code %d\n", vkResult);
			return vkResult;
		}
		else
		{
			FileIO("CreateFramebuffers(): vkCreateFramebuffer() succedded\n");
		}
	}

	return vkResult;
}

VkResult CreateSemaphores(void)
{

	VkResult vkResult = VK_SUCCESS;

	VkSemaphoreCreateInfo vkSemaphoreCreateInfo;
	memset((void*)&vkSemaphoreCreateInfo, 0, sizeof(VkSemaphoreCreateInfo));
	vkSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	vkSemaphoreCreateInfo.pNext = NULL;
	vkSemaphoreCreateInfo.flags = 0;

	vkResult = vkCreateSemaphore(vkDevice, &vkSemaphoreCreateInfo, NULL, &vkSemaphore_BackBuffer);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateSemaphores(): vkCreateSemaphore() function failed with error code %d for vkSemaphore_BackBuffer\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateSemaphores(): vkCreateSemaphore() succedded for vkSemaphore_BackBuffer\n");
	}

	vkResult = vkCreateSemaphore(vkDevice, &vkSemaphoreCreateInfo, NULL, &vkSemaphore_RenderComplete);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateSemaphores(): vkCreateSemaphore() function failed with error code %d for vkSemaphore_RenderComplete\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateSemaphores(): vkCreateSemaphore() succedded for vkSemaphore_RenderComplete\n");
	}

	return vkResult;
}

VkResult CreateFences(void)
{

	VkResult vkResult = VK_SUCCESS;

	VkFenceCreateInfo  vkFenceCreateInfo;
	memset((void*)&vkFenceCreateInfo, 0, sizeof(VkFenceCreateInfo));
	vkFenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	vkFenceCreateInfo.pNext = NULL;
	vkFenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	vkFence_array = (VkFence*)malloc(sizeof(VkFence) * swapchainImageCount);

	for(uint32_t i =0; i < swapchainImageCount; i++)
	{

		vkResult = vkCreateFence(vkDevice, &vkFenceCreateInfo, NULL, &vkFence_array[i]);
		if (vkResult != VK_SUCCESS)
		{
			FileIO("CreateFences(): vkCreateFence() function failed with error code %d at %d iteration\n", vkResult, i);
			return vkResult;
		}
		else
		{
			FileIO("CreateFences(): vkCreateFence() succedded at %d iteration\n", i);
		}
	}

	return vkResult;
}

VkResult buildCommandBuffers(void)
{

	VkResult vkResult = VK_SUCCESS;

	VkCommandBuffer *vkCommandBuffer_array = vkCommandBuffer_for_1024_x_1024_graphics_array;

	for(uint32_t i =0; i< swapchainImageCount; i++)
	{

		vkResult = vkResetCommandBuffer(vkCommandBuffer_array[i], 0);
		if (vkResult != VK_SUCCESS)
		{
			FileIO("buildCommandBuffers(): vkResetCommandBuffer() function failed with error code %d at %d iteration\n", vkResult, i);
			return vkResult;
		}

		VkCommandBufferBeginInfo vkCommandBufferBeginInfo;
		memset((void*)&vkCommandBufferBeginInfo, 0, sizeof(VkCommandBufferBeginInfo));
		vkCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		vkCommandBufferBeginInfo.pNext = NULL;
		vkCommandBufferBeginInfo.flags = 0;

		vkCommandBufferBeginInfo.pInheritanceInfo = NULL;

		vkResult = vkBeginCommandBuffer(vkCommandBuffer_array[i], &vkCommandBufferBeginInfo);
		if (vkResult != VK_SUCCESS)
		{
			FileIO("buildCommandBuffers(): vkBeginCommandBuffer() function failed with error code %d at %d iteration\n", vkResult, i);
			return vkResult;
		}

		VkClearValue vkClearValue_array[2];
		memset((void*)vkClearValue_array, 0, sizeof(VkClearValue) * _ARRAYSIZE(vkClearValue_array));
		// Black background for white wireframe
		vkClearValue_array[0].color = vkClearColorValue;
		vkClearValue_array[1].depthStencil = vkClearDepthStencilValue;

		VkRenderPassBeginInfo vkRenderPassBeginInfo;
		memset((void*)&vkRenderPassBeginInfo, 0, sizeof(VkRenderPassBeginInfo));
		vkRenderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		vkRenderPassBeginInfo.pNext = NULL;
		vkRenderPassBeginInfo.renderPass = vkRenderPass;

		vkRenderPassBeginInfo.renderArea.offset.x = 0;
		vkRenderPassBeginInfo.renderArea.offset.y = 0;
		vkRenderPassBeginInfo.renderArea.extent.width = vkExtent2D_SwapChain.width;
		vkRenderPassBeginInfo.renderArea.extent.height = vkExtent2D_SwapChain.height;

		vkRenderPassBeginInfo.clearValueCount = _ARRAYSIZE(vkClearValue_array);
		vkRenderPassBeginInfo.pClearValues = vkClearValue_array;

		vkRenderPassBeginInfo.framebuffer = vkFramebuffer_array[i];

		vkCmdBeginRenderPass(vkCommandBuffer_array[i], &vkRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(vkCommandBuffer_array[i], VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipeline);

		vkCmdBindDescriptorSets(vkCommandBuffer_array[i], VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipelineLayout, 0, 1, &vkDescriptorSet, 0, NULL);

		VkDeviceSize vkDeviceSize_offset_array[1];
		memset((void*)vkDeviceSize_offset_array, 0, sizeof(VkDeviceSize) * _ARRAYSIZE(vkDeviceSize_offset_array));

		vkCmdBindVertexBuffers(vkCommandBuffer_array[i], 0, 1, &vertexData_external.vkBuffer, vkDeviceSize_offset_array);

		vkCmdDrawIndirect(vkCommandBuffer_array[i], vertexdata_indirect_buffer.vkBuffer, 0, 1, 0);

		vkCmdEndRenderPass(vkCommandBuffer_array[i]);

		vkResult = vkEndCommandBuffer(vkCommandBuffer_array[i]);
		if (vkResult != VK_SUCCESS)
		{
			FileIO("buildCommandBuffers(): vkEndCommandBuffer() function failed with error code %d at %d iteration\n", vkResult, i);
			return vkResult;
		}

	}

	return vkResult;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallback(VkDebugReportFlagsEXT vkDebugReportFlagsEXT, VkDebugReportObjectTypeEXT vkDebugReportObjectTypeEXT, uint64_t object, size_t location,  int32_t messageCode,const char* pLayerPrefix, const char* pMessage, void* pUserData)
{

	FileIO("Anjaneya_VALIDATION:debugReportCallback():%s(%d) = %s\n", pLayerPrefix, messageCode, pMessage);
    return (VK_FALSE);
}