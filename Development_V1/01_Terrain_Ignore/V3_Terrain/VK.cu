#include <stdio.h>		
#include <stdlib.h>	
#include <windows.h>	
#include <math.h>	

#include "VK.h"			
#define LOG_FILE (char*)"Log.txt" 

//Vulkan related header files
#define VK_USE_PLATFORM_WIN32_KHR // XLIB_KHR, MACOS_KHR & MOLTEN something
#include <vulkan/vulkan.h> //(Only those members are enabled connected with above macro {conditional compilation using #ifdef internally})

//GLM related macro and header files
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

//Vulkan related libraries
#pragma comment(lib, "vulkan-1.lib")

//Cuda related headers
#include <cuda_runtime.h>

// Global Function Declarations
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

const char* gpszAppName = "ARTR";

HWND ghwnd = NULL;
BOOL gbActive = FALSE;
DWORD dwStyle = 0;
//WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) }; //dont do this as cpp style
WINDOWPLACEMENT wpPrev;
BOOL gbFullscreen = FALSE;
BOOL bWindowMinimize = FALSE;

// Global Variable Declarations

//Vulkan related global variables

//Instance extension related variables
uint32_t enabledInstanceExtensionsCount = 0;
/*
VK_KHR_SURFACE_EXTENSION_NAME
VK_KHR_WIN32_SURFACE_EXTENSION_NAME
and
Added in 21_validation: VK_EXT_DEBUG_REPORT_EXTENSION_NAME (https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_debug_report.html)
*/
//const char* enabledInstanceExtensionNames_array[2];
const char* enabledInstanceExtensionNames_array[3];

//Vulkan Instance
VkInstance vkInstance = VK_NULL_HANDLE;

//Vulkan Presentation Surface
/*
Declare a global variable to hold presentation surface object
*/
VkSurfaceKHR vkSurfaceKHR = VK_NULL_HANDLE;

/*
Vulkan Physical device related global variables
*/
VkPhysicalDevice vkPhysicalDevice_selected = VK_NULL_HANDLE;//https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDevice.html
uint32_t graphicsQuequeFamilyIndex_selected = UINT32_MAX; //ata max aahe mag apan proper count deu
VkPhysicalDeviceMemoryProperties vkPhysicalDeviceMemoryProperties; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceMemoryProperties.html (Itha nahi lagnaar, staging ani non staging buffers la lagel)

/*
PrintVulkanInfo() changes
1. Remove local declaration of physicalDeviceCount and physicalDeviceArray from GetPhysicalDevice() and do it globally.
*/
uint32_t physicalDeviceCount = 0;
VkPhysicalDevice *vkPhysicalDevice_array = NULL; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDevice.html

//Device extension related variables {In MAC , we need to add portability etensions, so there will be 2 extensions. Similarly for ray tracing there will be atleast 8 extensions.}
uint32_t enabledDeviceExtensionsCount = 0;
/*
VK_KHR_SWAPCHAIN_EXTENSION_NAME
VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME
VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME - for CUDA-Vulkan timeline semaphore interop
*/
const char* enabledDeviceExtensionNames_array[3]; // Added for external semaphore support

/*
Vulkan Device
*/
VkDevice vkDevice = VK_NULL_HANDLE; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkDevice.html

/*
Device Queque
*/
VkQueue vkQueue =  VK_NULL_HANDLE; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueue.html

/*
Color Format and Color Space
*/
VkFormat vkFormat_color = VK_FORMAT_UNDEFINED; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormat.html {Will be also needed for depth later}
VkColorSpaceKHR vkColorSpaceKHR = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkColorSpaceKHR.html

/*
Presentation Mode
https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSurfacePresentModesKHR.html
https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentModeKHR.html
*/
VkPresentModeKHR vkPresentModeKHR = VK_PRESENT_MODE_FIFO_KHR; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentModeKHR.html

/*
SwapChain Related Global variables
*/
int winWidth = WIN_WIDTH;
int winHeight = WIN_HEIGHT;

//https://registry.khronos.org/vulkan/specs/latest/man/html/VkSwapchainKHR.html
VkSwapchainKHR vkSwapchainKHR =  VK_NULL_HANDLE;

//https://registry.khronos.org/vulkan/specs/latest/man/html/VkExtent2D.html
VkExtent2D vkExtent2D_SwapChain;

/*
Swapchain images and Swapchain image views related variables
*/
uint32_t swapchainImageCount = UINT32_MAX;

//https://registry.khronos.org/vulkan/specs/latest/man/html/VkImage.html
VkImage *swapChainImage_array = NULL;

//https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageView.html
VkImageView *swapChainImageView_array = NULL;

/*
Command Pool
*/
//https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandPool.html
VkCommandPool vkCommandPool = VK_NULL_HANDLE; 

/*
RenderPass
*/
//https://registry.khronos.org/vulkan/specs/latest/man/html/VkRenderPass.html
VkRenderPass vkRenderPass = VK_NULL_HANDLE;

/*
Framebuffers
The number framebuffers should be equal to number of swapchain images
*/
//https://registry.khronos.org/vulkan/specs/latest/man/html/VkFramebuffer.html
VkFramebuffer *vkFramebuffer_array = NULL;

/*
Fences and Semaphores
18_1. Globally declare an array of fences of pointer type VkFence (https://registry.khronos.org/vulkan/specs/latest/man/html/VkFence.html).
	Additionally declare 2 semaphore objects of type VkSemaphore (https://registry.khronos.org/vulkan/specs/latest/man/html/VkSemaphore.html)
*/

//https://registry.khronos.org/vulkan/specs/latest/man/html/VkSemaphore.html
VkSemaphore vkSemaphore_BackBuffer = VK_NULL_HANDLE;
VkSemaphore vkSemaphore_RenderComplete = VK_NULL_HANDLE;

//https://registry.khronos.org/vulkan/specs/latest/man/html/VkFence.html
VkFence *vkFence_array = NULL;

/*
19_Build_Command_Buffers: Clear Colors
*/

/*
// Provided by VK_VERSION_1_0
typedef union VkClearColorValue {
    float       float32[4]; //RGBA member to be used if vkFormat is float //In our case vkFormat it is unmorm, so we will use float one
    int32_t     int32[4]; //RGBA member to be used if vkFormat is int
    uint32_t    uint32[4]; //RGBA member to be used if vkFormat is uint32_t
} VkClearColorValue;
*/
VkClearColorValue vkClearColorValue;

//https://registry.khronos.org/vulkan/specs/latest/man/html/VkClearDepthStencilValue.html
VkClearDepthStencilValue vkClearDepthStencilValue;

/*
20_Render
*/
BOOL bInitialized = FALSE;
uint32_t currentImageIndex = UINT32_MAX; //UINT_MAX is also ok

/*
21_Validation
*/
BOOL bValidation = TRUE;
uint32_t enabledValidationLayerCount = 0;
const char* enabledValidationlayerNames_array[1]; //For VK_LAYER_KHRONOS_validation
VkDebugReportCallbackEXT vkDebugReportCallbackEXT = VK_NULL_HANDLE; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugReportCallbackEXT.html

//https://registry.khronos.org/vulkan/specs/latest/man/html/PFN_vkDebugReportCallbackEXT.html 
PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT_fnptr = NULL; 

//22. Vertex Buffer related steps
/*
1. Globally Declare a structure holding Vertex buffer related two things
 a. VkBuffer Object
 b. VkDeviceMemory Object
	We will call it as struct VertexData and declare a global variable of this structure named vertexData_position.
*/
typedef struct 
{
	VkBuffer vkBuffer; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkBuffer.html
	VkDeviceMemory vkDeviceMemory; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceMemory.html
}VertexData;

//31-Ortho: Uniform Buffer (Uniform related declarations)
//31.1
struct MyUniformData
{
	glm::mat4 modelMatrix;
	glm::mat4 viewMatrix;
	glm::mat4 projectionMatrix;
	glm::vec4 color;
};

// Tessellation Parameters for LOD-based dynamic tessellation
struct TessellationParams
{
	glm::vec4 cameraPos;      // Camera world position (xyz) + padding (w)
	float minTessLevel;       // Minimum tessellation level (close to camera)
	float maxTessLevel;       // Maximum tessellation level (far from camera)
	float minDistance;        // Distance for maximum tessellation
	float maxDistance;        // Distance for minimum tessellation
};

//31.1
struct UniformData
{
	VkBuffer vkBuffer; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkBuffer.html
	VkDeviceMemory vkDeviceMemory; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceMemory.html
};

//31.1
struct UniformData uniformData;

// Tessellation uniform buffer
struct UniformData tessUniformData;

// Height map texture for tessellation displacement
VkImage vkImage_heightMap = VK_NULL_HANDLE;
VkDeviceMemory vkDeviceMemory_heightMap = VK_NULL_HANDLE;
VkImageView vkImageView_heightMap = VK_NULL_HANDLE;
VkSampler vkSampler_heightMap = VK_NULL_HANDLE;

// CUDA external memory for height map
cudaExternalMemory_t cuExternalMemory_heightMap;
float* heightMap_CUDA = NULL;

// Height map dimensions - 4096x4096 for 8km x 8km terrain
#define HEIGHTMAP_WIDTH 4096
#define HEIGHTMAP_HEIGHT 4096

// Terrain scale in meters (8km x 8km)
#define TERRAIN_SIZE_METERS 8000.0f
#define TERRAIN_MAX_HEIGHT_METERS 1500.0f  // Maximum mountain height

// Patch grid dimensions for tessellation (coarse grid)
#define PATCH_GRID_WIDTH 257
#define PATCH_GRID_HEIGHT 257

//23. Shader related variables
/*
1. Write Shaders  and compile them to SPIRV using shader compilation tools that we receive in Vulkan SDK.
2. Globally declate 2 shader object module variables of VkShaderModule type to hold Vulkan compatible vertex shader module object and fragment shader module object respectively.
*/
VkShaderModule vkShaderMoudule_vertex_shader = VK_NULL_HANDLE;
VkShaderModule vkShaderMoudule_fragment_shader = VK_NULL_HANDLE;

// Tessellation shader modules
VkShaderModule vkShaderModule_tesc = VK_NULL_HANDLE;  // Tessellation Control Shader
VkShaderModule vkShaderModule_tese = VK_NULL_HANDLE;  // Tessellation Evaluation Shader 

/*24. Descriptor Set Layout
https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorSetLayout.html
24.1. Globally declare Vulkan object of type VkDescriptorSetLayout and initialize it to VK_NULL_HANDLE.
*/
VkDescriptorSetLayout vkDescriptorSetLayout = VK_NULL_HANDLE;

/* 25. Pipeline layout
25.1. Globally declare Vulkan object of type VkPipelineLayout and initialize it to VK_NULL_HANDLE.
*/
VkPipelineLayout vkPipelineLayout = VK_NULL_HANDLE; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineLayout.html

//31.1
//Descriptor Pool : https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorPool.html
VkDescriptorPool vkDescriptorPool = VK_NULL_HANDLE;

//31.1
//Descriptor Set : https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorSet.html
VkDescriptorSet vkDescriptorSet = VK_NULL_HANDLE;

/*
26 Pipeline
*/

/*
//https://registry.khronos.org/vulkan/specs/latest/man/html/VkViewport.html
typedef struct VkViewport {
    float    x;
    float    y;
    float    width;
    float    height;
    float    minDepth;
    float    maxDepth;
} VkViewport;
*/
VkViewport vkViewPort;

/*
https://registry.khronos.org/vulkan/specs/latest/man/html/VkRect2D.html
// Provided by VK_VERSION_1_0
typedef struct VkRect2D {
    VkOffset2D    offset; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkOffset2D.html
    VkExtent2D    extent; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkExtent2D.html
} VkRect2D;

// Provided by VK_VERSION_1_0
typedef struct VkOffset2D {
    int32_t    x;
    int32_t    y;
} VkOffset2D;

// Provided by VK_VERSION_1_0
typedef struct VkExtent2D {
    uint32_t    width;
    uint32_t    height;
} VkExtent2D;
*/
VkRect2D vkRect2D_scissor;

VkPipeline vkPipeline = VK_NULL_HANDLE; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipeline.html

// Depth changes
VkFormat vkFormat_depth = VK_FORMAT_UNDEFINED;
VkImage vkImage_depth = VK_NULL_HANDLE;
VkDeviceMemory vkDeviceMemory_depth = VK_NULL_HANDLE;
VkImageView vkImageView_depth = VK_NULL_HANDLE;

VkCommandBuffer *vkCommandBuffer_for_1024_x_1024_graphics_array = NULL;
BOOL bMesh1024_chosen = true;

char colorFromKey = 'O';
float animationTime = 0.0f;

//CUDA based global variables
cudaError_t cudaResult;

//https://registry.khronos.org/VulkanSC/specs/1.0-extensions/man/html/VkExternalMemoryHandleTypeFlagBits.html
VkExternalMemoryHandleTypeFlagBits vkExternalMemoryHandleTypeFlagBits;

//https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html
cudaExternalMemory_t cuExternalMemory_t;

void* pos_CUDA = NULL;

// CUDA Stream for async operations
cudaStream_t cudaStream_compute = NULL;

// CUDA-Vulkan Timeline Semaphore interop
cudaExternalSemaphore_t cudaExtSemaphore_timelineVulkan = NULL;
VkSemaphore vkSemaphore_timeline = VK_NULL_HANDLE;
uint64_t timelineSemaphoreValue = 0;
BOOL bTimelineSemaphoreSupported = FALSE;

// Optimized thread block configuration constants
#define CUDA_BLOCK_SIZE_X 16
#define CUDA_BLOCK_SIZE_Y 16
#define CUDA_THREADS_PER_BLOCK (CUDA_BLOCK_SIZE_X * CUDA_BLOCK_SIZE_Y) // 256 threads for better occupancy

// Double/Triple Buffering Configuration
// NUM_VERTEX_BUFFERS: Set to 2 for double buffering, 3 for triple buffering
// Double buffering: One buffer for rendering while other is being updated by CUDA
// Triple buffering: Allows better pipelining with one buffer ready, one rendering, one computing
#define NUM_VERTEX_BUFFERS 3  // Triple buffering for optimal performance

// Shared memory stencil configuration
// STENCIL_RADIUS: The radius of the stencil operation (1 = 3x3 kernel, 2 = 5x5 kernel)
// STENCIL_TILE_SIZE: Tile size for shared memory caching
#define STENCIL_RADIUS 1
#define STENCIL_TILE_SIZE 16
#define STENCIL_BLOCK_SIZE (STENCIL_TILE_SIZE + 2 * STENCIL_RADIUS) // 18 for radius 1

// Multiple vertex buffers for double/triple buffering
VertexData vertexData_buffers[NUM_VERTEX_BUFFERS]; // Array of vertex buffers
void* pos_CUDA_buffers[NUM_VERTEX_BUFFERS] = {NULL}; // CUDA pointers for each buffer
cudaExternalMemory_t cuExternalMemory_buffers[NUM_VERTEX_BUFFERS]; // External memory handles

// Current buffer indices for ping-pong buffering
// computeBufferIndex: Buffer being written to by CUDA kernel
// renderBufferIndex: Buffer being read by Vulkan renderer
unsigned int computeBufferIndex = 0;
unsigned int renderBufferIndex = 0;

// Legacy single buffer (kept for backward compatibility)
VertexData vertexData_external; //will be mapped with pos_CUDA later

VertexData vertexdata_indirect_buffer; //Indirect buffer for vkCmdDrawIndirect
BOOL bIndirectBufferMemoryCoherent = TRUE; //Track if indirect buffer memory is HOST_COHERENT

int iResult = 0;

void FileIO(const char* format, ...)
{
	FILE* file = fopen(LOG_FILE, "a"); // Open in append mode
	if (file)
	{
		va_list args;
		va_start(args, format);
		vfprintf(file, format, args);
		va_end(args);
		fclose(file);
	}
}

// sinewave kernel - OPTIMIZED for memory coalescing and performance
// OPTIMIZATION 4: Memory Coalescing Improvements
// - Uses __restrict__ for pointer aliasing optimization
// - Bounds checking for robustness with any mesh size
// - Row-major memory access pattern ensures coalesced writes
// - Pre-computed reciprocals to reduce division operations
// - sinf/cosf intrinsics for fast math
__global__ void sinewave(float4 * __restrict__ pos, unsigned int width, unsigned int height,
                         float animTime) {
  // Calculate global thread indices
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Bounds checking - allows kernel to work with any mesh dimensions
  // This is essential when grid dimensions don't perfectly match mesh size
  if(x >= width || y >= height)
    return;

  // Pre-compute reciprocals for division optimization
  // Division is expensive; multiplication by reciprocal is faster
  const float invWidth = 1.0f / (float)width;
  const float invHeight = 1.0f / (float)height;

  // Normalize coordinates to [0,1] range then scale to [-1,1]
  float u = x * invWidth;
  float v = y * invHeight;

  // Transform to [-1, 1] range using fused multiply-add for efficiency
  u = __fmaf_rn(u, 2.0f, -1.0f);  // u = u * 2.0 - 1.0
  v = __fmaf_rn(v, 2.0f, -1.0f);  // v = v * 2.0 - 1.0

  // Compute sine wave height with frequency modulation
  const float freq = 4.0f;

  // Pre-compute the phase terms for both sin and cos
  const float phaseU = __fmaf_rn(freq, u, animTime);  // freq * u + animTime
  const float phaseV = __fmaf_rn(freq, v, animTime);  // freq * v + animTime

  // Use fast intrinsic sin/cos functions
  const float w = __sinf(phaseU) * __cosf(phaseV) * 0.5f;

  // MEMORY COALESCING: Write to row-major linear memory
  // When threads in a warp write to consecutive addresses (y * width + x),
  // memory transactions are coalesced into fewer operations
  // This is optimal because adjacent threads (x, x+1, x+2...) in a warp
  // write to adjacent memory locations
  const unsigned int idx = y * width + x;
  pos[idx] = make_float4(u, w, v, 1.0f);
}

// ============================================================================
// SHARED MEMORY STENCIL KERNEL - Smoothing filter using 2D stencil operation
// ============================================================================
// This kernel implements a 3x3 averaging (box filter) stencil operation using
// shared memory for efficient neighbor access. Stencil operations are common
// in image processing, physics simulations, and mesh smoothing.
//
// OPTIMIZATION: Shared Memory Tiling
// - Each thread block loads a tile of data into shared memory
// - Halo regions (apron) are loaded for boundary handling
// - Reduces global memory bandwidth by factor of ~9x for 3x3 stencil
// - Avoids redundant global memory reads for overlapping neighborhoods
//
// Memory layout: Each tile includes STENCIL_RADIUS border pixels on each side
// ============================================================================
__global__ void stencilSmooth(float4 * __restrict__ dst,
                               const float4 * __restrict__ src,
                               unsigned int width, unsigned int height) {
    // Shared memory tile with halo/apron for stencil neighborhood
    // Size: (TILE + 2*RADIUS)^2 to accommodate border pixels
    __shared__ float4 tile[STENCIL_BLOCK_SIZE][STENCIL_BLOCK_SIZE];

    // Global coordinates this thread is responsible for
    const int gx = blockIdx.x * STENCIL_TILE_SIZE + threadIdx.x;
    const int gy = blockIdx.y * STENCIL_TILE_SIZE + threadIdx.y;

    // Local coordinates within the shared memory tile (offset by RADIUS for halo)
    const int lx = threadIdx.x + STENCIL_RADIUS;
    const int ly = threadIdx.y + STENCIL_RADIUS;

    // ========================================================================
    // PHASE 1: Load tile data into shared memory (including halo regions)
    // ========================================================================

    // Load center pixel (main tile)
    if(gx < width && gy < height) {
        tile[ly][lx] = src[gy * width + gx];
    } else {
        tile[ly][lx] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    // Load halo regions (border pixels needed for stencil computation)
    // Left halo
    if(threadIdx.x < STENCIL_RADIUS) {
        int haloX = gx - STENCIL_RADIUS;
        if(haloX >= 0 && gy < height) {
            tile[ly][threadIdx.x] = src[gy * width + haloX];
        } else {
            tile[ly][threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    // Right halo
    if(threadIdx.x >= STENCIL_TILE_SIZE - STENCIL_RADIUS) {
        int haloX = gx + STENCIL_RADIUS;
        if(haloX < width && gy < height) {
            tile[ly][lx + STENCIL_RADIUS] = src[gy * width + haloX];
        } else {
            tile[ly][lx + STENCIL_RADIUS] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    // Top halo
    if(threadIdx.y < STENCIL_RADIUS) {
        int haloY = gy - STENCIL_RADIUS;
        if(haloY >= 0 && gx < width) {
            tile[threadIdx.y][lx] = src[haloY * width + gx];
        } else {
            tile[threadIdx.y][lx] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    // Bottom halo
    if(threadIdx.y >= STENCIL_TILE_SIZE - STENCIL_RADIUS) {
        int haloY = gy + STENCIL_RADIUS;
        if(haloY < height && gx < width) {
            tile[ly + STENCIL_RADIUS][lx] = src[haloY * width + gx];
        } else {
            tile[ly + STENCIL_RADIUS][lx] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    // Corner halos (top-left, top-right, bottom-left, bottom-right)
    // Top-left corner
    if(threadIdx.x < STENCIL_RADIUS && threadIdx.y < STENCIL_RADIUS) {
        int haloX = gx - STENCIL_RADIUS;
        int haloY = gy - STENCIL_RADIUS;
        if(haloX >= 0 && haloY >= 0) {
            tile[threadIdx.y][threadIdx.x] = src[haloY * width + haloX];
        } else {
            tile[threadIdx.y][threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    // Top-right corner
    if(threadIdx.x >= STENCIL_TILE_SIZE - STENCIL_RADIUS && threadIdx.y < STENCIL_RADIUS) {
        int haloX = gx + STENCIL_RADIUS;
        int haloY = gy - STENCIL_RADIUS;
        if(haloX < width && haloY >= 0) {
            tile[threadIdx.y][lx + STENCIL_RADIUS] = src[haloY * width + haloX];
        } else {
            tile[threadIdx.y][lx + STENCIL_RADIUS] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    // Bottom-left corner
    if(threadIdx.x < STENCIL_RADIUS && threadIdx.y >= STENCIL_TILE_SIZE - STENCIL_RADIUS) {
        int haloX = gx - STENCIL_RADIUS;
        int haloY = gy + STENCIL_RADIUS;
        if(haloX >= 0 && haloY < height) {
            tile[ly + STENCIL_RADIUS][threadIdx.x] = src[haloY * width + haloX];
        } else {
            tile[ly + STENCIL_RADIUS][threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    // Bottom-right corner
    if(threadIdx.x >= STENCIL_TILE_SIZE - STENCIL_RADIUS && threadIdx.y >= STENCIL_TILE_SIZE - STENCIL_RADIUS) {
        int haloX = gx + STENCIL_RADIUS;
        int haloY = gy + STENCIL_RADIUS;
        if(haloX < width && haloY < height) {
            tile[ly + STENCIL_RADIUS][lx + STENCIL_RADIUS] = src[haloY * width + haloX];
        } else {
            tile[ly + STENCIL_RADIUS][lx + STENCIL_RADIUS] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    // Synchronize to ensure all shared memory loads are complete
    __syncthreads();

    // ========================================================================
    // PHASE 2: Apply stencil operation using shared memory data
    // ========================================================================

    // Bounds check for output
    if(gx >= width || gy >= height)
        return;

    // Apply 3x3 box filter (averaging) stencil
    // Weights: All 9 neighbors contribute equally (1/9 each)
    // This smooths the height (Y) values while preserving X/Z coordinates
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;

    #pragma unroll
    for(int dy = -STENCIL_RADIUS; dy <= STENCIL_RADIUS; dy++) {
        #pragma unroll
        for(int dx = -STENCIL_RADIUS; dx <= STENCIL_RADIUS; dx++) {
            float4 neighbor = tile[ly + dy][lx + dx];
            // Only smooth the height (y component), keep x and z from center
            sum.y += neighbor.y;
            weightSum += 1.0f;
        }
    }

    // Get center pixel for x, z coordinates (we only smooth height)
    float4 center = tile[ly][lx];

    // Output: Keep original x, z positions but use smoothed height
    // w component is always 1.0 for homogeneous coordinates
    dst[gy * width + gx] = make_float4(center.x, sum.y / weightSum, center.z, 1.0f);
}

// ============================================================================
// Combined Sinewave + Stencil kernel - Generates wave then applies smoothing
// ============================================================================
// This kernel demonstrates the double buffering pattern:
// 1. Generate sinewave into temporary shared memory buffer
// 2. Apply stencil smoothing and write to output
// All done within a single kernel to minimize memory traffic
// ============================================================================
__global__ void sinewaveWithStencil(float4 * __restrict__ pos,
                                     unsigned int width, unsigned int height,
                                     float animTime, float smoothFactor) {
    // Shared memory for the tile with halo
    __shared__ float4 tile[STENCIL_BLOCK_SIZE][STENCIL_BLOCK_SIZE];

    // Calculate global thread indices
    const unsigned int gx = blockIdx.x * STENCIL_TILE_SIZE + threadIdx.x;
    const unsigned int gy = blockIdx.y * STENCIL_TILE_SIZE + threadIdx.y;

    // Local indices within shared memory tile
    const int lx = threadIdx.x + STENCIL_RADIUS;
    const int ly = threadIdx.y + STENCIL_RADIUS;

    // Pre-compute reciprocals
    const float invWidth = 1.0f / (float)width;
    const float invHeight = 1.0f / (float)height;
    const float freq = 4.0f;

    // ========================================================================
    // PHASE 1: Generate sinewave values into shared memory
    // ========================================================================

    // Helper lambda to compute sinewave height at given coordinates
    #define COMPUTE_WAVE(px, py) \
        (((px) < width && (py) < height) ? \
            (__sinf(__fmaf_rn(freq, __fmaf_rn((px) * invWidth, 2.0f, -1.0f), animTime)) * \
             __cosf(__fmaf_rn(freq, __fmaf_rn((py) * invHeight, 2.0f, -1.0f), animTime)) * 0.5f) : 0.0f)

    // Load center value
    if(gx < width && gy < height) {
        float u = __fmaf_rn(gx * invWidth, 2.0f, -1.0f);
        float v = __fmaf_rn(gy * invHeight, 2.0f, -1.0f);
        float w = __sinf(__fmaf_rn(freq, u, animTime)) * __cosf(__fmaf_rn(freq, v, animTime)) * 0.5f;
        tile[ly][lx] = make_float4(u, w, v, 1.0f);
    } else {
        tile[ly][lx] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    // Load halo regions - compute sinewave for neighbor pixels
    // Left halo
    if(threadIdx.x < STENCIL_RADIUS) {
        int hx = gx - STENCIL_RADIUS;
        if(hx >= 0 && gy < height) {
            float u = __fmaf_rn(hx * invWidth, 2.0f, -1.0f);
            float v = __fmaf_rn(gy * invHeight, 2.0f, -1.0f);
            float w = __sinf(__fmaf_rn(freq, u, animTime)) * __cosf(__fmaf_rn(freq, v, animTime)) * 0.5f;
            tile[ly][threadIdx.x] = make_float4(u, w, v, 1.0f);
        } else {
            tile[ly][threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    // Right halo
    if(threadIdx.x >= STENCIL_TILE_SIZE - STENCIL_RADIUS) {
        int hx = gx + STENCIL_RADIUS;
        if(hx < width && gy < height) {
            float u = __fmaf_rn(hx * invWidth, 2.0f, -1.0f);
            float v = __fmaf_rn(gy * invHeight, 2.0f, -1.0f);
            float w = __sinf(__fmaf_rn(freq, u, animTime)) * __cosf(__fmaf_rn(freq, v, animTime)) * 0.5f;
            tile[ly][lx + STENCIL_RADIUS] = make_float4(u, w, v, 1.0f);
        } else {
            tile[ly][lx + STENCIL_RADIUS] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    // Top halo
    if(threadIdx.y < STENCIL_RADIUS) {
        int hy = gy - STENCIL_RADIUS;
        if(hy >= 0 && gx < width) {
            float u = __fmaf_rn(gx * invWidth, 2.0f, -1.0f);
            float v = __fmaf_rn(hy * invHeight, 2.0f, -1.0f);
            float w = __sinf(__fmaf_rn(freq, u, animTime)) * __cosf(__fmaf_rn(freq, v, animTime)) * 0.5f;
            tile[threadIdx.y][lx] = make_float4(u, w, v, 1.0f);
        } else {
            tile[threadIdx.y][lx] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    // Bottom halo
    if(threadIdx.y >= STENCIL_TILE_SIZE - STENCIL_RADIUS) {
        int hy = gy + STENCIL_RADIUS;
        if(hy < height && gx < width) {
            float u = __fmaf_rn(gx * invWidth, 2.0f, -1.0f);
            float v = __fmaf_rn(hy * invHeight, 2.0f, -1.0f);
            float w = __sinf(__fmaf_rn(freq, u, animTime)) * __cosf(__fmaf_rn(freq, v, animTime)) * 0.5f;
            tile[ly + STENCIL_RADIUS][lx] = make_float4(u, w, v, 1.0f);
        } else {
            tile[ly + STENCIL_RADIUS][lx] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    // Corners
    if(threadIdx.x < STENCIL_RADIUS && threadIdx.y < STENCIL_RADIUS) {
        int hx = gx - STENCIL_RADIUS, hy = gy - STENCIL_RADIUS;
        if(hx >= 0 && hy >= 0) {
            float u = __fmaf_rn(hx * invWidth, 2.0f, -1.0f);
            float v = __fmaf_rn(hy * invHeight, 2.0f, -1.0f);
            float w = __sinf(__fmaf_rn(freq, u, animTime)) * __cosf(__fmaf_rn(freq, v, animTime)) * 0.5f;
            tile[threadIdx.y][threadIdx.x] = make_float4(u, w, v, 1.0f);
        } else {
            tile[threadIdx.y][threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    if(threadIdx.x >= STENCIL_TILE_SIZE - STENCIL_RADIUS && threadIdx.y < STENCIL_RADIUS) {
        int hx = gx + STENCIL_RADIUS, hy = gy - STENCIL_RADIUS;
        if(hx < width && hy >= 0) {
            float u = __fmaf_rn(hx * invWidth, 2.0f, -1.0f);
            float v = __fmaf_rn(hy * invHeight, 2.0f, -1.0f);
            float w = __sinf(__fmaf_rn(freq, u, animTime)) * __cosf(__fmaf_rn(freq, v, animTime)) * 0.5f;
            tile[threadIdx.y][lx + STENCIL_RADIUS] = make_float4(u, w, v, 1.0f);
        } else {
            tile[threadIdx.y][lx + STENCIL_RADIUS] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    if(threadIdx.x < STENCIL_RADIUS && threadIdx.y >= STENCIL_TILE_SIZE - STENCIL_RADIUS) {
        int hx = gx - STENCIL_RADIUS, hy = gy + STENCIL_RADIUS;
        if(hx >= 0 && hy < height) {
            float u = __fmaf_rn(hx * invWidth, 2.0f, -1.0f);
            float v = __fmaf_rn(hy * invHeight, 2.0f, -1.0f);
            float w = __sinf(__fmaf_rn(freq, u, animTime)) * __cosf(__fmaf_rn(freq, v, animTime)) * 0.5f;
            tile[ly + STENCIL_RADIUS][threadIdx.x] = make_float4(u, w, v, 1.0f);
        } else {
            tile[ly + STENCIL_RADIUS][threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    if(threadIdx.x >= STENCIL_TILE_SIZE - STENCIL_RADIUS && threadIdx.y >= STENCIL_TILE_SIZE - STENCIL_RADIUS) {
        int hx = gx + STENCIL_RADIUS, hy = gy + STENCIL_RADIUS;
        if(hx < width && hy < height) {
            float u = __fmaf_rn(hx * invWidth, 2.0f, -1.0f);
            float v = __fmaf_rn(hy * invHeight, 2.0f, -1.0f);
            float w = __sinf(__fmaf_rn(freq, u, animTime)) * __cosf(__fmaf_rn(freq, v, animTime)) * 0.5f;
            tile[ly + STENCIL_RADIUS][lx + STENCIL_RADIUS] = make_float4(u, w, v, 1.0f);
        } else {
            tile[ly + STENCIL_RADIUS][lx + STENCIL_RADIUS] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    #undef COMPUTE_WAVE

    // Synchronize to ensure all shared memory is populated
    __syncthreads();

    // ========================================================================
    // PHASE 2: Apply stencil smoothing and write output
    // ========================================================================

    if(gx >= width || gy >= height)
        return;

    // Compute smoothed height using 3x3 box filter from shared memory
    float smoothedY = 0.0f;
    float weightSum = 0.0f;

    #pragma unroll
    for(int dy = -STENCIL_RADIUS; dy <= STENCIL_RADIUS; dy++) {
        #pragma unroll
        for(int dx = -STENCIL_RADIUS; dx <= STENCIL_RADIUS; dx++) {
            smoothedY += tile[ly + dy][lx + dx].y;
            weightSum += 1.0f;
        }
    }
    smoothedY /= weightSum;

    // Get center pixel data
    float4 center = tile[ly][lx];

    // Blend between original and smoothed based on smoothFactor
    // smoothFactor = 0.0: No smoothing (original sinewave)
    // smoothFactor = 1.0: Full smoothing
    float finalY = __fmaf_rn(smoothFactor, smoothedY - center.y, center.y);

    // Write output with coalesced memory access
    const unsigned int idx = gy * width + gx;
    pos[idx] = make_float4(center.x, finalY, center.z, 1.0f);
}

// ============================================================================
// Perlin Noise Implementation for Unigine Valley-style Terrain
// ============================================================================
// GPU-optimized Perlin noise with multiple octaves (fBm)
// ============================================================================

// Permutation table for Perlin noise (static, pre-computed)
__constant__ unsigned char perm[512] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

// Fade function for smooth interpolation: 6t^5 - 15t^4 + 10t^3
__device__ __forceinline__ float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// Linear interpolation
__device__ __forceinline__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// Smoothstep function (GLSL-compatible) - performs Hermite interpolation
__device__ __forceinline__ float smoothstep(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Gradient function - returns dot product of gradient vector and distance vector
__device__ __forceinline__ float grad(int hash, float x, float y) {
    int h = hash & 7;
    float u = h < 4 ? x : y;
    float v = h < 4 ? y : x;
    return ((h & 1) ? -u : u) + ((h & 2) ? -2.0f * v : 2.0f * v);
}

// 2D Perlin noise function
__device__ float perlinNoise2D(float x, float y) {
    // Find unit grid cell containing point
    int X = (int)floorf(x) & 255;
    int Y = (int)floorf(y) & 255;

    // Get relative position within cell
    x -= floorf(x);
    y -= floorf(y);

    // Compute fade curves
    float u = fade(x);
    float v = fade(y);

    // Hash coordinates of the 4 corners
    int A = perm[X] + Y;
    int B = perm[X + 1] + Y;

    // Interpolate between corner gradients
    float res = lerp(
        lerp(grad(perm[A], x, y), grad(perm[B], x - 1.0f, y), u),
        lerp(grad(perm[A + 1], x, y - 1.0f), grad(perm[B + 1], x - 1.0f, y - 1.0f), u),
        v
    );

    return res;
}

// Fractional Brownian Motion (fBm) - multiple octaves of Perlin noise
__device__ float fbm(float x, float y, int octaves, float persistence, float lacunarity) {
    float total = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float maxValue = 0.0f;

    for (int i = 0; i < octaves; i++) {
        total += perlinNoise2D(x * frequency, y * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    return total / maxValue;
}

// Ridge noise - creates sharp ridges like mountain ranges
__device__ float ridgeNoise(float x, float y, int octaves, float persistence, float lacunarity) {
    float total = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float maxValue = 0.0f;
    float prev = 1.0f;

    for (int i = 0; i < octaves; i++) {
        float n = fabsf(perlinNoise2D(x * frequency, y * frequency));
        n = 1.0f - n;  // Invert
        n = n * n;     // Square for sharper ridges
        n *= prev;     // Weight by previous octave
        prev = n;

        total += n * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    return total / maxValue;
}

// ============================================================================
// Height Map Generation Kernel for Tessellation
// ============================================================================
// Generates Unigine Valley-style terrain: mountain valley with ridges,
// varied terrain, and natural-looking landscape
// ============================================================================
__global__ void generateHeightMap(float* __restrict__ heightMap,
                                   unsigned int width, unsigned int height,
                                   float animTime) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Compute UV coordinates in [0, 1]
    const float u = (float)x / (float)(width - 1);
    const float v = (float)y / (float)(height - 1);

    // Map to coordinate space for terrain features
    const float terrainScale = 6.0f;
    const float px = u * terrainScale;
    const float py = v * terrainScale;

    // ==========================================================================
    // RUGGED TERRAIN: Random elevations using multiple noise octaves
    // ==========================================================================

    // Base terrain with high-frequency noise for rugged look
    float baseTerrain = fbm(px * 2.0f, py * 2.0f, 6, 0.5f, 2.0f);

    // Add sharp ridge features for ruggedness
    float ridges = ridgeNoise(px * 1.5f, py * 1.5f, 4, 0.5f, 2.0f);

    // High-frequency detail for rough surface
    float detail = fbm(px * 8.0f, py * 8.0f, 3, 0.4f, 2.5f) * 0.2f;

    // Random spikes and variations
    float spikes = ridgeNoise(px * 3.0f, py * 3.0f, 3, 0.6f, 2.0f) * 0.3f;

    // ==========================================================================
    // COMBINE FOR RUGGED LANDSCAPE
    // ==========================================================================
    float heightVal = baseTerrain * 0.4f + ridges * 0.4f + detail + spikes;

    // Scale to appropriate height range
    heightVal = heightVal * 0.6f;

    // Write height to texture (R32_SFLOAT format)
    heightMap[y * width + x] = heightVal;
}

// ============================================================================
// Patch Grid Generation Kernel for Tessellation
// ============================================================================
// Generates a coarse grid of patch corner vertices for tessellation
// Each vertex is a corner of a quad patch
// The TES shader will subdivide these patches based on LOD
// ============================================================================
__global__ void generatePatchGrid(float4* __restrict__ patchVertices,
                                   unsigned int gridWidth, unsigned int gridHeight) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= gridWidth || y >= gridHeight)
        return;

    // Compute UV coordinates in [0, 1]
    const float u = (float)x / (float)(gridWidth - 1);
    const float v = (float)y / (float)(gridHeight - 1);

    // Map to terrain coordinates [-1, 1] for XZ plane
    const float px = u * 2.0f - 1.0f;
    const float pz = v * 2.0f - 1.0f;

    // Store position (Y will be computed in TES from height map)
    // Set Y to 0 as placeholder - TES will sample height map for actual height
    const unsigned int idx = y * gridWidth + x;
    patchVertices[idx] = make_float4(px, 0.0f, pz, 1.0f);
}

// ============================================================================
// Triangle Patch Grid Generation Kernel for Tessellation
// ============================================================================
// Generates triangle patches from a grid for tessellation rendering
// Each quad cell in the grid is split into 2 triangles
// Output: 6 vertices per quad cell (2 triangles * 3 vertices)
// For a grid of NxN points: (N-1)*(N-1)*2 triangles = (N-1)*(N-1)*6 vertices
// ============================================================================
#define PATCH_GRID_SIZE 256  // 256x256 grid points = 255x255 quads = 130050 triangles for 8km terrain
#define PATCH_TRIANGLE_COUNT ((PATCH_GRID_SIZE - 1) * (PATCH_GRID_SIZE - 1) * 2)
#define PATCH_VERTEX_COUNT (PATCH_TRIANGLE_COUNT * 3)  // 390150 vertices

__global__ void generateTrianglePatchGrid(float4* __restrict__ patchVertices,
                                           unsigned int gridSize, float animTime) {
    // Each thread handles one quad cell (2 triangles, 6 vertices)
    const unsigned int quadX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int quadY = blockIdx.y * blockDim.y + threadIdx.y;

    // Grid has (gridSize-1) x (gridSize-1) quad cells
    if (quadX >= gridSize - 1 || quadY >= gridSize - 1)
        return;

    // Calculate UV coordinates for the 4 corners of this quad
    const float invGrid = 1.0f / (float)(gridSize - 1);

    // Corner UVs in [0, 1]
    const float u0 = (float)quadX * invGrid;
    const float u1 = (float)(quadX + 1) * invGrid;
    const float v0 = (float)quadY * invGrid;
    const float v1 = (float)(quadY + 1) * invGrid;

    // Map to terrain coordinates [-1, 1] for XZ plane
    const float x0 = u0 * 2.0f - 1.0f;
    const float x1 = u1 * 2.0f - 1.0f;
    const float z0 = v0 * 2.0f - 1.0f;
    const float z1 = v1 * 2.0f - 1.0f;

    // Y coordinate is 0 - TES will sample height map for actual height
    // The 4 corners of the quad:
    // v0 (x0, z0), v1 (x1, z0), v2 (x0, z1), v3 (x1, z1)

    // Output index: each quad produces 6 vertices
    // Quads are numbered row by row: quad at (quadX, quadY) has index (quadY * (gridSize-1) + quadX)
    const unsigned int quadIdx = quadY * (gridSize - 1) + quadX;
    const unsigned int baseIdx = quadIdx * 6;

    // Triangle 1: v0, v1, v2 (bottom-left, bottom-right, top-left)
    patchVertices[baseIdx + 0] = make_float4(x0, 0.0f, z0, 1.0f);  // v0
    patchVertices[baseIdx + 1] = make_float4(x1, 0.0f, z0, 1.0f);  // v1
    patchVertices[baseIdx + 2] = make_float4(x0, 0.0f, z1, 1.0f);  // v2

    // Triangle 2: v1, v3, v2 (bottom-right, top-right, top-left)
    patchVertices[baseIdx + 3] = make_float4(x1, 0.0f, z0, 1.0f);  // v1
    patchVertices[baseIdx + 4] = make_float4(x1, 0.0f, z1, 1.0f);  // v3
    patchVertices[baseIdx + 5] = make_float4(x0, 0.0f, z1, 1.0f);  // v2
}

// ============================================================================
// Combined Height Map and Patch Grid Update Kernel
// ============================================================================
// Updates the height map - uses static Unigine Valley terrain
// animTime parameter kept for API compatibility but not used
// ============================================================================
__global__ void updateHeightMapAnimated(float* __restrict__ heightMap,
                                         unsigned int width, unsigned int height,
                                         float animTime) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Compute UV coordinates in [0, 1]
    const float u = (float)x / (float)(width - 1);
    const float v = (float)y / (float)(height - 1);

    // Map to larger coordinate space for terrain features
    const float terrainScale = 8.0f;
    const float px = u * terrainScale;
    const float py = v * terrainScale;

    // Valley shape - central valley running through terrain
    float distFromCenter = fabsf(v - 0.5f) * 2.0f;
    float valleyFactor = distFromCenter * distFromCenter * (3.0f - 2.0f * distFromCenter);

    // Valley floor with undulation
    float valleyFloor = fbm(px * 2.0f, py * 0.5f, 4, 0.5f, 2.0f) * 0.05f;

    // Mountain ridges
    float ridges = ridgeNoise(px * 0.8f, py * 0.8f, 5, 0.5f, 2.2f);
    float ridgeVariation = fbm(px * 0.3f + 100.0f, py * 0.3f + 100.0f, 3, 0.6f, 2.0f);
    ridges *= (0.7f + ridgeVariation * 0.5f);

    // Base terrain
    float baseTerrain = fbm(px * 1.5f, py * 1.5f, 6, 0.5f, 2.0f);
    float largeFeaturesX = fbm(px * 0.4f + 50.0f, py * 0.4f, 3, 0.6f, 2.0f);
    float largeFeaturesY = fbm(px * 0.4f, py * 0.4f + 50.0f, 3, 0.6f, 2.0f);
    float largeFeatures = (largeFeaturesX + largeFeaturesY) * 0.5f;

    // Details
    float detail = fbm(px * 4.0f, py * 4.0f, 4, 0.4f, 2.5f) * 0.15f;
    float erosion = fbm(px * 6.0f + baseTerrain * 2.0f,
                        py * 6.0f + baseTerrain * 2.0f, 3, 0.5f, 2.0f) * 0.05f;

    // Combine features
    float valleyHeight = valleyFloor - 0.3f;
    float mountainHeight = ridges * 0.6f + baseTerrain * 0.25f + largeFeatures * 0.2f;
    mountainHeight += detail + erosion;

    float heightVal = lerp(valleyHeight, mountainHeight, valleyFactor);

    // River bed
    float riverWidth = 0.08f;
    float riverDist = fabsf(v - 0.5f);
    if (riverDist < riverWidth) {
        float riverFactor = 1.0f - (riverDist / riverWidth);
        riverFactor = riverFactor * riverFactor;
        float meander = sinf(u * 12.0f) * 0.02f;
        if (fabsf(v - 0.5f - meander) < riverWidth * 0.5f) {
            heightVal -= riverFactor * 0.08f;
        }
    }

    // Edge fade
    float edgeFadeX = smoothstep(0.0f, 0.1f, u) * smoothstep(0.0f, 0.1f, 1.0f - u);
    float edgeFadeY = smoothstep(0.0f, 0.1f, v) * smoothstep(0.0f, 0.1f, 1.0f - v);
    heightVal *= edgeFadeX * edgeFadeY;

    // Final scaling
    heightVal = heightVal * 0.8f;

    // Write height to texture (R32_SFLOAT format)
    heightMap[y * width + x] = heightVal;
}

// Entry-Point Function
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	// Function Declarations
	VkResult initialize(void);
	void uninitialize(void);
	VkResult display(void);
	void update(void);

	// Local Variable Declarations
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

	// Code

	// Clear Log File at startup
	FILE* file = fopen(LOG_FILE, "w");
	if (!file)
	{
		MessageBox(NULL, TEXT("Program cannot open log file!"), TEXT("Error"), MB_OK | MB_ICONERROR);
		exit(0);
	}
	fclose(file);
	FileIO("WinMain()-> Program started successfully\n");
	
	wsprintf(szAppName, TEXT("%s"), gpszAppName);

	// WNDCLASSEX Initilization 
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

	// Register WNDCLASSEX
	RegisterClassEx(&wndclass);


	// Create Window								// glutCreateWindow
	hwnd = CreateWindowEx(WS_EX_APPWINDOW,			// to above of taskbar for fullscreen
						szAppName,
						TEXT("05_PhysicalDevice"),
						WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
						xCoordinate,				// glutWindowPosition 1st Parameter
						yCoordinate,				// glutWindowPosition 2nd Parameter
						WIN_WIDTH,					// glutWindowSize 1st Parameter
						WIN_HEIGHT,					// glutWindowSize 2nd Parameter
						NULL,
						NULL,
						hInstance,
						NULL);

	ghwnd = hwnd;

	// Initialization
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

	// Show The Window
	ShowWindow(hwnd, iCmdShow);
	UpdateWindow(hwnd);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	// Game Loop
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
					if ((vkResult != VK_FALSE) && (vkResult != VK_SUCCESS) && (vkResult != VK_ERROR_OUT_OF_DATE_KHR) && ((vkResult != VK_SUBOPTIMAL_KHR))) //VK_ERROR_OUT_OF_DATE_KHR and VK_SUBOPTIMAL_KHR are meant for future issues.You can remove them.
					{
						FileIO("WinMain(): display() function failed\n");
						bDone = TRUE;
					}
					update();
				}
			}
		}
	}

	// Uninitialization
	uninitialize();	

	return((int)msg.wParam);
}

// CALLBACK Function
LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	// Function Declarations
	void ToggleFullscreen( void );
	VkResult resize(int, int);
	void uninitialize(void);
	
	//Variable Declarations
	VkResult vkResult;

	// Code
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
				bWindowMinimize = FALSE; //Any sequence is OK
				vkResult = resize(LOWORD(lParam), HIWORD(lParam)); //No need of error checking
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

		/*
		case WM_ERASEBKGND:
			return(0);
		*/

		case WM_KEYDOWN:
			switch (LOWORD(wParam))
			{
			case VK_ESCAPE:
				FileIO("WndProc() VK_ESCAPE-> Program ended successfully.\n");
				DestroyWindow(hwnd);
				break;
				
			case 0x52: //R key 
				FileIO("WndProc() 0x52 pressed.\n");
				bMesh1024_chosen = true;
				break;
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

			case 'O':
			case 'o':
				colorFromKey = 'O';
				break;

			case 'K':
			case 'k':
				colorFromKey = 'K';
				break;

			case 'R':
			case 'r':
				colorFromKey = 'R';
				break;

			case 'G':
			case 'g':
				colorFromKey = 'G';
				break;

			case 'B':
			case 'b':
				colorFromKey = 'B';
				break;

			case 'C':
			case 'c':
				colorFromKey = 'C';
				break;

			case 'M':
			case 'm':
				colorFromKey = 'M';
				break;

			case 'Y':
			case 'y':
				colorFromKey = 'Y';
				break;

			case 'W':
			case 'w':
				colorFromKey = 'W';
				break;

			case '1':
				bMesh1024_chosen = true;
				FileIO("WndProc() WM_CHAR(1 key)-> 1024 x 1024 mesh selected.\n");
				break;

			}
			break;

		case WM_RBUTTONDOWN:								
			DestroyWindow(hwnd);
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
	// Local Variable Declarations
	MONITORINFO mi = { sizeof(MONITORINFO) };

	// Code
	if (gbFullscreen == FALSE)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);

		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);

				SetWindowPos(ghwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);
				// HWND_TOP ~ WS_OVERLAPPED, rc ~ RECT, SWP_FRAMECHANGED ~ WM_NCCALCSIZE msg
			}
		}

		ShowCursor(FALSE);
	}
	else {
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
		// SetWindowPos has greater priority than SetWindowPlacement and SetWindowStyle for Z-Order
		ShowCursor(TRUE);
	}
}

VkResult initialize(void)
{
	//Function declaration
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

	//31.2
	VkResult CreateUniformBuffer(void);

	// Tessellation-related function declarations
	VkResult CreateTessUniformBuffer(void);
	VkResult CreateHeightMapTexture(void);

	/*
	23.3 Declare prototype of UDF say CreateShaders() in initialize(), following a convention i.e after CreateVertexBuffer() and before CreateRenderPass().
	*/
	VkResult CreateShaders(void);
	
	/*
	24.2. In initialize(), declare and call UDF CreateDescriptorSetLayout() maintaining the convention of declaring and calling it after CreateShaders() and before CreateRenderPass().
	*/
	VkResult CreateDescriptorSetLayout(void);
	
	/*
	25.2. In initialize(), declare and call UDF CreatePipelineLayout() maintaining the convention of declaring and calling it after CreatDescriptorSetLayout() and before CreateRenderPass().
	*/
	VkResult CreatePipelineLayout(void);
	
	//31.2
	VkResult CreateDescriptorPool(void);
	VkResult CreateDescriptorSet(void);
	
	VkResult CreateRenderPass(void);
	
	/*
	26. Pipeline
	*/
	VkResult CreatePipeline(void);
	
	VkResult CreateFramebuffers(void);
	VkResult CreateSemaphores(void);
	VkResult CreateFences(void);
	VkResult buildCommandBuffers(void);
	
	//CUDA related function declarations
	cudaError_t initialize_cuda(void);
	VkResult CreateExternalVertexBuffer(unsigned int, unsigned int, VertexData*);
	VkResult CreateMultipleExternalVertexBuffers(unsigned int, unsigned int);
	
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;
	
	// Code
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
	
	//Create Vulkan Presentation Surface
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
	
	//Enumerate and select physical device and its queque family index
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
	
	//Print Vulkan Info ;
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
	
	//Create Vulkan Device (Logical Device)
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
	
	//get Device Queque
	GetDeviceQueque();
	
	//CUDA initialization
	
	cudaResult = initialize_cuda();
	if (cudaResult != cudaSuccess)
	{
	  vkResult = VK_ERROR_INITIALIZATION_FAILED; //return hardcoded failure
	  FileIO("initialize(): initialize_cuda() function failed with error code %d\n", vkResult);
	  return vkResult;
    }
	else
	{
	   FileIO("initialize(): initialize_cuda() succedded\n");
	}
	
	vkResult = CreateSwapChain(VK_FALSE); //https://registry.khronos.org/vulkan/specs/latest/man/html/VK_FALSE.html
	if (vkResult != VK_SUCCESS)
	{
		/*
		Why are we giving hardcoded error when returbn value is vkResult?
		Answer sir will give in swapchain
		*/
		vkResult = VK_ERROR_INITIALIZATION_FAILED; //return hardcoded failure
		FileIO("initialize(): CreateSwapChain() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateSwapChain() succedded\n");
	}
	
	//1. Get Swapchain image count in a global variable using vkGetSwapchainImagesKHR() API (https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetSwapchainImagesKHR.html).
	//Create Vulkan images and image views
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

	// Create multiple vertex buffers for double/triple buffering
	// This creates NUM_VERTEX_BUFFERS buffers for ping-pong rendering
	vkResult = CreateMultipleExternalVertexBuffers(1024, 1024);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateMultipleExternalVertexBuffers() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateMultipleExternalVertexBuffers() succedded with %d buffers\n", NUM_VERTEX_BUFFERS);
	}

	// Legacy single buffer creation (kept for backward compatibility, shares buffer 0)
	// vertexData_external is now set to point to vertexData_buffers[0] by CreateMultipleExternalVertexBuffers
	FileIO("initialize(): Legacy vertexData_external now points to buffer 0\n");

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

	/*
	31.3 CreateUniformBuffer()
	*/
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

	// Create Tessellation Uniform Buffer
	vkResult = CreateTessUniformBuffer();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateTessUniformBuffer() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateTessUniformBuffer() succedded\n");
	}

	// Create Height Map Texture for Tessellation
	vkResult = CreateHeightMapTexture();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("initialize(): CreateHeightMapTexture() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("initialize(): CreateHeightMapTexture() succedded\n");
	}

	/*
	23.4. Using same above convention, call CreateShaders() between calls of above two.
	*/
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
	
	/*
	24.2. In initialize(), declare and call UDF CreateDescriptorSetLayout() maintaining the convention of declaring and calling it after CreateShaders() and before CreateRenderPass().
	*/
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
	
	//31.4 CreateDescriptorPool
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
	
	//31.5 CreateDescriptorSet
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

	// Import Vulkan timeline semaphore into CUDA for GPU-GPU synchronization
	{
		cudaError_t ImportVulkanTimelineSemaphoreToCUDA(void);
		cudaResult = ImportVulkanTimelineSemaphoreToCUDA();
		if(cudaResult != cudaSuccess)
		{
			FileIO("initialize(): ImportVulkanTimelineSemaphoreToCUDA() failed (non-fatal)\n");
			// Non-fatal - continue without timeline semaphore
		}
		else
		{
			FileIO("initialize(): ImportVulkanTimelineSemaphoreToCUDA() succeeded\n");
		}
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
	
	/*
	Initialize Clear Color values
	*/
	memset((void*)&vkClearColorValue, 0, sizeof(VkClearColorValue));
	//Following step is analogus to glClearColor. This is more analogus to DirectX 11.
	vkClearColorValue.float32[0] = 0.0f;
	vkClearColorValue.float32[1] = 0.0f;
	vkClearColorValue.float32[2] = 0.0f;
	vkClearColorValue.float32[3] = 1.0f;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkClearDepthStencilValue.html
	memset((void*)&vkClearDepthStencilValue, 0, sizeof(VkClearDepthStencilValue));
	//Set default clear depth value
	vkClearDepthStencilValue.depth = 1.0f; //type float
	//Set default clear stencil value
	vkClearDepthStencilValue.stencil = 0; //type uint32_t
	
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
	
	/*
	Initialization is completed here..........................
	*/
	bInitialized = TRUE;
	
	FileIO("initialize(): initialize() completed sucessfully");
	
	return vkResult;
}

cudaError_t initialize_cuda(void)
{
	/*
	Code
	*/
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
	
	//Check equality of UUID between Vulkan and CUDA
	
	//Get Vulkan UUID of device
	//https://registry.khronos.org/VulkanSC/specs/1.0-extensions/man/html/VkPhysicalDeviceIDProperties.html
	/*
	// Provided by VK_VERSION_1_1
	typedef struct VkPhysicalDeviceIDProperties {
    VkStructureType    sType;
    void*              pNext;
    uint8_t            deviceUUID[VK_UUID_SIZE];
    uint8_t            driverUUID[VK_UUID_SIZE];
    uint8_t            deviceLUID[VK_LUID_SIZE];
    uint32_t           deviceNodeMask;
    VkBool32           deviceLUIDValid;
	} VkPhysicalDeviceIDProperties;
	*/
	VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties;
	memset((void*)&vkPhysicalDeviceIDProperties, 0, sizeof(VkPhysicalDeviceIDProperties));
	
	vkPhysicalDeviceIDProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
	vkPhysicalDeviceIDProperties.pNext = NULL;
	//vkPhysicalDeviceIDProperties.deviceUUID = ;
	//vkPhysicalDeviceIDProperties.driverUUID = ;
	//vkPhysicalDeviceIDProperties.deviceLUID = ;
	//vkPhysicalDeviceIDProperties.deviceNodeMask = ;
	//vkPhysicalDeviceIDProperties.deviceLUIDValid = ;
	
	//https://registry.khronos.org/VulkanSC/specs/1.0-extensions/man/html/VkPhysicalDeviceProperties2.html
	/*
	// Provided by VK_VERSION_1_1
	typedef struct VkPhysicalDeviceProperties2 {
    VkStructureType               sType;
    void*                         pNext;
    VkPhysicalDeviceProperties    properties;
	} VkPhysicalDeviceProperties2;
	*/
	VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2;
	memset((void*)&vkPhysicalDeviceProperties2, 0, sizeof(VkPhysicalDeviceProperties2));
	vkPhysicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;
	//vkPhysicalDeviceProperties2.properties = ;
	
	//https://registry.khronos.org/VulkanSC/specs/1.0-extensions/man/html/vkGetPhysicalDeviceProperties2.html
	/*
	// Provided by VK_VERSION_1_1
	void vkGetPhysicalDeviceProperties2(
    VkPhysicalDevice                            physicalDevice,
    VkPhysicalDeviceProperties2*                pProperties);
	*/
	vkGetPhysicalDeviceProperties2(vkPhysicalDevice_selected, &vkPhysicalDeviceProperties2);
	uint8_t vulkanDeviceUUID[VK_UUID_SIZE];
	memcpy((void*)&vulkanDeviceUUID, vkPhysicalDeviceIDProperties.deviceUUID, VK_UUID_SIZE);
	
	/*
	As we have atleast 1 CUDA enabled device, so find out its CUDA equivakent UUID and compare with Vulkan UUID for equality,
	so find out its CUDA equivalent UUID and compare with above Vulkan UUID for equality
	*/
	int iVulkanCUDAInterOpDeviceFound = -1;
	for(int i=0; i < devCount; i++)
	{
		//First select that device whose compute mode is not prohibited
		int compute_Mode;
		//https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151
		//https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g49e2f8c2c0bd6fe264f2fc970912e5cd
		//https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7eb25f5413a962faad0956d92bae10d0
		cudaResult = cudaDeviceGetAttribute(&compute_Mode, cudaDevAttrComputeMode, i);
		if(cudaResult != CUDA_SUCCESS)
		{
			continue;
		}
		
		if(compute_Mode == cudaComputeModeProhibited)
		{
			continue;
		}
		
		//Get CUDA Device Properties
		cudaDeviceProp devProp;
		memset((void*)&devProp, 0, sizeof(cudaDeviceProp));
		cudaResult = cudaGetDeviceProperties(&devProp, i);
		if(cudaResult != CUDA_SUCCESS)
		{
			continue;
		}
		
		//Now compare both UUID's
		int iResult = memcmp((void*)&devProp.uuid, (void*)&vulkanDeviceUUID, VK_UUID_SIZE);
		if(iResult != 0)
		{
			continue;
		}
		
		//Otherwise required device is found. Now set it
		cudaResult = cudaSetDevice(i);
		if(cudaResult != CUDA_SUCCESS)
		{
			continue;
		}
		
		FileIO("initialize_cuda(): selected device is %s\n", devProp.name);
		iVulkanCUDAInterOpDeviceFound = 1;
		break;
	}
	
	//If no such interop device is found, return with error
	if(iVulkanCUDAInterOpDeviceFound == -1)
	{
		FileIO("initialize_cuda(): no device found\n");
		return cudaErrorUnknown; //cuda hardcoded failure
	}
	
	//Assuming we are already using Windows OS greater than 8.1, but with Win32 application not Win64 UWP/Metro application
	vkExternalMemoryHandleTypeFlagBits = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

	// Create CUDA stream for async operations
	// Using cudaStreamNonBlocking for better concurrency with Vulkan
	cudaResult = cudaStreamCreateWithFlags(&cudaStream_compute, cudaStreamNonBlocking);
	if(cudaResult != cudaSuccess)
	{
		FileIO("initialize_cuda(): cudaStreamCreateWithFlags() failed with error: %s\n", cudaGetErrorString(cudaResult));
		return cudaResult;
	}
	FileIO("initialize_cuda(): CUDA stream created successfully for async operations\n");

	// Log optimized thread block configuration
	FileIO("initialize_cuda(): Using optimized thread block config: %dx%d = %d threads/block\n",
		CUDA_BLOCK_SIZE_X, CUDA_BLOCK_SIZE_Y, CUDA_THREADS_PER_BLOCK);

	return cudaSuccess;
}

VkResult resize(int width, int height)
{
	//Function declarations
	VkResult CreateSwapChain(VkBool32);
	VkResult CreateImagesAndImageViews(void);
	VkResult CreateRenderPass(void);
	VkResult CreatePipelineLayout(void);
	VkResult CreatePipeline(void);
	VkResult CreateFramebuffers(void);
	VkResult CreateCommandBuffers(VkCommandBuffer**);
	VkResult buildCommandBuffers(void);
	
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;
	
	// Code
	if(height <= 0)
	{
		height = 1;
	}
	
	//30.1
	//Check the bInitialized variable
	if(bInitialized == FALSE)
	{
		//throw error
		FileIO("resize(): initialization yet not completed or failed\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	
	//30.2 
	//As recreation of swapchain is needed, we are going to repeat many steps of initialize() again.
	//Hence set bInitialized = FALSE again.
	bInitialized = FALSE;
	
	/*
	call can go to display() and code for resize() here
	*/
	
	//30.4 
	//Set global WIN_WIDTH and WIN_HEIGHT variables
	winWidth = width;
	winHeight = height;
	
	//30.5
	//Wait for device to complete in-hand tasks
	if(vkDevice)
	{
		vkDeviceWaitIdle(vkDevice);
		FileIO("resize(): vkDeviceWaitIdle() is done\n");
	}
	
	//Destroy and recreate Swapchain, Swapchain image and image views functions, Swapchain count functions, Renderpass, Framebuffer, Pipeline, Pipeline Layout, CommandBuffer
	
	//30.6
	//Check presence of swapchain
	if(vkSwapchainKHR == VK_NULL_HANDLE)
	{
		FileIO("resize(): vkSwapchainKHR is already NULL, cannot proceed\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	
	//30.7
	//Destroy framebuffer: destroy framebuffers in a loop for swapchainImageCount
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyFramebuffer.html
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
	
	//30.11
	//Destroy Commandbuffer: In unitialize(), free each command buffer by using vkFreeCommandBuffers()(https://registry.khronos.org/vulkan/specs/latest/man/html/vkFreeCommandBuffers.html) in a loop of size swapchainImage count.
	for(uint32_t i =0; i < swapchainImageCount; i++)
	{
		vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_for_1024_x_1024_graphics_array[i]);
		FileIO("resize(): vkFreeCommandBuffers() is done \n");
	}
			
	//Free actual command buffer array.
	if(vkCommandBuffer_for_1024_x_1024_graphics_array)
	{
		free(vkCommandBuffer_for_1024_x_1024_graphics_array);
		vkCommandBuffer_for_1024_x_1024_graphics_array = NULL;
		FileIO("resize(): vkCommandBuffer_for_1024_x_1024_graphics_array is freed\n");
	}

	//30.9
	//Destroy Pipeline
	if(vkPipeline)
	{
		vkDestroyPipeline(vkDevice, vkPipeline, NULL);
		vkPipeline = VK_NULL_HANDLE;
		FileIO("resize(): vkPipeline is freed\n");
	}
	
	//30.10
	//Destroy PipelineLayout
	if(vkPipelineLayout)
	{
		vkDestroyPipelineLayout(vkDevice, vkPipelineLayout, NULL);
		vkPipelineLayout = VK_NULL_HANDLE;
		FileIO("resize(): vkPipelineLayout is freed\n");
	}
	
	//30.8
	//Destroy Renderpass : In uninitialize , destroy the renderpass by 
	//using vkDestrorRenderPass() (https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyRenderPass.html).
	if(vkRenderPass)
	{
		vkDestroyRenderPass(vkDevice, vkRenderPass, NULL); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyRenderPass.html
		vkRenderPass = VK_NULL_HANDLE;
		FileIO("resize(): vkDestroyRenderPass() is done\n");
	}
	
	//destroy depth image view
	if(vkImageView_depth)
	{
		vkDestroyImageView(vkDevice, vkImageView_depth, NULL); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyImageView.html
		vkImageView_depth = VK_NULL_HANDLE;
	}
			
	//destroy device memory for depth image
	if(vkDeviceMemory_depth)
	{
		vkFreeMemory(vkDevice, vkDeviceMemory_depth, NULL); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkFreeMemory.html
		vkDeviceMemory_depth = VK_NULL_HANDLE;
	}
			
	//destroy depth image
	if(vkImage_depth)
	{
		//https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyImage.html
		vkDestroyImage(vkDevice, vkImage_depth, NULL);
		vkImage_depth = VK_NULL_HANDLE;
	}
	
	//30.12
	//Destroy Swapchain image and image view: Keeping the "destructor logic aside" for a while , first destroy image views from imagesViews array in a loop using vkDestroyImageViews() api.
	//(https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyImageView.html)
	for(uint32_t i =0; i < swapchainImageCount; i++)
	{
		vkDestroyImageView(vkDevice, swapChainImageView_array[i], NULL);
		FileIO("resize(): vkDestroyImageView() is done\n");
	}
	
	//Now actually free imageView array using free().
	//free imageView array
	if(swapChainImageView_array)
	{
		free(swapChainImageView_array);
		swapChainImageView_array = NULL;
		FileIO("resize(): swapChainImageView_array is freed\n");
	}
	
	//Now actually free swapchain image array using free().
	/*
	for(uint32_t i = 0; i < swapchainImageCount; i++)
	{
		vkDestroyImage(vkDevice, swapChainImage_array[i], NULL);
		FileIO("resize(): vkDestroyImage() is done\n");
	}
	*/
	
	if(swapChainImage_array)
	{
		free(swapChainImage_array);
		swapChainImage_array = NULL;
		FileIO("resize(): swapChainImage_array is freed\n");
	}
	
	//30.13
	//Destroy swapchain : destroy it uninitilialize() by using vkDestroySwapchainKHR() (https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroySwapchainKHR.html) Vulkan API.
	vkDestroySwapchainKHR(vkDevice, vkSwapchainKHR, NULL);
	vkSwapchainKHR = VK_NULL_HANDLE;
	FileIO("resize(): vkDestroySwapchainKHR() is done\n");
	
	//RECREATE FOR RESIZE
	
	//30.14 Create Swapchain
	vkResult = CreateSwapChain(VK_FALSE); //https://registry.khronos.org/vulkan/specs/latest/man/html/VK_FALSE.html
	if (vkResult != VK_SUCCESS)
	{
		vkResult = VK_ERROR_INITIALIZATION_FAILED; //return hardcoded failure
		FileIO("resize(): CreateSwapChain() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	
	//30.15 Create Swapchain image and Image Views
	vkResult =  CreateImagesAndImageViews();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("resize(): CreateImagesAndImageViews() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	
	//30.18 Create renderPass
	vkResult =  CreateRenderPass();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("resize(): CreateRenderPass() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	
	//30.16 Create PipelineLayout
	vkResult = CreatePipelineLayout();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("resize(): CreatePipelineLayout() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	
	//30.17 Create Pipeline
	vkResult = CreatePipeline();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("resize(): CreatePipeline() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	
	//30.19 Create framebuffers
	vkResult = CreateFramebuffers();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("resize(): CreateFramebuffers() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	
	
	//30.16 Create CommandBuffers for 1024 x 1024
	vkResult  = CreateCommandBuffers(&vkCommandBuffer_for_1024_x_1024_graphics_array);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("resize(): CreateCommandBuffers() function failed with error code %d for 1024 x 1024\n", vkResult);
		return vkResult;
	}

	//30.20 Build Commandbuffers
	vkResult = buildCommandBuffers();
	if (vkResult != VK_SUCCESS)
	{
		FileIO("resize(): buildCommandBuffers() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	
	//30.3
	//Do this
	bInitialized = TRUE;
	
	return vkResult;
}

//31.12
VkResult UpdateUniformBuffer(void)
{
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;
	
	//Code
	MyUniformData myUniformData;
	memset((void*)&myUniformData, 0, sizeof(struct MyUniformData));
	
	//Update matrices
	// Scale model matrix to transform [-1, 1] terrain to [-4000, 4000] meters (8km x 8km)
	// Also scale Y by max height to get proper mountain heights
	myUniformData.modelMatrix = glm::mat4(1.0f);
	float terrainScaleXZ = TERRAIN_SIZE_METERS / 2.0f;  // 4000 meters
	float terrainScaleY = TERRAIN_MAX_HEIGHT_METERS;     // 1500 meters max height
	myUniformData.modelMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(terrainScaleXZ, terrainScaleY, terrainScaleXZ));

	// Set up view matrix - camera positioned to see 8km x 8km terrain
	// Camera at high elevation and distance to see the whole terrain
	glm::vec3 cameraPos = glm::vec3(0.0f, 6000.0f, 12000.0f);    // 6km up, 12km back
	glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);         // Looking at terrain center
	glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);             // Up vector
	myUniformData.viewMatrix = glm::lookAt(cameraPos, cameraTarget, cameraUp);

	// Set up orthographic projection matrix for 8km terrain
	// The terrain spans 8km in X and Z after model matrix scaling
	float aspectRatio = (float)winWidth / (float)winHeight;
	float orthoSize = 10000.0f; // 10km visible range to see full 8km terrain
	float left = -orthoSize * aspectRatio;
	float right = orthoSize * aspectRatio;
	float bottom = -orthoSize;
	float top = orthoSize;
	float nearPlane = 1.0f;
	float farPlane = 50000.0f;  // 50km far plane for large terrain

	glm::mat4 orthoProjectionMatrix = glm::ortho(left, right, bottom, top, nearPlane, farPlane);
	orthoProjectionMatrix[1][1] = orthoProjectionMatrix[1][1] * (-1.0f); // Y-flip for Vulkan

	myUniformData.projectionMatrix = orthoProjectionMatrix;
	
	//For color (Only case where background changes to white)
	if(colorFromKey == 'K')
	{
		myUniformData.color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	}
	else //for all other colors background will be black 
	{
		if(colorFromKey == 'R')
		{
			myUniformData.color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f); //Red
		}
		else if(colorFromKey == 'G')
		{
			myUniformData.color = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f); //Green
		}
		else if(colorFromKey == 'B')
		{
			myUniformData.color = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f); //Green
		}
		else if(colorFromKey == 'C')
		{
			myUniformData.color = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f); //Cyan
		}
		else if(colorFromKey == 'M')
		{
			myUniformData.color = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f); //Magenta
		}
		else if(colorFromKey == 'Y')
		{
			myUniformData.color = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f); //Yellow
		}
		else if(colorFromKey == 'W')
		{
			myUniformData.color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f); //White
		}
		else if(colorFromKey == 'O')
		{
			myUniformData.color = glm::vec4(1.0f, 0.647f, 0.0f, 1.0f); //Orange
		}
		else
		{
			myUniformData.color = glm::vec4(1.0f, 0.647f, 0.0f, 1.0f); //Default is Orange
		}
	}
	
	//Map Uniform Buffer
	/*
	This will allow us to do memory mapped IO means when we write on void* buffer data, 
	it will get automatically written/copied on to device memory represented by device memory object handle.
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkMapMemory.html
	// Provided by VK_VERSION_1_0
	VkResult vkMapMemory(
    VkDevice                                    device,
    VkDeviceMemory                              memory,
    VkDeviceSize                                offset,
    VkDeviceSize                                size,
    VkMemoryMapFlags                            flags,
    void**                                      ppData);
	*/
	void* data = NULL;
	vkResult = vkMapMemory(vkDevice, uniformData.vkDeviceMemory, 0, sizeof(struct MyUniformData), 0, &data);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("UpdateUniformBuffer(): vkMapMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	
	//Copy the data to the mapped buffer
	/*
	31.12. Now to do actual memory mapped IO, call memcpy.
	*/
	memcpy(data, &myUniformData, sizeof(struct MyUniformData));
	
	/*
	31.12. To complete this memory mapped IO. finally call vkUmmapMemory() API.
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkUnmapMemory.html
	// Provided by VK_VERSION_1_0
	void vkUnmapMemory(
    VkDevice                                    device,
    VkDeviceMemory                              memory);
	*/
	vkUnmapMemory(vkDevice, uniformData.vkDeviceMemory);
	
	return vkResult;
}

VkResult display(void)
{
	//Function declarations
	VkResult resize(int, int);
	//31.6
	VkResult UpdateUniformBuffer(void);
	VkResult UpdateIndirectBuffer(void);
	VkResult buildCommandBuffers(void);

	//Variable declarations
	VkResult vkResult = VK_SUCCESS;
	
	/*
	Command Buffer
	*/
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandBuffer.html
	VkCommandBuffer *vkCommandBuffer_array = NULL;
	
	// Code
	
	// If control comes here , before initialization is completed , return false
	if(bInitialized == FALSE)
	{
		FileIO("display(): initialization not completed yet\n");
		return (VkResult)VK_FALSE;
	}
	
	if(bMesh1024_chosen == TRUE)
	{
		vkCommandBuffer_array = vkCommandBuffer_for_1024_x_1024_graphics_array;
	}

	if(bMesh1024_chosen == TRUE)
	{
		// ====================================================================
		// TESSELLATION PATCH GRID GENERATION
		// ====================================================================
		// This implementation generates a coarse grid of triangle patches
		// for GPU tessellation. The TES shader will subdivide these patches
		// and sample height from the height map texture.
		// ====================================================================

		// Get the current compute buffer CUDA pointer
		float4* computeBuffer = (float4*)pos_CUDA_buffers[computeBufferIndex];

		// Grid/block configuration for triangle patch generation
		// Each thread handles one quad cell (2 triangles)
		const unsigned int numQuadsX = PATCH_GRID_SIZE - 1;
		const unsigned int numQuadsY = PATCH_GRID_SIZE - 1;
		dim3 patchBlock(16, 16, 1);
		dim3 patchGrid((numQuadsX + patchBlock.x - 1) / patchBlock.x,
		               (numQuadsY + patchBlock.y - 1) / patchBlock.y, 1);

		// Generate triangle patch vertices
		generateTrianglePatchGrid<<<patchGrid, patchBlock, 0, cudaStream_compute>>>(
			computeBuffer, PATCH_GRID_SIZE, animationTime);

		// Update legacy pointer to point to the render buffer for Vulkan
		// (renderBufferIndex points to the previously completed buffer)
		pos_CUDA = pos_CUDA_buffers[renderBufferIndex];
		vertexData_external = vertexData_buffers[renderBufferIndex];
	}

	//Common to all above mesh if blocks
	cudaResult = cudaGetLastError();
	if(cudaResult != CUDA_SUCCESS)
	{
		FileIO("display(): kernel failed: %s\n", cudaGetErrorString(cudaResult));
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}

	// OPTIMIZATION 3: Use timeline semaphore for CUDA-Vulkan synchronization
	// This provides more efficient GPU-GPU sync than device-wide synchronization
	if(bTimelineSemaphoreSupported && cudaExtSemaphore_timelineVulkan)
	{
		// Signal the timeline semaphore from CUDA after kernel completion
		timelineSemaphoreValue++;

		cudaExternalSemaphoreSignalParams signalParams;
		memset(&signalParams, 0, sizeof(cudaExternalSemaphoreSignalParams));
		signalParams.params.fence.value = timelineSemaphoreValue;
		signalParams.flags = 0;

		cudaResult = cudaSignalExternalSemaphoresAsync(&cudaExtSemaphore_timelineVulkan, &signalParams, 1, cudaStream_compute);
		if(cudaResult != cudaSuccess)
		{
			FileIO("display(): cudaSignalExternalSemaphoresAsync failed: %s\n", cudaGetErrorString(cudaResult));
			// Fall back to device sync on failure
			cudaResult = cudaStreamSynchronize(cudaStream_compute);
		}
	}
	else
	{
		// Fallback: synchronize on the specific stream instead of entire device
		// This is more efficient than cudaDeviceSynchronize() as it only waits
		// for this stream's operations to complete
		cudaResult = cudaStreamSynchronize(cudaStream_compute);
		if(cudaResult != CUDA_SUCCESS)
		{
			FileIO("display(): cudaStreamSynchronize failed\n");
			vkResult = VK_ERROR_INITIALIZATION_FAILED;
			return vkResult;
		}
	}

	// ========================================================================
	// BUFFER ROTATION for Double/Triple Buffering
	// ========================================================================
	// After CUDA kernel completion (sync done above), rotate the buffer indices
	// The buffer just written becomes the new render buffer for next frame
	// The old render buffer becomes the new compute buffer
	//
	// For double buffering (N=2): Simple swap between 0 and 1
	// For triple buffering (N=3): Circular rotation 0->1->2->0
	// ========================================================================
	{
		unsigned int previousComputeBuffer = computeBufferIndex;

		// Rotate compute buffer to next index (circular)
		computeBufferIndex = (computeBufferIndex + 1) % NUM_VERTEX_BUFFERS;

		// The just-completed compute buffer becomes the render buffer
		renderBufferIndex = previousComputeBuffer;

		// Update legacy pointers to point to current render buffer
		// This ensures Vulkan reads from the just-computed data
		vertexData_external = vertexData_buffers[renderBufferIndex];
		pos_CUDA = pos_CUDA_buffers[renderBufferIndex];
	}

	//UpdateIndirectBuffer
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
	
	// Acquire index of next swapChainImage
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkAcquireNextImageKHR.html
	/*
	// Provided by VK_KHR_swapchain
	VkResult vkAcquireNextImageKHR(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    uint64_t                                    timeout, // Waiting time from our side for swapchain to give the image for device. (Time in nanoseconds)
    VkSemaphore                                 semaphore, // Waiting for another queque to release the image held by another queque demanded by swapchain
    VkFence                                     fence, // ask host to wait image to be given by swapchain
    uint32_t*                                   pImageIndex);
	
	If this function  will not get image index from swapchain within gven time or timeout, then vkAcquireNextImageKHR() will return VK_NOT_READY
	4th paramater is waiting for another queque to release the image held by another queque demanded by swapchain
	*/
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
	
	/*
	Use fence to allow host to wait for completion of execution of previous commandbuffer.
	Magacha command buffer cha operation complete vhava mhanun vaprat aahe he fence
	*/
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkWaitForFences.html
	/*
	// Provided by VK_VERSION_1_0
	VkResult vkWaitForFences(
    VkDevice                                    device,
    uint32_t                                    fenceCount,
    const VkFence*                              pFences,
    VkBool32                                    waitAll,
    uint64_t                                    timeout);
	*/
	vkResult = vkWaitForFences(vkDevice, 1, &vkFence_array[currentImageIndex], VK_TRUE, UINT64_MAX);
	if(vkResult != VK_SUCCESS)
	{
		FileIO("display(): vkWaitForFences() failed\n");
		return vkResult;
	}
	
	//Now ready the fences for next commandbuffer.
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkResetFences.html
	/*
	// Provided by VK_VERSION_1_0
	VkResult vkResetFences(
    VkDevice                                    device,
    uint32_t                                    fenceCount,
    const VkFence*                              pFences);
	*/
	vkResult = vkResetFences(vkDevice, 1, &vkFence_array[currentImageIndex]);
	if(vkResult != VK_SUCCESS)
	{
		FileIO("display(): vkResetFences() failed\n");
		return vkResult;
	}
	
	//One of the memebers of VkSubmitInfo structure requires array of pipeline stages. We have only one of completion of color attachment output.
	//Still we need 1 member array.
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineStageFlags.html
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineStageFlagBits.html
	const VkPipelineStageFlags waitDstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		
	// https://registry.khronos.org/vulkan/specs/latest/man/html/VkSubmitInfo.html
	// Declare, memset and initialize VkSubmitInfo structure
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
	
	//Now submit above work to the queque
	vkResult = vkQueueSubmit(vkQueue, 1, &vkSubmitInfo, vkFence_array[currentImageIndex]); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueSubmit.html
	if(vkResult != VK_SUCCESS)
	{
		FileIO("display(): vkQueueSubmit() failed\n");
		return vkResult;
	}
	
	//We are going to present the rendered image after declaring  and initializing VkPresentInfoKHR struct
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentInfoKHR.html
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
	
	//Present the queque
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueuePresentKHR.html
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
	
	//31.7
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
	// Code
	animationTime = animationTime + 0.005f;
}

/*
void uninitialize(void)
{
		// Function Declarations
		void ToggleFullScreen(void);


		if (gbFullscreen == TRUE)
		{
			ToggleFullscreen();
			gbFullscreen = FALSE;
		}

		// Destroy Window
		if (ghwnd)
		{
			DestroyWindow(ghwnd);
			ghwnd = NULL;
		}
		
		
		//10. When done destroy it uninitilialize() by using vkDestroySwapchainKHR() (https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroySwapchainKHR.html) Vulkan API.
		//Destroy swapchain
		vkDestroySwapchainKHR(vkDevice, vkSwapchainKHR, NULL);
		vkSwapchainKHR = VK_NULL_HANDLE;
		FileIO("uninitialize(): vkDestroySwapchainKHR() is done\n");
		
		//Destroy Vulkan device
		
		//No need to destroy/uninitialize device queque
		
		//No need to destroy selected physical device
		if(vkDevice)
		{
			vkDeviceWaitIdle(vkDevice); //First synchronization function
			FileIO("uninitialize(): vkDeviceWaitIdle() is done\n");
			vkDestroyDevice(vkDevice, NULL); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyDevice.html
			vkDevice = VK_NULL_HANDLE;
			FileIO("uninitialize(): vkDestroyDevice() is done\n");
		}
		
		if(vkSurfaceKHR)
		{
			// The destroy() of vkDestroySurfaceKHR() generic not platform specific
			vkDestroySurfaceKHR(vkInstance, vkSurfaceKHR, NULL); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroySurfaceKHR.html
			vkSurfaceKHR = VK_NULL_HANDLE;
			FileIO("uninitialize(): vkDestroySurfaceKHR() sucedded\n");
		}

		// Destroy VkInstance in uninitialize()
		if(vkInstance)
		{
			vkDestroyInstance(vkInstance, NULL); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyInstance.html
			vkInstance = VK_NULL_HANDLE;
			FileIO("uninitialize(): vkDestroyInstance() sucedded\n");
		}

		FileIO("uninitialize()-> Program ended successfully.\n");
}
*/

void uninitialize(void)
{
		// Function Declarations
		void ToggleFullScreen(void);
		
		//Cuda related function declarations
		cudaError_t uninitialize_cuda(void);

		if (gbFullscreen == TRUE)
		{
			ToggleFullscreen();
			gbFullscreen = FALSE;
		}
		
	
		// Destroy Window
		if (ghwnd)
		{
			DestroyWindow(ghwnd);
			ghwnd = NULL;
		}
		
		//Destroy Vulkan device
		if(vkDevice)
		{
			vkDeviceWaitIdle(vkDevice); //First synchronization function
			FileIO("uninitialize(): vkDeviceWaitIdle() is done\n");
			
			/*
			18_7. In uninitialize(), first in a loop with swapchain image count as counter, destroy frnce array objects using vkDestroyFence() {https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyFence.html} and then
				actually free allocated fences array by using free().
			*/
			//Destroying fences
			for(uint32_t i = 0; i< swapchainImageCount; i++)
			{
				vkDestroyFence(vkDevice, vkFence_array[i], NULL); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyFence.html
				FileIO("uninitialize(): vkFence_array[%d] is freed\n", i);
			}
			
			if(vkFence_array)
			{
				free(vkFence_array);
				vkFence_array = NULL;
				FileIO("uninitialize(): vkFence_array is freed\n");
			}
			
			/*
			18_8. Destroy both global semaphore objects  with two separate calls to vkDestroySemaphore() {https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroySemaphore.html}.
			*/
			//https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroySemaphore.html
			if(vkSemaphore_RenderComplete)
			{
				vkDestroySemaphore(vkDevice, vkSemaphore_RenderComplete, NULL);
				vkSemaphore_RenderComplete = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkSemaphore_RenderComplete is freed\n");
			}
			
			if(vkSemaphore_BackBuffer)
			{
				vkDestroySemaphore(vkDevice, vkSemaphore_BackBuffer, NULL);
				vkSemaphore_BackBuffer = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkSemaphore_BackBuffer is freed\n");
			}

			// Destroy timeline semaphore for CUDA-Vulkan interop
			if(vkSemaphore_timeline)
			{
				vkDestroySemaphore(vkDevice, vkSemaphore_timeline, NULL);
				vkSemaphore_timeline = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkSemaphore_timeline is freed\n");
			}
			
			/*
			Step_17_3. In unitialize destroy framebuffers in a loop for swapchainImageCount.
			https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyFramebuffer.html
			*/
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
			
			/*
			24.5. In uninitialize, call vkDestroyDescriptorSetlayout() Vulkan API to destroy this Vulkan object.
			//https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyDescriptorSetLayout.html
			// Provided by VK_VERSION_1_0
			void vkDestroyDescriptorSetLayout(
			VkDevice                                    device,
			VkDescriptorSetLayout                       descriptorSetLayout,
			const VkAllocationCallbacks*                pAllocator);
			*/
			if(vkDescriptorSetLayout)
			{
				vkDestroyDescriptorSetLayout(vkDevice, vkDescriptorSetLayout, NULL);
				vkDescriptorSetLayout = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkDescriptorSetLayout is freed\n");
			}
			
			/*
			25.5. In uninitialize, call vkDestroyPipelineLayout() Vulkan API to destroy this vkPipelineLayout Vulkan object.
			//https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyPipelineLayout.html
			// Provided by VK_VERSION_1_0
			void vkDestroyPipelineLayout(
				VkDevice                                    device,
				VkPipelineLayout                            pipelineLayout,
				const VkAllocationCallbacks*                pAllocator);
			*/
			if(vkPipelineLayout)
			{
				vkDestroyPipelineLayout(vkDevice, vkPipelineLayout, NULL);
				vkPipelineLayout = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkPipelineLayout is freed\n");
			}
			
			//Step_16_6. In uninitialize , destroy the renderpass by using vkDestrorRenderPass() (https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyRenderPass.html).
			if(vkRenderPass)
			{
				vkDestroyRenderPass(vkDevice, vkRenderPass, NULL); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyRenderPass.html
				vkRenderPass = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkDestroyRenderPass() is done\n");
			}
			
			//31.8 Destroy descriptorpool (When descriptor pool is destroyed, descriptor sets created by that pool are also destroyed implicitly)
			if(vkDescriptorPool)
			{
				//https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyDescriptorPool.html
				vkDestroyDescriptorPool(vkDevice, vkDescriptorPool, NULL);
				vkDescriptorPool = VK_NULL_HANDLE;
				vkDescriptorSet = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkDestroyDescriptorPool() is done for vkDescriptorPool and vkDescriptorSet both\n");
			}
			
			/*
			23.11. In uninitialize , destroy both global shader objects using vkDestroyShaderModule() Vulkan API.
			//https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyShaderModule.html
			// Provided by VK_VERSION_1_0
			void vkDestroyShaderModule(
			VkDevice device,
			VkShaderModule shaderModule,
			const VkAllocationCallbacks* pAllocator);
			*/
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

			// Destroy tessellation shader modules
			if(vkShaderModule_tesc)
			{
				vkDestroyShaderModule(vkDevice, vkShaderModule_tesc, NULL);
				vkShaderModule_tesc = VK_NULL_HANDLE;
				FileIO("uninitialize(): VkShaderModule for tessellation control shader is done\n");
			}

			if(vkShaderModule_tese)
			{
				vkDestroyShaderModule(vkDevice, vkShaderModule_tese, NULL);
				vkShaderModule_tese = VK_NULL_HANDLE;
				FileIO("uninitialize(): VkShaderModule for tessellation evaluation shader is done\n");
			}

			//31.9 Destroy uniform buffer
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

			// Destroy tessellation uniform buffer
			if(tessUniformData.vkBuffer)
			{
				vkDestroyBuffer(vkDevice, tessUniformData.vkBuffer, NULL);
				tessUniformData.vkBuffer = VK_NULL_HANDLE;
				FileIO("uninitialize(): tessUniformData.vkBuffer is freed\n");
			}

			if(tessUniformData.vkDeviceMemory)
			{
				vkFreeMemory(vkDevice, tessUniformData.vkDeviceMemory, NULL);
				tessUniformData.vkDeviceMemory = VK_NULL_HANDLE;
				FileIO("uninitialize(): tessUniformData.vkDeviceMemory is freed\n");
			}

			// Destroy height map texture resources
			if(vkSampler_heightMap)
			{
				vkDestroySampler(vkDevice, vkSampler_heightMap, NULL);
				vkSampler_heightMap = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkSampler_heightMap is freed\n");
			}

			if(vkImageView_heightMap)
			{
				vkDestroyImageView(vkDevice, vkImageView_heightMap, NULL);
				vkImageView_heightMap = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkImageView_heightMap is freed\n");
			}

			if(vkDeviceMemory_heightMap)
			{
				vkFreeMemory(vkDevice, vkDeviceMemory_heightMap, NULL);
				vkDeviceMemory_heightMap = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkDeviceMemory_heightMap is freed\n");
			}

			if(vkImage_heightMap)
			{
				vkDestroyImage(vkDevice, vkImage_heightMap, NULL);
				vkImage_heightMap = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkImage_heightMap is freed\n");
			}

			//uninitialize CUDA
			cudaResult = uninitialize_cuda();
			if(cudaResult != CUDA_SUCCESS)
			{
				FileIO("uninitialize(): uninitialize_cuda() failed\n");
			}
			else
			{
				FileIO("uninitialize(): uninitialize_cuda() suceeded\n");
			}
			
			/*
			22.14. In uninitialize()
			First Free the ".vkDeviceMemory" memory of our global structure using vkFreeMemory() and then destroy ".vkBuffer" member of our global structure by using vkDestroyBuffer().
			
			//https://registry.khronos.org/vulkan/specs/latest/man/html/vkFreeMemory.html
			// Provided by VK_VERSION_1_0
			void vkFreeMemory(
				VkDevice                                    device,
				VkDeviceMemory                              memory,
				const VkAllocationCallbacks*                pAllocator);
				
			https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyBuffer.html
			// Provided by VK_VERSION_1_0
			void vkDestroyBuffer(
				VkDevice                                    device,
				VkBuffer                                    buffer,
				const VkAllocationCallbacks*                pAllocator);
			*/
			// ================================================================
			// Cleanup all multiple vertex buffers (double/triple buffering)
			// ================================================================
			FileIO("uninitialize(): Cleaning up %d vertex buffers for double/triple buffering\n", NUM_VERTEX_BUFFERS);

			for(int bufIdx = 0; bufIdx < NUM_VERTEX_BUFFERS; bufIdx++)
			{
				if(vertexData_buffers[bufIdx].vkDeviceMemory)
				{
					vkFreeMemory(vkDevice, vertexData_buffers[bufIdx].vkDeviceMemory, NULL);
					vertexData_buffers[bufIdx].vkDeviceMemory = VK_NULL_HANDLE;
					FileIO("uninitialize(): vertexData_buffers[%d].vkDeviceMemory is freed\n", bufIdx);
				}

				if(vertexData_buffers[bufIdx].vkBuffer)
				{
					vkDestroyBuffer(vkDevice, vertexData_buffers[bufIdx].vkBuffer, NULL);
					vertexData_buffers[bufIdx].vkBuffer = VK_NULL_HANDLE;
					FileIO("uninitialize(): vertexData_buffers[%d].vkBuffer is freed\n", bufIdx);
				}
			}

			// Clear legacy pointers (they pointed to one of the buffers above)
			vertexData_external.vkDeviceMemory = VK_NULL_HANDLE;
			vertexData_external.vkBuffer = VK_NULL_HANDLE;
			FileIO("uninitialize(): Legacy vertexData_external pointers cleared\n");
			
			/*
			22.14. In uninitialize()
			First Free the ".vkDeviceMemory" memory of our global structure using vkFreeMemory() and then destroy ".vkBuffer" member of our global structure by using vkDestroyBuffer().
			
			//https://registry.khronos.org/vulkan/specs/latest/man/html/vkFreeMemory.html
			// Provided by VK_VERSION_1_0
			void vkFreeMemory(
				VkDevice                                    device,
				VkDeviceMemory                              memory,
				const VkAllocationCallbacks*                pAllocator);
				
			https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyBuffer.html
			// Provided by VK_VERSION_1_0
			void vkDestroyBuffer(
				VkDevice                                    device,
				VkBuffer                                    buffer,
				const VkAllocationCallbacks*                pAllocator);
			*/
			//Free indirect buffer
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

			//Step_15_4. In unitialize(), free each command buffer by using vkFreeCommandBuffers()(https://registry.khronos.org/vulkan/specs/latest/man/html/vkFreeCommandBuffers.html) in a loop of size swapchainImage count.
			for(uint32_t i =0; i < swapchainImageCount; i++)
			{
				vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_for_1024_x_1024_graphics_array[i]);
				FileIO("uninitialize(): vkFreeCommandBuffers() is done\n");
			}
			
				//Step_15_5. Free actual command buffer array.
			if(vkCommandBuffer_for_1024_x_1024_graphics_array)
			{
				free(vkCommandBuffer_for_1024_x_1024_graphics_array);
				vkCommandBuffer_for_1024_x_1024_graphics_array = NULL;
				FileIO("uninitialize(): vkCommandBuffer_for_1024_x_1024_graphics_array is freed\n");
			}

			//Step_14_3 In uninitialize(), destroy commandpool using VkDestroyCommandPool.
			// https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyCommandPool.html
			if(vkCommandPool)
			{
				vkDestroyCommandPool(vkDevice, vkCommandPool, NULL);
				vkCommandPool = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkDestroyCommandPool() is done\n");
			}
			
			//destroy depth image view
			if(vkImageView_depth)
			{
				vkDestroyImageView(vkDevice, vkImageView_depth, NULL); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyImageView.html
				vkImageView_depth = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkImageView_depth is done\n");
			}
			
			//destroy device memory for depth image
			if(vkDeviceMemory_depth)
			{
				vkFreeMemory(vkDevice, vkDeviceMemory_depth, NULL); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkFreeMemory.html
				vkDeviceMemory_depth = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkDeviceMemory_depth is done\n");
			}
			
			//destroy depth image
			if(vkImage_depth)
			{
				//https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyImage.html
				vkDestroyImage(vkDevice, vkImage_depth, NULL);
				vkImage_depth = VK_NULL_HANDLE;
				FileIO("uninitialize(): vkImage_depth is done\n");
			}
			
			/*
			9. In unitialize(), keeping the "destructor logic aside" for a while , first destroy image views from imagesViews array in a loop using vkDestroyImageViews() api.
			(https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyImageView.html)
			*/
			for(uint32_t i =0; i < swapchainImageCount; i++)
			{
				vkDestroyImageView(vkDevice, swapChainImageView_array[i], NULL);
				FileIO("uninitialize(): vkDestroyImageView() is done\n");
			}
			
			/*
			10. In uninitialize() , now actually free imageView array using free().
			free imageView array
			*/
			if(swapChainImageView_array)
			{
				free(swapChainImageView_array);
				swapChainImageView_array = NULL;
				FileIO("uninitialize():swapChainImageView_array is freed\n");
			}
			
			/*
			7. In unitialize(), keeping the "destructor logic aside" for a while , first destroy swapchain images from swap chain images array in a loop using vkDestroyImage() api. 
			(https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyImage.html)
			//Free swap chain images
			*/
			/*
			for(uint32_t i = 0; i < swapchainImageCount; i++)
			{
				vkDestroyImage(vkDevice, swapChainImage_array[i], NULL);
				FileIO("uninitialize(): vkDestroyImage() is done\n");
			}
			*/
			
			/*
			8. In uninitialize() , now actually free swapchain image array using free().
			*/
			if(swapChainImage_array)
			{
				free(swapChainImage_array);
				swapChainImage_array = NULL;
				FileIO("uninitialize():swapChainImage_array is freed\n");
			}
			
			/*
			10. When done destroy it uninitilialize() by using vkDestroySwapchainKHR() (https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroySwapchainKHR.html) Vulkan API.
			Destroy swapchain
			*/
			vkDestroySwapchainKHR(vkDevice, vkSwapchainKHR, NULL);
			vkSwapchainKHR = VK_NULL_HANDLE;
			FileIO("uninitialize(): vkDestroySwapchainKHR() is done\n");
			
			
			vkDestroyDevice(vkDevice, NULL); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyDevice.html
			vkDevice = VK_NULL_HANDLE;
			FileIO("uninitialize(): vkDestroyDevice() is done\n");
		}
		
		//No need to destroy/uninitialize device queque
		
		//No need to destroy selected physical device
		
		if(vkSurfaceKHR)
		{
			/*
			The destroy() of vkDestroySurfaceKHR() generic not platform specific
			*/
			vkDestroySurfaceKHR(vkInstance, vkSurfaceKHR, NULL); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroySurfaceKHR.html
			vkSurfaceKHR = VK_NULL_HANDLE;
			FileIO("uninitialize(): vkDestroySurfaceKHR() sucedded\n");
		}

		//21_Validation
		if(vkDebugReportCallbackEXT && vkDestroyDebugReportCallbackEXT_fnptr)
		{
			vkDestroyDebugReportCallbackEXT_fnptr(vkInstance, vkDebugReportCallbackEXT, NULL); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyDebugReportCallbackEXT.html
			vkDebugReportCallbackEXT = VK_NULL_HANDLE;
			vkDestroyDebugReportCallbackEXT_fnptr = NULL; //Nahi kel tari chalel
		}

		/*
		Destroy VkInstance in uninitialize()
		*/
		if(vkInstance)
		{
			vkDestroyInstance(vkInstance, NULL); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyInstance.html
			vkInstance = VK_NULL_HANDLE;
			FileIO("uninitialize(): vkDestroyInstance() sucedded\n");
		}

		FileIO("uninitialize()-> Program ended successfully.\n");
}

cudaError_t uninitialize_cuda(void)
{
	// ========================================================================
	// Cleanup all CUDA resources for double/triple buffering
	// ========================================================================

	FileIO("uninitialize_cuda()-> Starting cleanup of %d vertex buffers\n", NUM_VERTEX_BUFFERS);

	// Free all CUDA mapped buffer pointers for multiple buffers
	for(int i = 0; i < NUM_VERTEX_BUFFERS; i++)
	{
		if(pos_CUDA_buffers[i])
		{
			cudaResult = cudaFree(pos_CUDA_buffers[i]);
			if(cudaResult != cudaSuccess)
			{
				FileIO("uninitialize_cuda()-> cudaFree() failed for buffer %d: %s\n", i, cudaGetErrorString(cudaResult));
				// Continue cleanup despite error
			}
			else
			{
				FileIO("uninitialize_cuda()-> cudaFree() succeeded for buffer %d\n", i);
			}
			pos_CUDA_buffers[i] = NULL;
		}
	}

	// Clear legacy pointer
	pos_CUDA = NULL;

	// Destroy all external memory handles for multiple buffers
	for(int i = 0; i < NUM_VERTEX_BUFFERS; i++)
	{
		if(cuExternalMemory_buffers[i])
		{
			cudaResult = cudaDestroyExternalMemory(cuExternalMemory_buffers[i]);
			if(cudaResult != cudaSuccess)
			{
				FileIO("uninitialize_cuda()-> cudaDestroyExternalMemory() failed for buffer %d: %s\n", i, cudaGetErrorString(cudaResult));
				// Continue cleanup despite error
			}
			else
			{
				FileIO("uninitialize_cuda()-> cudaDestroyExternalMemory() succeeded for buffer %d\n", i);
			}
			cuExternalMemory_buffers[i] = NULL;
		}
	}

	// Clear legacy handle
	cuExternalMemory_t = NULL;

	// Destroy CUDA stream
	if(cudaStream_compute)
	{
		cudaResult = cudaStreamDestroy(cudaStream_compute);
		if(cudaResult != cudaSuccess)
		{
			FileIO("uninitialize_cuda()-> cudaStreamDestroy() failed\n");
			return cudaResult;
		}
		else
		{
			cudaStream_compute = NULL;
			FileIO("uninitialize_cuda()-> cudaStreamDestroy() succeeded\n");
		}
	}

	// Destroy CUDA external semaphore
	if(cudaExtSemaphore_timelineVulkan)
	{
		cudaResult = cudaDestroyExternalSemaphore(cudaExtSemaphore_timelineVulkan);
		if(cudaResult != cudaSuccess)
		{
			FileIO("uninitialize_cuda()-> cudaDestroyExternalSemaphore() failed\n");
			return cudaResult;
		}
		else
		{
			cudaExtSemaphore_timelineVulkan = NULL;
			FileIO("uninitialize_cuda()-> cudaDestroyExternalSemaphore() succeeded\n");
		}
	}

	return cudaSuccess;
}

// Function to import Vulkan timeline semaphore into CUDA for GPU-GPU synchronization
cudaError_t ImportVulkanTimelineSemaphoreToCUDA(void)
{
	cudaError_t result = cudaSuccess;

	if(bTimelineSemaphoreSupported == FALSE || vkSemaphore_timeline == VK_NULL_HANDLE)
	{
		FileIO("ImportVulkanTimelineSemaphoreToCUDA(): Timeline semaphore not available, skipping CUDA import\n");
		return cudaSuccess; // Not an error, just skip
	}

	// Get Win32 handle from Vulkan semaphore
	HANDLE semaphoreHandle = NULL;

	// Get the function pointer for vkGetSemaphoreWin32HandleKHR
	PFN_vkGetSemaphoreWin32HandleKHR vkGetSemaphoreWin32HandleKHR_fnptr =
		(PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(vkDevice, "vkGetSemaphoreWin32HandleKHR");

	if(vkGetSemaphoreWin32HandleKHR_fnptr == NULL)
	{
		FileIO("ImportVulkanTimelineSemaphoreToCUDA(): vkGetSemaphoreWin32HandleKHR not available\n");
		bTimelineSemaphoreSupported = FALSE;
		return cudaSuccess; // Non-fatal
	}

	VkSemaphoreGetWin32HandleInfoKHR vkSemaphoreGetWin32HandleInfoKHR;
	memset(&vkSemaphoreGetWin32HandleInfoKHR, 0, sizeof(VkSemaphoreGetWin32HandleInfoKHR));
	vkSemaphoreGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
	vkSemaphoreGetWin32HandleInfoKHR.pNext = NULL;
	vkSemaphoreGetWin32HandleInfoKHR.semaphore = vkSemaphore_timeline;
	vkSemaphoreGetWin32HandleInfoKHR.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

	VkResult vkResult = vkGetSemaphoreWin32HandleKHR_fnptr(vkDevice, &vkSemaphoreGetWin32HandleInfoKHR, &semaphoreHandle);
	if(vkResult != VK_SUCCESS || semaphoreHandle == NULL)
	{
		FileIO("ImportVulkanTimelineSemaphoreToCUDA(): vkGetSemaphoreWin32HandleKHR failed with error %d\n", vkResult);
		bTimelineSemaphoreSupported = FALSE;
		return cudaSuccess; // Non-fatal
	}

	FileIO("ImportVulkanTimelineSemaphoreToCUDA(): Got Win32 handle for timeline semaphore\n");

	// Import the semaphore handle into CUDA
	cudaExternalSemaphoreHandleDesc extSemaphoreHandleDesc;
	memset(&extSemaphoreHandleDesc, 0, sizeof(cudaExternalSemaphoreHandleDesc));
	extSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
	extSemaphoreHandleDesc.handle.win32.handle = semaphoreHandle;
	extSemaphoreHandleDesc.flags = 0;

	result = cudaImportExternalSemaphore(&cudaExtSemaphore_timelineVulkan, &extSemaphoreHandleDesc);
	if(result != cudaSuccess)
	{
		FileIO("ImportVulkanTimelineSemaphoreToCUDA(): cudaImportExternalSemaphore failed: %s\n", cudaGetErrorString(result));
		bTimelineSemaphoreSupported = FALSE;
		return cudaSuccess; // Non-fatal, will fall back to device sync
	}

	FileIO("ImportVulkanTimelineSemaphoreToCUDA(): Successfully imported timeline semaphore for CUDA-Vulkan sync\n");
	return cudaSuccess;
}

//Definition of Vulkan related functions

VkResult CreateVulkanInstance(void)
{
	/*
		As explained before fill and initialize required extension names and count in 2 respective global variables (Lasst 8 steps mhanje instance cha first step)
	*/
	//Function declarations
	VkResult FillInstanceExtensionNames(void);
	
	//Added in 21_Validation 
	VkResult FillValidationLayerNames(void);
	VkResult CreateValidationCallbackFunction(void);
	
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;

	// Code
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
	
	//21_Validation
	if(bValidation == TRUE)
	{
		//21_Validation
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
	
	/*
	Initialize struct VkApplicationInfo (Somewhat limbu timbu)
	*/
	struct VkApplicationInfo vkApplicationInfo;
	memset((void*)&vkApplicationInfo, 0, sizeof(struct VkApplicationInfo)); //Dont use ZeroMemory to keep parity across all OS
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkApplicationInfo.html/
	vkApplicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; //First member of all Vulkan structure, for genericness and typesafety
	vkApplicationInfo.pNext = NULL;
	vkApplicationInfo.pApplicationName = gpszAppName; //any string will suffice
	vkApplicationInfo.applicationVersion = 1; //any number will suffice
	vkApplicationInfo.pEngineName = gpszAppName; //any string will suffice
	vkApplicationInfo.engineVersion = 1; //any number will suffice
	/*
	Mahatavacha aahe, 
	on fly risk aahe Sir used VK_API_VERSION_1_3 as installed 1.3.296 version
	Those using 1.4.304 must use VK_API_VERSION_1_4
	*/
	vkApplicationInfo.apiVersion = VK_API_VERSION_1_4; 
	
	/*
	Initialize struct VkInstanceCreateInfo by using information from Step1 and Step2 (Important)
	*/
	struct VkInstanceCreateInfo vkInstanceCreateInfo;
	memset((void*)&vkInstanceCreateInfo, 0, sizeof(struct VkInstanceCreateInfo));
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkInstanceCreateInfo.html
	vkInstanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	vkInstanceCreateInfo.pNext = NULL;
	vkInstanceCreateInfo.pApplicationInfo = &vkApplicationInfo;
	//folowing 2 members important
	vkInstanceCreateInfo.enabledExtensionCount = enabledInstanceExtensionsCount;
	vkInstanceCreateInfo.ppEnabledExtensionNames = enabledInstanceExtensionNames_array;
	//21_Validation
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

	/*
	Call vkCreateInstance() to get VkInstance in a global variable and do error checking
	*/
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateInstance.html
	//2nd parameters is NULL as saying tuza memory allocator vapar , mazyakade custom memory allocator nahi
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
	
	//21_validation: do for validation callbacks
	if(bValidation)
	{
		//21_Validation
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
	// Code
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;

	/*
	1. Find how many instance extensions are supported by Vulkan driver of/for this version and keept the count in a local variable.
	1.3.296 madhe ek instance navta , je aata add zala aahe 1.4.304 madhe , VK_NV_DISPLAY_STEREO
	*/
	uint32_t instanceExtensionCount = 0;

	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkEnumerateInstanceExtensionProperties.html
	vkResult = vkEnumerateInstanceExtensionProperties(NULL, &instanceExtensionCount, NULL);
	/* like in OpenCL
	1st - which layer extension required, as want all so NULL (akha driver supported kelleli extensions)
	2nd - count de mala
	3rd - Extension cha property cha array, NULL aahe karan count nahi ajun aplyakade
	*/
	if (vkResult != VK_SUCCESS)
	{
		FileIO("FillInstanceExtensionNames(): First call to vkEnumerateInstanceExtensionProperties()  function failed\n");
		return vkResult;
	}
	else
	{
		FileIO("FillInstanceExtensionNames(): First call to vkEnumerateInstanceExtensionProperties() succedded\n");
	}

	/*
	 Allocate and fill struct VkExtensionProperties 
	 (https://registry.khronos.org/vulkan/specs/latest/man/html/VkExtensionProperties.html) structure array, 
	 corresponding to above count
	*/
	VkExtensionProperties* vkExtensionProperties_array = NULL;
	vkExtensionProperties_array = (VkExtensionProperties*)malloc(sizeof(VkExtensionProperties) * instanceExtensionCount);
	if (vkExtensionProperties_array != NULL)
	{
		//Add log here later for failure
		//exit(-1);
	}
	else
	{
		//Add log here later for success
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

	/*
	Fill and display a local string array of extension names obtained from VkExtensionProperties structure array
	*/
	char** instanceExtensionNames_array = NULL;
	instanceExtensionNames_array = (char**)malloc(sizeof(char*) * instanceExtensionCount);
	if (instanceExtensionNames_array != NULL)
	{
		//Add log here later for failure
		//exit(-1);
	}
	else
	{
		//Add log here later for success
	}

	for (uint32_t i =0; i < instanceExtensionCount; i++)
	{
		//https://registry.khronos.org/vulkan/specs/latest/man/html/VkExtensionProperties.html
		instanceExtensionNames_array[i] = (char*)malloc( sizeof(char) * (strlen(vkExtensionProperties_array[i].extensionName) + 1));
		memcpy(instanceExtensionNames_array[i], vkExtensionProperties_array[i].extensionName, (strlen(vkExtensionProperties_array[i].extensionName) + 1));
		FileIO("FillInstanceExtensionNames(): Vulkan Instance Extension Name = %s\n", instanceExtensionNames_array[i]);
	}

	/*
	As not required here onwards, free VkExtensionProperties array
	*/
	if (vkExtensionProperties_array)
	{
		free(vkExtensionProperties_array);
		vkExtensionProperties_array = NULL;
	}

	/*
	Find whether above extension names contain our required two extensions
	VK_KHR_SURFACE_EXTENSION_NAME
	VK_KHR_WIN32_SURFACE_EXTENSION_NAME
	VK_EXT_DEBUG_REPORT_EXTENSION_NAME (added for 21_Validation)
	Accordingly set two global variables, "required extension count" and "required extension names array"
	*/
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkBool32.html -> Vulkan cha bool
	VkBool32 vulkanSurfaceExtensionFound = VK_FALSE;
	VkBool32 vulkanWin32SurfaceExtensionFound = VK_FALSE;
	
	//21_Validation
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
				//array will not have entry so no code here
				//enabledInstanceExtensionNames_array[enabledInstanceExtensionsCount++] = VK_EXT_DEBUG_REPORT_EXTENSION_NAME;
			}
		}
	}

	/*
	As not needed hence forth , free local string array
	*/
	for (uint32_t i =0 ; i < instanceExtensionCount; i++)
	{
		free(instanceExtensionNames_array[i]);
	}
	free(instanceExtensionNames_array);

	/*
	Print whether our required instance extension names or not (He log madhe yenar. Jithe print asel sarv log madhe yenar)
	*/
	if (vulkanSurfaceExtensionFound == VK_FALSE)
	{
		//Type mismatch in return VkResult and VKBool32, so return hardcoded failure
		vkResult = VK_ERROR_INITIALIZATION_FAILED; //return hardcoded failure
		FileIO("FillInstanceExtensionNames(): VK_KHR_SURFACE_EXTENSION_NAME not found\n");
		return vkResult;
	}
	else
	{
		FileIO("FillInstanceExtensionNames(): VK_KHR_SURFACE_EXTENSION_NAME is found\n");
	}

	if (vulkanWin32SurfaceExtensionFound == VK_FALSE)
	{
		//Type mismatch in return VkResult and VKBool32, so return hardcoded failure
		vkResult = VK_ERROR_INITIALIZATION_FAILED; //return hardcoded failure
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
			//Type mismatch in return VkResult and VKBool32, so return hardcoded failure
			vkResult = VK_ERROR_INITIALIZATION_FAILED; //return hardcoded failure
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
			//Type mismatch in return VkResult and VKBool32, so return hardcoded failure
			//vkResult = VK_ERROR_INITIALIZATION_FAILED; //return hardcoded failure
			FileIO("FillInstanceExtensionNames(): Validation is ON, but required VK_EXT_DEBUG_REPORT_EXTENSION_NAME is also supported\n");
			//return vkResult;
		}
		else
		{
			FileIO("FillInstanceExtensionNames(): Validation is OFF, but VK_EXT_DEBUG_REPORT_EXTENSION_NAME is also supported\n");
		}
	}

	/*
	Print only enabled extension names
	*/
	for (uint32_t i = 0; i < enabledInstanceExtensionsCount; i++)
	{
		FileIO("FillInstanceExtensionNames(): Enabled Vulkan Instance Extension Name = %s\n", enabledInstanceExtensionNames_array[i]);
	}

	return vkResult;
}

VkResult FillValidationLayerNames(void)
{
	//Code
	
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;
	
	uint32_t validationLayerCount = 0;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkEnumerateInstanceLayerProperties.html
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
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkLayerProperties.html
	VkLayerProperties* vkLayerProperties_array = NULL;
	vkLayerProperties_array = (VkLayerProperties*)malloc(sizeof(VkLayerProperties) * validationLayerCount);
	if (vkLayerProperties_array != NULL)
	{
		//Add log here later for failure
		//exit(-1);
	}
	else
	{
		//Add log here later for success
	}
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkEnumerateInstanceLayerProperties.html
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
		//Add log here later for failure
		//exit(-1);
	}
	else
	{
		//Add log here later for success
	}

	for (uint32_t i =0; i < validationLayerCount; i++)
	{
		//https://registry.khronos.org/vulkan/specs/latest/man/html/VkLayerProperties.html
		validationLayerNames_array[i] = (char*)malloc( sizeof(char) * (strlen(vkLayerProperties_array[i].layerName) + 1));
		memcpy(validationLayerNames_array[i], vkLayerProperties_array[i].layerName, (strlen(vkLayerProperties_array[i].layerName) + 1));
		FileIO("FillValidationLayerNames(): Vulkan Instance Layer Name = %s\n", validationLayerNames_array[i]);
	}

	if (vkLayerProperties_array)
	{
		free(vkLayerProperties_array);
		vkLayerProperties_array = NULL;
	}
	
	//For required 1 validation layer VK_LAYER_KHRONOS_validation
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
		//Type mismatch in return VkResult and VKBool32, so return hardcoded failure
		vkResult = VK_ERROR_INITIALIZATION_FAILED; //return hardcoded failure
		FileIO("FillValidationLayerNames(): VK_LAYER_KHRONOS_validation not supported\n");
		return vkResult;
	}
	else
	{
		FileIO("FillValidationLayerNames(): VK_LAYER_KHRONOS_validation is supported\n");
	}
	
	/*
	Print only enabled extension names
	*/
	for (uint32_t i = 0; i < enabledValidationLayerCount; i++)
	{
		FileIO("FillValidationLayerNames(): Enabled Vulkan Validation Layer Name = %s\n", enabledValidationlayerNames_array[i]);
	}
	
	return vkResult;
}

VkResult CreateValidationCallbackFunction(void)
{
	//Function declaration
	/*
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugReportFlagsEXT.html
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VKAPI_ATTR.html
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugReportObjectTypeEXT.html
	//https://registry.khronos.org/vulkan/specs/latest/man/html/PFN_vkDebugReportCallbackEXT.html
	*/
	VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallback(VkDebugReportFlagsEXT, VkDebugReportObjectTypeEXT, uint64_t, size_t, int32_t, const char*, const char*, void*);
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateDebugReportCallbackEXT.html
	PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT_fnptr = NULL;
	
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;
	
	
	//Code
	//get required function pointers
	/*
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetInstanceProcAddr.html
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateDebugReportCallbackEXT.html
	*/
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
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyDebugReportCallbackEXT.html
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
	
	//get VulkanDebugReportCallback object
	/*
	VkDebugReportCallbackEXT *vkDebugReportCallbackEXT = VK_NULL_HANDLE; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugReportCallbackEXT.html

	//https://registry.khronos.org/vulkan/specs/latest/man/html/PFN_vkDebugReportCallbackEXT.html 
	PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT_fnptr = NULL; 
	*/
	VkDebugReportCallbackCreateInfoEXT vkDebugReportCallbackCreateInfoEXT ; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugReportCallbackCreateInfoEXT.html
	memset((void*)&vkDebugReportCallbackCreateInfoEXT, 0, sizeof(VkDebugReportCallbackCreateInfoEXT));
	/*
	// Provided by VK_EXT_debug_report
	typedef struct VkDebugReportCallbackCreateInfoEXT {
		VkStructureType                 sType;
		const void*                     pNext;
		VkDebugReportFlagsEXT           flags;
		PFN_vkDebugReportCallbackEXT    pfnCallback;
		void*                           pUserData;
	} VkDebugReportCallbackCreateInfoEXT;
	*/
	vkDebugReportCallbackCreateInfoEXT.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
	vkDebugReportCallbackCreateInfoEXT.pNext = NULL;
	vkDebugReportCallbackCreateInfoEXT.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT|VK_DEBUG_REPORT_WARNING_BIT_EXT|VK_DEBUG_REPORT_INFORMATION_BIT_EXT|VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT|VK_DEBUG_REPORT_DEBUG_BIT_EXT; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugReportFlagBitsEXT.html
	vkDebugReportCallbackCreateInfoEXT.pfnCallback = debugReportCallback;
	vkDebugReportCallbackCreateInfoEXT.pUserData = NULL;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateDebugReportCallbackEXT.html
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
	//Code
	
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;
	
	/*
	Declare and memset a platform(Windows, Linux , Android etc) specific SurfaceInfoCreate structure
	*/
	VkWin32SurfaceCreateInfoKHR vkWin32SurfaceCreateInfoKHR;
	memset((void*)&vkWin32SurfaceCreateInfoKHR, 0 , sizeof(struct VkWin32SurfaceCreateInfoKHR));
	
	/*
	Initialize it , particularly its HINSTANCE and HWND members
	*/
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkWin32SurfaceCreateInfoKHR.html
	vkWin32SurfaceCreateInfoKHR.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
	vkWin32SurfaceCreateInfoKHR.pNext = NULL;
	vkWin32SurfaceCreateInfoKHR.flags = 0;
	vkWin32SurfaceCreateInfoKHR.hinstance = (HINSTANCE)GetWindowLongPtr(ghwnd, GWLP_HINSTANCE); //This member can also be initialized by using (HINSTANCE)GetModuleHandle(NULL); {typecasted HINSTANCE}
	vkWin32SurfaceCreateInfoKHR.hwnd = ghwnd;
	
	/*
	Now call VkCreateWin32SurfaceKHR() to create the presentation surface object
	*/
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateWin32SurfaceKHR.html
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
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;
	
	/*
	2. Call vkEnumeratePhysicalDevices() to get Physical device count
	*/
	vkResult = vkEnumeratePhysicalDevices(vkInstance, &physicalDeviceCount, NULL); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkEnumeratePhysicalDevices.html (first call)
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
	
	/*
	3. Allocate VkPhysicalDeviceArray object according to above count
	*/
	vkPhysicalDevice_array = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * physicalDeviceCount);
	//for sake of brevity no error checking
	
	/*
	4. Call vkEnumeratePhysicalDevices() again to fill above array
	*/
	vkResult = vkEnumeratePhysicalDevices(vkInstance, &physicalDeviceCount, vkPhysicalDevice_array); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkEnumeratePhysicalDevices.html (seocnd call)
	if (vkResult != VK_SUCCESS)
	{
		FileIO("GetPhysicalDevice(): vkEnumeratePhysicalDevices() second call failed with error code %d\n", vkResult);
		vkResult = VK_ERROR_INITIALIZATION_FAILED; //return hardcoded failure
		return vkResult;
	}
	else
	{
		FileIO("GetPhysicalDevice(): vkEnumeratePhysicalDevices() second call succedded\n");
	}
	
	/*
	5. Start a loop using physical device count and physical device, array obtained above (Note: declare a boolean bFound variable before this loop which will decide whether we found desired physical device or not)
	Inside this loop, 
	a. Declare a local variable to hold queque count
	b. Call vkGetPhysicalDeviceQuequeFamilyProperties() to initialize above queque count variable
	c. Allocate VkQuequeFamilyProperties array according to above count
	d. Call vkGetPhysicalDeviceQuequeFamilyProperties() again to fill above array
	e. Declare VkBool32 type array and allocate it using the same above queque count
	f. Start a nested loop and fill above VkBool32 type array by calling vkGetPhysicalDeviceSurfaceSupportKHR()
	g. Start another nested loop(not inside above loop , but nested in main loop) and check whether physical device
	   in its array with its queque family "has"(Sir told to underline) graphics bit or not. 
	   If yes then this is a selected physical device and assign it to global variable. 
	   Similarly this index is the selected queque family index and assign it to global variable too and set bFound to true
	   and break out from second nested loop
	h. Now we are back in main loop, so free queque family array and VkBool32 type array
	i. Still being in main loop, acording to bFound variable break out from main loop
	j. free physical device array 
	*/
	VkBool32 bFound = VK_FALSE; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkBool32.html
	for(uint32_t i = 0; i < physicalDeviceCount; i++)
	{
		/*
		a. Declare a local variable to hold queque count
		*/
		uint32_t quequeCount = UINT32_MAX;
		
		
		/*
		b. Call vkGetPhysicalDeviceQuequeFamilyProperties() to initialize above queque count variable
		*/
		//Strange call returns void
		//Error checking not done above as yacha VkResult nahi aahe
		//Kiti physical devices denar , jevde array madhe aahet tevda -> Second parameter
		//If physical device is present , then it must separate atleast one qurque family
		vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice_array[i], &quequeCount, NULL);//https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceQueueFamilyProperties.html
		
		/*
		c. Allocate VkQuequeFamilyProperties array according to above count
		*/
		struct VkQueueFamilyProperties *vkQueueFamilyProperties_array = NULL;//https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueueFamilyProperties.html
		vkQueueFamilyProperties_array = (struct VkQueueFamilyProperties*) malloc(sizeof(struct VkQueueFamilyProperties) * quequeCount);
		//for sake of brevity no error checking
		
		/*
		d. Call vkGetPhysicalDeviceQuequeFamilyProperties() again to fill above array
		*/
		vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice_array[i], &quequeCount, vkQueueFamilyProperties_array);//https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceQueueFamilyProperties.html
		
		/*
		e. Declare VkBool32 type array and allocate it using the same above queque count
		*/
		VkBool32 *isQuequeSurfaceSupported_array = NULL;
		isQuequeSurfaceSupported_array = (VkBool32*) malloc(sizeof(VkBool32) * quequeCount);
		//for sake of brevity no error checking
		
		/*
		f. Start a nested loop and fill above VkBool32 type array by calling vkGetPhysicalDeviceSurfaceSupportKHR()
		*/
		for(uint32_t j =0; j < quequeCount ; j++)
		{
			//vkGetPhysicalDeviceSurfaceSupportKHR ->Supported surface la tumhi dilela surface support karto ka?
			//vkPhysicalDevice_array[i] -> ya device cha
			//j -> ha index
			//vkSurfaceKHR -> ha surface
			//isQuequeSurfaceSupported_array-> support karto ki nahi bhar
			vkResult = vkGetPhysicalDeviceSurfaceSupportKHR(vkPhysicalDevice_array[i], j, vkSurfaceKHR, &isQuequeSurfaceSupported_array[j]); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSurfaceSupportKHR.html
		}
		
		/*
		g. Start another nested loop(not inside above loop , but nested in main loop) and check whether physical device
		   in its array with its queque family "has"(Sir told to underline) graphics bit or not. 
		   If yes then this is a selected physical device and assign it to global variable. 
		   Similarly this index is the selected queque family index and assign it to global variable too and set bFound to true
		   and break out from second nested loop
		*/
		for(uint32_t j =0; j < quequeCount ; j++)
		{
			//https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueueFamilyProperties.html
			//https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueueFlagBits.html
			if(vkQueueFamilyProperties_array[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				//select ith graphic card, queque familt at j, bFound la TRUE karun break vha
				if(isQuequeSurfaceSupported_array[j] == VK_TRUE)
				{
					vkPhysicalDevice_selected = vkPhysicalDevice_array[i];
					graphicsQuequeFamilyIndex_selected = j;
					bFound = VK_TRUE;
					break;
				}
			}
		}
		
		/*
		h. Now we are back in main loop, so free queque family array and VkBool32 type array
		*/
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
		
		/*
		i. Still being in main loop, acording to bFound variable break out from main loop
		*/
		if(bFound == VK_TRUE)
		{
			break;
		}
	}
	
	/*
	6. Do error checking according to value of bFound
	*/
	if(bFound == VK_TRUE)
	{
		FileIO("GetPhysicalDevice(): GetPhysicalDevice() suceeded to select required physical device with graphics enabled\n");
		
		/*
		PrintVulkanInfo() changes
		2. Accordingly remove physicaldevicearray freeing block from if(bFound == VK_TRUE) block and we will later write this freeing block in printVkInfo().
		*/
		
		/*
		//j. free physical device array 
		if(vkPhysicalDevice_array)
		{
			free(vkPhysicalDevice_array);
			vkPhysicalDevice_array = NULL;
			FileIO("GetPhysicalDevice(): succedded to free vkPhysicalDevice_array\n");
		}
		*/
	}
	else
	{
		FileIO("GetPhysicalDevice(): GetPhysicalDevice() failed to obtain graphics supported physical device\n");
		
		/*
		j. free physical device array 
		*/
		if(vkPhysicalDevice_array)
		{
			free(vkPhysicalDevice_array);
			vkPhysicalDevice_array = NULL;
			FileIO("GetPhysicalDevice(): succedded to free vkPhysicalDevice_array\n");
		}
		
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	
	/*
	7. memset the global physical device memory property structure
	*/
	memset((void*)&vkPhysicalDeviceMemoryProperties, 0, sizeof(struct VkPhysicalDeviceMemoryProperties)); //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceMemoryProperties.html
	
	/*
	8. initialize above structure by using vkGetPhysicalDeviceMemoryProperties() //https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceMemoryProperties.html
	No need of error checking as we already have physical device
	*/
	vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice_selected, &vkPhysicalDeviceMemoryProperties);
	
	/*
	9. Declare a local structure variable VkPhysicalDeviceFeatures, memset it  and initialize it by calling vkGetPhysicalDeviceFeatures() 
	// https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceFeatures.html
	// //https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceFeatures.html
	*/
	VkPhysicalDeviceFeatures vkPhysicalDeviceFeatures;
	memset((void*)&vkPhysicalDeviceFeatures, 0, sizeof(VkPhysicalDeviceFeatures));
	vkGetPhysicalDeviceFeatures(vkPhysicalDevice_selected, &vkPhysicalDeviceFeatures);
	
	/*
	10. By using "tescellation shader" member of above structure check selected device's tescellation shader support
	11. By using "geometry shader" member of above structure check selected device's geometry shader support
	*/
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
	
	/*
	12. There is no need to free/uninitialize/destroy selected physical device?
	Bcoz later we will create Vulkan logical device which need to be destroyed and its destruction will automatically destroy selected physical device.
	*/
	
	return vkResult;
}

/*
PrintVkInfo() changes
3. Write printVkInfo() user defined function with following steps
3a. Start a loop using global physical device count and inside it declare  and memset VkPhysicalDeviceProperties struct variable (https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceProperties.html).
3b. Initialize this struct variable by calling vkGetPhysicalDeviceProperties() (https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceProperties.html) vulkan api.
3c. Print Vulkan API version using apiVersion member of above struct.
	This requires 3 Vulkan macros.
3d. Print device name by using "deviceName" member of above struct.
3e. Use "deviceType" member of above struct in a switch case block and accordingly print device type.
3f. Print hexadecimal Vendor Id of device using "vendorId" member of above struct.
3g. Print hexadecimal deviceID of device using "deviceId" member of struct.
Note*: For sake of completeness, we can repeat a to h points from GetPhysicalDevice() {05-GetPhysicalDevice notes},
but now instead of assigning selected queque and selected device, print whether this device supports graphic bit, compute bit, transfer bit using if else if else if blocks
Similarly we also can repeat device features from GetPhysicalDevice() and can print all around 50 plus device features including support to tescellation shader and geometry shader.
3h. Free physicaldevice array here which we removed from if(bFound == VK_TRUE) block of GetPhysicalDevice().
*/
VkResult PrintVulkanInfo(void)
{
	VkResult vkResult = VK_SUCCESS;
	
	//Code
	FileIO("************************* Shree Ganesha******************************\n");
	
	/*
	PrintVkInfo() changes
	3a. Start a loop using global physical device count and inside it declare  and memset VkPhysicalDeviceProperties struct variable
	*/
	for(uint32_t i = 0; i < physicalDeviceCount; i++)
	{
		/*
		PrintVkInfo() changes
		3b. Initialize this struct variable by calling vkGetPhysicalDeviceProperties()
		*/
		VkPhysicalDeviceProperties vkPhysicalDeviceProperties; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceProperties.html
		memset((void*)&vkPhysicalDeviceProperties, 0, sizeof(VkPhysicalDeviceProperties));
		vkGetPhysicalDeviceProperties(vkPhysicalDevice_array[i], &vkPhysicalDeviceProperties ); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceProperties.html
		
		/*
		PrintVkInfo() changes
		3c. Print Vulkan API version using apiVersion member of above struct.
		This requires 3 Vulkan macros.
		*/
		//uint32_t majorVersion,minorVersion,patchVersion;
		//https://registry.khronos.org/vulkan/specs/latest/man/html/VK_VERSION_MAJOR.html -> api deprecation for which we changed to VK_API_VERSION_XXXXX
		uint32_t majorVersion = VK_API_VERSION_MAJOR(vkPhysicalDeviceProperties.apiVersion); //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceProperties.html
		uint32_t minorVersion = VK_API_VERSION_MINOR(vkPhysicalDeviceProperties.apiVersion);
		uint32_t patchVersion = VK_API_VERSION_PATCH(vkPhysicalDeviceProperties.apiVersion);
		
		//API Version
		FileIO("apiVersion = %d.%d.%d\n", majorVersion, minorVersion, patchVersion);
		
		/*
		PrintVkInfo() changes
		3d. Print device name by using "deviceName" member of above struct.
		*/
		FileIO("deviceName = %s\n", vkPhysicalDeviceProperties.deviceName);
		
		/*
		PrintVkInfo() changes
		3e. Use "deviceType" member of above struct in a switch case block and accordingly print device type.
		*/
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
		
		/*
		PrintVkInfo() changes
		3f. Print hexadecimal Vendor Id of device using "vendorId" member of above struct.
		*/
		FileIO("vendorID = 0x%04x\n", vkPhysicalDeviceProperties.vendorID);
		
		/*
		PrintVkInfo() changes
		3g. Print hexadecimal deviceID of device using "deviceId" member of struct.
		*/
		FileIO("deviceID = 0x%04x\n", vkPhysicalDeviceProperties.deviceID);
	}
	
	/*
	PrintVkInfo() changes
	3h. Free physicaldevice array here which we removed from if(bFound == VK_TRUE) block of GetPhysicalDevice().
	*/
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
	// Code
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;

	/*
	1. Find how many device extensions are supported by Vulkan driver of/for this version and keept the count in a local variable.
	*/
	uint32_t deviceExtensionCount = 0;

	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkEnumerateDeviceExtensionProperties.html
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

	/*
	 Allocate and fill struct VkExtensionProperties 
	 (https://registry.khronos.org/vulkan/specs/latest/man/html/VkExtensionProperties.html) structure array, 
	 corresponding to above count
	*/
	VkExtensionProperties* vkExtensionProperties_array = NULL;
	vkExtensionProperties_array = (VkExtensionProperties*)malloc(sizeof(VkExtensionProperties) * deviceExtensionCount);
	if (vkExtensionProperties_array != NULL)
	{
		//Add log here later for failure
		//exit(-1);
	}
	else
	{
		//Add log here later for success
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

	/*
	Fill and display a local string array of extension names obtained from VkExtensionProperties structure array
	*/
	char** deviceExtensionNames_array = NULL;
	deviceExtensionNames_array = (char**)malloc(sizeof(char*) * deviceExtensionCount);
	if (deviceExtensionNames_array != NULL)
	{
		//Add log here later for failure
		//exit(-1);
	}
	else
	{
		//Add log here later for success
	}

	for (uint32_t i =0; i < deviceExtensionCount; i++)
	{
		//https://registry.khronos.org/vulkan/specs/latest/man/html/VkExtensionProperties.html
		deviceExtensionNames_array[i] = (char*)malloc( sizeof(char) * (strlen(vkExtensionProperties_array[i].extensionName) + 1));
		memcpy(deviceExtensionNames_array[i], vkExtensionProperties_array[i].extensionName, (strlen(vkExtensionProperties_array[i].extensionName) + 1));
		FileIO("FillDeviceExtensionNames(): Vulkan Device Extension Name = %s\n", deviceExtensionNames_array[i]);
	}

	/*
	As not required here onwards, free VkExtensionProperties array
	*/
	if (vkExtensionProperties_array)
	{
		free(vkExtensionProperties_array);
		vkExtensionProperties_array = NULL;
	}

	/*
	Find whether above extension names contain our required extensions
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
	VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME is macro for "VK_KHR_external_memory_win32"
	VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME - for CUDA-Vulkan timeline semaphore interop
	Accordingly set two global variables, "required extension count" and "required extension names array"
	*/
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkBool32.html -> Vulkan cha bool
	VkBool32 vulkanSwapchainExtensionFound = VK_FALSE;
	VkBool32 vulkanExternalMemoryWin32ExtensionFound = VK_FALSE;
	VkBool32 vulkanExternalSemaphoreWin32ExtensionFound = VK_FALSE;
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

		// Check for external semaphore extension (optional - for CUDA-Vulkan timeline semaphore sync)
		if (strcmp(deviceExtensionNames_array[i], VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME) == 0)
		{
			vulkanExternalSemaphoreWin32ExtensionFound = VK_TRUE;
			enabledDeviceExtensionNames_array[enabledDeviceExtensionsCount++] = VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME;
		}
	}

	/*
	As not needed hence forth , free local string array
	*/
	for (uint32_t i =0 ; i < deviceExtensionCount; i++)
	{
		free(deviceExtensionNames_array[i]);
	}
	free(deviceExtensionNames_array);

	/*
	Print whether our required device extension names or not (He log madhe yenar. Jithe print asel sarv log madhe yenar)
	*/
	if (vulkanSwapchainExtensionFound == VK_FALSE)
	{
		//Type mismatch in return VkResult and VKBool32, so return hardcoded failure
		vkResult = VK_ERROR_INITIALIZATION_FAILED; //return hardcoded failure
		FileIO("FillDeviceExtensionNames(): VK_KHR_SWAPCHAIN_EXTENSION_NAME not found\n");
		return vkResult;
	}
	else
	{
		FileIO("FillDeviceExtensionNames(): VK_KHR_SWAPCHAIN_EXTENSION_NAME is found\n");
	}
	
	if (vulkanExternalMemoryWin32ExtensionFound == VK_FALSE)
	{
		//Type mismatch in return VkResult and VKBool32, so return hardcoded failure
		vkResult = VK_ERROR_INITIALIZATION_FAILED; //return hardcoded failure
		FileIO("FillDeviceExtensionNames(): VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME not found\n");
		return vkResult;
	}
	else
	{
		FileIO("FillDeviceExtensionNames(): VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME is found\n");
	}

	// External semaphore extension is optional - used for CUDA-Vulkan timeline semaphore sync
	if (vulkanExternalSemaphoreWin32ExtensionFound == VK_FALSE)
	{
		// Non-fatal: timeline semaphores are an optional optimization
		bTimelineSemaphoreSupported = FALSE;
		FileIO("FillDeviceExtensionNames(): VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME not found (optional - CUDA will use stream sync)\n");
	}
	else
	{
		FileIO("FillDeviceExtensionNames(): VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME is found (timeline semaphore sync enabled)\n");
	}

	/*
	Print only enabled device extension names
	*/
	for (uint32_t i = 0; i < enabledDeviceExtensionsCount; i++)
	{
		FileIO("FillDeviceExtensionNames(): Enabled Vulkan Device Extension Name = %s\n", enabledDeviceExtensionNames_array[i]);
	}

	return vkResult;
}

VkResult CreateVulKanDevice(void)
{
	//function declaration
	VkResult FillDeviceExtensionNames(void);
	
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;
	
	/*
	fill device extensions
	2. Call previously created FillDeviceExtensionNames() in it.
	*/
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
	
	/*
	Newly added code
	*/
	//float queuePriorities[1]  = {1.0};
	float queuePriorities[1];
	queuePriorities[0] = 1.0f;
	VkDeviceQueueCreateInfo vkDeviceQueueCreateInfo; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceQueueCreateInfo.html
	memset(&vkDeviceQueueCreateInfo, 0, sizeof(VkDeviceQueueCreateInfo));
	
	vkDeviceQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	vkDeviceQueueCreateInfo.pNext = NULL;
	vkDeviceQueueCreateInfo.flags = 0;
	vkDeviceQueueCreateInfo.queueFamilyIndex = graphicsQuequeFamilyIndex_selected;
	vkDeviceQueueCreateInfo.queueCount = 1;
	vkDeviceQueueCreateInfo.pQueuePriorities = queuePriorities;
	
	/*
	3. Declare and initialize VkDeviceCreateInfo structure (https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceCreateInfo.html).
	*/
	VkDeviceCreateInfo vkDeviceCreateInfo;
	memset(&vkDeviceCreateInfo, 0, sizeof(VkDeviceCreateInfo));

	/*
	3.1 Enable required device features (tessellationShader for tessellation pipeline, fillModeNonSolid for wireframe rendering)
	*/
	VkPhysicalDeviceFeatures vkPhysicalDeviceFeatures;
	memset(&vkPhysicalDeviceFeatures, 0, sizeof(VkPhysicalDeviceFeatures));
	vkPhysicalDeviceFeatures.tessellationShader = VK_TRUE;
	vkPhysicalDeviceFeatures.fillModeNonSolid = VK_TRUE;

	/*
	3.2 Enable Vulkan 1.2 features (timelineSemaphore for CUDA-Vulkan interop)
	*/
	VkPhysicalDeviceVulkan12Features vkPhysicalDeviceVulkan12Features;
	memset(&vkPhysicalDeviceVulkan12Features, 0, sizeof(VkPhysicalDeviceVulkan12Features));
	vkPhysicalDeviceVulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	vkPhysicalDeviceVulkan12Features.pNext = NULL;
	vkPhysicalDeviceVulkan12Features.timelineSemaphore = VK_TRUE;

	/*
	4. Use previously obtained device extension count and device extension array to initialize this structure.
	*/
	vkDeviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	vkDeviceCreateInfo.pNext = &vkPhysicalDeviceVulkan12Features;
	vkDeviceCreateInfo.flags = 0;
	vkDeviceCreateInfo.enabledExtensionCount = enabledDeviceExtensionsCount;
	vkDeviceCreateInfo.ppEnabledExtensionNames = enabledDeviceExtensionNames_array;
	vkDeviceCreateInfo.enabledLayerCount = 0;
	vkDeviceCreateInfo.ppEnabledLayerNames = NULL;
	vkDeviceCreateInfo.pEnabledFeatures = &vkPhysicalDeviceFeatures;
	vkDeviceCreateInfo.queueCreateInfoCount = 1;
	vkDeviceCreateInfo.pQueueCreateInfos = &vkDeviceQueueCreateInfo;
	
	/*
	5. Now call vkCreateDevice to create actual Vulkan device and do error checking.
	*/
	vkResult = vkCreateDevice(vkPhysicalDevice_selected, &vkDeviceCreateInfo, NULL, &vkDevice); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateDevice.html
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
	//Code
	vkGetDeviceQueue(vkDevice, graphicsQuequeFamilyIndex_selected, 0, &vkQueue); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetDeviceQueue.html
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
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;
	
	//Code
	//Get count of supported surface color formats
	uint32_t FormatCount = 0;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSurfaceFormatsKHR.html
	vkResult = vkGetPhysicalDeviceSurfaceFormatsKHR(vkPhysicalDevice_selected, vkSurfaceKHR, &FormatCount, NULL);
	if(vkResult != VK_SUCCESS)
	{
		FileIO("getPhysicalDeviceSurfaceFormatAndColorSpace(): First call to vkGetPhysicalDeviceSurfaceFormatsKHR() failed\n");
		return vkResult;
	}
	else if(FormatCount == 0)
	{
		vkResult = VK_ERROR_INITIALIZATION_FAILED; //return hardcoded failure
		FileIO("vkGetPhysicalDeviceSurfaceFormatsKHR():: First call to vkGetPhysicalDeviceSurfaceFormatsKHR() returned FormatCount as 0\n");
		return vkResult;
	}
	else
	{
		FileIO("getPhysicalDeviceSurfaceFormatAndColorSpace(): First call to vkGetPhysicalDeviceSurfaceFormatsKHR() succedded\n");
	}
	
	//Declare and allocate VkSurfaceKHR array
	VkSurfaceFormatKHR *vkSurfaceFormatKHR_array = (VkSurfaceFormatKHR*)malloc(FormatCount * sizeof(VkSurfaceFormatKHR)); //https://registry.khronos.org/vulkan/specs/latest/man/html/VkSurfaceFormatKHR.html
	//For sake of brevity  no error checking
	
	//Filling the array
	vkResult = vkGetPhysicalDeviceSurfaceFormatsKHR(vkPhysicalDevice_selected, vkSurfaceKHR, &FormatCount, vkSurfaceFormatKHR_array); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSurfaceFormatsKHR.html
	if(vkResult != VK_SUCCESS)
	{
		FileIO("getPhysicalDeviceSurfaceFormatAndColorSpace(): Second call to vkGetPhysicalDeviceSurfaceFormatsKHR()  function failed\n");
		return vkResult;
	}
	else
	{
		FileIO("getPhysicalDeviceSurfaceFormatAndColorSpace():  Second call to vkGetPhysicalDeviceSurfaceFormatsKHR() succedded\n");
	}
	
	//According to contents of array , we have to decide surface format and color space
	//Decide surface format first
	if( (1 == FormatCount) && (vkSurfaceFormatKHR_array[0].format == VK_FORMAT_UNDEFINED) )
	{
		vkFormat_color = VK_FORMAT_B8G8R8A8_UNORM; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormat.html
	}
	else 
	{
		vkFormat_color = vkSurfaceFormatKHR_array[0].format; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormat.html
	}
	
	//Decide color space second
	vkColorSpaceKHR = vkSurfaceFormatKHR_array[0].colorSpace;
	
	//free the array
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
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;
	
	//Code
	//mailbox bhetel aata , fifo milel android la kadachit
	uint32_t presentModeCount = 0;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSurfacePresentModesKHR.html
	vkResult = vkGetPhysicalDeviceSurfacePresentModesKHR(vkPhysicalDevice_selected, vkSurfaceKHR, &presentModeCount, NULL);
	if(vkResult != VK_SUCCESS)
	{
		FileIO("getPhysicalDevicePresentMode(): First call to vkGetPhysicalDeviceSurfaceFormatsKHR() failed\n");
		return vkResult;
	}
	else if(presentModeCount == 0)
	{
		vkResult = VK_ERROR_INITIALIZATION_FAILED; //return hardcoded failure
		FileIO("getPhysicalDevicePresentMode():: First call to vkGetPhysicalDeviceSurfaceFormatsKHR() returned presentModeCount as 0\n");
		return vkResult;
	}
	else
	{
		FileIO("getPhysicalDevicePresentMode(): First call to vkGetPhysicalDeviceSurfaceFormatsKHR() succedded\n");
	}
	
	//Declare and allocate VkPresentModeKHR array
	VkPresentModeKHR  *vkPresentModeKHR_array = (VkPresentModeKHR*)malloc(presentModeCount * sizeof(VkPresentModeKHR)); //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentModeKHR.html
	//For sake of brevity  no error checking
	
	//Filling the array
	vkResult = vkGetPhysicalDeviceSurfacePresentModesKHR(vkPhysicalDevice_selected, vkSurfaceKHR, &presentModeCount, vkPresentModeKHR_array); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSurfaceFormatsKHR.html
	if(vkResult != VK_SUCCESS)
	{
		FileIO("getPhysicalDevicePresentMode(): Second call to vkGetPhysicalDeviceSurfacePresentModesKHR()  function failed\n");
		return vkResult;
	}
	else
	{
		FileIO("getPhysicalDevicePresentMode():  Second call to vkGetPhysicalDeviceSurfacePresentModesKHR() succedded\n");
	}
	
	//According to contents of array , we have to decide presentation mode
	for(uint32_t i=0; i < presentModeCount; i++)
	{
		if(vkPresentModeKHR_array[i] == VK_PRESENT_MODE_MAILBOX_KHR)
		{
			vkPresentModeKHR = VK_PRESENT_MODE_MAILBOX_KHR; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentModeKHR.html
			break;
		}
	}
	
	if(vkPresentModeKHR != VK_PRESENT_MODE_MAILBOX_KHR)
	{
		vkPresentModeKHR = VK_PRESENT_MODE_FIFO_KHR; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentModeKHR.html
	}
	
	FileIO("getPhysicalDevicePresentMode(): vkPresentModeKHR is %d\n", vkPresentModeKHR);
	
	//free the array
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
	/*
	Function Declaration
	*/
	VkResult getPhysicalDeviceSurfaceFormatAndColorSpace(void);
	VkResult getPhysicalDevicePresentMode(void);
	
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;
	
	/*
	Code
	*/
	
	/*
	Surface Format and Color Space
	1. Get Physical Device Surface supported color format and physical device surface supported color space , using Step 10.
	*/
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
	
	/*
	2. Get Physical Device Surface capabilities by using Vulkan API vkGetPhysicalDeviceSurfaceCapabilitiesKHR (https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceSurfaceCapabilitiesKHR.html)
    and accordingly initialize VkSurfaceCapabilitiesKHR structure (https://registry.khronos.org/vulkan/specs/latest/man/html/VkSurfaceCapabilitiesKHR.html).
	*/
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
	
	/*
	3. By using minImageCount and maxImageCount members of above structure , decide desired ImageCount for swapchain.
	*/
	uint32_t testingNumerOfSwapChainImages = vkSurfaceCapabilitiesKHR.minImageCount + 1;
	uint32_t desiredNumerOfSwapChainImages = 0; //To find this
	if( (vkSurfaceCapabilitiesKHR.maxImageCount > 0) && (vkSurfaceCapabilitiesKHR.maxImageCount < testingNumerOfSwapChainImages) )
	{
		desiredNumerOfSwapChainImages = vkSurfaceCapabilitiesKHR.maxImageCount;
	}
	else
	{
		desiredNumerOfSwapChainImages = vkSurfaceCapabilitiesKHR.minImageCount;
	}
		
	/*
	4. By using currentExtent.width and currentExtent.height members of above structure and comparing them with current width and height of window, decide image width and image height of swapchain.
	Choose size of swapchain image
	*/
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
	
		/*
		If surface size is already defined, then swapchain image size must match with it.
		*/
		VkExtent2D vkExtent2D;
		memset((void*)&vkExtent2D, 0, sizeof(VkExtent2D));
		vkExtent2D.width = (uint32_t)winWidth;
		vkExtent2D.height = (uint32_t)winHeight;
		
		vkExtent2D_SwapChain.width = glm::max(vkSurfaceCapabilitiesKHR.minImageExtent.width, glm::min(vkSurfaceCapabilitiesKHR.maxImageExtent.width, vkExtent2D.width));
		vkExtent2D_SwapChain.height = glm::max(vkSurfaceCapabilitiesKHR.minImageExtent.height, glm::min(vkSurfaceCapabilitiesKHR.maxImageExtent.height, vkExtent2D.height));
		FileIO("CreateSwapChain(): Swapchain Image Width x SwapChain  Image Height = %d X %d\n", vkExtent2D_SwapChain.width, vkExtent2D_SwapChain.height);
	}
	
	/*
	5. Decide how we are going to use swapchain images, means whether we we are going to store image data and use it later (Deferred Rendering) or we are going to use it immediatly as color attachment.
	Set Swapchain image usage flag
	Image usage flag hi concept aahe
	*/
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageUsageFlagBits.html
	VkImageUsageFlagBits vkImageUsageFlagBits = (VkImageUsageFlagBits) (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT); // VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT -> Imp, VK_IMAGE_USAGE_TRANSFER_SRC_BIT->Optional
	/*
	Although VK_IMAGE_USAGE_TRANSFER_SRC_BIT is not usefule here for triangle application.
	It is useful for texture, fbo, compute shader
	*/
	
	
	/*
	6. Swapchain  is capable of storing transformed image before presentation, which is called as PreTransform. 
    While creating swapchain , we can decide whether to pretransform or not the swapchain images. (Pre transform also includes flipping of image)
   
    Whether to consider pretransform/flipping or not?
	*/
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkSurfaceTransformFlagBitsKHR.html
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkSurfaceCapabilitiesKHR.html
	VkSurfaceTransformFlagBitsKHR vkSurfaceTransformFlagBitsKHR;
	if(vkSurfaceCapabilitiesKHR.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
	{
		vkSurfaceTransformFlagBitsKHR = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	}
	else
	{
		vkSurfaceTransformFlagBitsKHR = vkSurfaceCapabilitiesKHR.currentTransform;
	}
	
	/*
	Presentation Mode
	7. Get Presentation mode for swapchain images using Step 11.
	*/
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
	
	/*
	8. According to above data, declare ,memset and initialize VkSwapchainCreateInfoKHR  structure (https://registry.khronos.org/vulkan/specs/latest/man/html/VkSwapchainCreateInfoKHR.html)
	bas aata structure bharaycha aahe
	*/
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
	vkSwapchainCreateInfoKHR.imageArrayLayers = 1; //concept
	vkSwapchainCreateInfoKHR.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkSharingMode.html
	vkSwapchainCreateInfoKHR.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkCompositeAlphaFlagBitsKHR.html
	vkSwapchainCreateInfoKHR.presentMode = vkPresentModeKHR;
	vkSwapchainCreateInfoKHR.clipped = VK_TRUE;
	//vkSwapchainCreateInfoKHR.oldSwapchain is of no use in this application. Will be used in resize.
	
	/*
	9. At the end , call vkCreateSwapchainKHR() (https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateSwapchainKHR.html) Vulkan API to create the swapchain
	*/
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
	//Function Declarations 
	VkResult GetSupportedDepthFormat(void);

	//Variable declarations
	VkResult vkResult = VK_SUCCESS;
	
	//Code
	
	//1. Get Swapchain image count in a global variable using vkGetSwapchainImagesKHR() API (https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetSwapchainImagesKHR.html).
	vkResult = vkGetSwapchainImagesKHR(vkDevice, vkSwapchainKHR, &swapchainImageCount, NULL);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateImagesAndImageViews(): first call to vkGetSwapchainImagesKHR() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else if(swapchainImageCount == 0)
	{
		vkResult = vkResult = VK_ERROR_INITIALIZATION_FAILED; //return hardcoded failure
		FileIO("CreateImagesAndImageViews(): first call to vkGetSwapchainImagesKHR() function returned swapchain Image Count as 0\n");
		return vkResult;
	}
	else
	{
		FileIO("CreateImagesAndImageViews(): first call to vkGetSwapchainImagesKHR() succedded with swapchainImageCount as %d\n", swapchainImageCount);
	}
	
	//2. Declare a global VkImage type array and allocate it to swapchain image count using malloc. (https://registry.khronos.org/vulkan/specs/latest/man/html/VkImage.html)
	// Allocate swapchain image array
	swapChainImage_array = (VkImage*)malloc(sizeof(VkImage) * swapchainImageCount);
	if(swapChainImage_array == NULL)
	{
			FileIO("CreateImagesAndImageViews(): swapChainImage_array is NULL. malloc() failed\n");
	}
	
	//3. Now call same function again which we called in Step 1 and fill this array.
	//Fill this array by swapchain images
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
	
	//4. Declare another global array of type VkImageView(https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageView.html) and allocate it to sizeof Swapchain image count.
	// Allocate array of swapchain image view
	swapChainImageView_array = (VkImageView*)malloc(sizeof(VkImageView) * swapchainImageCount);
	if(swapChainImageView_array == NULL)
	{
			FileIO("CreateImagesAndImageViews(): swapChainImageView_array is NULL. malloc() failed\n");
	}
	
	//5. Declare  and initialize VkImageViewCreateInfo struct (https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageViewCreateInfo.html) except its ".image" member.
	//Initialize VkImageViewCreateInfo struct
	VkImageViewCreateInfo vkImageViewCreateInfo;
	memset((void*)&vkImageViewCreateInfo, 0, sizeof(VkImageViewCreateInfo));
	
	/*
	typedef struct VkImageViewCreateInfo {
    VkStructureType            sType;
    const void*                pNext;
    VkImageViewCreateFlags     flags;
    VkImage                    image;
    VkImageViewType            viewType;
    VkFormat                   format;
    VkComponentMapping         components;
    VkImageSubresourceRange    subresourceRange;
	} VkImageViewCreateInfo;
	*/
	
	vkImageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	vkImageViewCreateInfo.pNext = NULL;
	vkImageViewCreateInfo.flags = 0;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormat.html
	vkImageViewCreateInfo.format = vkFormat_color;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkComponentMapping.html
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkComponentSwizzle.html
	vkImageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_R;
	vkImageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_G;
	vkImageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_B;
	vkImageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_A;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageSubresourceRange.html
	/*
	typedef struct VkImageSubresourceRange {
    VkImageAspectFlags    aspectMask;
    uint32_t              baseMipLevel;
    uint32_t              levelCount;
    uint32_t              baseArrayLayer;
    uint32_t              layerCount;
	} VkImageSubresourceRange;
	*/
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageAspectFlags.html
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageAspectFlagBits.html
	vkImageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	vkImageViewCreateInfo.subresourceRange.baseMipLevel = 0;
	vkImageViewCreateInfo.subresourceRange.levelCount = 1;
	vkImageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
	vkImageViewCreateInfo.subresourceRange.layerCount = 1;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageViewType.html
	vkImageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	
	
	//6. Now start a loop for swapchain image count and inside this loop, initialize above ".image" member to swapchain image array index we obtained above and then call vkCreateImage() to fill  above ImageView array.
	//Fill image view array using above struct
	for(uint32_t i = 0; i < swapchainImageCount; i++)
	{
		vkImageViewCreateInfo.image = swapChainImage_array[i];
		
		//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateImageView.html
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
	
	//For depth image
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
	
	//For depth image, initialize VkImageCreateInfo
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageCreateInfo.html
	/*
	// Provided by VK_VERSION_1_0
	typedef struct VkImageCreateInfo {
		VkStructureType          sType;
		const void*              pNext;
		VkImageCreateFlags       flags;
		VkImageType              imageType;
		VkFormat                 format;
		VkExtent3D               extent;
		uint32_t                 mipLevels;
		uint32_t                 arrayLayers;
		VkSampleCountFlagBits    samples;
		VkImageTiling            tiling;
		VkImageUsageFlags        usage;
		VkSharingMode            sharingMode;
		uint32_t                 queueFamilyIndexCount;
		const uint32_t*          pQueueFamilyIndices;
		VkImageLayout            initialLayout;
	} VkImageCreateInfo;
	*/
	VkImageCreateInfo vkImageCreateInfo;
	memset((void*)&vkImageCreateInfo, 0, sizeof(VkImageCreateInfo));
	vkImageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	vkImageCreateInfo.pNext = NULL;
	vkImageCreateInfo.flags = 0;
	vkImageCreateInfo.imageType = VK_IMAGE_TYPE_2D; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageType.html
	vkImageCreateInfo.format = vkFormat_depth;
	
	vkImageCreateInfo.extent.width = (uint32_t)winWidth; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkExtent3D.html
	vkImageCreateInfo.extent.height = (uint32_t)winHeight; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkExtent3D.html
	vkImageCreateInfo.extent.depth = 1; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkExtent3D.html
	
	vkImageCreateInfo.mipLevels = 1;
	vkImageCreateInfo.arrayLayers = 1;
	vkImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkSampleCountFlagBits.html
	vkImageCreateInfo.tiling =  VK_IMAGE_TILING_OPTIMAL; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageTiling.html
	vkImageCreateInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageUsageFlags.html
	//vkImageCreateInfo.sharingMode = ;
	//vkImageCreateInfo.queueFamilyIndexCount = ;
	//vkImageCreateInfo.pQueueFamilyIndices = ;
	//vkImageCreateInfo.initialLayout = ;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateImage.html
	/*
	// Provided by VK_VERSION_1_0
	VkResult vkCreateImage(
    VkDevice                                    device,
    const VkImageCreateInfo*                    pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkImage*                                    pImage);
	*/
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
	
	//Memory requirements for depth Image
	/*
	// Provided by VK_VERSION_1_0
	typedef struct VkMemoryRequirements {
		VkDeviceSize    size;
		VkDeviceSize    alignment;
		uint32_t        memoryTypeBits;
	} VkMemoryRequirements;
	*/
	VkMemoryRequirements vkMemoryRequirements;
	memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetBufferMemoryRequirements.html
	/*
	// Provided by VK_VERSION_1_0
	void vkGetBufferMemoryRequirements(
    VkDevice                                    device,
    VkBuffer                                    buffer,
    VkMemoryRequirements*                       pMemoryRequirements);
	*/
	vkGetImageMemoryRequirements(vkDevice, vkImage_depth, &vkMemoryRequirements);
	
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryAllocateInfo.html
	/*
	// Provided by VK_VERSION_1_0
	typedef struct VkMemoryAllocateInfo {
		VkStructureType    sType;
		const void*        pNext;
		VkDeviceSize       allocationSize;
		uint32_t           memoryTypeIndex;
	} VkMemoryAllocateInfo;
	*/
	VkMemoryAllocateInfo vkMemoryAllocateInfo;
	memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
	vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	vkMemoryAllocateInfo.pNext = NULL;
	vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceSize.html (vkMemoryRequirements allocates memory in regions.)
	
	vkMemoryAllocateInfo.memoryTypeIndex = 0; //Initial value before entering into the loop
	for(uint32_t i =0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++) //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceMemoryProperties.html
	{
		if((vkMemoryRequirements.memoryTypeBits & 1) == 1) //https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryRequirements.html
		{
			//https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryType.html
			//https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryPropertyFlagBits.html
			if(vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
			{
				vkMemoryAllocateInfo.memoryTypeIndex = i;
				break;
			}			
		}
		vkMemoryRequirements.memoryTypeBits >>= 1;
	}
	
	/*
	// Provided by VK_VERSION_1_0
	VkResult vkAllocateMemory(
    VkDevice                                    device,
    const VkMemoryAllocateInfo*                 pAllocateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDeviceMemory*                             pMemory);
	*/
	vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vkDeviceMemory_depth); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkAllocateMemory.html
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateImagesAndImageViews(): vkAllocateMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateImagesAndImageViews(): vkAllocateMemory() succedded\n");
	}
	
	/*
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkBindBufferMemory.html
	// Provided by VK_VERSION_1_0
	VkResult vkBindBufferMemory(
    VkDevice                                    device,
    VkBuffer                                    buffer, //whom to bind
    VkDeviceMemory                              memory, //what to bind
    VkDeviceSize                                memoryOffset);
	*/
	vkResult = vkBindImageMemory(vkDevice, vkImage_depth, vkDeviceMemory_depth, 0); // We are binding device memory object handle with Vulkan buffer object handle. 
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateImagesAndImageViews(): vkBindBufferMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateImagesAndImageViews(): vkBindBufferMemory() succedded\n");
	}
	
	//Create ImageView for above depth image
	//Declare  and initialize VkImageViewCreateInfo struct (https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageViewCreateInfo.html) except its ".image" member.
	//Initialize VkImageViewCreateInfo struct
	memset((void*)&vkImageViewCreateInfo, 0, sizeof(VkImageViewCreateInfo));
	
	/*
	typedef struct VkImageViewCreateInfo {
    VkStructureType            sType;
    const void*                pNext;
    VkImageViewCreateFlags     flags;
    VkImage                    image;
    VkImageViewType            viewType;
    VkFormat                   format;
    VkComponentMapping         components;
    VkImageSubresourceRange    subresourceRange;
	} VkImageViewCreateInfo;
	*/
	
	vkImageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	vkImageViewCreateInfo.pNext = NULL;
	vkImageViewCreateInfo.flags = 0;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormat.html
	vkImageViewCreateInfo.format = vkFormat_depth;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkComponentMapping.html
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkComponentSwizzle.html
	//vkImageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_R;
	//vkImageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_G;
	//vkImageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_B;
	//vkImageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_A;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageSubresourceRange.html
	/*
	typedef struct VkImageSubresourceRange {
    VkImageAspectFlags    aspectMask;
    uint32_t              baseMipLevel;
    uint32_t              levelCount;
    uint32_t              baseArrayLayer;
    uint32_t              layerCount;
	} VkImageSubresourceRange;
	*/
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageAspectFlags.html
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageAspectFlagBits.html
	vkImageViewCreateInfo.subresourceRange.aspectMask =  VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT;
	vkImageViewCreateInfo.subresourceRange.baseMipLevel = 0;
	vkImageViewCreateInfo.subresourceRange.levelCount = 1;
	vkImageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
	vkImageViewCreateInfo.subresourceRange.layerCount = 1;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageViewType.html
	vkImageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	vkImageViewCreateInfo.image = vkImage_depth;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateImageView.html
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
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;
	
	////https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormat.html
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
		//https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormatProperties.html
		//https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormatFeatureFlags.html
		//https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormatFeatureFlagBits.html
		VkFormatProperties vkFormatProperties;
		memset((void*)&vkFormatProperties, 0, sizeof(vkFormatProperties));
		
		//https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormatProperties.html
		//https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetPhysicalDeviceFormatProperties.html
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
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;
	
	/*
	1. Declare and initialize VkCreateCommandPoolCreateInfo structure.
	https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandPoolCreateInfo.html
	
	typedef struct VkCommandPoolCreateInfo {
    VkStructureType             sType;
    const void*                 pNext;
    VkCommandPoolCreateFlags    flags;
    uint32_t                    queueFamilyIndex;
	} VkCommandPoolCreateInfo;
	
	*/
	VkCommandPoolCreateInfo vkCommandPoolCreateInfo;
	memset((void*)&vkCommandPoolCreateInfo, 0, sizeof(VkCommandPoolCreateInfo));
	
	vkCommandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	vkCommandPoolCreateInfo.pNext = NULL;
	/*
	This flag states that Vulkan should create such command pools which will contain such command buffers capable of reset and restart.
	These command buffers are long lived.
	Other transient one{transfer one} is short lived.
	*/
	vkCommandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandPoolCreateFlagBits.html
	vkCommandPoolCreateInfo.queueFamilyIndex = graphicsQuequeFamilyIndex_selected;
	
	/*
	2. Call VkCreateCommandPool to create command pool.
	https://registry.khronos.org/VulkanSC/specs/1.0-extensions/man/html/vkCreateCommandPool.html
	*/
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
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;
	
	/*
	Command Buffer
	*/
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandBuffer.html
	VkCommandBuffer *vkCommandBuffer_array = NULL;
	
	/*
	Code
	*/
	
	/*
	1. Declare and initialize struct VkCommandBufferAllocateInfo (https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandBufferAllocateInfo.html)
	The number of command buffers are coventionally equal to number of swapchain images.
	
	typedef struct VkCommandBufferAllocateInfo {
    VkStructureType         sType;
    const void*             pNext;
    VkCommandPool           commandPool;
    VkCommandBufferLevel    level;
    uint32_t                commandBufferCount;
	} VkCommandBufferAllocateInfo;
	*/
	VkCommandBufferAllocateInfo vkCommandBufferAllocateInfo;
	memset((void*)&vkCommandBufferAllocateInfo, 0, sizeof(VkCommandBufferAllocateInfo));
	vkCommandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	vkCommandBufferAllocateInfo.pNext = NULL;
	//vkCommandBufferAllocateInfo.flags = 0;
	vkCommandBufferAllocateInfo.commandPool = vkCommandPool;
	vkCommandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; //https://docs.vulkan.org/spec/latest/chapters/cmdbuffers.html#VkCommandBufferAllocateInfo
	vkCommandBufferAllocateInfo.commandBufferCount = 1;
	
	/*
	2. Declare command buffer array globally and allocate it to swapchain image count.
	*/
	vkCommandBuffer_array = (VkCommandBuffer*)malloc(sizeof(VkCommandBuffer) * swapchainImageCount);
	//skipping error check for brevity
	
	/*
	3. In a loop , which is equal to swapchainImageCount, allocate each command buffer in above array by using vkAllocateCommandBuffers(). //https://registry.khronos.org/vulkan/specs/latest/man/html/vkAllocateCommandBuffers.html
   Remember at time of allocation all commandbuffers will be empty.
   Later we will record graphic/compute commands into them.
	*/
	for(uint32_t i = 0; i < swapchainImageCount; i++)
	{
		//https://registry.khronos.org/vulkan/specs/latest/man/html/vkAllocateCommandBuffers.html
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
	//Variable declarations	
	VkResult vkResult = VK_SUCCESS;
	
	//22. Position
	VertexData vertexdata_position;
	
	unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
	
	/*
	Code
	*/

	//Initialize external memory buffer info
	/*
	//https://registry.khronos.org/VulkanSC/specs/1.0-extensions/man/html/VkExternalMemoryBufferCreateInfo.html
	// Provided by VK_VERSION_1_1
	typedef struct VkExternalMemoryBufferCreateInfo {
		VkStructureType                    sType;
		const void*                        pNext;
		VkExternalMemoryHandleTypeFlags    handleTypes;
	} VkExternalMemoryBufferCreateInfo;
	*/
	VkExternalMemoryBufferCreateInfo vkExternalMemoryBufferCreateInfo;
	memset((void*)&vkExternalMemoryBufferCreateInfo, 0, sizeof(VkExternalMemoryBufferCreateInfo));
	vkExternalMemoryBufferCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
	vkExternalMemoryBufferCreateInfo.pNext = NULL;
	vkExternalMemoryBufferCreateInfo.handleTypes = vkExternalMemoryHandleTypeFlagBits;

	/*
	22.4. memset our global vertexData_position.
	*/
	memset((void*)&vertexdata_position, 0, sizeof(VertexData));
	
	/*
	22.5. Declare and memset VkBufferCreateInfo struct.
	It has 8 members, we will use 5
	Out of them, 2 are very important (Usage and Size)
	*/
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferCreateInfo.html
	VkBufferCreateInfo vkBufferCreateInfo;
	memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
	
	/*
	// Provided by VK_VERSION_1_0
	typedef struct VkBufferCreateInfo {
		VkStructureType        sType;
		const void*            pNext;
		VkBufferCreateFlags    flags;
		VkDeviceSize           size;
		VkBufferUsageFlags     usage;
		VkSharingMode          sharingMode;
		uint32_t               queueFamilyIndexCount;
		const uint32_t*        pQueueFamilyIndices;
	} VkBufferCreateInfo;
	*/
	vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vkBufferCreateInfo.pNext = &vkExternalMemoryBufferCreateInfo;
	vkBufferCreateInfo.flags = 0; //Valid flags are used in scattered(sparse) buffer
	vkBufferCreateInfo.size = size;
	vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferUsageFlagBits.html;
	/* //when one buffer shared in multiple queque's
	vkBufferCreateInfo.sharingMode =;
	vkBufferCreateInfo.queueFamilyIndexCount =;
	vkBufferCreateInfo.pQueueFamilyIndices =; 
	*/
	
	
	/*
	22.6. Call vkCreateBuffer() vulkan API in the ".vkBuffer" member of our global struct
	// Provided by VK_VERSION_1_0
	VkResult vkCreateBuffer(
    VkDevice                                    device,
    const VkBufferCreateInfo*                   pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkBuffer*                                   pBuffer);
	*/
	vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexdata_position.vkBuffer); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateBuffer.html
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateExternalVertexBuffer(): vkCreateBuffer() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateExternalVertexBuffer(): vkCreateBuffer() succedded\n");
	}
	
	/*
	22.7. Declare and member memset struct VkMemoryRequirements and then call vkGetBufferMemoryRequirements() API to get the memory requirements.
	// Provided by VK_VERSION_1_0
	typedef struct VkMemoryRequirements {
		VkDeviceSize    size;
		VkDeviceSize    alignment;
		uint32_t        memoryTypeBits;
	} VkMemoryRequirements;
	*/
	VkMemoryRequirements vkMemoryRequirements;
	memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetBufferMemoryRequirements.html
	/*
	// Provided by VK_VERSION_1_0
	void vkGetBufferMemoryRequirements(
    VkDevice                                    device,
    VkBuffer                                    buffer,
    VkMemoryRequirements*                       pMemoryRequirements);
	*/
	vkGetBufferMemoryRequirements(vkDevice, vertexdata_position.vkBuffer, &vkMemoryRequirements);
	
	//initialize exportable meory allocation
	//https://registry.khronos.org/VulkanSC/specs/1.0-extensions/man/html/VkExportMemoryAllocateInfo.html
	/*
	// Provided by VK_VERSION_1_1
	typedef struct VkExportMemoryAllocateInfo {
		VkStructureType                    sType;
		const void*                        pNext;
		VkExternalMemoryHandleTypeFlags    handleTypes;
	} VkExportMemoryAllocateInfo;
	*/
	VkExportMemoryAllocateInfo vkExportMemoryAllocateInfo;
	memset((void*)&vkExportMemoryAllocateInfo, 0, sizeof(VkExportMemoryAllocateInfo));
	vkExportMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO; //Chaining 2
	vkExportMemoryAllocateInfo.pNext = NULL;
	vkExportMemoryAllocateInfo.handleTypes = vkExternalMemoryHandleTypeFlagBits;
	
	
	/*
	   22.8. To actually allocate the required memory, we need to call vkAllocateMemory().
	   But before that we need to declare and memset VkMemoryAllocateInfo structure.
	   Important members of this structure are ".memoryTypeIndex" and ".allocationSize".
	   For ".allocationSize", use the size obtained from vkGetBufferMemoryRequirements().
	   For ".memoryTypeIndex" : 
	   a. Start a loop with count as vkPkysicalDeviceMemoryProperties.memoryTypeCount.
	   b. Inside the loop check vkMemoryRequiremts.memoryTypeBits contain 1 or not.
	   c. If yes, Check vkPhysicalDeviceMemoryProperties.memeoryTypes[i].propertyFlags member contains enum VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.
	   d. Then this ith index will be our ".memoryTypeIndex".
		  If found, break out of the loop.
	   e. If not continue the loop by right shifting VkMemoryRequirements.memoryTypeBits by 1, over each iteration.
	*/
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryAllocateInfo.html
	/*
	// Provided by VK_VERSION_1_0
	typedef struct VkMemoryAllocateInfo {
		VkStructureType    sType;
		const void*        pNext;
		VkDeviceSize       allocationSize;
		uint32_t           memoryTypeIndex;
	} VkMemoryAllocateInfo;
	*/
	VkMemoryAllocateInfo vkMemoryAllocateInfo;
	memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
	vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	vkMemoryAllocateInfo.pNext = &vkExportMemoryAllocateInfo; //Chaining 3
	vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceSize.html (vkMemoryRequirements allocates memory in regions.)
	
	/*
	   22.8. To actually allocate the required memory, we need to call vkAllocateMemory().
	   But before that we need to declare and memset VkMemoryAllocateInfo structure.
	   Important members of this structure are ".memoryTypeIndex" and ".allocationSize".
	   For ".allocationSize", use the size obtained from vkGetBufferMemoryRequirements().
	   For ".memoryTypeIndex" : 
	   a. Start a loop with count as vkPhysicalDeviceMemoryProperties.memoryTypeCount.
	   b. Inside the loop check vkMemoryRequiremts.memoryTypeBits contain 1 or not.
	   c. If yes, Check vkPhysicalDeviceMemoryProperties.memeoryTypes[i].propertyFlags member contains enum VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.
	   d. Then this ith index will be our ".memoryTypeIndex".
		  If found, break out of the loop.
	   e. If not continue the loop by right shifting VkMemoryRequirements.memoryTypeBits by 1, over each iteration.
	*/
	vkMemoryAllocateInfo.memoryTypeIndex = 0; //Initial value before entering into the loop
	for(uint32_t i =0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++) //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceMemoryProperties.html
	{
		if((vkMemoryRequirements.memoryTypeBits & 1) == 1) //https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryRequirements.html
		{
			//https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryType.html
			//https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryPropertyFlagBits.html
			if(vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
			{
				vkMemoryAllocateInfo.memoryTypeIndex = i;
				break;
			}			
		}
		vkMemoryRequirements.memoryTypeBits >>= 1;
	}
	
	/*
	22.9. Now call vkAllocateMemory()  and get the required Vulkan memory objects handle into the ".vkDeviceMemory" member of put global structure.
	// Provided by VK_VERSION_1_0
	VkResult vkAllocateMemory(
    VkDevice                                    device,
    const VkMemoryAllocateInfo*                 pAllocateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDeviceMemory*                             pMemory);
	*/
	vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vertexdata_position.vkDeviceMemory); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkAllocateMemory.html
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateExternalVertexBuffer(): vkAllocateMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateExternalVertexBuffer(): vkAllocateMemory() succedded\n");
	}
	
	/*
	22.10. Now we have our required deviceMemory handle as well as VkBuffer Handle.
	Bind this device memory handle to VkBuffer Handle by using vkBindBufferMemory().
	Declare a void* buffer say "data" and call vkMapMemory() to map our device memory object handle to this void* buffer data.
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkBindBufferMemory.html
	// Provided by VK_VERSION_1_0
	VkResult vkBindBufferMemory(
    VkDevice                                    device,
    VkBuffer                                    buffer, //whom to bind
    VkDeviceMemory                              memory, //what to bind
    VkDeviceSize                                memoryOffset);
	*/
	vkResult = vkBindBufferMemory(vkDevice, vertexdata_position.vkBuffer, vertexdata_position.vkDeviceMemory, 0); // We are binding device memory object handle with Vulkan buffer object handle. 
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateExternalVertexBuffer(): vkBindBufferMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateExternalVertexBuffer(): vkBindBufferMemory() succedded\n");
	}
	
	//Completely new code
	HANDLE hMemoryWin32Handle = NULL;

	//https://docs.vulkan.org/refpages/latest/refpages/source/VkMemoryGetWin32HandleInfoKHR.html
	/*
	// Provided by VK_KHR_external_memory_win32
	typedef struct VkMemoryGetWin32HandleInfoKHR {
    VkStructureType                       sType;
    const void*                           pNext;
    VkDeviceMemory                        memory;
    VkExternalMemoryHandleTypeFlagBits    handleType;
	} VkMemoryGetWin32HandleInfoKHR;
	*/
	VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR;
	memset((void*)&vkMemoryGetWin32HandleInfoKHR, 0, sizeof(VkMemoryGetWin32HandleInfoKHR));
	vkMemoryGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
	vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
	vkMemoryGetWin32HandleInfoKHR.memory = vertexdata_position.vkDeviceMemory;
	vkMemoryGetWin32HandleInfoKHR.handleType = vkExternalMemoryHandleTypeFlagBits;
	
	//https://docs.vulkan.org/refpages/latest/refpages/source/vkGetMemoryWin32HandleKHR.html
	//https://docs.vulkan.org/refpages/latest/refpages/source/vkGetDeviceProcAddr.html
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
	
	//Call above function pointer
	// https://docs.vulkan.org/refpages/latest/refpages/source/vkGetMemoryWin32HandleKHR.html
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
	
	//Import above external buffer's memory (vkDeviceMemory) into CUDA
	cudaExternalMemoryHandleDesc cuExternalMemoryHandleDesc;
	memset((void*)&cuExternalMemoryHandleDesc, 0, sizeof(cudaExternalMemoryHandleDesc));
	cuExternalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32; //This is CUDA's inbuilt constant for Windows 8.1 or greater
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
	
	//Close tha handle as its job is done
	CloseHandle(hMemoryWin32Handle);
	hMemoryWin32Handle = NULL;
	
	//Use above external imported memory to get mapped device pointer from CUDA
	//As we have buffer, we need mapped pointer to buffer
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

// ============================================================================
// CreateMultipleExternalVertexBuffers - Creates multiple buffers for ping-pong
// ============================================================================
// This function creates NUM_VERTEX_BUFFERS external vertex buffers for
// double/triple buffering. Each buffer can be used independently for
// CUDA-Vulkan interop, allowing overlapped compute and rendering.
//
// Double Buffering (NUM_VERTEX_BUFFERS = 2):
// - Frame N: Render from buffer 0, compute into buffer 1
// - Frame N+1: Render from buffer 1, compute into buffer 0
//
// Triple Buffering (NUM_VERTEX_BUFFERS = 3):
// - More fluid frame pacing, better GPU utilization
// - One buffer always ready, one rendering, one computing
// ============================================================================
VkResult CreateMultipleExternalVertexBuffers(unsigned int mesh_width, unsigned int mesh_height)
{
	// Variable declarations
	VkResult vkResult = VK_SUCCESS;
	unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);

	FileIO("CreateMultipleExternalVertexBuffers(): Creating %d vertex buffers for %s buffering\n",
	       NUM_VERTEX_BUFFERS, NUM_VERTEX_BUFFERS == 2 ? "double" : "triple");

	// Create each buffer in the ping-pong set
	for(int bufferIndex = 0; bufferIndex < NUM_VERTEX_BUFFERS; bufferIndex++)
	{
		FileIO("CreateMultipleExternalVertexBuffers(): Creating buffer %d of %d\n",
		       bufferIndex + 1, NUM_VERTEX_BUFFERS);

		// Initialize the vertex data structure
		memset((void*)&vertexData_buffers[bufferIndex], 0, sizeof(VertexData));

		// Initialize external memory buffer info for CUDA interop
		VkExternalMemoryBufferCreateInfo vkExternalMemoryBufferCreateInfo;
		memset((void*)&vkExternalMemoryBufferCreateInfo, 0, sizeof(VkExternalMemoryBufferCreateInfo));
		vkExternalMemoryBufferCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
		vkExternalMemoryBufferCreateInfo.pNext = NULL;
		vkExternalMemoryBufferCreateInfo.handleTypes = vkExternalMemoryHandleTypeFlagBits;

		// Create Vulkan buffer
		VkBufferCreateInfo vkBufferCreateInfo;
		memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
		vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		vkBufferCreateInfo.pNext = &vkExternalMemoryBufferCreateInfo;
		vkBufferCreateInfo.flags = 0;
		vkBufferCreateInfo.size = size;
		vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

		vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL,
		                          &vertexData_buffers[bufferIndex].vkBuffer);
		if(vkResult != VK_SUCCESS)
		{
			FileIO("CreateMultipleExternalVertexBuffers(): vkCreateBuffer() failed for buffer %d\n", bufferIndex);
			return vkResult;
		}

		// Get memory requirements
		VkMemoryRequirements vkMemoryRequirements;
		memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
		vkGetBufferMemoryRequirements(vkDevice, vertexData_buffers[bufferIndex].vkBuffer, &vkMemoryRequirements);

		// Set up export memory allocation
		VkExportMemoryAllocateInfo vkExportMemoryAllocateInfo;
		memset((void*)&vkExportMemoryAllocateInfo, 0, sizeof(VkExportMemoryAllocateInfo));
		vkExportMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
		vkExportMemoryAllocateInfo.pNext = NULL;
		vkExportMemoryAllocateInfo.handleTypes = vkExternalMemoryHandleTypeFlagBits;

		// Memory allocation info
		VkMemoryAllocateInfo vkMemoryAllocateInfo;
		memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
		vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		vkMemoryAllocateInfo.pNext = &vkExportMemoryAllocateInfo;
		vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;

		// Find suitable device local memory type
		vkMemoryAllocateInfo.memoryTypeIndex = 0;
		uint32_t memTypeBits = vkMemoryRequirements.memoryTypeBits;
		for(uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
		{
			if((memTypeBits & 1) == 1)
			{
				if(vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
				{
					vkMemoryAllocateInfo.memoryTypeIndex = i;
					break;
				}
			}
			memTypeBits >>= 1;
		}

		// Allocate memory
		vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL,
		                            &vertexData_buffers[bufferIndex].vkDeviceMemory);
		if(vkResult != VK_SUCCESS)
		{
			FileIO("CreateMultipleExternalVertexBuffers(): vkAllocateMemory() failed for buffer %d\n", bufferIndex);
			return vkResult;
		}

		// Bind buffer to memory
		vkResult = vkBindBufferMemory(vkDevice, vertexData_buffers[bufferIndex].vkBuffer,
		                              vertexData_buffers[bufferIndex].vkDeviceMemory, 0);
		if(vkResult != VK_SUCCESS)
		{
			FileIO("CreateMultipleExternalVertexBuffers(): vkBindBufferMemory() failed for buffer %d\n", bufferIndex);
			return vkResult;
		}

		// Get Win32 handle for CUDA interop
		HANDLE hMemoryWin32Handle = NULL;

		VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR;
		memset((void*)&vkMemoryGetWin32HandleInfoKHR, 0, sizeof(VkMemoryGetWin32HandleInfoKHR));
		vkMemoryGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
		vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
		vkMemoryGetWin32HandleInfoKHR.memory = vertexData_buffers[bufferIndex].vkDeviceMemory;
		vkMemoryGetWin32HandleInfoKHR.handleType = vkExternalMemoryHandleTypeFlagBits;

		PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR =
			(PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(vkDevice, "vkGetMemoryWin32HandleKHR");
		if(vkGetMemoryWin32HandleKHR == NULL)
		{
			FileIO("CreateMultipleExternalVertexBuffers(): vkGetMemoryWin32HandleKHR api not obtained\n");
			return VK_ERROR_INITIALIZATION_FAILED;
		}

		vkResult = vkGetMemoryWin32HandleKHR(vkDevice, &vkMemoryGetWin32HandleInfoKHR, &hMemoryWin32Handle);
		if(vkResult != VK_SUCCESS)
		{
			FileIO("CreateMultipleExternalVertexBuffers(): vkGetMemoryWin32HandleKHR() failed for buffer %d\n", bufferIndex);
			return vkResult;
		}

		// Import external memory into CUDA
		cudaExternalMemoryHandleDesc cuExternalMemoryHandleDesc;
		memset((void*)&cuExternalMemoryHandleDesc, 0, sizeof(cudaExternalMemoryHandleDesc));
		cuExternalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
		cuExternalMemoryHandleDesc.handle.win32.handle = hMemoryWin32Handle;
		cuExternalMemoryHandleDesc.size = size;
		cuExternalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;

		cudaResult = cudaImportExternalMemory(&cuExternalMemory_buffers[bufferIndex], &cuExternalMemoryHandleDesc);
		if(cudaResult != cudaSuccess)
		{
			FileIO("CreateMultipleExternalVertexBuffers(): cudaImportExternalMemory() failed for buffer %d: %s\n",
			       bufferIndex, cudaGetErrorString(cudaResult));
			CloseHandle(hMemoryWin32Handle);
			return VK_ERROR_INITIALIZATION_FAILED;
		}

		// Close Win32 handle after import
		CloseHandle(hMemoryWin32Handle);

		// Get CUDA mapped buffer pointer
		cudaExternalMemoryBufferDesc cuExternalMemoryBufferDesc;
		memset((void*)&cuExternalMemoryBufferDesc, 0, sizeof(cudaExternalMemoryBufferDesc));
		cuExternalMemoryBufferDesc.offset = 0;
		cuExternalMemoryBufferDesc.size = size;
		cuExternalMemoryBufferDesc.flags = 0;

		cudaResult = cudaExternalMemoryGetMappedBuffer(&pos_CUDA_buffers[bufferIndex],
		                                               cuExternalMemory_buffers[bufferIndex],
		                                               &cuExternalMemoryBufferDesc);
		if(cudaResult != cudaSuccess)
		{
			FileIO("CreateMultipleExternalVertexBuffers(): cudaExternalMemoryGetMappedBuffer() failed for buffer %d: %s\n",
			       bufferIndex, cudaGetErrorString(cudaResult));
			return VK_ERROR_INITIALIZATION_FAILED;
		}

		FileIO("CreateMultipleExternalVertexBuffers(): Buffer %d created successfully\n", bufferIndex);
	}

	// Initialize buffer indices for ping-pong
	computeBufferIndex = 0;
	renderBufferIndex = (NUM_VERTEX_BUFFERS > 1) ? 1 : 0;

	// Set legacy pointers to first buffer for backward compatibility
	vertexData_external = vertexData_buffers[0];
	pos_CUDA = pos_CUDA_buffers[0];
	cuExternalMemory_t = cuExternalMemory_buffers[0];

	FileIO("CreateMultipleExternalVertexBuffers(): All %d buffers created successfully\n", NUM_VERTEX_BUFFERS);
	FileIO("CreateMultipleExternalVertexBuffers(): Initial compute buffer: %d, render buffer: %d\n",
	       computeBufferIndex, renderBufferIndex);

	return vkResult;
}

VkResult CreateIndirectBuffer(void)
{
	VkResult UpdateIndirectBuffer(void);

	//Variable declarations
	VkResult vkResult = VK_SUCCESS;

	/*
	Code
	*/

	/*
	memset our global vertexdata_indirect_buffer.
	*/
	memset((void*)&vertexdata_indirect_buffer, 0, sizeof(VertexData));

	/*
	Declare and memset VkBufferCreateInfo struct.
	*/
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferCreateInfo.html
	VkBufferCreateInfo vkBufferCreateInfo;
	memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));

	vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vkBufferCreateInfo.pNext = NULL;
	vkBufferCreateInfo.flags = 0;
	vkBufferCreateInfo.size = sizeof(VkDrawIndirectCommand); //https://registry.khronos.org/VulkanSC/specs/1.0-extensions/man/html/VkDrawIndirectCommand.html
	vkBufferCreateInfo.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferUsageFlagBits.html

	/*
	Call vkCreateBuffer() vulkan API
	*/
	vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexdata_indirect_buffer.vkBuffer); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateBuffer.html
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateIndirectBuffer(): vkCreateBuffer() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateIndirectBuffer(): vkCreateBuffer() succedded\n");
	}

	/*
	Get memory requirements
	*/
	VkMemoryRequirements vkMemoryRequirements;
	memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));

	vkGetBufferMemoryRequirements(vkDevice, vertexdata_indirect_buffer.vkBuffer, &vkMemoryRequirements);

	/*
	Allocate memory
	*/
	VkMemoryAllocateInfo vkMemoryAllocateInfo;
	memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
	vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	vkMemoryAllocateInfo.pNext = NULL;
	vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;

	/*
	Find suitable memory type index
	First try HOST_VISIBLE + HOST_COHERENT, then fall back to HOST_VISIBLE only
	If using non-coherent memory, we must call vkFlushMappedMemoryRanges() after writes
	*/
	BOOL bFoundMemoryType = FALSE;
	uint32_t memoryTypeBits = vkMemoryRequirements.memoryTypeBits;

	// First pass: try to find HOST_VISIBLE + HOST_COHERENT
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

	// Second pass: fall back to HOST_VISIBLE only if no coherent type found
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

	// Error if no suitable memory type found
	if(!bFoundMemoryType)
	{
		FileIO("CreateIndirectBuffer(): Failed to find suitable HOST_VISIBLE memory type\n");
		return VK_ERROR_FEATURE_NOT_PRESENT;
	}

	/*
	Call vkAllocateMemory()
	*/
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

	/*
	Bind buffer memory
	*/
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

	//Update indirect buffer
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
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;

	//https://registry.khronos.org/VulkanSC/specs/1.0-extensions/man/html/VkDrawIndirectCommand.html
	/*
	// Provided by VK_VERSION_1_0
	typedef struct VkDrawIndirectCommand {
		uint32_t    vertexCount;
		uint32_t    instanceCount;
		uint32_t    firstVertex;
		uint32_t    firstInstance;
	} VkDrawIndirectCommand;
	*/
	VkDrawIndirectCommand vkDrawIndirectCommand;
	memset((void*)&vkDrawIndirectCommand, 0, sizeof(VkDrawIndirectCommand));
	vkDrawIndirectCommand.vertexCount = PATCH_VERTEX_COUNT; // Triangle patches: 63*63*2*3 = 23814 vertices
	vkDrawIndirectCommand.instanceCount = 1;
	vkDrawIndirectCommand.firstVertex = 0;
	vkDrawIndirectCommand.firstInstance = 0;

	/*
	Map memory and copy VkDrawIndirectCommand
	*/
	void* data = NULL;
	vkResult = vkMapMemory(vkDevice, vertexdata_indirect_buffer.vkDeviceMemory, 0, VK_WHOLE_SIZE, 0, &data);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("UpdateIndirectBuffer(): vkMapMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}

	memcpy(data, &vkDrawIndirectCommand, sizeof(VkDrawIndirectCommand));

	// Flush memory if not HOST_COHERENT to ensure GPU sees the updated data
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

//31.11
VkResult CreateUniformBuffer()
{
	//Function Declaration
	VkResult UpdateUniformBuffer(void);
	
	//Variable declarations	
	VkResult vkResult = VK_SUCCESS;
	
	//Code
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferCreateInfo.html
	VkBufferCreateInfo vkBufferCreateInfo;
	memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
	
	/*
	// Provided by VK_VERSION_1_0
	typedef struct VkBufferCreateInfo {
		VkStructureType        sType;
		const void*            pNext;
		VkBufferCreateFlags    flags;
		VkDeviceSize           size;
		VkBufferUsageFlags     usage;
		VkSharingMode          sharingMode;
		uint32_t               queueFamilyIndexCount;
		const uint32_t*        pQueueFamilyIndices;
	} VkBufferCreateInfo;
	*/
	vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vkBufferCreateInfo.pNext = NULL;
	vkBufferCreateInfo.flags = 0; //Valid flags are used in scattered(sparse) buffer
	vkBufferCreateInfo.size = sizeof(struct MyUniformData);
	vkBufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferUsageFlagBits.html;
	/* //when one buffer shared in multiple queque's
	vkBufferCreateInfo.sharingMode =;
	vkBufferCreateInfo.queueFamilyIndexCount =;
	vkBufferCreateInfo.pQueueFamilyIndices =; 
	*/
	
	memset((void*)&uniformData, 0, sizeof(struct UniformData));
	
	/*
	// Provided by VK_VERSION_1_0
	VkResult vkCreateBuffer(
    VkDevice                                    device,
    const VkBufferCreateInfo*                   pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkBuffer*                                   pBuffer);
	*/
	vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &uniformData.vkBuffer); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateBuffer.html
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateUniformBuffer(): vkCreateBuffer() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateUniformBuffer(): vkCreateBuffer() succedded\n");
	}
	
	/*
	// Provided by VK_VERSION_1_0
	typedef struct VkMemoryRequirements {
		VkDeviceSize    size;
		VkDeviceSize    alignment;
		uint32_t        memoryTypeBits;
	} VkMemoryRequirements;
	*/
	VkMemoryRequirements vkMemoryRequirements;
	memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetBufferMemoryRequirements.html
	/*
	// Provided by VK_VERSION_1_0
	void vkGetBufferMemoryRequirements(
    VkDevice                                    device,
    VkBuffer                                    buffer,
    VkMemoryRequirements*                       pMemoryRequirements);
	*/
	vkGetBufferMemoryRequirements(vkDevice, uniformData.vkBuffer, &vkMemoryRequirements);
	
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryAllocateInfo.html
	/*
	// Provided by VK_VERSION_1_0
	typedef struct VkMemoryAllocateInfo {
		VkStructureType    sType;
		const void*        pNext;
		VkDeviceSize       allocationSize;
		uint32_t           memoryTypeIndex;
	} VkMemoryAllocateInfo;
	*/
	VkMemoryAllocateInfo vkMemoryAllocateInfo;
	memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
	vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	vkMemoryAllocateInfo.pNext = NULL;
	vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceSize.html (vkMemoryRequirements allocates memory in regions.)
	
	vkMemoryAllocateInfo.memoryTypeIndex = 0; //Initial value before entering into the loop
	for(uint32_t i =0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++) //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceMemoryProperties.html
	{
		if((vkMemoryRequirements.memoryTypeBits & 1) == 1) //https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryRequirements.html
		{
			//https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryType.html
			//https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryPropertyFlagBits.html
			if(vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
			{
				vkMemoryAllocateInfo.memoryTypeIndex = i;
				break;
			}			
		}
		vkMemoryRequirements.memoryTypeBits >>= 1;
	}
	
	/*
	// Provided by VK_VERSION_1_0
	VkResult vkAllocateMemory(
    VkDevice                                    device,
    const VkMemoryAllocateInfo*                 pAllocateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDeviceMemory*                             pMemory);
	*/
	vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &uniformData.vkDeviceMemory); //https://registry.khronos.org/vulkan/specs/latest/man/html/vkAllocateMemory.html
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateUniformBuffer(): vkAllocateMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateUniformBuffer(): vkAllocateMemory() succedded\n");
	}
	
	/*
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkBindBufferMemory.html
	// Provided by VK_VERSION_1_0
	VkResult vkBindBufferMemory(
    VkDevice                                    device,
    VkBuffer                                    buffer, //whom to bind
    VkDeviceMemory                              memory, //what to bind
    VkDeviceSize                                memoryOffset);
	*/
	vkResult = vkBindBufferMemory(vkDevice, uniformData.vkBuffer, uniformData.vkDeviceMemory, 0); // We are binding device memory object handle with Vulkan buffer object handle. 
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateUniformBuffer(): vkBindBufferMemory() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateUniformBuffer(): vkBindBufferMemory() succedded\n");
	}
	
	//Call updateUniformBuffer() here
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

// ============================================================================
// Create Tessellation Parameters Uniform Buffer
// ============================================================================
VkResult CreateTessUniformBuffer(void)
{
	VkResult vkResult = VK_SUCCESS;

	// Create buffer for tessellation parameters
	VkBufferCreateInfo vkBufferCreateInfo;
	memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
	vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vkBufferCreateInfo.pNext = NULL;
	vkBufferCreateInfo.flags = 0;
	vkBufferCreateInfo.size = sizeof(struct TessellationParams);
	vkBufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	vkBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	vkBufferCreateInfo.queueFamilyIndexCount = 0;
	vkBufferCreateInfo.pQueueFamilyIndices = NULL;

	vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &tessUniformData.vkBuffer);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateTessUniformBuffer(): vkCreateBuffer() failed with error code %d\n", vkResult);
		return vkResult;
	}
	FileIO("CreateTessUniformBuffer(): vkCreateBuffer() succeeded\n");

	// Get memory requirements
	VkMemoryRequirements vkMemoryRequirements;
	vkGetBufferMemoryRequirements(vkDevice, tessUniformData.vkBuffer, &vkMemoryRequirements);

	// Allocate memory
	VkMemoryAllocateInfo vkMemoryAllocateInfo;
	memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
	vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	vkMemoryAllocateInfo.pNext = NULL;
	vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;

	vkMemoryAllocateInfo.memoryTypeIndex = 0;
	for(uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
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

	vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &tessUniformData.vkDeviceMemory);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateTessUniformBuffer(): vkAllocateMemory() failed with error code %d\n", vkResult);
		return vkResult;
	}
	FileIO("CreateTessUniformBuffer(): vkAllocateMemory() succeeded\n");

	// Bind memory to buffer
	vkResult = vkBindBufferMemory(vkDevice, tessUniformData.vkBuffer, tessUniformData.vkDeviceMemory, 0);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateTessUniformBuffer(): vkBindBufferMemory() failed with error code %d\n", vkResult);
		return vkResult;
	}
	FileIO("CreateTessUniformBuffer(): vkBindBufferMemory() succeeded\n");

	// Initialize tessellation parameters for 8km x 8km terrain
	struct TessellationParams tessParams;
	tessParams.cameraPos = glm::vec4(0.0f, 6000.0f, 12000.0f, 1.0f);  // Camera position matching 8km terrain view
	tessParams.minTessLevel = 1.0f;    // Minimum tessellation for far objects
	tessParams.maxTessLevel = 64.0f;   // Maximum tessellation for close objects
	tessParams.minDistance = 500.0f;   // Distance for max tessellation (500m)
	tessParams.maxDistance = 20000.0f; // Distance for min tessellation (20km)

	// Map and copy initial data
	void* pData = NULL;
	vkResult = vkMapMemory(vkDevice, tessUniformData.vkDeviceMemory, 0, sizeof(struct TessellationParams), 0, &pData);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateTessUniformBuffer(): vkMapMemory() failed with error code %d\n", vkResult);
		return vkResult;
	}
	memcpy(pData, &tessParams, sizeof(struct TessellationParams));
	vkUnmapMemory(vkDevice, tessUniformData.vkDeviceMemory);

	FileIO("CreateTessUniformBuffer(): Tessellation uniform buffer created successfully\n");
	return vkResult;
}

// ============================================================================
// Create Height Map Texture for Tessellation
// ============================================================================
VkResult CreateHeightMapTexture(void)
{
	VkResult vkResult = VK_SUCCESS;

	// Create image for height map
	VkImageCreateInfo vkImageCreateInfo;
	memset((void*)&vkImageCreateInfo, 0, sizeof(VkImageCreateInfo));
	vkImageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	vkImageCreateInfo.pNext = NULL;
	vkImageCreateInfo.flags = 0;
	vkImageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
	vkImageCreateInfo.format = VK_FORMAT_R32_SFLOAT;  // Single channel float for height
	vkImageCreateInfo.extent.width = HEIGHTMAP_WIDTH;
	vkImageCreateInfo.extent.height = HEIGHTMAP_HEIGHT;
	vkImageCreateInfo.extent.depth = 1;
	vkImageCreateInfo.mipLevels = 1;
	vkImageCreateInfo.arrayLayers = 1;
	vkImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	vkImageCreateInfo.tiling = VK_IMAGE_TILING_LINEAR;  // Linear for CPU access
	vkImageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	vkImageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	vkImageCreateInfo.queueFamilyIndexCount = 0;
	vkImageCreateInfo.pQueueFamilyIndices = NULL;
	vkImageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;

	vkResult = vkCreateImage(vkDevice, &vkImageCreateInfo, NULL, &vkImage_heightMap);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateHeightMapTexture(): vkCreateImage() failed with error code %d\n", vkResult);
		return vkResult;
	}
	FileIO("CreateHeightMapTexture(): vkCreateImage() succeeded\n");

	// Get memory requirements for the image
	VkMemoryRequirements vkMemoryRequirements;
	vkGetImageMemoryRequirements(vkDevice, vkImage_heightMap, &vkMemoryRequirements);

	// Allocate memory for the image
	VkMemoryAllocateInfo vkMemoryAllocateInfo;
	memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
	vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	vkMemoryAllocateInfo.pNext = NULL;
	vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;

	// Find memory type with HOST_VISIBLE for CPU writes
	vkMemoryAllocateInfo.memoryTypeIndex = 0;
	for(uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if((vkMemoryRequirements.memoryTypeBits & (1 << i)))
		{
			if(vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags &
			   (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
			{
				vkMemoryAllocateInfo.memoryTypeIndex = i;
				break;
			}
		}
	}

	vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vkDeviceMemory_heightMap);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateHeightMapTexture(): vkAllocateMemory() failed with error code %d\n", vkResult);
		return vkResult;
	}
	FileIO("CreateHeightMapTexture(): vkAllocateMemory() succeeded\n");

	// Bind memory to image
	vkResult = vkBindImageMemory(vkDevice, vkImage_heightMap, vkDeviceMemory_heightMap, 0);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateHeightMapTexture(): vkBindImageMemory() failed with error code %d\n", vkResult);
		return vkResult;
	}
	FileIO("CreateHeightMapTexture(): vkBindImageMemory() succeeded\n");

	// Create image view
	VkImageViewCreateInfo vkImageViewCreateInfo;
	memset((void*)&vkImageViewCreateInfo, 0, sizeof(VkImageViewCreateInfo));
	vkImageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	vkImageViewCreateInfo.pNext = NULL;
	vkImageViewCreateInfo.flags = 0;
	vkImageViewCreateInfo.image = vkImage_heightMap;
	vkImageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	vkImageViewCreateInfo.format = VK_FORMAT_R32_SFLOAT;
	vkImageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_R;
	vkImageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_ZERO;
	vkImageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_ZERO;
	vkImageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_ONE;
	vkImageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	vkImageViewCreateInfo.subresourceRange.baseMipLevel = 0;
	vkImageViewCreateInfo.subresourceRange.levelCount = 1;
	vkImageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
	vkImageViewCreateInfo.subresourceRange.layerCount = 1;

	vkResult = vkCreateImageView(vkDevice, &vkImageViewCreateInfo, NULL, &vkImageView_heightMap);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateHeightMapTexture(): vkCreateImageView() failed with error code %d\n", vkResult);
		return vkResult;
	}
	FileIO("CreateHeightMapTexture(): vkCreateImageView() succeeded\n");

	// Create sampler for height map
	VkSamplerCreateInfo vkSamplerCreateInfo;
	memset((void*)&vkSamplerCreateInfo, 0, sizeof(VkSamplerCreateInfo));
	vkSamplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	vkSamplerCreateInfo.pNext = NULL;
	vkSamplerCreateInfo.flags = 0;
	vkSamplerCreateInfo.magFilter = VK_FILTER_LINEAR;
	vkSamplerCreateInfo.minFilter = VK_FILTER_LINEAR;
	vkSamplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	vkSamplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	vkSamplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	vkSamplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	vkSamplerCreateInfo.mipLodBias = 0.0f;
	vkSamplerCreateInfo.anisotropyEnable = VK_FALSE;
	vkSamplerCreateInfo.maxAnisotropy = 1.0f;
	vkSamplerCreateInfo.compareEnable = VK_FALSE;
	vkSamplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
	vkSamplerCreateInfo.minLod = 0.0f;
	vkSamplerCreateInfo.maxLod = 0.0f;
	vkSamplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
	vkSamplerCreateInfo.unnormalizedCoordinates = VK_FALSE;

	vkResult = vkCreateSampler(vkDevice, &vkSamplerCreateInfo, NULL, &vkSampler_heightMap);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateHeightMapTexture(): vkCreateSampler() failed with error code %d\n", vkResult);
		return vkResult;
	}
	FileIO("CreateHeightMapTexture(): vkCreateSampler() succeeded\n");

	// Initialize height map with procedural terrain (fBm - fractal Brownian motion)
	// This creates realistic terrain with mountains and valleys as shown in OpenGL Insights
	void* pData = NULL;
	vkResult = vkMapMemory(vkDevice, vkDeviceMemory_heightMap, 0, HEIGHTMAP_WIDTH * HEIGHTMAP_HEIGHT * sizeof(float), 0, &pData);
	if (vkResult == VK_SUCCESS)
	{
		float* heightData = (float*)pData;

		// Noise function using simple value noise with smooth interpolation
		// Based on techniques from OpenGL Insights Chapter %.2
		auto smoothstep = [](float t) -> float {
			return t * t * (3.0f - 2.0f * t);
		};

		auto lerp = [](float a, float b, float t) -> float {
			return a + t * (b - a);
		};

		// Simple hash function for pseudo-random values
		auto hash = [](int x, int y) -> float {
			int n = x + y * 57;
			n = (n << 13) ^ n;
			return (1.0f - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f);
		};

		// Value noise function with smooth interpolation
		auto noise = [&](float x, float y) -> float {
			int ix = (int)floorf(x);
			int iy = (int)floorf(y);
			float fx = x - ix;
			float fy = y - iy;

			// Smooth interpolation
			float sx = smoothstep(fx);
			float sy = smoothstep(fy);

			// Four corners
			float n00 = hash(ix, iy);
			float n10 = hash(ix + 1, iy);
			float n01 = hash(ix, iy + 1);
			float n11 = hash(ix + 1, iy + 1);

			// Bilinear interpolation
			float nx0 = lerp(n00, n10, sx);
			float nx1 = lerp(n01, n11, sx);
			return lerp(nx0, nx1, sy);
		};

		// fBm (fractal Brownian motion) for realistic terrain
		auto fbm = [&](float x, float y, int octaves, float persistence, float lacunarity) -> float {
			float total = 0.0f;
			float amplitude = 1.0f;
			float frequency = 1.0f;
			float maxValue = 0.0f;

			for (int i = 0; i < octaves; i++)
			{
				total += noise(x * frequency, y * frequency) * amplitude;
				maxValue += amplitude;
				amplitude *= persistence;
				frequency *= lacunarity;
			}

			return total / maxValue; // Normalize to [-1, 1]
		};

		// Generate terrain heightmap
		const float terrainScale = 8.0f;  // Controls terrain feature size
		const int octaves = 6;            // Number of noise layers
		const float persistence = 0.5f;   // Amplitude decay per octave
		const float lacunarity = 2.0f;    // Frequency increase per octave
		const float heightScale = 0.5f;   // Maximum terrain height

		for (unsigned int y = 0; y < HEIGHTMAP_HEIGHT; y++)
		{
			for (unsigned int x = 0; x < HEIGHTMAP_WIDTH; x++)
			{
				// Compute terrain coordinates
				float tx = (float)x / (float)(HEIGHTMAP_WIDTH - 1) * terrainScale;
				float ty = (float)y / (float)(HEIGHTMAP_HEIGHT - 1) * terrainScale;

				// Generate height using fBm
				float heightVal = fbm(tx, ty, octaves, persistence, lacunarity) * heightScale;

				heightData[y * HEIGHTMAP_WIDTH + x] = heightVal;
			}
		}
		FileIO("CreateHeightMapTexture(): Initialized height map with fBm terrain (OpenGL Insights style)\n");
		vkUnmapMemory(vkDevice, vkDeviceMemory_heightMap);
	}

	// Transition image layout from PREINITIALIZED to SHADER_READ_ONLY_OPTIMAL
	// Create a temporary command buffer for the layout transition
	VkCommandBufferAllocateInfo vkCommandBufferAllocateInfo;
	memset(&vkCommandBufferAllocateInfo, 0, sizeof(VkCommandBufferAllocateInfo));
	vkCommandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	vkCommandBufferAllocateInfo.pNext = NULL;
	vkCommandBufferAllocateInfo.commandPool = vkCommandPool;
	vkCommandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	vkCommandBufferAllocateInfo.commandBufferCount = 1;

	VkCommandBuffer vkCommandBuffer_transition;
	vkResult = vkAllocateCommandBuffers(vkDevice, &vkCommandBufferAllocateInfo, &vkCommandBuffer_transition);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateHeightMapTexture(): Failed to allocate command buffer for layout transition\n");
		return vkResult;
	}

	// Begin command buffer
	VkCommandBufferBeginInfo vkCommandBufferBeginInfo;
	memset(&vkCommandBufferBeginInfo, 0, sizeof(VkCommandBufferBeginInfo));
	vkCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	vkCommandBufferBeginInfo.pNext = NULL;
	vkCommandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkCommandBufferBeginInfo.pInheritanceInfo = NULL;

	vkResult = vkBeginCommandBuffer(vkCommandBuffer_transition, &vkCommandBufferBeginInfo);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateHeightMapTexture(): Failed to begin command buffer for layout transition\n");
		vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition);
		return vkResult;
	}

	// Create image memory barrier for layout transition
	VkImageMemoryBarrier vkImageMemoryBarrier;
	memset(&vkImageMemoryBarrier, 0, sizeof(VkImageMemoryBarrier));
	vkImageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	vkImageMemoryBarrier.pNext = NULL;
	vkImageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
	vkImageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	vkImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
	vkImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	vkImageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	vkImageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	vkImageMemoryBarrier.image = vkImage_heightMap;
	vkImageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	vkImageMemoryBarrier.subresourceRange.baseMipLevel = 0;
	vkImageMemoryBarrier.subresourceRange.levelCount = 1;
	vkImageMemoryBarrier.subresourceRange.baseArrayLayer = 0;
	vkImageMemoryBarrier.subresourceRange.layerCount = 1;

	vkCmdPipelineBarrier(
		vkCommandBuffer_transition,
		VK_PIPELINE_STAGE_HOST_BIT,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		0,
		0, NULL,
		0, NULL,
		1, &vkImageMemoryBarrier
	);

	vkResult = vkEndCommandBuffer(vkCommandBuffer_transition);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateHeightMapTexture(): Failed to end command buffer for layout transition\n");
		vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition);
		return vkResult;
	}

	// Submit command buffer
	VkSubmitInfo vkSubmitInfo;
	memset(&vkSubmitInfo, 0, sizeof(VkSubmitInfo));
	vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	vkSubmitInfo.pNext = NULL;
	vkSubmitInfo.waitSemaphoreCount = 0;
	vkSubmitInfo.pWaitSemaphores = NULL;
	vkSubmitInfo.pWaitDstStageMask = NULL;
	vkSubmitInfo.commandBufferCount = 1;
	vkSubmitInfo.pCommandBuffers = &vkCommandBuffer_transition;
	vkSubmitInfo.signalSemaphoreCount = 0;
	vkSubmitInfo.pSignalSemaphores = NULL;

	vkResult = vkQueueSubmit(vkQueue, 1, &vkSubmitInfo, VK_NULL_HANDLE);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateHeightMapTexture(): Failed to submit layout transition command buffer\n");
		vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition);
		return vkResult;
	}

	// Wait for the queue to complete
	vkQueueWaitIdle(vkQueue);

	// Free the temporary command buffer
	vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition);

	FileIO("CreateHeightMapTexture(): Image layout transitioned to SHADER_READ_ONLY_OPTIMAL\n");
	FileIO("CreateHeightMapTexture(): Height map texture created successfully\n");
	return vkResult;
}

/*
23.5. Maintaining the same baove convention while defining CreateShaders() between definition of above two.
*/
VkResult CreateShaders(void)
{
	//Variable declarations	
	VkResult vkResult = VK_SUCCESS;
	
	/*
	Code for Vertex Shader
	*/
	
	/*
	6. Inside our function, 
	first open shader file, 
	set the file pointer at end of file,
	find the byte size of shader file data,
	reset the file pointer at begining of the file,
	allocate a character buffer of file size and read Shader file data into it,
	and finally close the file.
	Do all these things using conventional fileIO.
	*/
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
	
	/*
	23.7. Declare and memset struct VkShaderModuleCreateInfo and specify above file size and buffer while initializing it.
	// Provided by VK_VERSION_1_0
	typedef struct VkShaderModuleCreateInfo {
		VkStructureType              sType;
		const void*                  pNext;
		VkShaderModuleCreateFlags    flags;
		size_t                       codeSize;
		const uint32_t*              pCode;
	} VkShaderModuleCreateInfo;
	*/
	VkShaderModuleCreateInfo vkShaderModuleCreateInfo; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkShaderModuleCreateInfo.html
	memset((void*)&vkShaderModuleCreateInfo, 0, sizeof(VkShaderModuleCreateInfo));
	vkShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	vkShaderModuleCreateInfo.pNext = NULL;
	vkShaderModuleCreateInfo.flags = 0; //Reserved for future use. Hence must be 0
	vkShaderModuleCreateInfo.codeSize = size;
	vkShaderModuleCreateInfo.pCode = (uint32_t*)shaderData;
	
	/*
	8. Call vkCreateShaderModule() Vulkan API, pass above struct's pointer to it as parameter and obtain shader module object in global variable, that we declared in Step 2.
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateShaderModule.html
	// Provided by VK_VERSION_1_0
	VkResult vkCreateShaderModule(
    VkDevice                                    device,
    const VkShaderModuleCreateInfo*             pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkShaderModule*                             pShaderModule);
	*/
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
	
	/*
	9. Free the ShaderCode buffer which we allocated in Step 6.
	*/
	if(shaderData)
	{
		free(shaderData);
		shaderData = NULL;
	}
	FileIO("CreateShaders(): vertex Shader module successfully created\n");
	
	/*
	23.10. Assuming we did above 4 steps 6 to 9 for Vertex Shader, Repeat them all for fragment shader too.
	Code for Fragment Shader
	*/
	
	/*
	6. Inside our function, 
	first open shader file, 
	set the file pointer at end of file,
	find the byte size of shader file data,
	reset the file pointer at begining of the file,
	allocate a character buffer of file size and read Shader file data into it,
	and finally close the file.
	Do all these things using conventional fileIO.
	*/
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
	
	/*
	23.7. Declare and memset struct VkShaderModuleCreateInfo and specify above file size and buffer while initializing it.
	// Provided by VK_VERSION_1_0
	typedef struct VkShaderModuleCreateInfo {
		VkStructureType              sType;
		const void*                  pNext;
		VkShaderModuleCreateFlags    flags;
		size_t                       codeSize;
		const uint32_t*              pCode;
	} VkShaderModuleCreateInfo;
	*/
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkShaderModuleCreateInfo.html
	memset((void*)&vkShaderModuleCreateInfo, 0, sizeof(VkShaderModuleCreateInfo));
	vkShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	vkShaderModuleCreateInfo.pNext = NULL;
	vkShaderModuleCreateInfo.flags = 0; //Reserved for future use. Hence must be 0
	vkShaderModuleCreateInfo.codeSize = size;
	vkShaderModuleCreateInfo.pCode = (uint32_t*)shaderData;
	
	/*
	8. Call vkCreateShaderModule() Vulkan API, pass above struct's pointer to it as parameter and obtain shader module object in global variable, that we declared in Step 2.
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateShaderModule.html
	// Provided by VK_VERSION_1_0
	VkResult vkCreateShaderModule(
    VkDevice                                    device,
    const VkShaderModuleCreateInfo*             pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkShaderModule*                             pShaderModule);
	*/
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
	
	/*
	9. Free the ShaderCode buffer which we allocated in Step 6.
	*/
	if(shaderData)
	{
		free(shaderData);
		shaderData = NULL;
	}
	FileIO("CreateShaders(): fragment Shader module successfully created\n");

	// ================================================================
	// Load Tessellation Control Shader (TCS)
	// ================================================================
	szFileName = "Shader.tesc.spv";
	size = 0;
	fp = NULL;

	fp = fopen(szFileName, "rb");
	if(fp == NULL)
	{
		FileIO("CreateShaders(): failed to open TCS SPIRV file\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): succeeded to open TCS SPIRV file\n");
	}

	fseek(fp, 0L, SEEK_END);
	size = ftell(fp);
	if(size == 0)
	{
		FileIO("CreateShaders(): TCS SPIRV file size is 0\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}

	fseek(fp, 0L, SEEK_SET);

	shaderData = (char*)malloc(sizeof(char) * size);
	if(shaderData == NULL)
	{
		FileIO("CreateShaders(): malloc for TCS SPIRV file failed\n");
	}
	else
	{
		FileIO("CreateShaders(): malloc for TCS SPIRV file done\n");
	}

	retVal = fread(shaderData, size, 1, fp);
	if(retVal != 1)
	{
		FileIO("CreateShaders(): failed to read TCS SPIRV file\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): succeeded to read TCS SPIRV file\n");
	}

	fclose(fp);

	memset((void*)&vkShaderModuleCreateInfo, 0, sizeof(VkShaderModuleCreateInfo));
	vkShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	vkShaderModuleCreateInfo.pNext = NULL;
	vkShaderModuleCreateInfo.flags = 0;
	vkShaderModuleCreateInfo.codeSize = size;
	vkShaderModuleCreateInfo.pCode = (uint32_t*)shaderData;

	vkResult = vkCreateShaderModule(vkDevice, &vkShaderModuleCreateInfo, NULL, &vkShaderModule_tesc);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateShaders(): vkCreateShaderModule() for TCS failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): vkCreateShaderModule() for TCS succeeded\n");
	}

	if(shaderData)
	{
		free(shaderData);
		shaderData = NULL;
	}
	FileIO("CreateShaders(): TCS Shader module successfully created\n");

	// ================================================================
	// Load Tessellation Evaluation Shader (TES)
	// ================================================================
	szFileName = "Shader.tese.spv";
	size = 0;
	fp = NULL;

	fp = fopen(szFileName, "rb");
	if(fp == NULL)
	{
		FileIO("CreateShaders(): failed to open TES SPIRV file\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): succeeded to open TES SPIRV file\n");
	}

	fseek(fp, 0L, SEEK_END);
	size = ftell(fp);
	if(size == 0)
	{
		FileIO("CreateShaders(): TES SPIRV file size is 0\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}

	fseek(fp, 0L, SEEK_SET);

	shaderData = (char*)malloc(sizeof(char) * size);
	if(shaderData == NULL)
	{
		FileIO("CreateShaders(): malloc for TES SPIRV file failed\n");
	}
	else
	{
		FileIO("CreateShaders(): malloc for TES SPIRV file done\n");
	}

	retVal = fread(shaderData, size, 1, fp);
	if(retVal != 1)
	{
		FileIO("CreateShaders(): failed to read TES SPIRV file\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): succeeded to read TES SPIRV file\n");
	}

	fclose(fp);

	memset((void*)&vkShaderModuleCreateInfo, 0, sizeof(VkShaderModuleCreateInfo));
	vkShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	vkShaderModuleCreateInfo.pNext = NULL;
	vkShaderModuleCreateInfo.flags = 0;
	vkShaderModuleCreateInfo.codeSize = size;
	vkShaderModuleCreateInfo.pCode = (uint32_t*)shaderData;

	vkResult = vkCreateShaderModule(vkDevice, &vkShaderModuleCreateInfo, NULL, &vkShaderModule_tese);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateShaders(): vkCreateShaderModule() for TES failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateShaders(): vkCreateShaderModule() for TES succeeded\n");
	}

	if(shaderData)
	{
		free(shaderData);
		shaderData = NULL;
	}
	FileIO("CreateShaders(): TES Shader module successfully created\n");

	return vkResult;
}

/*
24.2. In initialize(), declare and call UDF CreateDescriptorSetLayout() maintaining the convention of declaring and calling it after CreateShaders() and before CreateRenderPass().
*/
VkResult CreateDescriptorSetLayout()
{
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;

	/*
	Code
	*/

	// Array of descriptor set layout bindings for tessellation pipeline:
	// Binding 0: MVP uniform buffer (vertex, TCS, TES, fragment stages)
	// Binding 1: Tessellation parameters uniform buffer (TCS, fragment stages)
	// Binding 2: Height map sampler (TES stage)
	VkDescriptorSetLayoutBinding vkDescriptorSetLayoutBinding_array[3];
	memset((void*)vkDescriptorSetLayoutBinding_array, 0, sizeof(VkDescriptorSetLayoutBinding) * 3);

	// Binding 0: MVP Matrix uniform buffer
	vkDescriptorSetLayoutBinding_array[0].binding = 0;
	vkDescriptorSetLayoutBinding_array[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	vkDescriptorSetLayoutBinding_array[0].descriptorCount = 1;
	vkDescriptorSetLayoutBinding_array[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
	                                                    VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
	                                                    VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
	                                                    VK_SHADER_STAGE_FRAGMENT_BIT;
	vkDescriptorSetLayoutBinding_array[0].pImmutableSamplers = NULL;

	// Binding 1: Tessellation parameters uniform buffer
	vkDescriptorSetLayoutBinding_array[1].binding = 1;
	vkDescriptorSetLayoutBinding_array[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	vkDescriptorSetLayoutBinding_array[1].descriptorCount = 1;
	vkDescriptorSetLayoutBinding_array[1].stageFlags = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
	                                                    VK_SHADER_STAGE_FRAGMENT_BIT;
	vkDescriptorSetLayoutBinding_array[1].pImmutableSamplers = NULL;

	// Binding 2: Height map combined image sampler
	vkDescriptorSetLayoutBinding_array[2].binding = 2;
	vkDescriptorSetLayoutBinding_array[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	vkDescriptorSetLayoutBinding_array[2].descriptorCount = 1;
	vkDescriptorSetLayoutBinding_array[2].stageFlags = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
	vkDescriptorSetLayoutBinding_array[2].pImmutableSamplers = NULL;

	/*
	24.3. Create descriptor set layout
	*/
	VkDescriptorSetLayoutCreateInfo vkDescriptorSetLayoutCreateInfo;
	memset((void*)&vkDescriptorSetLayoutCreateInfo, 0, sizeof(VkDescriptorSetLayoutCreateInfo));
	vkDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	vkDescriptorSetLayoutCreateInfo.pNext = NULL;
	vkDescriptorSetLayoutCreateInfo.flags = 0;

	vkDescriptorSetLayoutCreateInfo.bindingCount = 3; // Three bindings for tessellation
	vkDescriptorSetLayoutCreateInfo.pBindings = vkDescriptorSetLayoutBinding_array;
	
	/*
	24.4. Then call vkCreateDescriptorSetLayout() Vulkan API with adress of above initialized structure and get the required global Vulkan object vkDescriptorSetLayout in its last parameter.
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateDescriptorSetLayout.html
	// Provided by VK_VERSION_1_0
	VkResult vkCreateDescriptorSetLayout(
    VkDevice                                    device,
    const VkDescriptorSetLayoutCreateInfo*      pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDescriptorSetLayout*                      pSetLayout);
	*/
	vkResult = vkCreateDescriptorSetLayout(vkDevice, &vkDescriptorSetLayoutCreateInfo, NULL, &vkDescriptorSetLayout);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateDescriptorSetLayout(): vkCreateDescriptorSetLayout() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateDescriptorSetLayout(): vkCreateDescriptorSetLayout() function succedded\n");
	}
	
	return vkResult;
}

/*
25.2. In initialize(), declare and call UDF CreatePipelineLayout() maintaining the convention of declaring and calling it after CreatDescriptorSetLayout() and before CreateRenderPass().
*/
VkResult CreatePipelineLayout(void)
{
	//Variable declarations	
	VkResult vkResult = VK_SUCCESS;
	
	/*
	Code
	*/
	
	/*
	25.3. While writing the definition of UDF, declare, memset and initialize struct VkPipelineLayoutCreateInfo , particularly its 4 important members 
	   1. .setLayoutCount
	   2. .pSetLayouts array
	   3. .pushConstantRangeCount
	   4. .pPushConstantRanges array
	//https://registry.khronos.org/VulkanSC/specs/1.0-extensions/man/html/VkPipelineLayoutCreateInfo.html
	// Provided by VK_VERSION_1_0
	typedef struct VkPipelineLayoutCreateInfo {
		VkStructureType                 sType;
		const void*                     pNext;
		VkPipelineLayoutCreateFlags     flags;
		uint32_t                        setLayoutCount;
		const VkDescriptorSetLayout*    pSetLayouts;
		uint32_t                        pushConstantRangeCount;
		const VkPushConstantRange*      pPushConstantRanges;
	} VkPipelineLayoutCreateInfo;
	*/
	VkPipelineLayoutCreateInfo vkPipelineLayoutCreateInfo;
	memset((void*)&vkPipelineLayoutCreateInfo, 0, sizeof(VkPipelineLayoutCreateInfo));
	vkPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	vkPipelineLayoutCreateInfo.pNext = NULL;
	vkPipelineLayoutCreateInfo.flags = 0; /* Reserved*/
	vkPipelineLayoutCreateInfo.setLayoutCount = 1;
	vkPipelineLayoutCreateInfo.pSetLayouts = &vkDescriptorSetLayout;
	vkPipelineLayoutCreateInfo.pushConstantRangeCount = 0;
	vkPipelineLayoutCreateInfo.pPushConstantRanges = NULL;
	
	/*
	25.4. Then call vkCreatePipelineLayout() Vulkan API with adress of above initialized structure and get the required global Vulkan object vkPipelineLayout in its last parameter.
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreatePipelineLayout.html
	// Provided by VK_VERSION_1_0
	VkResult vkCreatePipelineLayout(
    VkDevice                                    device,
    const VkPipelineLayoutCreateInfo*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkPipelineLayout*                           pPipelineLayout);
	*/
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

//31.13
VkResult CreateDescriptorPool(void)
{
	//Variable declarations
	VkResult vkResult = VK_SUCCESS;

	/*
	Code
	*/
	// Pool sizes for tessellation pipeline:
	// - 2 uniform buffers (MVP + Tessellation params)
	// - 1 combined image sampler (Height map)
	VkDescriptorPoolSize vkDescriptorPoolSize_array[2];
	memset((void*)vkDescriptorPoolSize_array, 0, sizeof(VkDescriptorPoolSize) * 2);

	// Uniform buffers pool size
	vkDescriptorPoolSize_array[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	vkDescriptorPoolSize_array[0].descriptorCount = 2; // MVP + Tessellation params

	// Combined image sampler pool size
	vkDescriptorPoolSize_array[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	vkDescriptorPoolSize_array[1].descriptorCount = 1; // Height map

	/*
	//Create the pool
	*/
	VkDescriptorPoolCreateInfo vkDescriptorPoolCreateInfo;
	memset((void*)&vkDescriptorPoolCreateInfo, 0, sizeof(VkDescriptorPoolCreateInfo));
	vkDescriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	vkDescriptorPoolCreateInfo.pNext = NULL;
	vkDescriptorPoolCreateInfo.flags = 0;
	vkDescriptorPoolCreateInfo.maxSets = 1;
	vkDescriptorPoolCreateInfo.poolSizeCount = 2; // Two pool size entries
	vkDescriptorPoolCreateInfo.pPoolSizes = vkDescriptorPoolSize_array;
	
	/*
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateDescriptorPool.html
	// Provided by VK_VERSION_1_0
	VkResult vkCreateDescriptorPool(
    VkDevice                                    device,
    const VkDescriptorPoolCreateInfo*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDescriptorPool*                           pDescriptorPool);
	*/
	vkResult = vkCreateDescriptorPool(vkDevice, &vkDescriptorPoolCreateInfo, NULL, &vkDescriptorPool);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateDescriptorPool(): vkCreateDescriptorPool() function failed with error code %d\n", vkResult);
		return vkResult;
	}
	else
	{
		FileIO("CreateDescriptorPool(): vkCreateDescriptorPool() succedded\n");
	}
	
	return vkResult;
}

//31.14
VkResult CreateDescriptorSet(void)
{
	//Variable declarations	
	VkResult vkResult = VK_SUCCESS;
	
	/*
	Code
	*/
	/*
	//Initialize descriptor set allocation info
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorSetAllocateInfo.html
	// Provided by VK_VERSION_1_0
	typedef struct VkDescriptorSetAllocateInfo {
		VkStructureType                 sType;
		const void*                     pNext;
		VkDescriptorPool                descriptorPool;
		uint32_t                        descriptorSetCount;
		const VkDescriptorSetLayout*    pSetLayouts;
	} VkDescriptorSetAllocateInfo;
	*/
	VkDescriptorSetAllocateInfo vkDescriptorSetAllocateInfo;
	memset((void*)&vkDescriptorSetAllocateInfo, 0, sizeof(VkDescriptorSetAllocateInfo));
	vkDescriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	vkDescriptorSetAllocateInfo.pNext = NULL;
	vkDescriptorSetAllocateInfo.descriptorPool = vkDescriptorPool;
	
	vkDescriptorSetAllocateInfo.descriptorSetCount = 1; //We are passing only 1 struct so put 1 here
	//we are giving descriptor setlayout's here for first time after Pipeline
	//Now plate is not empty, it has 1 descriptor
	//to bharnyasathi allocate karun de , 1 descriptor set bharnya sathi
	vkDescriptorSetAllocateInfo.pSetLayouts = &vkDescriptorSetLayout; 
	
	/*
	//Jitha structure madhe point ani counter ekatra astat, tithe array expected astoch
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkAllocateDescriptorSets.html
	// Provided by VK_VERSION_1_0
	VkResult vkAllocateDescriptorSets(
    VkDevice                                    device,
    const VkDescriptorSetAllocateInfo*          pAllocateInfo,
    VkDescriptorSet*                            pDescriptorSets);
	*/
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

	// ================================================================
	// Descriptor buffer info for MVP uniform (binding 0)
	// ================================================================
	VkDescriptorBufferInfo vkDescriptorBufferInfo_mvp;
	memset((void*)&vkDescriptorBufferInfo_mvp, 0, sizeof(VkDescriptorBufferInfo));
	vkDescriptorBufferInfo_mvp.buffer = uniformData.vkBuffer;
	vkDescriptorBufferInfo_mvp.offset = 0;
	vkDescriptorBufferInfo_mvp.range = sizeof(struct MyUniformData);

	// ================================================================
	// Descriptor buffer info for Tessellation params (binding 1)
	// ================================================================
	VkDescriptorBufferInfo vkDescriptorBufferInfo_tess;
	memset((void*)&vkDescriptorBufferInfo_tess, 0, sizeof(VkDescriptorBufferInfo));
	vkDescriptorBufferInfo_tess.buffer = tessUniformData.vkBuffer;
	vkDescriptorBufferInfo_tess.offset = 0;
	vkDescriptorBufferInfo_tess.range = sizeof(struct TessellationParams);

	// ================================================================
	// Descriptor image info for Height map sampler (binding 2)
	// ================================================================
	VkDescriptorImageInfo vkDescriptorImageInfo_heightMap;
	memset((void*)&vkDescriptorImageInfo_heightMap, 0, sizeof(VkDescriptorImageInfo));
	vkDescriptorImageInfo_heightMap.sampler = vkSampler_heightMap;
	vkDescriptorImageInfo_heightMap.imageView = vkImageView_heightMap;
	vkDescriptorImageInfo_heightMap.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	// ================================================================
	// Write descriptor sets array (3 descriptors)
	// ================================================================
	VkWriteDescriptorSet vkWriteDescriptorSet_array[3];
	memset((void*)vkWriteDescriptorSet_array, 0, sizeof(VkWriteDescriptorSet) * 3);

	// Write descriptor for MVP uniform buffer (binding 0)
	vkWriteDescriptorSet_array[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	vkWriteDescriptorSet_array[0].pNext = NULL;
	vkWriteDescriptorSet_array[0].dstSet = vkDescriptorSet;
	vkWriteDescriptorSet_array[0].dstBinding = 0;
	vkWriteDescriptorSet_array[0].dstArrayElement = 0;
	vkWriteDescriptorSet_array[0].descriptorCount = 1;
	vkWriteDescriptorSet_array[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	vkWriteDescriptorSet_array[0].pImageInfo = NULL;
	vkWriteDescriptorSet_array[0].pBufferInfo = &vkDescriptorBufferInfo_mvp;
	vkWriteDescriptorSet_array[0].pTexelBufferView = NULL;

	// Write descriptor for Tessellation params uniform buffer (binding 1)
	vkWriteDescriptorSet_array[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	vkWriteDescriptorSet_array[1].pNext = NULL;
	vkWriteDescriptorSet_array[1].dstSet = vkDescriptorSet;
	vkWriteDescriptorSet_array[1].dstBinding = 1;
	vkWriteDescriptorSet_array[1].dstArrayElement = 0;
	vkWriteDescriptorSet_array[1].descriptorCount = 1;
	vkWriteDescriptorSet_array[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	vkWriteDescriptorSet_array[1].pImageInfo = NULL;
	vkWriteDescriptorSet_array[1].pBufferInfo = &vkDescriptorBufferInfo_tess;
	vkWriteDescriptorSet_array[1].pTexelBufferView = NULL;

	// Write descriptor for Height map combined image sampler (binding 2)
	vkWriteDescriptorSet_array[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	vkWriteDescriptorSet_array[2].pNext = NULL;
	vkWriteDescriptorSet_array[2].dstSet = vkDescriptorSet;
	vkWriteDescriptorSet_array[2].dstBinding = 2;
	vkWriteDescriptorSet_array[2].dstArrayElement = 0;
	vkWriteDescriptorSet_array[2].descriptorCount = 1;
	vkWriteDescriptorSet_array[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	vkWriteDescriptorSet_array[2].pImageInfo = &vkDescriptorImageInfo_heightMap;
	vkWriteDescriptorSet_array[2].pBufferInfo = NULL;
	vkWriteDescriptorSet_array[2].pTexelBufferView = NULL;

	// Update all 3 descriptors at once
	vkUpdateDescriptorSets(vkDevice, 3, vkWriteDescriptorSet_array, 0, NULL);

	FileIO("CreateDescriptorSet(): vkUpdateDescriptorSets() for 3 descriptors succeeded\n");
	
	return vkResult;
}

VkResult CreateRenderPass(void)
{
	//Variable declarations	
	VkResult vkResult = VK_SUCCESS;
	
	/*
	Code
	*/
	
	/*
	1. Declare and initialize VkAttachmentDescription Struct array. (https://registry.khronos.org/vulkan/specs/latest/man/html/VkAttachmentDescription.html)
    Number of elements in Array depends on number of attachments.
   (Although we have only 1 attachment i.e color attachment in this example, we will consider it as array)
   
   typedef struct VkAttachmentDescription {
    VkAttachmentDescriptionFlags    flags;
    VkFormat                        format;
    VkSampleCountFlagBits           samples;
    VkAttachmentLoadOp              loadOp;
    VkAttachmentStoreOp             storeOp;
    VkAttachmentLoadOp              stencilLoadOp;
    VkAttachmentStoreOp             stencilStoreOp;
    VkImageLayout                   initialLayout;
    VkImageLayout                   finalLayout;
	} VkAttachmentDescription;
	*/
	VkAttachmentDescription  vkAttachmentDescription_array[2]; //color and depth when added array will be of 2
	memset((void*)vkAttachmentDescription_array, 0, sizeof(VkAttachmentDescription) * _ARRAYSIZE(vkAttachmentDescription_array));
	
	//For Color
	/*
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkAttachmentDescriptionFlagBits.html
	
	// Provided by VK_VERSION_1_0
	typedef enum VkAttachmentDescriptionFlagBits {
		VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT = 0x00000001,
	} VkAttachmentDescriptionFlagBits;
	
	Info on Sony japan company documentation of paper presentation.
	Mostly 0 , only for manging memory in embedded devices
	Multiple attachments jar astil , tar eka mekanchi memory vapru shaktat.
	*/
	vkAttachmentDescription_array[0].flags = 0; 
	
	vkAttachmentDescription_array[0].format = vkFormat_color;

	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkSampleCountFlagBits.html
	/*
	// Provided by VK_VERSION_1_0
	typedef enum VkSampleCountFlagBits {
    VK_SAMPLE_COUNT_1_BIT = 0x00000001,
    VK_SAMPLE_COUNT_2_BIT = 0x00000002,
    VK_SAMPLE_COUNT_4_BIT = 0x00000004,
    VK_SAMPLE_COUNT_8_BIT = 0x00000008,
    VK_SAMPLE_COUNT_16_BIT = 0x00000010,
    VK_SAMPLE_COUNT_32_BIT = 0x00000020,
    VK_SAMPLE_COUNT_64_BIT = 0x00000040,
	} VkSampleCountFlagBits;
	
	https://www.google.com/search?q=sampling+meaning+in+texturw&oq=sampling+meaning+in+texturw&gs_lcrp=EgZjaHJvbWUyBggAEEUYOdIBCTYzMjlqMGoxNagCCLACAQ&sourceid=chrome&ie=UTF-8
	*/
	vkAttachmentDescription_array[0].samples = VK_SAMPLE_COUNT_1_BIT; // No MSAA
	
	// https://registry.khronos.org/vulkan/specs/latest/man/html/VkAttachmentLoadOp.html
	/*
	// Provided by VK_VERSION_1_0
	typedef enum VkAttachmentLoadOp {
		VK_ATTACHMENT_LOAD_OP_LOAD = 0,
		VK_ATTACHMENT_LOAD_OP_CLEAR = 1,
		VK_ATTACHMENT_LOAD_OP_DONT_CARE = 2,
	  // Provided by VK_VERSION_1_4
		VK_ATTACHMENT_LOAD_OP_NONE = 1000400000,
	  // Provided by VK_EXT_load_store_op_none
		VK_ATTACHMENT_LOAD_OP_NONE_EXT = VK_ATTACHMENT_LOAD_OP_NONE,
	  // Provided by VK_KHR_load_store_op_none
		VK_ATTACHMENT_LOAD_OP_NONE_KHR = VK_ATTACHMENT_LOAD_OP_NONE,
	} VkAttachmentLoadOp;
	
	ya structure chi mahiti direct renderpass la jata.
	*/
	vkAttachmentDescription_array[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; //Render pass madhe aat aalyavar kay karu attachment cha image data sobat
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkAttachmentStoreOp.html
	/*
	// Provided by VK_VERSION_1_0
	typedef enum VkAttachmentStoreOp {
    VK_ATTACHMENT_STORE_OP_STORE = 0,
    VK_ATTACHMENT_STORE_OP_DONT_CARE = 1,
  // Provided by VK_VERSION_1_3
    VK_ATTACHMENT_STORE_OP_NONE = 1000301000,
  // Provided by VK_KHR_dynamic_rendering, VK_KHR_load_store_op_none
    VK_ATTACHMENT_STORE_OP_NONE_KHR = VK_ATTACHMENT_STORE_OP_NONE,
  // Provided by VK_QCOM_render_pass_store_ops
    VK_ATTACHMENT_STORE_OP_NONE_QCOM = VK_ATTACHMENT_STORE_OP_NONE,
  // Provided by VK_EXT_load_store_op_none
    VK_ATTACHMENT_STORE_OP_NONE_EXT = VK_ATTACHMENT_STORE_OP_NONE,
	} VkAttachmentStoreOp;
	*/
	vkAttachmentDescription_array[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE; //Render pass madhun baher gelyavar kay karu attachment image data sobat
	
	vkAttachmentDescription_array[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; // For both depth and stencil, dont go on name
	vkAttachmentDescription_array[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; // For both depth and stencil, dont go on name
	
	/*
	https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageLayout.html
	he sarv attachment madhla data cha arrangement cha aahe
	Unpacking athva RTR cha , karan color attachment mhnaje mostly texture
	*/
	vkAttachmentDescription_array[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; //Renderpass cha aat aalyavar , attachment cha data arrangemnent cha kay karu
	vkAttachmentDescription_array[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; //Renderpass cha baher gelyavar , attachment cha data arrangemnent cha kay karu
	/*
	jya praname soure image aage , taasach layout thevun present kar.
	Madhe kahi changes zale, source praname thev
	*/
	
	//For Depth
	/*
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkAttachmentDescriptionFlagBits.html
	
	// Provided by VK_VERSION_1_0
	typedef enum VkAttachmentDescriptionFlagBits {
		VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT = 0x00000001,
	} VkAttachmentDescriptionFlagBits;
	
	Info on Sony japan company documentation of paper presentation.
	Mostly 0 , only for manging memory in embedded devices
	Multiple attachments jar astil , tar eka mekanchi memory vapru shaktat.
	*/
	vkAttachmentDescription_array[1].flags = 0; 
	
	vkAttachmentDescription_array[1].format = vkFormat_depth;

	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkSampleCountFlagBits.html
	/*
	// Provided by VK_VERSION_1_0
	typedef enum VkSampleCountFlagBits {
    VK_SAMPLE_COUNT_1_BIT = 0x00000001,
    VK_SAMPLE_COUNT_2_BIT = 0x00000002,
    VK_SAMPLE_COUNT_4_BIT = 0x00000004,
    VK_SAMPLE_COUNT_8_BIT = 0x00000008,
    VK_SAMPLE_COUNT_16_BIT = 0x00000010,
    VK_SAMPLE_COUNT_32_BIT = 0x00000020,
    VK_SAMPLE_COUNT_64_BIT = 0x00000040,
	} VkSampleCountFlagBits;
	
	https://www.google.com/search?q=sampling+meaning+in+texturw&oq=sampling+meaning+in+texturw&gs_lcrp=EgZjaHJvbWUyBggAEEUYOdIBCTYzMjlqMGoxNagCCLACAQ&sourceid=chrome&ie=UTF-8
	*/
	vkAttachmentDescription_array[1].samples = VK_SAMPLE_COUNT_1_BIT; // No MSAA
	
	// https://registry.khronos.org/vulkan/specs/latest/man/html/VkAttachmentLoadOp.html
	/*
	// Provided by VK_VERSION_1_0
	typedef enum VkAttachmentLoadOp {
		VK_ATTACHMENT_LOAD_OP_LOAD = 0,
		VK_ATTACHMENT_LOAD_OP_CLEAR = 1,
		VK_ATTACHMENT_LOAD_OP_DONT_CARE = 2,
	  // Provided by VK_VERSION_1_4
		VK_ATTACHMENT_LOAD_OP_NONE = 1000400000,
	  // Provided by VK_EXT_load_store_op_none
		VK_ATTACHMENT_LOAD_OP_NONE_EXT = VK_ATTACHMENT_LOAD_OP_NONE,
	  // Provided by VK_KHR_load_store_op_none
		VK_ATTACHMENT_LOAD_OP_NONE_KHR = VK_ATTACHMENT_LOAD_OP_NONE,
	} VkAttachmentLoadOp;
	
	ya structure chi mahiti direct renderpass la jata.
	*/
	vkAttachmentDescription_array[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; //Render pass madhe aat aalyavar kay karu attachment cha image data sobat
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkAttachmentStoreOp.html
	/*
	// Provided by VK_VERSION_1_0
	typedef enum VkAttachmentStoreOp {
    VK_ATTACHMENT_STORE_OP_STORE = 0,
    VK_ATTACHMENT_STORE_OP_DONT_CARE = 1,
  // Provided by VK_VERSION_1_3
    VK_ATTACHMENT_STORE_OP_NONE = 1000301000,
  // Provided by VK_KHR_dynamic_rendering, VK_KHR_load_store_op_none
    VK_ATTACHMENT_STORE_OP_NONE_KHR = VK_ATTACHMENT_STORE_OP_NONE,
  // Provided by VK_QCOM_render_pass_store_ops
    VK_ATTACHMENT_STORE_OP_NONE_QCOM = VK_ATTACHMENT_STORE_OP_NONE,
  // Provided by VK_EXT_load_store_op_none
    VK_ATTACHMENT_STORE_OP_NONE_EXT = VK_ATTACHMENT_STORE_OP_NONE,
	} VkAttachmentStoreOp;
	*/
	vkAttachmentDescription_array[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE; //Render pass madhun baher gelyavar kay karu attachment image data sobat
	
	vkAttachmentDescription_array[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; // For both depth and stencil, dont go on name
	vkAttachmentDescription_array[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; // For both depth and stencil, dont go on name
	
	/*
	https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageLayout.html
	he sarv attachment madhla data cha arrangement cha aahe
	Unpacking athva RTR cha , karan color attachment mhnaje mostly texture
	*/
	vkAttachmentDescription_array[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; //Renderpass cha aat aalyavar , attachment cha data arrangemnent cha kay karu
	vkAttachmentDescription_array[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; //Renderpass cha baher gelyavar , attachment cha data arrangemnent cha kay karu
	/*
	jya praname soure image aage , taasach layout thevun present kar.
	Madhe kahi changes zale, source praname thev
	*/
	
	/*
	/////////////////////////////////
	//For Color attachment
	2. Declare and initialize VkAttachmentReference struct (https://registry.khronos.org/vulkan/specs/latest/man/html/VkAttachmentReference.html) , which will have information about the attachment we described above.
	(jevha depth baghu , tevha proper ek extra element add hoil array madhe)
	*/
	VkAttachmentReference vkAttachmentReference_color;
	memset((void*)&vkAttachmentReference_color, 0, sizeof(VkAttachmentReference));
	vkAttachmentReference_color.attachment = 0; //It is index. 0th is color attchment , 1st will be depth attachment
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageLayout.html
	//he image ksa vapraycha aahe , sang mala
	vkAttachmentReference_color.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; //layout kasa thevaycha aahe , vapraycha aahe ? i.e yacha layout asa thev ki mi he attachment , color attachment mhanun vapru shakel
	
	/*
	/////////////////////////////////
	//For Depth attachmnent
	Declare and initialize VkAttachmentReference struct (https://registry.khronos.org/vulkan/specs/latest/man/html/VkAttachmentReference.html) , which will have information about the attachment we described above.
	(jevha depth baghu , tevha proper ek extra element add hoil array madhe)
	*/
	VkAttachmentReference vkAttachmentReference_depth;
	memset((void*)&vkAttachmentReference_depth, 0, sizeof(VkAttachmentReference));
	vkAttachmentReference_depth.attachment = 1; //It is index. 0th is color attchment , 1st will be depth attachment
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageLayout.html
	//he image ksa vapraycha aahe , sang mala
	vkAttachmentReference_depth.layout =  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; //layout kasa thevaycha aahe , vapraycha aahe ? i.e yacha layout asa thev ki mi he attachment , color attachment mhanun vapru shakel
	
	/*
	/////////////////////////////////
	3. Declare and initialize VkSubpassDescription struct (https://registry.khronos.org/vulkan/specs/latest/man/html/VkSubpassDescription.html) and keep reference about above VkAttachmentReference structe in it.
	*/
	VkSubpassDescription vkSubpassDescription; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkSubpassDescription.html
	memset((void*)&vkSubpassDescription, 0, sizeof(VkSubpassDescription));
	
	vkSubpassDescription.flags = 0;
	vkSubpassDescription.pipelineBindPoint =  VK_PIPELINE_BIND_POINT_GRAPHICS; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineBindPoint.html
	vkSubpassDescription.inputAttachmentCount = 0;
	vkSubpassDescription.pInputAttachments = NULL;
	vkSubpassDescription.colorAttachmentCount = 1; //This count should be count of VkAttachmentReference used for color
	vkSubpassDescription.pColorAttachments = (const VkAttachmentReference*)&vkAttachmentReference_color;
	vkSubpassDescription.pResolveAttachments = NULL;
	vkSubpassDescription.pDepthStencilAttachment = (const VkAttachmentReference*)&vkAttachmentReference_depth;
	vkSubpassDescription.preserveAttachmentCount = 0;
	vkSubpassDescription.pPreserveAttachments = NULL;
	
	/*
	/////////////////////////////////
	4. Declare and initialize VkRenderPassCreatefo struct (https://registry.khronos.org/vulkan/specs/latest/man/html/VkRenderPassCreateInfo.html)  and referabove VkAttachmentDescription struct and VkSubpassDescription struct into it.
    Remember here also we need attachment information in form of Image Views, which will be used by framebuffer later.
    We also need to specify interdependancy between subpasses if needed.
	*/
	// https://registry.khronos.org/vulkan/specs/latest/man/html/VkRenderPassCreateInfo.html
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
	
	/*
	/////////////////////////////////
	5. Now call vkCreateRenderPass() (https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateRenderPass.html) to create actual RenderPass.
	*/
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
	//Variable declarations	
	VkResult vkResult = VK_SUCCESS;
	
	/*
	Code
	*/
	/*
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkVertexInputBindingDescription.html
	// Provided by VK_VERSION_1_0
	typedef struct VkVertexInputBindingDescription {
		uint32_t             binding;
		uint32_t             stride;
		VkVertexInputRate    inputRate; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkVertexInputRate.html
	} VkVertexInputBindingDescription;
	
	// Provided by VK_VERSION_1_0
	typedef enum VkVertexInputRate {
		VK_VERTEX_INPUT_RATE_VERTEX = 0,
		VK_VERTEX_INPUT_RATE_INSTANCE = 1,
	} VkVertexInputRate;
	*/
	VkVertexInputBindingDescription vkVertexInputBindingDescription_array[1];
	memset((void*)vkVertexInputBindingDescription_array, 0,  sizeof(VkVertexInputBindingDescription) * _ARRAYSIZE(vkVertexInputBindingDescription_array));
	vkVertexInputBindingDescription_array[0].binding = 0; //Equivalent to GL_ARRAY_BUFFER
	vkVertexInputBindingDescription_array[0].stride = sizeof(float) * 4; //cause in shader we have vec4
	vkVertexInputBindingDescription_array[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX; //vertices maan, indices nako
	
	/*
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkVertexInputAttributeDescription.html
	// Provided by VK_VERSION_1_0
	typedef struct VkVertexInputAttributeDescription {
		uint32_t    location;
		uint32_t    binding;
		VkFormat    format; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormat.html
		uint32_t    offset;
	} VkVertexInputAttributeDescription;
	*/
	VkVertexInputAttributeDescription vkVertexInputAttributeDescription_array[1];
	memset((void*)vkVertexInputAttributeDescription_array, 0,  sizeof(VkVertexInputAttributeDescription) * _ARRAYSIZE(vkVertexInputAttributeDescription_array));
	vkVertexInputAttributeDescription_array[0].location = 0;
	vkVertexInputAttributeDescription_array[0].binding = 0;
	vkVertexInputAttributeDescription_array[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
	vkVertexInputAttributeDescription_array[0].offset = 0;
	
	/*
	Vertex Input State PSO
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineVertexInputStateCreateInfo.html
	// Provided by VK_VERSION_1_0
	typedef struct VkPipelineVertexInputStateCreateInfo {
		VkStructureType                             sType;
		const void*                                 pNext;
		VkPipelineVertexInputStateCreateFlags       flags;
		uint32_t                                    vertexBindingDescriptionCount;
		const VkVertexInputBindingDescription*      pVertexBindingDescriptions;
		uint32_t                                    vertexAttributeDescriptionCount;
		const VkVertexInputAttributeDescription*    pVertexAttributeDescriptions;
	} VkPipelineVertexInputStateCreateInfo;
	*/
	VkPipelineVertexInputStateCreateInfo vkPipelineVertexInputStateCreateInfo;
	memset((void*)&vkPipelineVertexInputStateCreateInfo, 0,  sizeof(VkPipelineVertexInputStateCreateInfo));
	vkPipelineVertexInputStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vkPipelineVertexInputStateCreateInfo.pNext = NULL;
	vkPipelineVertexInputStateCreateInfo.flags = 0;
	vkPipelineVertexInputStateCreateInfo.vertexBindingDescriptionCount = _ARRAYSIZE(vkVertexInputBindingDescription_array);
	vkPipelineVertexInputStateCreateInfo.pVertexBindingDescriptions = vkVertexInputBindingDescription_array;
	vkPipelineVertexInputStateCreateInfo.vertexAttributeDescriptionCount = _ARRAYSIZE(vkVertexInputAttributeDescription_array);
	vkPipelineVertexInputStateCreateInfo.pVertexAttributeDescriptions = vkVertexInputAttributeDescription_array;
	
	/*
	Input Assembly State
	https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineInputAssemblyStateCreateInfo.html/
	// Provided by VK_VERSION_1_0
	typedef struct VkPipelineInputAssemblyStateCreateInfo {
		VkStructureType                            sType;
		const void*                                pNext;
		VkPipelineInputAssemblyStateCreateFlags    flags; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineInputAssemblyStateCreateFlags.html
		VkPrimitiveTopology                        topology;
		VkBool32                                   primitiveRestartEnable;
	} VkPipelineInputAssemblyStateCreateInfo;
	
	https://registry.khronos.org/vulkan/specs/latest/man/html/VkPrimitiveTopology.html
	// Provided by VK_VERSION_1_0
	typedef enum VkPrimitiveTopology {
		VK_PRIMITIVE_TOPOLOGY_POINT_LIST = 0,
		VK_PRIMITIVE_TOPOLOGY_LINE_LIST = 1,
		VK_PRIMITIVE_TOPOLOGY_LINE_STRIP = 2,
		VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 3,
		VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP = 4,
		VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN = 5,
		
		//For Geometry Shader
		VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY = 6,
		VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY = 7,
		VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY = 8,
		VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY = 9,
		
		//For Tescellation Shader
		VK_PRIMITIVE_TOPOLOGY_PATCH_LIST = 10,
	} VkPrimitiveTopology;
	
	*/
	VkPipelineInputAssemblyStateCreateInfo vkPipelineInputAssemblyStateCreateInfo;
	memset((void*)&vkPipelineInputAssemblyStateCreateInfo, 0,  sizeof(VkPipelineInputAssemblyStateCreateInfo));
	vkPipelineInputAssemblyStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	vkPipelineInputAssemblyStateCreateInfo.pNext = NULL;
	vkPipelineInputAssemblyStateCreateInfo.flags = 0;
	// Use PATCH_LIST for tessellation - each patch has 3 control points (triangle)
	vkPipelineInputAssemblyStateCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
	vkPipelineInputAssemblyStateCreateInfo.primitiveRestartEnable = VK_FALSE;
	
	/*
	//Rasterizer State
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineRasterizationStateCreateInfo.html
	// Provided by VK_VERSION_1_0
	typedef struct VkPipelineRasterizationStateCreateInfo {
		VkStructureType                            sType;
		const void*                                pNext;
		VkPipelineRasterizationStateCreateFlags    flags; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineRasterizationStateCreateFlags.html
		VkBool32                                   depthClampEnable;
		VkBool32                                   rasterizerDiscardEnable;
		VkPolygonMode                              polygonMode;
		VkCullModeFlags                            cullMode; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkCullModeFlags.html
		VkFrontFace                                frontFace;
		VkBool32                                   depthBiasEnable;
		float                                      depthBiasConstantFactor;
		float                                      depthBiasClamp;
		float                                      depthBiasSlopeFactor;
		float                                      lineWidth;
	} VkPipelineRasterizationStateCreateInfo;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkPolygonMode.html
	// Provided by VK_VERSION_1_0
	typedef enum VkPolygonMode {
		VK_POLYGON_MODE_FILL = 0,
		VK_POLYGON_MODE_LINE = 1,
		VK_POLYGON_MODE_POINT = 2,
	  // Provided by VK_NV_fill_rectangle
		VK_POLYGON_MODE_FILL_RECTANGLE_NV = 1000153000,
	} VkPolygonMode;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkFrontFace.html
	// Provided by VK_VERSION_1_0
	typedef enum VkFrontFace {
		VK_FRONT_FACE_COUNTER_CLOCKWISE = 0,
		VK_FRONT_FACE_CLOCKWISE = 1,
	} VkFrontFace;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkCullModeFlags.html
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkCullModeFlagBits.html
	// Provided by VK_VERSION_1_0
	typedef enum VkCullModeFlagBits {
		VK_CULL_MODE_NONE = 0,
		VK_CULL_MODE_FRONT_BIT = 0x00000001,
		VK_CULL_MODE_BACK_BIT = 0x00000002,
		VK_CULL_MODE_FRONT_AND_BACK = 0x00000003,
	} VkCullModeFlagBits;
	*/
	VkPipelineRasterizationStateCreateInfo vkPipelineRasterizationStateCreateInfo;
	memset((void*)&vkPipelineRasterizationStateCreateInfo, 0,  sizeof(VkPipelineRasterizationStateCreateInfo));
	vkPipelineRasterizationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	vkPipelineRasterizationStateCreateInfo.pNext = NULL;
	vkPipelineRasterizationStateCreateInfo.flags = 0;
	//vkPipelineRasterizationStateCreateInfo.depthClampEnable =;
	//vkPipelineRasterizationStateCreateInfo.rasterizerDiscardEnable =;
	vkPipelineRasterizationStateCreateInfo.polygonMode = VK_POLYGON_MODE_LINE; //Wireframe mode
	vkPipelineRasterizationStateCreateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
	vkPipelineRasterizationStateCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; //Triangle winding order 
	//vkPipelineRasterizationStateCreateInfo.depthBiasEnable =;
	//vkPipelineRasterizationStateCreateInfo.depthBiasConstantFactor =;
	//vkPipelineRasterizationStateCreateInfo.depthBiasClamp =;
	//vkPipelineRasterizationStateCreateInfo.depthBiasSlopeFactor =;
	vkPipelineRasterizationStateCreateInfo.lineWidth = 1.0f; //This is implementation dependant. So giving it is compulsary. Atleast give it 1.0
	
	/*
	//Color Blend state
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineColorBlendAttachmentState.html
	// Provided by VK_VERSION_1_0
	typedef struct VkPipelineColorBlendAttachmentState {
		VkBool32                 blendEnable;
		VkBlendFactor            srcColorBlendFactor; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkBlendFactor.html
		VkBlendFactor            dstColorBlendFactor; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkBlendFactor.html
		VkBlendOp                colorBlendOp; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkBlendOp.html
		VkBlendFactor            srcAlphaBlendFactor;
		VkBlendFactor            dstAlphaBlendFactor;
		VkBlendOp                alphaBlendOp;
		VkColorComponentFlags    colorWriteMask; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkColorComponentFlags.html
	} VkPipelineColorBlendAttachmentState;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkColorComponentFlags.html
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkColorComponentFlagBits.html
	// Provided by VK_VERSION_1_0
	typedef enum VkColorComponentFlagBits {
		VK_COLOR_COMPONENT_R_BIT = 0x00000001,
		VK_COLOR_COMPONENT_G_BIT = 0x00000002,
		VK_COLOR_COMPONENT_B_BIT = 0x00000004,
		VK_COLOR_COMPONENT_A_BIT = 0x00000008,
	} VkColorComponentFlagBits;
	*/
	VkPipelineColorBlendAttachmentState vkPipelineColorBlendAttachmentState_array[1];
	memset((void*)vkPipelineColorBlendAttachmentState_array, 0, sizeof(VkPipelineColorBlendAttachmentState) * _ARRAYSIZE(vkPipelineColorBlendAttachmentState_array));
	vkPipelineColorBlendAttachmentState_array[0].blendEnable = VK_FALSE;
	/*
	vkPipelineColorBlendAttachmentState_array[0].srcColorBlendFactor =;
	vkPipelineColorBlendAttachmentState_array[0].dstColorBlendFactor =;
	vkPipelineColorBlendAttachmentState_array[0].colorBlendOp =;
	vkPipelineColorBlendAttachmentState_array[0].srcAlphaBlendFactor =;
	vkPipelineColorBlendAttachmentState_array[0].dstAlphaBlendFactor =;
	vkPipelineColorBlendAttachmentState_array[0].alphaBlendOp=;
	*/
	vkPipelineColorBlendAttachmentState_array[0].colorWriteMask = 0xF;
	
	/*
	//Color Blend state
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineColorBlendStateCreateInfo.html
	// Provided by VK_VERSION_1_0
	typedef struct VkPipelineColorBlendStateCreateInfo {
		VkStructureType                               sType;
		const void*                                   pNext;
		VkPipelineColorBlendStateCreateFlags          flags; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineColorBlendStateCreateFlags.html
		VkBool32                                      logicOpEnable;
		VkLogicOp                                     logicOp;
		uint32_t                                      attachmentCount;
		const VkPipelineColorBlendAttachmentState*    pAttachments;
		float                                         blendConstants[4];
	} VkPipelineColorBlendStateCreateInfo;
	*/
	VkPipelineColorBlendStateCreateInfo vkPipelineColorBlendStateCreateInfo;
	memset((void*)&vkPipelineColorBlendStateCreateInfo, 0, sizeof(VkPipelineColorBlendStateCreateInfo));
	vkPipelineColorBlendStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	vkPipelineColorBlendStateCreateInfo.pNext = NULL;
	vkPipelineColorBlendStateCreateInfo.flags = 0;
	//vkPipelineColorBlendStateCreateInfo.logicOpEnable =;
	//vkPipelineColorBlendStateCreateInfo.logicOp = ;
	vkPipelineColorBlendStateCreateInfo.attachmentCount = _ARRAYSIZE(vkPipelineColorBlendAttachmentState_array);
	vkPipelineColorBlendStateCreateInfo.pAttachments = vkPipelineColorBlendAttachmentState_array;
	//vkPipelineColorBlendStateCreateInfo.blendConstants =;
	
	/*Viewport Scissor State
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineViewportStateCreateInfo.html
	// Provided by VK_VERSION_1_0
	typedef struct VkPipelineViewportStateCreateInfo {
		VkStructureType                       sType;
		const void*                           pNext;
		VkPipelineViewportStateCreateFlags    flags;
		uint32_t                              viewportCount;
		const VkViewport*                     pViewports;
		uint32_t                              scissorCount;
		const VkRect2D*                       pScissors;
	} VkPipelineViewportStateCreateInfo;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkViewport.html
	// Provided by VK_VERSION_1_0
	typedef struct VkViewport {
		float    x;
		float    y;
		float    width;
		float    height;
		float    minDepth;
		float    maxDepth;
	} VkViewport;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkRect2D.html
	// Provided by VK_VERSION_1_0
	typedef struct VkRect2D {
		VkOffset2D    offset;
		VkExtent2D    extent;
	} VkRect2D;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkOffset2D.html
	// Provided by VK_VERSION_1_0
	typedef struct VkOffset2D {
		int32_t    x;
		int32_t    y;
	} VkOffset2D;

	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkExtent2D.html
	// Provided by VK_VERSION_1_0
	typedef struct VkExtent2D {
		uint32_t    width;
		uint32_t    height;
	} VkExtent2D;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateGraphicsPipelines.html
	// Provided by VK_VERSION_1_0
	VkResult vkCreateGraphicsPipelines(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    uint32_t                                    createInfoCount,
    const VkGraphicsPipelineCreateInfo*         pCreateInfos,
    const VkAllocationCallbacks*                pAllocator,
    VkPipeline*                                 pPipelines);
	
	We can create multiple pipelines.
	The viewport and scissor count members of this structure must be same.
	*/
	VkPipelineViewportStateCreateInfo vkPipelineViewportStateCreateInfo;
	memset((void*)&vkPipelineViewportStateCreateInfo, 0, sizeof(VkPipelineViewportStateCreateInfo));
	vkPipelineViewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	vkPipelineViewportStateCreateInfo.pNext = NULL;
	vkPipelineViewportStateCreateInfo.flags = 0;
	
	////////////////
	vkPipelineViewportStateCreateInfo.viewportCount = 1; //We can specify multiple viewport as array;
	memset((void*)&vkViewPort, 0 , sizeof(VkViewport));
	vkViewPort.x = 0;
	vkViewPort.y = 0;
	vkViewPort.width = (float)vkExtent2D_SwapChain.width;
	vkViewPort.height = (float)vkExtent2D_SwapChain.height;
	
	//done link following parameters with glClearDepth()
	//viewport cha depth max kiti asu shakto deto ithe
	//depth buffer ani viewport cha depth cha sambandh nahi
	vkViewPort.minDepth = 0.0f;
	vkViewPort.maxDepth = 1.0f;
	
	vkPipelineViewportStateCreateInfo.pViewports = &vkViewPort;
	////////////////
	
	////////////////
	vkPipelineViewportStateCreateInfo.scissorCount = 1;
	memset((void*)&vkRect2D_scissor, 0 , sizeof(VkRect2D));
	vkRect2D_scissor.offset.x = 0;
	vkRect2D_scissor.offset.y = 0;
	vkRect2D_scissor.extent.width = vkExtent2D_SwapChain.width;
	vkRect2D_scissor.extent.height = vkExtent2D_SwapChain.height;
	
	vkPipelineViewportStateCreateInfo.pScissors = &vkRect2D_scissor;
	////////////////
	
	/* Depth Stencil State
	As we dont have depth yet, we can omit this step.
	*/
	
	/* Dynamic State
	Those states of PSO, which can be changed dynamically without recreating pipeline.
	ViewPort, Scissor, Depth Bias, Blend constants, Stencil Mask, LineWidth etc are some states which can be changed dynamically.
	We dont have any dynamic state in this code.
	*/
	
	/*
	MultiSampling State
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineMultisampleStateCreateInfo.html
	// Provided by VK_VERSION_1_0
	typedef struct VkPipelineMultisampleStateCreateInfo {
		VkStructureType                          sType;
		const void*                              pNext;
		VkPipelineMultisampleStateCreateFlags    flags;
		VkSampleCountFlagBits                    rasterizationSamples;
		VkBool32                                 sampleShadingEnable;
		float                                    minSampleShading;
		const VkSampleMask*                      pSampleMask;
		VkBool32                                 alphaToCoverageEnable;
		VkBool32                                 alphaToOneEnable;
	} VkPipelineMultisampleStateCreateInfo;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkSampleCountFlagBits.html
	// Provided by VK_VERSION_1_0
	typedef enum VkSampleCountFlagBits {
		VK_SAMPLE_COUNT_1_BIT = 0x00000001,
		VK_SAMPLE_COUNT_2_BIT = 0x00000002,
		VK_SAMPLE_COUNT_4_BIT = 0x00000004,
		VK_SAMPLE_COUNT_8_BIT = 0x00000008,
		VK_SAMPLE_COUNT_16_BIT = 0x00000010,
		VK_SAMPLE_COUNT_32_BIT = 0x00000020,
		VK_SAMPLE_COUNT_64_BIT = 0x00000040,
	} VkSampleCountFlagBits;
	*/
	VkPipelineMultisampleStateCreateInfo vkPipelineMultisampleStateCreateInfo;
	memset((void*)&vkPipelineMultisampleStateCreateInfo, 0, sizeof(VkPipelineMultisampleStateCreateInfo));
	vkPipelineMultisampleStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	vkPipelineMultisampleStateCreateInfo.pNext = NULL;
	vkPipelineMultisampleStateCreateInfo.flags = 0; //Reserved and kept for future use, so 0
	vkPipelineMultisampleStateCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT; // Need to give or validation error will come
	/*
	vkPipelineMultisampleStateCreateInfo.sampleShadingEnable =;
	vkPipelineMultisampleStateCreateInfo.minSampleShading =;
	vkPipelineMultisampleStateCreateInfo.pSampleMask =;
	vkPipelineMultisampleStateCreateInfo.alphaToCoverageEnable =;
	vkPipelineMultisampleStateCreateInfo.alphaToOneEnable =;
	*/
	
	/*
	Shader Stage - 4 stages for tessellation pipeline:
	0: Vertex Shader
	1: Tessellation Control Shader (TCS)
	2: Tessellation Evaluation Shader (TES)
	3: Fragment Shader
	*/
	VkPipelineShaderStageCreateInfo vkPipelineShaderStageCreateInfo_array[4];
	memset((void*)vkPipelineShaderStageCreateInfo_array, 0, sizeof(VkPipelineShaderStageCreateInfo) * 4);

	// Vertex Shader (Stage 0)
	vkPipelineShaderStageCreateInfo_array[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vkPipelineShaderStageCreateInfo_array[0].pNext = NULL;
	vkPipelineShaderStageCreateInfo_array[0].flags = 0;
	vkPipelineShaderStageCreateInfo_array[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	vkPipelineShaderStageCreateInfo_array[0].module = vkShaderMoudule_vertex_shader;
	vkPipelineShaderStageCreateInfo_array[0].pName = "main";
	vkPipelineShaderStageCreateInfo_array[0].pSpecializationInfo = NULL;

	// Tessellation Control Shader (Stage 1)
	vkPipelineShaderStageCreateInfo_array[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vkPipelineShaderStageCreateInfo_array[1].pNext = NULL;
	vkPipelineShaderStageCreateInfo_array[1].flags = 0;
	vkPipelineShaderStageCreateInfo_array[1].stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
	vkPipelineShaderStageCreateInfo_array[1].module = vkShaderModule_tesc;
	vkPipelineShaderStageCreateInfo_array[1].pName = "main";
	vkPipelineShaderStageCreateInfo_array[1].pSpecializationInfo = NULL;

	// Tessellation Evaluation Shader (Stage 2)
	vkPipelineShaderStageCreateInfo_array[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vkPipelineShaderStageCreateInfo_array[2].pNext = NULL;
	vkPipelineShaderStageCreateInfo_array[2].flags = 0;
	vkPipelineShaderStageCreateInfo_array[2].stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
	vkPipelineShaderStageCreateInfo_array[2].module = vkShaderModule_tese;
	vkPipelineShaderStageCreateInfo_array[2].pName = "main";
	vkPipelineShaderStageCreateInfo_array[2].pSpecializationInfo = NULL;

	// Fragment Shader (Stage 3)
	vkPipelineShaderStageCreateInfo_array[3].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vkPipelineShaderStageCreateInfo_array[3].pNext = NULL;
	vkPipelineShaderStageCreateInfo_array[3].flags = 0;
	vkPipelineShaderStageCreateInfo_array[3].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	vkPipelineShaderStageCreateInfo_array[3].module = vkShaderMoudule_fragment_shader;
	vkPipelineShaderStageCreateInfo_array[3].pName = "main";
	vkPipelineShaderStageCreateInfo_array[3].pSpecializationInfo = NULL;

	/*
	Tessellation State - Configure patch control points
	We use 4 control points per patch (quad tessellation)
	*/
	VkPipelineTessellationStateCreateInfo vkPipelineTessellationStateCreateInfo;
	memset((void*)&vkPipelineTessellationStateCreateInfo, 0, sizeof(VkPipelineTessellationStateCreateInfo));
	vkPipelineTessellationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
	vkPipelineTessellationStateCreateInfo.pNext = NULL;
	vkPipelineTessellationStateCreateInfo.flags = 0;
	vkPipelineTessellationStateCreateInfo.patchControlPoints = 3; // Triangle patches with 3 control points
	
	/*
	As pipelines are created from pipeline cache, we will now create pipeline cache object.
	Not in red book. But in spec.
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCacheCreateInfo.html
	// Provided by VK_VERSION_1_0
	typedef struct VkPipelineCacheCreateInfo {
		VkStructureType               sType;
		const void*                   pNext;
		VkPipelineCacheCreateFlags    flags;
		size_t                        initialDataSize;
		const void*                   pInitialData;
	} VkPipelineCacheCreateInfo;
	*/
	VkPipelineCacheCreateInfo vkPipelineCacheCreateInfo;
	memset((void*)&vkPipelineCacheCreateInfo, 0, sizeof(VkPipelineCacheCreateInfo));
	vkPipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	vkPipelineCacheCreateInfo.pNext = NULL;
	vkPipelineCacheCreateInfo.flags = 0;
	/*
	vkPipelineCacheCreateInfo.initialDataSize =;
	vkPipelineCacheCreateInfo.pInitialData =;
	*/
	
	/*
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreatePipelineCache.html
	// Provided by VK_VERSION_1_0
	VkResult vkCreatePipelineCache(
    VkDevice                                    device,
    const VkPipelineCacheCreateInfo*            pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkPipelineCache*                            pPipelineCache);
	*/
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
	
	/*
	Create actual graphics pipeline
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkGraphicsPipelineCreateInfo.html
	// Provided by VK_VERSION_1_0
	typedef struct VkGraphicsPipelineCreateInfo {
		VkStructureType                                  sType;
		const void*                                      pNext;
		VkPipelineCreateFlags                            flags;
		uint32_t                                         stageCount;
		const VkPipelineShaderStageCreateInfo*           pStages;
		const VkPipelineVertexInputStateCreateInfo*      pVertexInputState;
		const VkPipelineInputAssemblyStateCreateInfo*    pInputAssemblyState;
		const VkPipelineTessellationStateCreateInfo*     pTessellationState;
		const VkPipelineViewportStateCreateInfo*         pViewportState;
		const VkPipelineRasterizationStateCreateInfo*    pRasterizationState;
		const VkPipelineMultisampleStateCreateInfo*      pMultisampleState;
		const VkPipelineDepthStencilStateCreateInfo*     pDepthStencilState;
		const VkPipelineColorBlendStateCreateInfo*       pColorBlendState;
		const VkPipelineDynamicStateCreateInfo*          pDynamicState;
		VkPipelineLayout                                 layout;
		VkRenderPass                                     renderPass;
		uint32_t                                         subpass;
		VkPipeline                                       basePipelineHandle;
		int32_t                                          basePipelineIndex;
	} VkGraphicsPipelineCreateInfo;
	*/
	VkGraphicsPipelineCreateInfo vkGraphicsPipelineCreateInfo;
	memset((void*)&vkGraphicsPipelineCreateInfo, 0, sizeof(VkGraphicsPipelineCreateInfo));
	vkGraphicsPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	vkGraphicsPipelineCreateInfo.pNext = NULL;
	vkGraphicsPipelineCreateInfo.flags = 0;
	vkGraphicsPipelineCreateInfo.stageCount = 4; // Vertex, TCS, TES, Fragment
	vkGraphicsPipelineCreateInfo.pStages = vkPipelineShaderStageCreateInfo_array;
	vkGraphicsPipelineCreateInfo.pVertexInputState = &vkPipelineVertexInputStateCreateInfo;
	vkGraphicsPipelineCreateInfo.pInputAssemblyState = &vkPipelineInputAssemblyStateCreateInfo;
	vkGraphicsPipelineCreateInfo.pTessellationState = &vkPipelineTessellationStateCreateInfo; // Enable tessellation
	vkGraphicsPipelineCreateInfo.pViewportState = &vkPipelineViewportStateCreateInfo; //5
	vkGraphicsPipelineCreateInfo.pRasterizationState = &vkPipelineRasterizationStateCreateInfo; //3
	vkGraphicsPipelineCreateInfo.pMultisampleState = &vkPipelineMultisampleStateCreateInfo; //8
	//vkGraphicsPipelineCreateInfo.pDepthStencilState = NULL; //6
	
	/*
	// Provided by VK_VERSION_1_0
	typedef struct VkPipelineDepthStencilStateCreateInfo {
		VkStructureType                           sType;
		const void*                               pNext;
		VkPipelineDepthStencilStateCreateFlags    flags;
		VkBool32                                  depthTestEnable;
		VkBool32                                  depthWriteEnable;
		VkCompareOp                               depthCompareOp;
		VkBool32                                  depthBoundsTestEnable;
		VkBool32                                  stencilTestEnable;
		VkStencilOpState                          front;
		VkStencilOpState                          back;
		float                                     minDepthBounds;
		float                                     maxDepthBounds;
	} VkPipelineDepthStencilStateCreateInfo;
	*/
	VkPipelineDepthStencilStateCreateInfo vkPipelineDepthStencilStateCreateInfo;
	memset((void*)&vkPipelineDepthStencilStateCreateInfo, 0, sizeof(VkPipelineDepthStencilStateCreateInfo));
	vkPipelineDepthStencilStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	vkPipelineDepthStencilStateCreateInfo.pNext = NULL;
	vkPipelineDepthStencilStateCreateInfo.flags = 0;
	vkPipelineDepthStencilStateCreateInfo.depthTestEnable = VK_TRUE;
	vkPipelineDepthStencilStateCreateInfo.depthWriteEnable= VK_TRUE; 
	vkPipelineDepthStencilStateCreateInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkCompareOp.html
	vkPipelineDepthStencilStateCreateInfo.depthBoundsTestEnable= VK_FALSE;
	vkPipelineDepthStencilStateCreateInfo.stencilTestEnable = VK_FALSE;
	//vkPipelineDepthStencilStateCreateInfo.minDepthBounds = ;
	//vkPipelineDepthStencilStateCreateInfo.maxDepthBounds= ;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkStencilOpState.html
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkStencilOp.html
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkCompareOp.html
	vkPipelineDepthStencilStateCreateInfo.back.failOp = VK_STENCIL_OP_KEEP; 
	vkPipelineDepthStencilStateCreateInfo.back.passOp = VK_STENCIL_OP_KEEP;
	vkPipelineDepthStencilStateCreateInfo.back.compareOp = VK_COMPARE_OP_ALWAYS; // one of 8 tests 
	//vkPipelineDepthStencilStateCreateInfo.back.depthFailOp = ;
	//vkPipelineDepthStencilStateCreateInfo.back.compareMask = ;
	//vkPipelineDepthStencilStateCreateInfo.back.writeMask = ;
	//vkPipelineDepthStencilStateCreateInfo.back.reference = ;
	
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkStencilOpState.html
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkStencilOp.html
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkCompareOp.html
	vkPipelineDepthStencilStateCreateInfo.front = vkPipelineDepthStencilStateCreateInfo.back; 
	
	vkGraphicsPipelineCreateInfo.pDepthStencilState = &vkPipelineDepthStencilStateCreateInfo; //6
	
	
	vkGraphicsPipelineCreateInfo.pColorBlendState = &vkPipelineColorBlendStateCreateInfo; //4
	vkGraphicsPipelineCreateInfo.pDynamicState = NULL; //7
	vkGraphicsPipelineCreateInfo.layout = vkPipelineLayout; //11
	vkGraphicsPipelineCreateInfo.renderPass = vkRenderPass; //12
	vkGraphicsPipelineCreateInfo.subpass = 0; //13. 0 as no subpass as wehave only 1 renderpass and its default subpass(In Redbook)
	vkGraphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
	vkGraphicsPipelineCreateInfo.basePipelineIndex = 0;
	
	/*
	Now create the pipeline
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateGraphicsPipelines.html
	// Provided by VK_VERSION_1_0
	VkResult vkCreateGraphicsPipelines(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    uint32_t                                    createInfoCount,
    const VkGraphicsPipelineCreateInfo*         pCreateInfos,
    const VkAllocationCallbacks*                pAllocator,
    VkPipeline*                                 pPipelines);
	*/
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
	
	/*
	We are done with pipeline cache . So destroy it
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyPipelineCache.html
	// Provided by VK_VERSION_1_0
	void vkDestroyPipelineCache(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    const VkAllocationCallbacks*                pAllocator);
	*/
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
	//Variable declarations	
	VkResult vkResult = VK_SUCCESS;
	
	/*
	Code
	*/	
	vkFramebuffer_array = (VkFramebuffer*)malloc(sizeof(VkFramebuffer) * swapchainImageCount);
	//for sake of brevity, no error checking
	
	for(uint32_t i = 0 ; i < swapchainImageCount; i++)
	{
		/*
		1. Declare an array of VkImageView (https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageView.html) equal to number of attachments i.e in our example array of member.
		*/
		VkImageView vkImageView_attachment_array[2]; //was 1, made to 2 madhe for depth 
		memset((void*)vkImageView_attachment_array, 0, sizeof(VkImageView) * _ARRAYSIZE(vkImageView_attachment_array));
		
		/*
		2. Declare and initialize VkFramebufferCreateInfo structure (https://registry.khronos.org/vulkan/specs/latest/man/html/VkFramebufferCreateInfo.html).
		Allocate the framebuffer array by malloc eqal size to swapchainImageCount.
		 Start loop for  swapchainImageCount and call vkCreateFramebuffer() (https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateFramebuffer.html) to create framebuffers.
		*/
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
	//Variable declarations	
	VkResult vkResult = VK_SUCCESS;
	
	/*
	Code
	*/
	
	/*
	18_2. In CreateSemaphore() UDF(User defined function) , declare, memset and initialize VkSemaphoreCreateInfo  struct (https://registry.khronos.org/vulkan/specs/latest/man/html/VkSemaphoreCreateInfo.html)
	*/
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkSemaphoreCreateInfo.html
	VkSemaphoreCreateInfo vkSemaphoreCreateInfo;
	memset((void*)&vkSemaphoreCreateInfo, 0, sizeof(VkSemaphoreCreateInfo));
	vkSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	vkSemaphoreCreateInfo.pNext = NULL; //If no type is specified , the type of semaphore created is binary semaphore
	vkSemaphoreCreateInfo.flags = 0; //must be 0 as reserved
	
	/*
	18_3. Now call vkCreateSemaphore() {https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateSemaphore.html} 2 times to create our 2 semaphore objects.
    Remember both will use same  VkSemaphoreCreateInfo struct as defined in 2nd step.
	*/
	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateSemaphore.html
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

	//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateSemaphore.html
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

	// Create Timeline Semaphore for CUDA-Vulkan synchronization
	// Timeline semaphores allow for more efficient GPU-GPU synchronization
	VkSemaphoreTypeCreateInfo vkSemaphoreTypeCreateInfo;
	memset((void*)&vkSemaphoreTypeCreateInfo, 0, sizeof(VkSemaphoreTypeCreateInfo));
	vkSemaphoreTypeCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
	vkSemaphoreTypeCreateInfo.pNext = NULL;
	vkSemaphoreTypeCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	vkSemaphoreTypeCreateInfo.initialValue = 0;

	// Export info for CUDA interop - allows CUDA to import this semaphore
	VkExportSemaphoreCreateInfo vkExportSemaphoreCreateInfo;
	memset((void*)&vkExportSemaphoreCreateInfo, 0, sizeof(VkExportSemaphoreCreateInfo));
	vkExportSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
	vkExportSemaphoreCreateInfo.pNext = &vkSemaphoreTypeCreateInfo;
	vkExportSemaphoreCreateInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

	VkSemaphoreCreateInfo vkTimelineSemaphoreCreateInfo;
	memset((void*)&vkTimelineSemaphoreCreateInfo, 0, sizeof(VkSemaphoreCreateInfo));
	vkTimelineSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	vkTimelineSemaphoreCreateInfo.pNext = &vkExportSemaphoreCreateInfo;
	vkTimelineSemaphoreCreateInfo.flags = 0;

	vkResult = vkCreateSemaphore(vkDevice, &vkTimelineSemaphoreCreateInfo, NULL, &vkSemaphore_timeline);
	if (vkResult != VK_SUCCESS)
	{
		FileIO("CreateSemaphores(): vkCreateSemaphore() for timeline semaphore failed with error code %d\n", vkResult);
		// Non-fatal: timeline semaphores are optional optimization
		bTimelineSemaphoreSupported = FALSE;
		FileIO("CreateSemaphores(): Continuing without timeline semaphore (CUDA-Vulkan sync will use device sync)\n");
	}
	else
	{
		bTimelineSemaphoreSupported = TRUE;
		timelineSemaphoreValue = 0;
		FileIO("CreateSemaphores(): Timeline semaphore created successfully for CUDA-Vulkan interop\n");
	}

	return vkResult;
}

VkResult CreateFences(void)
{
	//Variable declarations	
	VkResult vkResult = VK_SUCCESS;
	
	/*
	Code
	*/
	
	/*
	18_4. In CreateFences() UDF(User defined function) declare, memset and initialize VkFenceCreateInfo struct (https://registry.khronos.org/vulkan/specs/latest/man/html/VkFenceCreateInfo.html).
	*/
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkFenceCreateInfo.html
	VkFenceCreateInfo  vkFenceCreateInfo;
	memset((void*)&vkFenceCreateInfo, 0, sizeof(VkFenceCreateInfo));
	vkFenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	vkFenceCreateInfo.pNext = NULL;
	vkFenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkFenceCreateFlagBits.html
	
	/*
	18_5. In this function, CreateFences() allocate our global fence array to size of swapchain image count using malloc.
	*/
	vkFence_array = (VkFence*)malloc(sizeof(VkFence) * swapchainImageCount);
	//error checking skipped due to brevity
	
	/*
	18_6. Now in a loop, call vkCreateFence() {https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateFence.html} to initialize our global fences array.
	*/
	for(uint32_t i =0; i < swapchainImageCount; i++)
	{
		//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateFence.html
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
	//Variable declarations	
	VkResult vkResult = VK_SUCCESS;
	
	/*
	Command Buffer
	*/
	//https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandBuffer.html
	VkCommandBuffer *vkCommandBuffer_array = NULL;
	
	/*
	Code
	*/

	if(bMesh1024_chosen == TRUE)
	{
		vkCommandBuffer_array = vkCommandBuffer_for_1024_x_1024_graphics_array;
	}

	/*
	1. Start a loop with swapchainImageCount as counter.
	   loop per swapchainImage
	*/
	for(uint32_t i =0; i< swapchainImageCount; i++)
	{
		/*
		2. Inside loop, call vkResetCommandBuffer to reset contents of command buffers.
		0 says dont release resource created by command pool for these command buffers, because we may reuse
		*/
		vkResult = vkResetCommandBuffer(vkCommandBuffer_array[i], 0);
		if (vkResult != VK_SUCCESS)
		{
			FileIO("buildCommandBuffers(): vkResetCommandBuffer() function failed with error code %d at %d iteration\n", vkResult, i);
			return vkResult;
		}
		/*
		else
		{
			FileIO("buildCommandBuffers(): vkResetCommandBuffer() succedded at %d iteration\n", i);
		}
		*/
		
		/*
		3. Then declare, memset and initialize VkCommandBufferBeginInfo struct.
		*/
		VkCommandBufferBeginInfo vkCommandBufferBeginInfo;
		memset((void*)&vkCommandBufferBeginInfo, 0, sizeof(VkCommandBufferBeginInfo));
		vkCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		vkCommandBufferBeginInfo.pNext = NULL;
		vkCommandBufferBeginInfo.flags = 0; 
		
		/*
		pInheritanceInfo is a pointer to a VkCommandBufferInheritanceInfo structure, used if commandBuffer is a secondary command buffer. If this is a primary command buffer, then this value is ignored.
		We are not going to use this command buffer simultaneouly between multiple threads.
		*/
		vkCommandBufferBeginInfo.pInheritanceInfo = NULL;
		
		/*
		4. Call vkBeginCommandBuffer() to record different Vulkan drawing related commands.
		Do Error Checking.
		*/
		vkResult = vkBeginCommandBuffer(vkCommandBuffer_array[i], &vkCommandBufferBeginInfo);
		if (vkResult != VK_SUCCESS)
		{
			FileIO("buildCommandBuffers(): vkBeginCommandBuffer() function failed with error code %d at %d iteration\n", vkResult, i);
			return vkResult;
		}
		/*
		else
		{
			FileIO("buildCommandBuffers(): vkBeginCommandBuffer() succedded at %d iteration\n", i);
		}
		*/
		
		/*
		5. Declare, memset and initialize struct array of VkClearValue type
		*/
		VkClearValue vkClearValue_array[2]; //https://registry.khronos.org/vulkan/specs/latest/man/html/VkClearValue.html
		memset((void*)vkClearValue_array, 0, sizeof(VkClearValue) * _ARRAYSIZE(vkClearValue_array));
		if(colorFromKey == 'K')
		{
			VkClearColorValue vkClearColorValue_white  = { 1.0f, 1.0f, 1.0f, 1.0f };
			vkClearValue_array[0].color = vkClearColorValue_white;
		}
		else
		{
			vkClearValue_array[0].color = vkClearColorValue;
		}
		vkClearValue_array[1].depthStencil = vkClearDepthStencilValue;
		
		/*
		6. Then declare , memset and initialize VkRenderPassBeginInfo struct.
		*/
		VkRenderPassBeginInfo vkRenderPassBeginInfo;
		memset((void*)&vkRenderPassBeginInfo, 0, sizeof(VkRenderPassBeginInfo));
		vkRenderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		vkRenderPassBeginInfo.pNext = NULL;
		vkRenderPassBeginInfo.renderPass = vkRenderPass;
		
		//https://registry.khronos.org/vulkan/specs/latest/man/html/VkRect2D.html
		//https://registry.khronos.org/vulkan/specs/latest/man/html/VkOffset2D.html
		//https://registry.khronos.org/vulkan/specs/latest/man/html/VkExtent2D.html
		//THis is like D3DViewport/glViewPort
		vkRenderPassBeginInfo.renderArea.offset.x = 0;
		vkRenderPassBeginInfo.renderArea.offset.y = 0;
		vkRenderPassBeginInfo.renderArea.extent.width = vkExtent2D_SwapChain.width;	
		vkRenderPassBeginInfo.renderArea.extent.height = vkExtent2D_SwapChain.height;	
		
		vkRenderPassBeginInfo.clearValueCount = _ARRAYSIZE(vkClearValue_array);
		vkRenderPassBeginInfo.pClearValues = vkClearValue_array;
		
		vkRenderPassBeginInfo.framebuffer = vkFramebuffer_array[i];
		
		/*
		7. Begin RenderPass by vkCmdBeginRenderPass() API.
		Remember, the code writtrn inside "BeginRenderPass" and "EndRenderPass" itself is code for subpass , if no subpass is explicitly created.
		In other words even if no subpass is declared explicitly , there is one subpass for renderpass.
		
		//https://registry.khronos.org/vulkan/specs/latest/man/html/VkSubpassContents.html
		//VK_SUBPASS_CONTENTS_INLINE specifies that the contents of the subpass will be recorded inline in the primary command buffer, and secondary command buffers must not be executed within the subpass.
		*/
		vkCmdBeginRenderPass(vkCommandBuffer_array[i], &vkRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE); 
		
		/*
		Bind with the pipeline
		//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindPipeline.html
		// Provided by VK_VERSION_1_0
		void vkCmdBindPipeline(
			VkCommandBuffer                             commandBuffer,
			VkPipelineBindPoint                         pipelineBindPoint,
			VkPipeline                                  pipeline);
			
		//https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineBindPoint.html
		// Provided by VK_VERSION_1_0
		typedef enum VkPipelineBindPoint {
			VK_PIPELINE_BIND_POINT_GRAPHICS = 0,
			VK_PIPELINE_BIND_POINT_COMPUTE = 1,
		#ifdef VK_ENABLE_BETA_EXTENSIONS
		  // Provided by VK_AMDX_shader_enqueue
			VK_PIPELINE_BIND_POINT_EXECUTION_GRAPH_AMDX = 1000134000,
		#endif
		  // Provided by VK_KHR_ray_tracing_pipeline
			VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR = 1000165000,
		  // Provided by VK_HUAWEI_subpass_shading
			VK_PIPELINE_BIND_POINT_SUBPASS_SHADING_HUAWEI = 1000369003,
		  // Provided by VK_NV_ray_tracing
			VK_PIPELINE_BIND_POINT_RAY_TRACING_NV = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
		} VkPipelineBindPoint;
		*/
		vkCmdBindPipeline(vkCommandBuffer_array[i], VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipeline);
		
		
		/*
		Bind our descriptor set with pipeline
		//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindDescriptorSets.html
		// Provided by VK_VERSION_1_0
		void vkCmdBindDescriptorSets(
		VkCommandBuffer                             commandBuffer,
		VkPipelineBindPoint                         pipelineBindPoint,
		VkPipelineLayout                            layout,
		uint32_t                                    firstSet,
		uint32_t                                    descriptorSetCount,
		const VkDescriptorSet*                      pDescriptorSets,
		uint32_t                                    dynamicOffsetCount, // Used for dynamic shader stages
		const uint32_t*                             pDynamicOffsets); // Used for dynamic shader stages
		*/
		vkCmdBindDescriptorSets(vkCommandBuffer_array[i], VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipelineLayout, 0, 1, &vkDescriptorSet, 0, NULL);
		
		/*
		Bind with vertex buffer
		//https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceSize.html
		
		//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBindVertexBuffers.html
		// Provided by VK_VERSION_1_0
		void vkCmdBindVertexBuffers(
			VkCommandBuffer                             commandBuffer,
			uint32_t                                    firstBinding,
			uint32_t                                    bindingCount,
			const VkBuffer*                             pBuffers,
			const VkDeviceSize*                         pOffsets);
		*/
		VkDeviceSize vkDeviceSize_offset_array[1];
		memset((void*)vkDeviceSize_offset_array, 0, sizeof(VkDeviceSize) * _ARRAYSIZE(vkDeviceSize_offset_array));
		
		if(bMesh1024_chosen == TRUE)
		{
			vkCmdBindVertexBuffers(vkCommandBuffer_array[i], 0, 1, &vertexData_external.vkBuffer, vkDeviceSize_offset_array);
		}

		/*
		Here we should call Vulkan drawing functions.
		*/

		/*
		//https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDraw.html
		void vkCmdDraw(
		VkCommandBuffer                             commandBuffer,
		uint32_t                                    vertexCount,
		uint32_t                                    instanceCount,
		uint32_t                                    firstVertex,
		uint32_t                                    firstInstance); //0th index cha instance
		*/
		if(bMesh1024_chosen == TRUE)
		{
			/*
			//https://registry.khronos.org/VulkanSC/specs/1.0-extensions/man/html/vkCmdDrawIndirect.html
			// Provided by VK_VERSION_1_0
			void vkCmdDrawIndirect(
			VkCommandBuffer                             commandBuffer,
			VkBuffer                                    buffer,
			VkDeviceSize                                offset,
			uint32_t                                    drawCount,
			uint32_t                                    stride);
			*/
			vkCmdDrawIndirect(vkCommandBuffer_array[i], vertexdata_indirect_buffer.vkBuffer, 0, 1, 0);
		}

		/*
		8. End the renderpass by calling vkCmdEndRenderpass.
		*/
		vkCmdEndRenderPass(vkCommandBuffer_array[i]);
		
		/*
		9. End the recording of commandbuffer by calling vkEndCommandBuffer() API.
		*/
		vkResult = vkEndCommandBuffer(vkCommandBuffer_array[i]);
		if (vkResult != VK_SUCCESS)
		{
			FileIO("buildCommandBuffers(): vkEndCommandBuffer() function failed with error code %d at %d iteration\n", vkResult, i);
			return vkResult;
		}
		/*
		else
		{
			FileIO("buildCommandBuffers(): vkEndCommandBuffer() succedded at %d iteration\n", i);
		}
		*/
		
		/*
		10. Close the loop.
		*/
	}
	
	return vkResult;
}


/*
VKAPI_ATTR VkBOOL32 VKAPI_CALL debugReportCallback(
	VkDebugReportFlagsEXT vkDebugReportFlagsEXT, //which flags gave this callback
	VkDebugReportObjectTypeEXT vkDebugReportObjectTypeEXT, //jyana ha callback trigger kela , tya object cha type
	uint64_t object, //Proper object
	size_t location,  //warning/error kutha aali tyacha location
	int32_t messageCode, // message cha id -> message code in hex 
	const char* pLayerPrefix, // kontya layer na ha dila (Purvi 5 layer hote, aata ek kila. So ekach yeil atta)
	const char* pMessage, //actual error message
	void* pUserData) //jar tumhi callback function la kahi parameter pass kela asel tar
{
	//Code
	FileIO("Anjaneya_VALIDATION:debugReportCallback():%s(%d) = %s\n", pLayerPrefix, messageCode, pMessage);  
    return (VK_FALSE);
}
*/

VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallback(VkDebugReportFlagsEXT vkDebugReportFlagsEXT, VkDebugReportObjectTypeEXT vkDebugReportObjectTypeEXT, uint64_t object, size_t location,  int32_t messageCode,const char* pLayerPrefix, const char* pMessage, void* pUserData)
{
	//Code
	FileIO("Anjaneya_VALIDATION:debugReportCallback():%s(%d) = %s\n", pLayerPrefix, messageCode, pMessage);  
    return (VK_FALSE);
}
