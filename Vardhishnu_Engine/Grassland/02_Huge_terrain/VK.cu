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
#include "QuaternionCamera.h"

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

#pragma comment(lib, "vulkan-1.lib")

#include <cuda_runtime.h>

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

const char* gpszAppName = "ARTR";

HWND ghwnd = NULL;
BOOL gbActive = FALSE;
DWORD dwStyle = 0;

WINDOWPLACEMENT wpPrev;
BOOL gbFullscreen = FALSE;
BOOL bWindowMinimize = FALSE;

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

char colorFromKey = 'O';
float animationTime = 0.0f;

// Quaternion-based camera system
QuaternionCamera g_camera;

// Mouse tracking for camera rotation
POINT g_lastMousePos = {0, 0};
bool g_mouseCapture = false;
bool g_firstMouse = true;

cudaError_t cudaResult;

VkExternalMemoryHandleTypeFlagBits vkExternalMemoryHandleTypeFlagBits;

cudaExternalMemory_t cuExternalMemory_t;

void* pos_CUDA = NULL;

VertexData vertexData_external;

VertexData vertexdata_indirect_buffer;
BOOL bIndirectBufferMemoryCoherent = TRUE;

int iResult = 0;

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
				if (g_mouseCapture)
				{
					ReleaseCapture();
					ShowCursor(TRUE);
					g_mouseCapture = false;
				}
				DestroyWindow(hwnd);
				break;

			case 0x52: // R key - reset camera
				FileIO("WndProc() 0x52 pressed - Reset camera.\n");
				g_camera.reset();
				break;

			// W - Move forward
			case 0x57:
				g_camera.moveForward(1.0f);
				break;

			// S - Move backward
			case 0x53:
				g_camera.moveBackward(1.0f);
				break;

			// A - Move left (strafe)
			case 0x41:
				g_camera.moveLeft(1.0f);
				break;

			// D - Move right (strafe)
			case 0x44:
				g_camera.moveRight(1.0f);
				break;

			// Q - Move up
			case 0x51:
				g_camera.moveUp(1.0f);
				break;

			// E - Move down
			case 0x45:
				g_camera.moveDown(1.0f);
				break;

			// Arrow keys for rotation
			case VK_UP:
				g_camera.pitch(-0.02f);
				break;

			case VK_DOWN:
				g_camera.pitch(0.02f);
				break;

			case VK_LEFT:
				g_camera.yaw(0.02f);
				break;

			case VK_RIGHT:
				g_camera.yaw(-0.02f);
				break;

			// Z - Roll left
			case 0x5A:
				if (g_camera.getMode() == CAMERA_MODE_FREE)
					g_camera.roll(-0.02f);
				break;

			// X - Roll right
			case 0x58:
				if (g_camera.getMode() == CAMERA_MODE_FREE)
					g_camera.roll(0.02f);
				break;

			// F1 - FPS camera mode
			case VK_F1:
				g_camera.setMode(CAMERA_MODE_FPS);
				FileIO("WndProc() Camera mode: FPS\n");
				break;

			// F2 - Free-fly camera mode
			case VK_F2:
				g_camera.setMode(CAMERA_MODE_FREE);
				FileIO("WndProc() Camera mode: Free-fly\n");
				break;

			// F3 - Orbit camera mode
			case VK_F3:
				g_camera.setMode(CAMERA_MODE_ORBIT);
				g_camera.setOrbitTarget(glm::vec3(0.0f, 0.0f, 0.0f));
				FileIO("WndProc() Camera mode: Orbit\n");
				break;

			// + / = - Increase movement speed
			case VK_OEM_PLUS:  // 0xBB
				g_camera.setMoveSpeed(g_camera.getMoveSpeed() + 1.0f);
				FileIO("WndProc() Camera speed: %.1f\n", g_camera.getMoveSpeed());
				break;

			// - - Decrease movement speed
			case VK_OEM_MINUS:  // 0xBD
				if (g_camera.getMoveSpeed() > 1.0f)
					g_camera.setMoveSpeed(g_camera.getMoveSpeed() - 1.0f);
				FileIO("WndProc() Camera speed: %.1f\n", g_camera.getMoveSpeed());
				break;
			}
			break;

		// Mouse button handling for camera control
		case WM_LBUTTONDOWN:
			SetCapture(hwnd);
			ShowCursor(FALSE);
			GetCursorPos(&g_lastMousePos);
			g_mouseCapture = true;
			g_firstMouse = true;
			break;

		case WM_LBUTTONUP:
			if (g_mouseCapture)
			{
				ReleaseCapture();
				ShowCursor(TRUE);
				g_mouseCapture = false;
			}
			break;

		case WM_MOUSEMOVE:
			if (g_mouseCapture)
			{
				POINT currentPos;
				GetCursorPos(&currentPos);

				if (g_firstMouse)
				{
					g_lastMousePos = currentPos;
					g_firstMouse = false;
				}
				else
				{
					float deltaX = (float)(currentPos.x - g_lastMousePos.x);
					float deltaY = (float)(currentPos.y - g_lastMousePos.y);

					if (g_camera.getMode() == CAMERA_MODE_ORBIT)
					{
						g_camera.orbit(deltaX * 0.005f, deltaY * 0.005f);
					}
					else
					{
						g_camera.processMouseMovement(deltaX, deltaY);
					}

					// Reset cursor to center to allow continuous rotation
					RECT rect;
					GetWindowRect(hwnd, &rect);
					int centerX = (rect.left + rect.right) / 2;
					int centerY = (rect.top + rect.bottom) / 2;
					SetCursorPos(centerX, centerY);
					g_lastMousePos.x = centerX;
					g_lastMousePos.y = centerY;
				}
			}
			break;

		// Mouse wheel for zoom
		case WM_MOUSEWHEEL:
			{
				short zDelta = GET_WHEEL_DELTA_WPARAM(wParam);
				g_camera.zoom((float)zDelta / 120.0f);
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

			case '1':
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

	// Initialize quaternion camera
	g_camera.setPosition(0.0f, 30.0f, 150.0f);
	g_camera.setMode(CAMERA_MODE_FPS);
	g_camera.setMoveSpeed(5.0f);
	g_camera.setRotationSpeed(0.005f);
	g_camera.setPerspective(glm::radians(45.0f), (float)WIN_WIDTH / (float)WIN_HEIGHT, 0.1f, 2000.0f);
	g_camera.lookAt(glm::vec3(0.0f, 0.0f, 0.0f)); // Look at center of terrain
	FileIO("initialize(): Quaternion camera initialized\n");

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

	// Update camera aspect ratio if window size changed
	g_camera.setAspectRatio((float)winWidth / (float)winHeight);

	// Get view matrix from quaternion camera
	myUniformData.viewMatrix = g_camera.getViewMatrix();

	// Get Vulkan-adjusted projection matrix (Y-flipped for Vulkan coordinate system)
	myUniformData.projectionMatrix = g_camera.getProjectionMatrixVulkan();

	if(colorFromKey == 'K')
	{
		myUniformData.color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	}
	else
	{
		if(colorFromKey == 'R')
		{
			myUniformData.color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
		}
		else if(colorFromKey == 'G')
		{
			myUniformData.color = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
		}
		else if(colorFromKey == 'B')
		{
			myUniformData.color = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
		}
		else if(colorFromKey == 'C')
		{
			myUniformData.color = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);
		}
		else if(colorFromKey == 'M')
		{
			myUniformData.color = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
		}
		else if(colorFromKey == 'Y')
		{
			myUniformData.color = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
		}
		else if(colorFromKey == 'W')
		{
			myUniformData.color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
		}
		else if(colorFromKey == 'O')
		{
			myUniformData.color = glm::vec4(1.0f, 0.647f, 0.0f, 1.0f);
		}
		else
		{
			myUniformData.color = glm::vec4(1.0f, 0.647f, 0.0f, 1.0f);
		}
	}

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

	dim3 block(8, 8, 1);
	dim3 grid((PATCH_GRID_SIZE + block.x - 1) / block.x, (PATCH_GRID_SIZE + block.y - 1) / block.y, 1);
	generateFlatTerrainPatches<<<grid, block>>>((float4*)pos_CUDA, PATCH_GRID_SIZE, TERRAIN_SIZE);

	cudaResult = cudaGetLastError();
	if(cudaResult != CUDA_SUCCESS)
	{
		FileIO("display(): kernel failed: %s\n", cudaGetErrorString(cudaResult));
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
	}

	cudaResult = cudaDeviceSynchronize();
	if(cudaResult != CUDA_SUCCESS)
	{
		FileIO("display(): cudaDeviceSynchronize failed\n");
		vkResult = VK_ERROR_INITIALIZATION_FAILED;
		return vkResult;
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

	VkDescriptorSetLayoutBinding vkDescriptorSetLayoutBinding;
	memset((void*)&vkDescriptorSetLayoutBinding, 0, sizeof(VkDescriptorSetLayoutBinding));

	vkDescriptorSetLayoutBinding.binding = 0;
	vkDescriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	vkDescriptorSetLayoutBinding.descriptorCount = 1;
	vkDescriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT|VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
	vkDescriptorSetLayoutBinding.pImmutableSamplers = NULL;

	VkDescriptorSetLayoutCreateInfo vkDescriptorSetLayoutCreateInfo;
	memset((void*)&vkDescriptorSetLayoutCreateInfo, 0, sizeof(VkDescriptorSetLayoutCreateInfo));
	vkDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	vkDescriptorSetLayoutCreateInfo.pNext = NULL;
	vkDescriptorSetLayoutCreateInfo.flags = 0;

	vkDescriptorSetLayoutCreateInfo.bindingCount = 1;
	vkDescriptorSetLayoutCreateInfo.pBindings = &vkDescriptorSetLayoutBinding;

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

	VkDescriptorPoolSize vkDescriptorPoolSize;
	memset((void*)&vkDescriptorPoolSize, 0, sizeof(VkDescriptorPoolSize));
	vkDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	vkDescriptorPoolSize.descriptorCount = 1;

	VkDescriptorPoolCreateInfo vkDescriptorPoolCreateInfo;
	memset((void*)&vkDescriptorPoolCreateInfo, 0, sizeof(VkDescriptorPoolCreateInfo));
	vkDescriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	vkDescriptorPoolCreateInfo.pNext = NULL;
	vkDescriptorPoolCreateInfo.flags = 0;
	vkDescriptorPoolCreateInfo.maxSets = 1;
	vkDescriptorPoolCreateInfo.poolSizeCount =  1;
	vkDescriptorPoolCreateInfo.pPoolSizes = &vkDescriptorPoolSize;

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

	VkDescriptorBufferInfo vkDescriptorBufferInfo;
	memset((void*)&vkDescriptorBufferInfo, 0, sizeof(VkDescriptorBufferInfo));
	vkDescriptorBufferInfo.buffer = uniformData.vkBuffer;
	vkDescriptorBufferInfo.offset = 0;
	vkDescriptorBufferInfo.range = sizeof(struct MyUniformData);

	VkWriteDescriptorSet vkWriteDescriptorSet;
	memset((void*)&vkWriteDescriptorSet, 0, sizeof(VkWriteDescriptorSet));
	vkWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	vkWriteDescriptorSet.pNext = NULL;
	vkWriteDescriptorSet.dstSet = vkDescriptorSet;
	vkWriteDescriptorSet.dstBinding = 0;
	vkWriteDescriptorSet.dstArrayElement = 0;
	vkWriteDescriptorSet.descriptorCount = 1;
	vkWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	vkWriteDescriptorSet.pImageInfo = NULL;
	vkWriteDescriptorSet.pBufferInfo =  &vkDescriptorBufferInfo;
	vkWriteDescriptorSet.pTexelBufferView = NULL;

	vkUpdateDescriptorSets(vkDevice, 1, &vkWriteDescriptorSet, 0, NULL);

	FileIO("CreateDescriptorSet(): vkUpdateDescriptorSets() succedded\n");

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

	vkPipelineRasterizationStateCreateInfo.polygonMode = VK_POLYGON_MODE_LINE;
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