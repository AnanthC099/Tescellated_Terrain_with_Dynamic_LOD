cls

del Vk.exe Log.txt Shader.vert.spv Shader.frag.spv

glslangValidator.exe -V -H -o Shader.vert.spv Shader.vert

glslangValidator.exe -V -H -o Shader.frag.spv Shader.frag

rc.exe Vk.rc

nvcc.exe -allow-unsupported-compiler -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\include" -I"C:\VulkanSDK\Anjaneya\Include" -L"C:\VulkanSDK\Anjaneya\Lib" -o Vk.exe  vulkan-1.lib User32.lib Gdi32.lib kernel32.lib Vk.res Vk.cu -diag-suppress 20012 -diag-suppress 20013 -diag-suppress 20014 -diag-suppress 20015

del Vk.obj Vk.pdb Vk.res 

Vk.exe

