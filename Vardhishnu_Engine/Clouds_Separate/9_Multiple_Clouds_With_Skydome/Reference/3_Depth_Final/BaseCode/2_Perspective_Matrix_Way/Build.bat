cls

del Vk.exe Log.txt Shader.vert.spv Shader.frag.spv

glslangValidator.exe -V -H -o Shader.vert.spv Shader.vert

glslangValidator.exe -V -H -o Shader.frag.spv Shader.frag

cl /I"C:\VulkanSDK\Anjaneya\Include" /c Vk.cpp /Fo"Vk.obj"

rc.exe Vk.rc

link Vk.obj Vk.res /LIBPATH:"C:\VulkanSDK\Anjaneya\Lib" vulkan-1.lib user32.lib gdi32.lib kernel32.lib /OUT:Vk.exe 

del Vk.obj Vk.pdb Vk.res

Vk.exe

