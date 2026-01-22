cls

del *.exe *.txt *.obj *.pdb *.res *.spv

glslangValidator.exe -V -H -o Shader.vert.spv Shader.vert

glslangValidator.exe -V -H -o Shader.frag.spv Shader.frag

glslangValidator.exe -V -H --target-env spirv1.4 -o CloudComputeRT.comp.spv CloudComputeRT.comp

cl /I"C:\VulkanSDK\Anjaneya\Include" /c Vk.cpp /Fo"Vk.obj"

rc.exe Vk.rc

link Vk.obj Vk.res /LIBPATH:"C:\VulkanSDK\Anjaneya\Lib" vulkan-1.lib user32.lib gdi32.lib kernel32.lib /OUT:Vk.exe 

del *.obj *.pdb *.res

Vk.exe

