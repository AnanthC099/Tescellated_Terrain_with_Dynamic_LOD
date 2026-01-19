@echo off
REM Shader compilation script for Dynamic Tessellation LOD Terrain
REM Requires Vulkan SDK to be installed and glslc in PATH

echo Compiling shaders...

REM Compile Vertex Shader
glslc -fshader-stage=vertex Shader.vert -o Shader.vert.spv
if %ERRORLEVEL% NEQ 0 (
    echo Error compiling vertex shader!
    pause
    exit /b 1
)
echo Vertex shader compiled successfully.

REM Compile Fragment Shader
glslc -fshader-stage=fragment Shader.frag -o Shader.frag.spv
if %ERRORLEVEL% NEQ 0 (
    echo Error compiling fragment shader!
    pause
    exit /b 1
)
echo Fragment shader compiled successfully.

REM Compile Tessellation Control Shader
glslc -fshader-stage=tesscontrol Shader.tesc -o Shader.tesc.spv
if %ERRORLEVEL% NEQ 0 (
    echo Error compiling tessellation control shader!
    pause
    exit /b 1
)
echo Tessellation control shader compiled successfully.

REM Compile Tessellation Evaluation Shader
glslc -fshader-stage=tesseval Shader.tese -o Shader.tese.spv
if %ERRORLEVEL% NEQ 0 (
    echo Error compiling tessellation evaluation shader!
    pause
    exit /b 1
)
echo Tessellation evaluation shader compiled successfully.

echo.
echo All shaders compiled successfully!
pause
