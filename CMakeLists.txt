cmake_minimum_required(VERSION 3.0)
project (NeuralNetworkCL VERSION 0.3 DESCRIPTION "An OpenCL tensor library with stuff")
set(CMAKE_CXX_STANDARD 17)

set(SFML_DIR C:/Libraries/SFML-2.5.1/lib/cmake/SFML)
set(SFML_LIB_DIR C:/Libraries/SFML-2.5.1/lib)
set(SFML_STATIC_LIBRARIES TRUE)
set(SFML_INCLUDE_DIR C:/Libraries/SFML-2.5.1/include)
set(SDK_DIRECTORY "C:/Program Files (x86)/Microsoft SDKs")
set(GLM_DIR C:/Libraries/glm)
set(GLCL_DIR C:/Libraries/GLCL)
set(ANTTWEAKBAR_DIR C:/Libraries/AntTweakBar)

find_package(SFML 2.5 COMPONENTS system window graphics REQUIRED)


include_directories(${SDK_DIRECTORY}/Windows/v7.1/Include)
include_directories(${ANTTWEAKBAR_DIR}/include)
include_directories(${GLM_DIR})
include_directories(${GLCL_DIR})

add_subdirectory(src/OPENCL)
add_subdirectory(src/CL_TENSOR)
add_subdirectory(src/PLOT)
add_subdirectory(src/MD)

configure_file(libparam.pc.in libparam.pc @ONLY)
					   

target_include_directories(OPENCL PUBLIC
  src/OPENCL
  ${GLCL_DIR}
)
	
target_include_directories(CL_TENSOR PUBLIC
  src/CL_TENSOR
  src/OPENCL
)	

target_include_directories(PLOTTER PUBLIC
  src/PLOT
  ${SFML_INCLUDE_DIR}
  ${ANTTWEAKBAR_DIR}/include
)

target_include_directories(MD PUBLIC
  src/CL_TENSOR
  src/OPENCL
  src/PLOT
  src/MD
  ${SFML_INCLUDE_DIR}
  ${ANTTWEAKBAR_DIR}/include
)

add_executable(NeuralNetworkCL src/Main.cpp)
  
add_custom_command(TARGET NeuralNetworkCL PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:NeuralNetworkCL>)

					   
target_compile_definitions(NeuralNetworkCL PRIVATE SFML_STATIC)
target_link_libraries(NeuralNetworkCL
  OPENCL
  CL_TENSOR
  PLOTTER
  MD
  sfml-system
  sfml-window
  sfml-graphics
  ${ANTTWEAKBAR_DIR}/lib/AntTweakBar.lib
  ${GLCL_DIR}/lib/x86/OpenCL.lib
  ${GLCL_DIR}/lib/x86/glew32.lib
  ${GLCL_DIR}/lib/x86/glut32.lib
  winmm
  gdi32
)
