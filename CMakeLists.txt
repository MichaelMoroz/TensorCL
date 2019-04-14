cmake_minimum_required(VERSION 3.0)
project (NeuralNetworkCLEngine)

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

add_subdirectory(src)

target_include_directories(NeuralNetworkCLSources PUBLIC
  src
  ${SFML_INCLUDE_DIR}
  ${ANTTWEAKBAR_DIR}/include
)


if(WIN32)
  add_executable(NeuralNetworkCLEngine WIN32 src/Main.cpp)
else()
  add_executable(NeuralNetworkCLEngine src/Main.cpp)
endif()

add_custom_command(TARGET NeuralNetworkCLEngine PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:NeuralNetworkCLEngine>)


target_compile_definitions(NeuralNetworkCLEngine PRIVATE SFML_STATIC)
target_link_libraries(NeuralNetworkCLEngine
  NeuralNetworkCLSources
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
