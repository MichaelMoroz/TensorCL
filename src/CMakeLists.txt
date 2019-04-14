add_library(NeuralNetworkCLSources
  CL_TENSOR.h
  CL_TENSOR.cpp

  CL_TENSOR_SORT.h
  CL_TENSOR_SORT.cpp

  OpenCL.h
  CLFunction.h

  SFML_plot.h
  SFML_plot.cpp
  
  CL_NN.h
  CL_NN.cpp
)

add_compile_definitions(SFML_STATIC)