add_library(CL_TENSOR
  CL_TENSOR.h
  CL_TENSOR.cpp
  
  Tensor.h
  Tensor.cpp

  CL_TENSOR_SORT.h
  CL_TENSOR_SORT.cpp

  OpenCL.h
  CLFunction.h
  
  CL_NN.h
  CL_NN.cpp
)

add_library(SFML_plot

  SFML_plot.h
  SFML_plot.cpp

)

add_compile_definitions(SFML_STATIC)