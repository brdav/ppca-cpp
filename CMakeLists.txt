cmake_minimum_required(VERSION 3.18)
project(lib-ppca LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Armadillo REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

### CARMA (NumPy <-> Armadillo conversions)
# Deterministic vendored approach: require git submodule at extern/carma.
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/extern/carma/CMakeLists.txt)
  message(FATAL_ERROR "Missing CARMA submodule. Run: git submodule update --init --recursive")
endif()
add_subdirectory(extern/carma)
set(USE_CARMA_TARGET TRUE)

# Core library
add_library(libppca_core STATIC
  cpp/src/ppca.cpp
  cpp/src/ppca_em.cpp
  cpp/src/ppca_infer.cpp
  cpp/src/detail/sampling.cpp
)
 target_include_directories(libppca_core PUBLIC cpp/include)
 if (TARGET Armadillo::Armadillo)
   target_link_libraries(libppca_core PUBLIC Armadillo::Armadillo)
 else()
   target_include_directories(libppca_core PUBLIC ${ARMADILLO_INCLUDE_DIRS})
   target_link_libraries(libppca_core PUBLIC ${ARMADILLO_LIBRARIES})
 endif()

# Python extension
pybind11_add_module(_bindings MODULE cpp/src/_bindings.cpp)
target_link_libraries(_bindings PRIVATE libppca_core carma::carma)
target_include_directories(_bindings PRIVATE cpp/include)
set_target_properties(_bindings PROPERTIES OUTPUT_NAME "_bindings")

install(TARGETS _bindings DESTINATION lib_ppca) # scikit-build will relocate under src layout
