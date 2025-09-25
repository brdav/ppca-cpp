# Build the Python extension for the ppca package.
cmake_minimum_required(VERSION 3.18)
project(ppca-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build the C++ core library
add_subdirectory(src/cpp)

# Ensure Python interpreter/dev and pybind11 are available.
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# --- CARMA (NumPy <-> Armadillo conversions) -----------------------------
# Assumes git submodule for simplicity and deterministic location
if (NOT EXISTS "${CMAKE_SOURCE_DIR}/extern/carma/CMakeLists.txt")
  message(FATAL_ERROR "Missing CARMA submodule. Run: git submodule update --init --recursive")
endif()
add_subdirectory(${CMAKE_SOURCE_DIR}/extern/carma)

# Make CARMA's generated_config.h also available as config.h. This is a bit
# hacky, but ensures that config.h is available after build even if the
# generated_config.h did not exist at configure time.
set(CARMA_BINARY_DIR "${CMAKE_BINARY_DIR}/extern/carma")
set(CARMA_GENERATED "${CARMA_BINARY_DIR}/include/carma_bits/generated_config.h")
set(CARMA_CONFIG "${CARMA_BINARY_DIR}/include/carma_bits/config.h")
if(EXISTS "${CARMA_GENERATED}")
  file(COPY "${CARMA_GENERATED}" DESTINATION "${CARMA_BINARY_DIR}/include/carma_bits/")
  file(RENAME "${CARMA_GENERATED}" "${CARMA_CONFIG}")
else()
  # If not present at configure time, add a simple build-step copy so the file will exist after build.
  add_custom_target(carma_copy_config ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CARMA_GENERATED}" "${CARMA_CONFIG}"
    BYPRODUCTS "${CARMA_CONFIG}"
    VERBATIM)
endif()

# The C++ core library 'ppca' is expected to exist (built by src/cpp).
# Link against it and CARMA.
pybind11_add_module(_ppca_bindings MODULE ${CMAKE_SOURCE_DIR}/src/python/ppca/_ppca_bindings.cpp)
target_link_libraries(_ppca_bindings PRIVATE ppca)
target_link_libraries(_ppca_bindings PRIVATE carma::carma)

# Install into the Python package directory so scikit-build maps it into the wheel
install(TARGETS _ppca_bindings DESTINATION ppca)
