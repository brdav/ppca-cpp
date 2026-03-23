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
find_package(Python3 COMPONENTS NumPy REQUIRED)

# The C++ core library 'ppca' is expected to exist (built by src/cpp).
pybind11_add_module(_ppca_bindings MODULE ${CMAKE_SOURCE_DIR}/src/python/ppca/_ppca_bindings.cpp)
target_link_libraries(_ppca_bindings PRIVATE ppca)
target_link_libraries(_ppca_bindings PRIVATE Python3::NumPy)

# Install into the Python package directory so scikit-build maps it into the wheel
install(TARGETS _ppca_bindings DESTINATION ppca)
