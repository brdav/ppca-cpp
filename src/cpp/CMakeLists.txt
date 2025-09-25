# Build the C++ core library for ppca and its dependencies.
cmake_minimum_required(VERSION 3.18)

project(ppca LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find and link BLAS/LAPACK (required by Armadillo)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

include(FetchContent)

## # --- Armadillo vendoring using FetchContent ------------------------------
set(ARMADILLO_VERSION "15.0.2")

# By setting SOURCE_SUBDIR to a non-existing path, we prevent automatic
# add_subdirectory in FetchContent_MakeAvailable.
# https://gitlab.kitware.com/cmake/cmake/-/issues/26220
FetchContent_Declare(
  armadillo_vendored
  URL "https://sourceforge.net/projects/arma/files/armadillo-${ARMADILLO_VERSION}.tar.xz/download"
  URL_HASH SHA256=990ab4ccb7eff1b6d70409e9aa7fa4119877ac5f5d10ba219e98460ab3e4d6eb
  DOWNLOAD_EXTRACT_TIMESTAMP true
  SOURCE_SUBDIR "PATH_THAT_DOES_NOT_EXIST" # Prevents automatic add_subdirectory
)
FetchContent_MakeAvailable(armadillo_vendored)
add_library(armadillo INTERFACE)
target_include_directories(armadillo INTERFACE "${armadillo_vendored_SOURCE_DIR}/include")

# --- Core library --------------------------------------------------------
add_library(ppca STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/ppca.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ppca_em.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ppca_infer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
)
target_include_directories(ppca PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Ensure the static library is built as PIC so it can be linked into a shared module
set_target_properties(ppca PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Link to Armadillo target and system BLAS/LAPACK
target_link_libraries(ppca PUBLIC armadillo ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
