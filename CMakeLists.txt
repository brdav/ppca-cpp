cmake_minimum_required(VERSION 3.18)
project(lib-ppca LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Allow selecting header-only Armadillo. Default to header-only to avoid runtime
# dylib dependencies for the Python extension.
option(HEADER_ONLY "Build Armadillo as header-only to avoid linking libarmadillo" ON)

include(FetchContent)

# --- Armadillo vendoring -------------------------------------------------
set(ARMADILLO_VERSION 15.0.2)
set(ARMADILLO_ARCHIVE "armadillo-${ARMADILLO_VERSION}.tar.xz")
set(ARMADILLO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/armadillo-${ARMADILLO_VERSION}")

# Download & extract the release tarball only when the source directory is missing.
if (NOT EXISTS ${ARMADILLO_SRC_DIR})
  message(STATUS "Downloading Armadillo ${ARMADILLO_VERSION} into ${CMAKE_CURRENT_SOURCE_DIR}/extern/")
  file(DOWNLOAD
    "https://sourceforge.net/projects/arma/files/${ARMADILLO_ARCHIVE}/download"
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/${ARMADILLO_ARCHIVE}"
    SHOW_PROGRESS STATUS _dl_status)
  list(GET _dl_status 0 _dl_code)
  if (_dl_code EQUAL 0)
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xJf "${CMAKE_CURRENT_SOURCE_DIR}/extern/${ARMADILLO_ARCHIVE}"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/extern"
      RESULT_VARIABLE _tar_result)
    if (NOT _tar_result EQUAL 0)
      message(FATAL_ERROR "Failed to extract Armadillo archive")
    endif()
  else()
    message(FATAL_ERROR "Failed downloading Armadillo archive")
  endif()
endif()

# Configure Armadillo build options before adding it as a subproject. These
# cache variables are read by Armadillo's CMakeLists.
set(HEADER_ONLY ${HEADER_ONLY} CACHE BOOL "Build Armadillo header-only" FORCE)

# Add Armadillo as a subdirectory only if we have the sources. If not present,
# we'll fall back to find_package() or use CARMA-provided headers.
if (EXISTS ${ARMADILLO_SRC_DIR})
  add_subdirectory(${ARMADILLO_SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/extern/armadillo-build EXCLUDE_FROM_ALL)
endif()

# Normalize Armadillo target/headers across different Armadillo CMake layouts.
# After this block, the logical target name is stored in ARMADILLO_TARGET and
# ARMADILLO_INCLUDE_DIRS contains include paths to use as a fallback.
unset(ARMADILLO_TARGET CACHE)
if (TARGET armadillo::armadillo)
  message(STATUS "Using Armadillo target: armadillo::armadillo")
  set(ARMADILLO_TARGET armadillo::armadillo)
elseif (TARGET armadillo)
  message(STATUS "Creating alias armadillo::armadillo -> armadillo")
  add_library(armadillo::armadillo ALIAS armadillo)
  set(ARMADILLO_TARGET armadillo::armadillo)
elseif (EXISTS "${ARMADILLO_SRC_DIR}/include")
  # Armadillo sources are present but don't export a library target. Expose
  # headers via an INTERFACE target so downstream code can link to
  # armadillo::armadillo uniformly.
  message(STATUS "No Armadillo target exported by upstream; creating INTERFACE target pointing to headers")
  set(ARMADILLO_INCLUDE_DIRS "${ARMADILLO_SRC_DIR}/include")
  add_library(armadillo INTERFACE)
  target_include_directories(armadillo INTERFACE ${ARMADILLO_INCLUDE_DIRS})
  add_library(armadillo::armadillo ALIAS armadillo)
  set(ARMADILLO_TARGET armadillo::armadillo)
endif()

# If we still don't have armadillo headers/target, defer to the system package.
if (NOT DEFINED ARMADILLO_TARGET)
  find_package(Armadillo QUIET)
  if (Armadillo_FOUND)
    if (TARGET armadillo::armadillo)
      set(ARMADILLO_TARGET armadillo::armadillo)
    endif()
  endif()
endif()

if (NOT DEFINED ARMADILLO_TARGET AND NOT DEFINED ARMADILLO_INCLUDE_DIRS)
  message(FATAL_ERROR "Armadillo not found. Either provide sources in extern/ or install Armadillo system-wide.")
endif()

# --- Python / pybind11 ---------------------------------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

### CARMA (NumPy <-> Armadillo conversions)
# Deterministic vendored approach: require git submodule at extern/carma.
if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/extern/carma/CMakeLists.txt)
  message(FATAL_ERROR "Missing CARMA submodule. Run: git submodule update --init --recursive")
endif()

# Add CARMA. CARMA will provide a `carma::carma` target
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/carma)

# Make CARMA's generated_config.h also available as config.h.
set(CARMA_BUILD_DIR "${CMAKE_BINARY_DIR}/extern/carma")
set(CARMA_GENERATED "${CARMA_BUILD_DIR}/include/carma_bits/generated_config.h")
set(CARMA_CONFIG   "${CARMA_BUILD_DIR}/include/carma_bits/config.h")
if(EXISTS "${CARMA_GENERATED}")
  file(COPY "${CARMA_GENERATED}" DESTINATION "${CARMA_BUILD_DIR}/include/carma_bits/")
  file(RENAME "${CARMA_GENERATED}" "${CARMA_CONFIG}")
else()
  # If not present at configure time, add a simple build-step copy so the file will exist after build.
  add_custom_target(carma_copy_config ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CARMA_GENERATED}" "${CARMA_CONFIG}"
    BYPRODUCTS "${CARMA_CONFIG}"
    VERBATIM)
endif()

# Core library
add_library(libppca_core STATIC
  cpp/src/ppca.cpp
  cpp/src/ppca_em.cpp
  cpp/src/ppca_infer.cpp
  cpp/src/detail/sampling.cpp
)
target_include_directories(libppca_core PUBLIC cpp/include)

# Link to normalized Armadillo target or include directories if we only have headers
if (DEFINED ARMADILLO_TARGET)
  target_link_libraries(libppca_core PUBLIC ${ARMADILLO_TARGET})
else()
  target_include_directories(libppca_core PUBLIC ${ARMADILLO_INCLUDE_DIRS})
endif()

# Python extension
pybind11_add_module(_bindings MODULE cpp/src/_bindings.cpp)
target_link_libraries(_bindings PRIVATE libppca_core)
target_link_libraries(_bindings PRIVATE carma::carma)
target_include_directories(_bindings PRIVATE cpp/include)
set_target_properties(_bindings PROPERTIES OUTPUT_NAME "_bindings")

install(TARGETS _bindings DESTINATION lib_ppca)
