cmake_minimum_required(VERSION 3.10)
project(pvfinder CUDA CXX)

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Optionally set compilers
if (NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "/usr/bin/gcc")
endif()
if (NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "/usr/bin/g++")
endif()

# Unset any Anaconda-related paths
unset(ENV{CONDA_PREFIX})
unset(ENV{CONDA_DEFAULT_ENV})

# CUDA and cuDNN paths
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.1" CACHE PATH "Path to CUDA toolkit")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDNN_INCLUDE_DIR "/usr/include" CACHE PATH "Path to cuDNN include")
set(CUDNN_LIBRARY_DIR "/usr/lib/x86_64-linux-gnu" CACHE PATH "Path to cuDNN libraries")

# Find cnpy
find_path(CNPY_INCLUDE_DIR cnpy.h
    PATHS
    $ENV{CNPY_INCLUDE_DIR}
    ${CMAKE_INSTALL_PREFIX}/include
    ~/local/include
    /usr/include
    /usr/local/include
)

find_library(CNPY_LIBRARY
    NAMES cnpy
    PATHS
    $ENV{CNPY_LIBRARY_DIR}
    ${CMAKE_INSTALL_PREFIX}/lib
    ~/local/lib
    /usr/lib
    /usr/local/lib
)

if(NOT CNPY_LIBRARY)
    message(FATAL_ERROR "cnpy library not found. Please install it first in ~/local or specify its path.")
endif()

# Find ZLIB
find_path(ZLIB_INCLUDE_DIR zlib.h
    PATHS
    $ENV{ZLIB_INCLUDE_DIR}
    ${CMAKE_INSTALL_PREFIX}/include
    ~/local/include
    /usr/include
)

find_library(ZLIB_LIBRARY
    NAMES z
    PATHS
    $ENV{ZLIB_LIBRARY_DIR}
    ${CMAKE_INSTALL_PREFIX}/lib
    ~/local/lib
    /usr/lib
)

if(NOT ZLIB_LIBRARY)
    message(FATAL_ERROR "ZLIB library not found. Please install it or provide its path.")
endif()

# Include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CNPY_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIR}
)

# Library directories
link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDNN_LIBRARY_DIR}
)

# Explicitly list all source files
set(SOURCES
    src/common/cuda_common.cu
    src/common/tensor.cu
    src/layers/conv.cu
    src/layers/pooling.cu
    src/layers/activation.cu
    src/layers/dropout.cu
    src/layers/conv_block.cu
    src/layers/transpose_conv.cu
    src/layers/batchnorm.cu
    src/model/unet.cu
    src/io/npy_loader.cu
    src/main.cu  # Make sure this is included
)

# Debug print of sources
message(STATUS "Source files: ${SOURCES}")

# Common CUDA properties
set(CUDA_COMPILE_OPTIONS
    --use_fast_math
    -lineinfo
    --expt-relaxed-constexpr
    -arch=sm_80
)

# Set properties for all CUDA files
foreach(CUDA_SOURCE ${SOURCES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE}")
        set_source_files_properties(
            ${CUDA_SOURCE}
            PROPERTIES
            LANGUAGE CUDA
        )
    else()
        message(WARNING "Source file not found: ${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE}")
    endif()
endforeach()

# Add executable
add_executable(pvfinder ${SOURCES})

target_include_directories(pvfinder
    PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CNPY_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIR}
)

target_link_libraries(pvfinder
    PRIVATE
    cudart
    cublas
    cudnn
    ${CNPY_LIBRARY}
    ${ZLIB_LIBRARY}
)

target_compile_options(pvfinder
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS}>
)

set(CMAKE_BUILD_TYPE Debug)

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA include: ${CUDA_INCLUDE_DIRS}")
message(STATUS "cuDNN include: ${CUDNN_INCLUDE_DIR}")
message(STATUS "Project source dir: ${CMAKE_SOURCE_DIR}")
message(STATUS "Project binary dir: ${CMAKE_BINARY_DIR}")