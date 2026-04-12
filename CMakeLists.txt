cmake_minimum_required(VERSION 3.20)

project(PhotonZip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYBIND11_FINDPYTHON ON)

option(PHOTONZIP_BUILD_PYTHON "Build the PhotonZip Python extension module" ON)
set(PHOTONZIP_MANS_TARGET_PLATFORM "cpu_nv" CACHE STRING "Forwarded TARGET_PLATFORM for the vendored MANS build")

if(NOT TARGET mans_common)
  set(BUILD_HDF5_PLUGIN ON CACHE BOOL "Build the vendored HDF5 plugin" FORCE)
  set(TARGET_PLATFORM "${PHOTONZIP_MANS_TARGET_PLATFORM}" CACHE STRING "Target platform to build: cpu / nv / amd / all / cpu_nv / cpu_amd" FORCE)
  add_subdirectory(3rdparty/lossless/MANS)
endif()

add_library(photonzip_core STATIC
  src/photonzip/core/codec_registry.cpp
  src/photonzip/codecs/mans/mans_codec.cpp
)

target_include_directories(photonzip_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lossless/MANS
)

target_link_libraries(photonzip_core PUBLIC mans_common)

if(TARGET mans_nv_core)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(photonzip_core PUBLIC MANS_ENABLE_NV)
  target_include_directories(photonzip_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(photonzip_core PUBLIC CUDA::cudart)
endif()

if(PHOTONZIP_BUILD_PYTHON)
  add_subdirectory(bindings/python)
endif()
