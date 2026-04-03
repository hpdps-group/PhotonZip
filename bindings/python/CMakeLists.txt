find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_native module.cpp)

target_link_libraries(_native PRIVATE photonzip_core)
set_target_properties(_native PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)

install(TARGETS _native DESTINATION photonzip)
