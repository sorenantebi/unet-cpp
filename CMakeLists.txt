cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(unet)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${CMAKE_SOURCE_DIR}/stb_image)

add_executable(unet unet.cpp encoder.cpp decoder.cpp main.cpp)
target_link_libraries(unet "${TORCH_LIBRARIES}")
set_property(TARGET unet PROPERTY CXX_STANDARD 17)