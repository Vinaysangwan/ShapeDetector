cmake_minimum_required(VERSION 3.28)
project(ShapeDetector)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIR})

add_executable(${PROJECT_NAME})

add_subdirectory(src)
target_precompile_headers(${PROJECT_NAME} PRIVATE src/pch.h)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
