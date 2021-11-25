cmake_minimum_required(VERSION 3.0.0)
project(FSG)

find_package(OpenCV REQUIRED)
message( STATUS "OpenCV_FOUND: " ${OpenCV_FOUND})
message( STATUS "OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE LIB_SOURCES "src/*.cpp" "src/*.h")
include_directories(src)
add_library(fsg ${LIB_SOURCES} src/main.cpp)
target_link_libraries(fsg ${OpenCV_LIBS})

add_executable(fsg_main src/main.cpp)
target_link_libraries(fsg_main fsg ${OpenCV_LIBS})