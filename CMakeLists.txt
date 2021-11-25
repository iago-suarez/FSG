cmake_minimum_required(VERSION 3.0.0)
project(FSG)


find_package(OpenCV REQUIRED opencv_core opencv_features2d opencv_highgui opencv_calib3d opencv_line_descriptor)
message( STATUS "OpenCV_FOUND: " ${OpenCV_FOUND})
message( STATUS "OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
message( STATUS "OpenCV_LIBS: " ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE LIB_SOURCES "src/*.cpp" "src/*.h")
include_directories(src)
add_library(fsg ${LIB_SOURCES} src/main.cpp)
target_link_libraries(fsg ${OpenCV_LIBS})

add_executable(fsg_main src/main.cpp)
target_link_libraries(fsg_main fsg ${OpenCV_LIBS})


#######################################################
ADD_SUBDIRECTORY(googletest)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")

add_executable(regression_tests ${TEST_SOURCES})
target_link_libraries(regression_tests fsg ${${OpenCV_LIBS}} gtest gmock)
target_include_directories(regression_tests PUBLIC tests)
add_test(NAME regression_tests COMMAND regression_tests)