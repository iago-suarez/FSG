cmake_minimum_required(VERSION 3.15...3.27)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

option(WITH_PYTHON "Compiles Pybind11 Python bindings" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
message( STATUS "OpenCV_FOUND: " ${OpenCV_FOUND})
message( STATUS "OpenCV_LIBS: " ${OpenCV_LIBS})
message( STATUS "OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})

set(LIB_SOURCES
        src/FastAcontrarioValidator.cpp
        src/FastAcontrarioValidator.h
        src/GreedyMerger.cpp
        src/GreedyMerger.h
        src/LsdOpenCV.cpp
        src/LsdOpenCV.h
        src/SegmentsGroup.cpp
        src/SegmentsGroup.h
        src/Utils.h)

add_library(fsg STATIC ${LIB_SOURCES})
target_include_directories(fsg PRIVATE src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(fsg ${OpenCV_LIBS})

add_executable(fsg_main src/main.cpp)
target_link_libraries(fsg_main fsg ${OpenCV_LIBS})


if(WITH_PYTHON)
    # Find the module development requirements (requires FindPython from 3.17 or
    # scikit-build-core's built-in backport)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)

    python_add_library(_pyfsg MODULE src/PYAPI.cpp WITH_SOABI)
    target_link_libraries(_pyfsg PRIVATE fsg pybind11::headers ${OpenCV_LIBS})

    # This is passing in the version as a define just as an example
    target_compile_definitions(_pyfsg PRIVATE VERSION_INFO=${PROJECT_VERSION})

    # The install directory is the output (wheel) directory
    install(TARGETS _pyfsg DESTINATION pyfsg)
endif()