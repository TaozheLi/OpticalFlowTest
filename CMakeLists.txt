cmake_minimum_required(VERSION 3.10)
project(OpticalFlowTest)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV lib: ${OpenCV_DIR}")
endif(OpenCV_FOUND)
include_directories("./include/")
add_executable(OpticalFlowTest main.cpp
        include/OpticalFlow.h
        include/utils.h)
target_link_libraries(OpticalFlowTest ${OpenCV_LIBS})
