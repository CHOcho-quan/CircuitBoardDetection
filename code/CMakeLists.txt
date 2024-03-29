cmake_minimum_required(VERSION 2.8)
project(vo1)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV 3 REQUIRED)
find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

add_executable(orb_cv orb_cv.cpp)
add_executable(sift_cv sift_cv.cpp)
add_executable(mad_cv MAD.cpp)
add_executable(quan_cv quan_cv.cpp)
target_link_libraries(quan_cv ${OpenCV_LIBS})
target_link_libraries(mad_cv ${OpenCV_LIBS})
target_link_libraries(orb_cv ${OpenCV_LIBS})
target_link_libraries(sift_cv ${OpenCV_LIBS})
