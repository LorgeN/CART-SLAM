cmake_minimum_required( VERSION 3.26 )

set(CXX_STANDARD 11)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

project( CART-SLAM LANGUAGES CUDA CXX )

find_package( OpenCV REQUIRED )
include_directories (${OpenCV_INCLUDE_DIRS} )

add_executable( cart_slam 
    src/cartslam.cpp 
    src/datasource.cpp 
    src/features.cpp 
)

target_include_directories( cart_slam PRIVATE include )
target_link_libraries( cart_slam ${OpenCV_LIBS} )