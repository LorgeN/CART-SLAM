cmake_minimum_required( VERSION 3.26 )

set(CXX_STANDARD 11)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

project( CART-SLAM LANGUAGES CUDA CXX )

set(Boost_USE_STATIC_LIBS ON) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost REQUIRED COMPONENTS system thread) 
include_directories(${Boost_INCLUDE_DIRS}) 

find_package( OpenCV REQUIRED )
include_directories (${OpenCV_INCLUDE_DIRS} )

find_package(log4cxx 1.1)

add_executable( cart_slam 
    src/main.cpp 
    src/cartslam.cpp     
    src/datasource.cpp 
    src/features.cpp 
    src/optflow.cpp
    src/logging.cpp
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions( cart_slam PRIVATE CARTSLAM_DEBUG )
    target_compile_definitions( cart_slam PRIVATE CARTSLAM_TIMING )
endif()

target_include_directories( cart_slam PRIVATE include )
target_link_libraries( cart_slam ${OpenCV_LIBS} )
target_link_libraries( cart_slam Boost::system Boost::thread )

target_include_directories( cart_slam PRIVATE $<TARGET_PROPERTY:log4cxx,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries( cart_slam log4cxx)