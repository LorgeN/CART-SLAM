cmake_minimum_required( VERSION 3.26 )

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

project( CART-SLAM LANGUAGES CUDA CXX )

set(Boost_USE_STATIC_LIBS ON) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 

find_package( Boost REQUIRED COMPONENTS system thread timer chrono ) 
find_package( OpenCV REQUIRED )
find_package( log4cxx 1.1 )
find_package( OpenMP )
find_package( ZED 4 )

add_executable( cart_slam 
    src/main.cpp 
    src/cartslam.cpp     
    src/datasource.cpp
    src/sources/kitti.cpp
    src/sources/zed.cpp
    src/logging.cpp 
    src/modules/features.cpp 
    src/modules/optflow.cpp
    src/modules/disparity/disparity.cu
    src/modules/disparity/interpolation.cu
    src/modules/planeseg/planeseg.cu
    src/modules/planeseg/planeseg_vis.cu
    src/modules/superpixels.cpp
    src/modules/superpixels/visualization.cu
    src/modules/superpixels/contourrelaxation/traversion.cpp
    src/modules/superpixels/contourrelaxation/contourrelaxation.cu
    src/modules/superpixels/contourrelaxation/initialization.cpp
    src/modules/superpixels/contourrelaxation/features/color.cpp
    src/modules/superpixels/contourrelaxation/features/compactness.cpp
    src/modules/superpixels/contourrelaxation/features/grayvalue.cpp
    src/utils/ui.cpp
    src/utils/data.cpp
    src/utils/colors.cpp
    src/utils/peaks.cpp
    src/utils/cuda.cu
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions( cart_slam PRIVATE CARTSLAM_DEBUG )
    target_compile_definitions( cart_slam PRIVATE CARTSLAM_TIMING )
endif()

if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_definitions( cart_slam PRIVATE CARTSLAM_DEBUG )
    target_compile_definitions( cart_slam PRIVATE CARTSLAM_TIMING )
endif()

target_compile_definitions( cart_slam PRIVATE BOOST_THREAD_VERSION=3 )
target_compile_definitions( cart_slam PRIVATE BOOST_THREAD_PROVIDES_FUTURE )
target_compile_definitions( cart_slam PRIVATE BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION )
target_compile_definitions( cart_slam PRIVATE BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY )
target_compile_definitions( cart_slam PRIVATE BOOST_THREAD_PROVIDES_FUTURE_UNWRAP )

target_include_directories( cart_slam PRIVATE include )
target_link_libraries( cart_slam PUBLIC ${OpenCV_LIBS} )
target_link_libraries( cart_slam PUBLIC Boost::system Boost::thread Boost::timer Boost::chrono )
target_link_libraries( cart_slam PUBLIC log4cxx)

target_include_directories( cart_slam PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )

if(OpenMP_CXX_FOUND)
    target_link_libraries( cart_slam PUBLIC OpenMP::OpenMP_CXX )
endif()

if(ZED_FOUND)
    target_compile_definitions( cart_slam PRIVATE CARTSLAM_ZED )
    target_include_directories( cart_slam PRIVATE ${ZED_INCLUDE_DIRS} )
    target_link_libraries( cart_slam PUBLIC ${ZED_LIBRARIES} )
endif()

target_include_directories( cart_slam PRIVATE $<TARGET_PROPERTY:log4cxx,INTERFACE_INCLUDE_DIRECTORIES>)