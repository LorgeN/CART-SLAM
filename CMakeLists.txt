cmake_minimum_required( VERSION 3.26 )

set ( CMAKE_CXX_STANDARD 17 )
set ( CMAKE_CUDA_STANDARD 17 )
set ( CMAKE_CUDA_ARCHITECTURES 70 )

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
find_package( log4cxx 1.2 )
find_package( OpenMP )
find_package( ZED 4 )
find_package( nlohmann_json 3.11.2 )

add_executable( cart_slam 
    src/main.cpp 
    src/cartslam.cpp     
    src/datasource.cpp
    src/sources/kitti.cpp
    src/sources/zed.cpp
    src/logging.cpp 
    src/cartconfig.cpp
    src/modules/module.cpp 
    src/modules/depth.cpp 
    src/modules/features.cpp 
    src/modules/optflow.cpp
    src/modules/visualization.cpp
    src/modules/disparity/derivative.cu
    src/modules/disparity/disparity.cu
    src/modules/disparity/interpolation.cu
    src/modules/planeseg/planeseg.cu
    src/modules/planeseg/planeseg_vis.cu
    src/modules/planeseg/sp_planeseg.cu
    src/modules/planefit.cu
    src/modules/planecluster.cpp
    src/modules/superpixels.cu
    src/modules/superpixels/visualization.cu
    src/modules/superpixels/contourrelaxation/features/compactness.cu
    src/modules/superpixels/contourrelaxation/features/gaussian.cu
    src/modules/superpixels/contourrelaxation/contourrelaxation.cu
    src/modules/superpixels/contourrelaxation/initialization.cu
    src/utils/ui.cpp
    src/utils/data.cpp
    src/utils/colors.cpp
    src/utils/peaks.cpp
    src/utils/cuda.cu
    src/utils/plane.cpp
    src/utils/random.cpp
    src/utils/sanity_check.cu
)

# Tells NVCC to generate device code for linking later
set_target_properties( cart_slam PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

OPTION(ENABLE_TIMING "Generate a timing file for each run" OFF)
OPTION(ENABLE_SAVE_SAMPLES "Save samples of visualizations to disk" OFF)
OPTION(ENABLE_RECORD_SAMPLES "Record samples of visualizations to disk" OFF)

if (ENABLE_TIMING)
    target_compile_definitions( cart_slam PRIVATE CARTSLAM_TIMING )
endif()

if (ENABLE_SAVE_SAMPLES)
    target_compile_definitions( cart_slam PRIVATE CARTSLAM_SAVE_SAMPLES )
endif()

if (ENABLE_RECORD_SAMPLES)
    target_compile_definitions( cart_slam PRIVATE CARTSLAM_RECORD_SAMPLES )
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions( cart_slam PRIVATE CARTSLAM_DEBUG ) 
    target_compile_options( cart_slam PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xcompiler=-fopenmp> ) 
endif()

if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_definitions( cart_slam PRIVATE CARTSLAM_DEBUG )
    target_compile_options( cart_slam PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xcompiler=-fopenmp> ) 
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

if(nlohmann_json_FOUND)
    target_compile_definitions( cart_slam PRIVATE CARTSLAM_JSON )
    target_include_directories( cart_slam PRIVATE ${nlohmann_json_INCLUDE_DIRS} )
    target_link_libraries( cart_slam PUBLIC nlohmann_json::nlohmann_json )
endif()

target_include_directories( cart_slam PRIVATE $<TARGET_PROPERTY:log4cxx,INTERFACE_INCLUDE_DIRECTORIES>)