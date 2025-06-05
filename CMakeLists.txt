cmake_minimum_required(VERSION 3.15)
project(flownet)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress=20012")
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CXX_STANDARD 20)

find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(TensorRT REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(spdlog REQUIRED)


add_executable(flownet src/flownet_node.cpp                           
                       src/flownet.cpp)
target_include_directories(flownet PRIVATE include
                                            ${OpenCV_INCLUDE_DIRS}
                                            ${CUDAToolkit_INCLUDE_DIR} 
                                            ${TensorRT_INCLUDE_DIRS})
ament_target_dependencies(flownet rclcpp cv_bridge image_transport)

target_link_libraries(flownet
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    ${CUDAToolkit_LIBRARY_DIR}/libcudart.so
    ${TensorRT_LIBRARY_DIRS}/libnvinfer.so
    ${TensorRT_LIBRARY_DIRS}/libnvinfer_plugin.so)

install(TARGETS flownet
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY config launch weights plugins
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
