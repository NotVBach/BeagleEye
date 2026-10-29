cmake_minimum_required(VERSION 3.16)
project(lan_camera C CXX)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)
find_package(JPEG REQUIRED)

set(TENSORFLOW_SOURCE_DIR "$ENV{HOME}/tensorflow-2.17.0" CACHE PATH "TensorFlow source tree")
add_subdirectory(${TENSORFLOW_SOURCE_DIR}/tensorflow/lite ${CMAKE_BINARY_DIR}/tensorflow-lite EXCLUDE_FROM_ALL)

add_executable(server
    server/server.c
    server/camera.c
    server/vision.cpp
)
# The sysroot ships OpenCV's libs but no OpenCVConfig.cmake.
target_include_directories(server PRIVATE ${CMAKE_SYSROOT}/usr/include/opencv4)
target_compile_options(server PRIVATE -Wall -Wextra)
target_link_libraries(server PRIVATE Threads::Threads tensorflow-lite opencv_core opencv_imgproc JPEG::JPEG)

set(BBB_HOST "debian@192.168.7.2" CACHE STRING "SSH user@host of the BeagleBone")
set(BBB_DIR  "camera"         CACHE STRING "Install folder on the BeagleBone, relative to home")

add_custom_target(deploy
    COMMAND ssh ${BBB_HOST} "mkdir -p ${BBB_DIR}/server ${BBB_DIR}/www"
    COMMAND scp $<TARGET_FILE:server> ${BBB_HOST}:${BBB_DIR}/server/server
    COMMAND scp ${CMAKE_SOURCE_DIR}/www/index.html ${BBB_HOST}:${BBB_DIR}/www/index.html
    DEPENDS server
    COMMENT "Copying server and www/ to ${BBB_HOST}:${BBB_DIR}"
    VERBATIM
)