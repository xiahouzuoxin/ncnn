
# find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
# if(NOT OpenCV_FOUND)
#     find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
# endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../../src)

add_executable(simple_fc simple_fc.cpp)
target_link_libraries(simple_fc ncnn)
