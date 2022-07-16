cmake_minimum_required(VERSION 2.8.3)
project(icra_planning)
add_compile_options(-std=c++11)

find_package(OpenCV 3 REQUIRED)
include_directories(
  include
        ${OpenCV_INCLUDE_DIRECTORIES}
)

add_library(icra_planning SHARED
        src/planning.cpp
        src/simulator_interface.cpp
        src/build_map.cpp
        src/astar.cpp
        src/smooth.cpp
        src/lookahead.cpp
        src/navigation.cpp
)


target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBRARIES}
        )
