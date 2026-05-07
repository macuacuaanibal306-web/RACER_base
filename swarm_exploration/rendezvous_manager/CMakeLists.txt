cmake_minimum_required(VERSION 3.0.2)
project(rendezvous_manager)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  visualization_msgs
)

# NOTE:
# This package subscribes to exploration_manager/DroneState, but we deliberately
# do NOT put exploration_manager in find_package(catkin COMPONENTS ...).
# The current exploration_manager package exports a non-existent library named
# "exploration_manager" in its catkin_package(), which makes downstream
# find_package(exploration_manager) fail during CMake configure.
#
# We keep the dependency in package.xml for catkin topological ordering, then
# include the generated message header from the devel include directory and add
# an explicit build dependency on the generated message target when available.
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs geometry_msgs visualization_msgs
)

include_directories(
  include
  SYSTEM
  ${catkin_INCLUDE_DIRS}
  ${CATKIN_DEVEL_PREFIX}/include
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(two_drone_st_rendezvous_node
  src/two_drone_st_rendezvous_node.cpp
)
add_dependencies(two_drone_st_rendezvous_node
  ${catkin_EXPORTED_TARGETS}
)

# Ensure exploration_manager/DroneState.h has been generated before compiling
# this independent rendezvous node. The target exists because package.xml keeps
# exploration_manager as a build dependency and catkin processes it first.
if(TARGET exploration_manager_generate_messages_cpp)
  add_dependencies(two_drone_st_rendezvous_node exploration_manager_generate_messages_cpp)
endif()
target_link_libraries(two_drone_st_rendezvous_node
  ${catkin_LIBRARIES}
)
