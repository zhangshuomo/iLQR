cmake_minimum_required(VERSION 3.0.2)
project(ilqr_single_leg)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
)

find_package(pinocchio REQUIRED)
find_package(PythonLibs REQUIRED)
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ilqr_single_leg
#  CATKIN_DEPENDS roscpp std_msgs tf2 tf2_geometry_msgs tf2_ros
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PINOCCHIO_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)

add_subdirectory(loss_function)
add_subdirectory(single_leg_without_point_trace)
add_subdirectory(core)
