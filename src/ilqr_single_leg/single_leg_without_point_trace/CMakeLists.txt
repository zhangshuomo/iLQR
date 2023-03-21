find_package(pinocchio REQUIRED)
find_package(casadi REQUIRED)

include_directories(
  ${PINOCCHIO_INCLUDE_DIRS}
)

add_library(single_leg_without_point_trace_lib single_leg.cpp)
target_link_libraries(single_leg_without_point_trace_lib ${PINOCCHIO_LIBRARIES} casadi)
