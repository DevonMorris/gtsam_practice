project(gtsam)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(GTSAM REQUIRED)

add_executable(kalman_smoother src/kalman_smoother.cpp)
target_link_libraries(kalman_smoother gtsam)
