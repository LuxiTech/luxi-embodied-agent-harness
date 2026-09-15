cmake_minimum_required(VERSION 3.16)
project(luxi_fastlio2_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED COMPONENTS common filters)
find_package(OpenMP QUIET)

if(NOT FASTLIO_DIR)
  message(FATAL_ERROR "FASTLIO_DIR must point at dimensionalOS/dimos-module-fastlio2")
endif()

add_executable(luxi_fastlio2_sim
  main.cpp
  ${FASTLIO_DIR}/src/preprocess.cpp
  ${FASTLIO_DIR}/include/ikd-Tree/ikd_Tree.cpp
)
target_include_directories(luxi_fastlio2_sim PRIVATE
  ${FASTLIO_DIR}/include
  ${FASTLIO_DIR}/src
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(luxi_fastlio2_sim PRIVATE ${PCL_LIBRARIES})
if(OpenMP_CXX_FOUND)
  target_link_libraries(luxi_fastlio2_sim PRIVATE OpenMP::OpenMP_CXX)
endif()
