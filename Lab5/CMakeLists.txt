cmake_minimum_required(VERSION 3.15)
project(Lab5)

set(CMAKE_CXX_STANDARD 17)

include_directories($ENV{MSMPI_INC})
message(STATUS $ENV{MSMPI_INC})
find_package(MPI REQUIRED)

add_executable(LOAD_BALANCING_MPI LOAD_BALANCING_MPI.cpp)
add_executable(MPI_THREADS_TEST MPI_THREADS_TEST.cpp)

set(LIB_ADDRESS "$ENV{MSMPI_LIB64}msmpi.lib")
string(REPLACE "\\" "/" LIBRARY ${LIB_ADDRESS})
target_link_libraries(LOAD_BALANCING_MPI ${LIBRARY})
target_link_libraries(MPI_THREADS_TEST ${LIBRARY})
