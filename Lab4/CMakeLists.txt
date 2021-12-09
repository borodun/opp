cmake_minimum_required(VERSION 3.15)
project(Lab4)

set(CMAKE_CXX_STANDARD 17)

include_directories($ENV{MSMPI_INC})
message(STATUS $ENV{MSMPI_INC})
find_package(MPI REQUIRED)

add_executable(JACOBI_MPI JACOBI_MPI.cpp)

set(LIB_ADDRESS "$ENV{MSMPI_LIB64}msmpi.lib")
string(REPLACE "\\" "/" LIBRARY ${LIB_ADDRESS})
target_link_libraries(JACOBI_MPI ${LIBRARY})

