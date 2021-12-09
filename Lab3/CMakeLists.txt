cmake_minimum_required(VERSION 3.15)
project(Lab3)

set(CMAKE_CXX_STANDARD 17)

include_directories($ENV{MSMPI_INC})
message(STATUS $ENV{MSMPI_INC})
find_package(MPI REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2") # for SSE

add_executable(MATMUL_MPI MATMUL_MPI.cpp)
add_executable(MATMUL_MPI_SendRecv MATMUL_MPI_SendRecv.cpp)
add_executable(Simple_MATMUL Simple_MATMUL.cpp)

set(LIB_ADDRESS "$ENV{MSMPI_LIB64}msmpi.lib")
string(REPLACE "\\" "/" LIBRARY ${LIB_ADDRESS})
target_link_libraries(MATMUL_MPI ${LIBRARY})
target_link_libraries(MATMUL_MPI_SendRecv ${LIBRARY})

