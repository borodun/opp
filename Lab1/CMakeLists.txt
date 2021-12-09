cmake_minimum_required(VERSION 3.15)
project(Lab1)

set(CMAKE_CXX_STANDARD 17)

include_directories($ENV{MSMPI_INC})
message(STATUS $ENV{MSMPI_INC})
find_package(MPI REQUIRED)

#SET(CMAKE_C_COMPILER mpicc)
#SET(CMAKE_CXX_COMPILER mpicxx)

add_executable(SLAE_MPI_1 SLAE_MPI_1.cpp)
add_executable(SLAE_MPI_2 SLAE_MPI_2.cpp)
add_executable(SLAE_MPI_CON SLAE_MPI_CON.cpp)

set(LIB_ADDRESS "$ENV{MSMPI_LIB64}msmpi.lib")
string(REPLACE "\\" "/" LIBRARY ${LIB_ADDRESS}) # replace '\' with '/' for mingw
target_link_libraries(SLAE_MPI_1 ${LIBRARY})
target_link_libraries(SLAE_MPI_2 ${LIBRARY})
