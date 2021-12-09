cmake_minimum_required(VERSION 3.17)
project(Lab2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2") #for SSE

find_package(OpenMP REQUIRED)
IF(OPENMP_FOUND)
    message("Found OpenMP")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ELSE()
    message(WARNING "Can't find OpenMP")
ENDIF()

add_executable(SLAE_OpenMP_1 SLAE_OpenMP_1.cpp)
add_executable(SLAE_OpenMP_2 SLAE_OpenMP_2.cpp)
add_executable(SLAE_OpenMP_CON_SSE SLAE_OpenMP_CON_SSE.cpp)
add_executable(SLAE_OpenMP_1_SSE SLAE_OpenMP_1_SSE.cpp)
add_executable(SLAE_OpenMP_2_SSE SLAE_OpenMP_2_SSE.cpp)