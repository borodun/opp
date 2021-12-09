#include <iostream>
#include <fstream>
#include <xmmintrin.h>
#include "mpi.h"

void printMat(double *mat, int rows, int columns, std::ofstream &stream) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            stream << mat[i * columns + j] << " ";
        }
        stream << std::endl;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    const int ndims = 2;
    const int X = 0;
    const int Y = 1;

    int dims[ndims] = {0};
    int periods[ndims] = {0};
    int coords[ndims];
    int procsCount;
    int rank;
    MPI_Comm gridComm;
    MPI_Comm rowComm;
    MPI_Comm colComm;

    if (argc == 7) {
        dims[X] = atoi(argv[5]);
        dims[Y] = atoi(argv[6]);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &procsCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Dims_create(procsCount, ndims, dims);

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &gridComm);
    MPI_Cart_coords(gridComm, rank, ndims, coords);
    MPI_Comm_split(gridComm, coords[Y], coords[X], &rowComm);
    MPI_Comm_split(gridComm, coords[X], coords[Y], &colComm);

    if (argc < 5 && rank == 0) {
        std::cout << "Usage: mpiexec -n 4 prog.exe 10 15 20 matmul [2] [2]" << std::endl;
        MPI_Finalize();
        return 0;
    }
    int N1 = atoi(argv[1]);
    int N2 = atoi(argv[2]);
    int N3 = atoi(argv[3]);
    std::ofstream fileStream(std::to_string(rank) + argv[4]);
    if (!fileStream && rank == 0) {
        std::cout << "error with output file" << std::endl;
        MPI_Finalize();
        return 0;
    }
    fileStream << "N1: " << N1 << " N2: " << N2 << " N3: " << N3 << std::endl;
    fileStream << "procCount: " << procsCount << std::endl;

    double *A;
    double *B;
    double *C;
    if (rank == 0) {
        std::cout << "N1: " << N1 << " N2: " << N2 << " N3: " << N3 << std::endl;
        A = new double[N1 * N2];
        B = new double[N2 * N3];
        C = new double[N1 * N3];
        std::fill(A, A + N1 * N2, 1);
        std::fill(B, B + N2 * N3, 2);

        for (int i = 0; i < N1; ++i) {
            A[i * N2 + i] = 10;
        }
        for (int i = 0; i < N2; ++i) {
            B[i * N3 + i] = 20;
        }
    }
    fileStream << "dims: (" + std::to_string(dims[X]) + ", " + std::to_string(dims[Y]) + ")" << std::endl
               << "cords: (" + std::to_string(coords[X]) + ", " + std::to_string(coords[Y]) + ")" << std::endl;

    int segmentRows = N1 / dims[Y];
    int segmentCols = N3 / dims[X];
    auto *segmentA = new double[segmentRows * N2];
    auto *segmentB = new double[N2 * segmentCols];
    auto *segmentC = new double[segmentRows * segmentCols];
    std::fill(segmentC, segmentC + segmentRows * segmentCols, 0);

    double matMulTime = -MPI_Wtime();

    /// Distribute matrices
    if (coords[X] == 0) {
        MPI_Scatter(A, segmentRows * N2, MPI_DOUBLE, segmentA, segmentRows * N2, MPI_DOUBLE, 0, colComm);
    }
    if (coords[Y] == 0) {
        MPI_Datatype sendSegment;

        MPI_Type_vector(N2, segmentCols, N3, MPI_DOUBLE, &sendSegment);
        MPI_Type_commit(&sendSegment);

        if (!rank) {
            for (int i = 0; i < N2; ++i) {
                for (int j = 0; j < segmentCols; ++j) {
                    segmentB[i * segmentCols + j] = B[i * N3 + j];
                }
            }
            for (int i = 1; i < dims[X]; ++i) {
                MPI_Send(B + segmentCols * i, 1, sendSegment, i, 1, rowComm);
            }
        } else {
            MPI_Recv(segmentB, N2 * segmentCols, MPI_DOUBLE, 0, 1, rowComm, MPI_STATUS_IGNORE);
        }

        MPI_Type_free(&sendSegment);
    }
    MPI_Bcast(segmentA, segmentRows * N2, MPI_DOUBLE, 0, rowComm);
    MPI_Bcast(segmentB, N2 * segmentCols, MPI_DOUBLE, 0, colComm);

    double segmentMulTime = -MPI_Wtime();
    /// Matrix multiplication
    for (int i = 0; i < segmentRows; ++i) {
        for (int j = 0; j < segmentCols; j += 2) {
            __m128d vR = _mm_setzero_pd();
            for (int k = 0; k < N2; k++) {
                __m128d vA = _mm_set1_pd(segmentA[i * N2 + k]);
                __m128d vB = _mm_loadu_pd(&segmentB[k * segmentCols + j]);
                vR += vA * vB;
            }
            _mm_storeu_pd(&segmentC[i * segmentCols + j], vR);
        }
    }
    /*for (int i = 0; i < segmentRows; ++i) {
        for (int k = 0; k < N2; ++k) {
            for (int j = 0; j < segmentCols; ++j) {
                segmentC[i * segmentCols + j] += segmentA[i * N2 + k] * segmentB[k * segmentCols + j];
            }
        }
    }*/
    segmentMulTime += MPI_Wtime();

    /// collect C
    MPI_Datatype recvSegment;

    MPI_Type_vector(segmentRows, segmentCols, N3, MPI_DOUBLE, &recvSegment);
    MPI_Type_commit(&recvSegment);

    int offsets[procsCount];
    for (int procRank = 0; procRank < procsCount; ++procRank) {
        MPI_Cart_coords(gridComm, procRank, ndims, coords);
        offsets[procRank] = coords[Y] * segmentRows * N3 + coords[X] * segmentCols;
    }

    if (rank) {
        MPI_Send(segmentC, segmentRows * segmentCols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    } else {
        for (int i = 0; i < segmentRows; ++i) {
            for (int j = 0; j < segmentCols; ++j) {
                C[i * N3 + j] = segmentC[i * segmentCols + j];
            }
        }
        for (int i = 1; i < procsCount; ++i) {
            MPI_Recv(C + offsets[i], 1, recvSegment, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Type_free(&recvSegment);

    MPI_Comm_free(&gridComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&rowComm);

    matMulTime += MPI_Wtime();

    /// Print info
    fileStream << std::endl << "segmentA: " << std::endl << "rows: " << segmentRows << " cols: " << N2 << std::endl;
    printMat(segmentA, segmentRows, N2, fileStream);
    fileStream << "segmentB: " << std::endl << "rows: " << N2 << " cols: " << segmentCols << std::endl;
    printMat(segmentB, N2, segmentCols, fileStream);
    fileStream << "segmentC: " << std::endl << "rows: " << segmentRows << " cols: " << segmentCols << std::endl;
    printMat(segmentC, segmentRows, segmentCols, fileStream);

    fileStream << std::endl << "segmentMulTime: " << segmentMulTime << "sec" << std::endl;

    /// Print matrices
    if (rank == 0) {
        std::cout << "segmentMulTime(0): " << segmentMulTime << "sec" << std::endl;
        std::cout << "matMulTime: " << matMulTime << "sec" << std::endl;
        fileStream << "matMulTime: " << matMulTime << "sec" << std::endl;
        fileStream << std::endl << "offsets: " << std::endl << std::endl;
        for (int j = 0; j < procsCount; ++j) {
            fileStream << offsets[j] << " ";
        }
        fileStream << std::endl;

        fileStream << std::endl << "A: " << std::endl << "rows " << N1 << " cols: " << N2 << std::endl;
        printMat(A, N1, N2, fileStream);
        fileStream << "B: " << std::endl << "rows: " << N2 << " cols: " << N3 << std::endl;
        printMat(B, N2, N3, fileStream);
        fileStream << "C: " << std::endl << "rows: " << N1 << " cols: " << N3 << std::endl;
        printMat(C, N1, N3, fileStream);
    }
    fileStream.close();

    if (rank == 0) {
        free(A);
        free(B);
        free(C);
    }
    free(segmentA);
    free(segmentB);
    free(segmentC);

    MPI_Finalize();
    return 0;
}