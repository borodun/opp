#include <iostream>
#include <fstream>
#include <chrono>

using namespace std::chrono;

void printMat(double *mat, int rows, int columns, std::ofstream &stream) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            stream << mat[i * columns + j] << " ";
        }
        stream << std::endl;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cout << "Program needs 4 arguments: N1, N2, N3, fileName" << std::endl;
        return 0;
    }
    int N1 = atoi(argv[1]);
    int N2 = atoi(argv[2]);
    int N3 = atoi(argv[3]);
    std::ofstream fileStream(argv[4]);
    if (!fileStream) {
        std::cout << "error with output file" << std::endl;
        return 0;
    }
    fileStream << "N1: " << N1 << " N2: " << N2 << " N3: " << N3 << std::endl;
    std::cout << "N1: " << N1 << " N2: " << N2 << " N3: " << N3 << std::endl;

    auto *A = new double[N1 * N2];
    auto *B = new double[N2 * N3];
    auto *C = new double[N1 * N3];
    std::fill(A, A + N1 * N2, 1.5);
    std::fill(B, B + N2 * N3, 2.5);
    for (int i = 0; i < N1; ++i) {
        A[i * N2 + i] = 10;
    }
    for (int i = 0; i < N2; ++i) {
        B[i * N3 + i] = 20;
    }
    std::fill(C, C + N1 * N3, 0);

    auto startTime = system_clock::now();
    for (int i = 0; i < N1; ++i) {
        for (int k = 0; k < N3; ++k) {
            for (int j = 0; j < N2; ++j) {
                C[i * N3 + k] += A[i * N2 + j] * B[j * N3 + k];
            }
        }
    }
    auto endTime = system_clock::now();
    auto duration = duration_cast<nanoseconds>(endTime - startTime);

    std::cout << "matMulTime: " << duration.count() / double(1000000000) << "sec" << std::endl;
    fileStream << "matMulTime: " << duration.count() / double(1000000000) << "sec" << std::endl;
    fileStream << std::endl << "A: " << std::endl << "rows " << N1 << " cols: " << N2 << std::endl;
    printMat(A, N1, N2, fileStream);
    fileStream << "B: " << std::endl << "rows: " << N2 << " cols: " << N3 << std::endl;
    printMat(B, N2, N3, fileStream);
    fileStream << "C: " << std::endl << "rows: " << N1 << " cols: " << N3 << std::endl;
    printMat(C, N1, N3, fileStream);

    free(A);
    free(B);
    free(C);

    return 0;
}