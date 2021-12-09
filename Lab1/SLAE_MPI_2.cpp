#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <winsock2.h>
#include "mpi.h"

using namespace std::chrono;

void matVecMul(const double *mat, double *vec, int *sizesPerThreads, int *dispositions, int N, double *newVec) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto *tmpVec = new double[N];
    MPI_Allgatherv(vec, sizesPerThreads[rank], MPI_DOUBLE, tmpVec,
                   sizesPerThreads, dispositions, MPI_DOUBLE, MPI_COMM_WORLD);
    for (int i = 0; i < sizesPerThreads[rank]; i++) {
        newVec[i] = 0;
        for (int j = 0; j < N; j++) {
            newVec[i] += mat[i * N + j] * tmpVec[j];
        }
    }
    delete[](tmpVec);
}

void mulByConst(const double *vec, double c, const int *sizesPerThreads, double *newVec) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (int i = 0; i < sizesPerThreads[rank]; i++) {
        newVec[i] = vec[i] * c;
    }
}

void subVec(const double *vec1, const double *vec2, const int *sizesPerThreads, double *newVec) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (int i = 0; i < sizesPerThreads[rank]; i++) {
        newVec[i] = vec1[i] - vec2[i];
    }
}

void sumVec(const double *vec1, const double *vec2, const int *sizesPerThreads, double *newVec) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (int i = 0; i < sizesPerThreads[rank]; i++) {
        newVec[i] = vec1[i] + vec2[i];
    }
}

double dotProduct(const double *vec1, const double *vec2, const int *sizesPerThreads) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double sum = 0;
    for (int i = 0; i < sizesPerThreads[rank]; ++i) {
        sum += vec1[i] * vec2[i];
    }
    double fullSum;
    MPI_Allreduce(&sum, &fullSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return fullSum;
}

void printMat(double *mat, int rows, int columns, std::ofstream &stream) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            stream << mat[i * columns + j] << " ";
        }
        stream << std::endl;
    }
}

double *solveSLAE(double *A, double *b, int *sizesPerThreads, int *dispositions, int N) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto *solution = new double[sizesPerThreads[rank]]; // xn+1
    std::fill(solution, solution + sizesPerThreads[rank], 0);
    auto *prevSolution = new double[sizesPerThreads[rank]]; // xn
    std::fill(prevSolution, prevSolution + sizesPerThreads[rank], 0);

    auto *Atmp = new double[sizesPerThreads[rank]];
    auto *r = new double[sizesPerThreads[rank]];
    auto *z = new double[sizesPerThreads[rank]];
    auto *rNext = new double[sizesPerThreads[rank]];
    auto *zNext = new double[sizesPerThreads[rank]];
    auto *alphaZ = new double[sizesPerThreads[rank]];
    auto *betaZ = new double[sizesPerThreads[rank]];

    double alpha;
    double beta;

    const double EPSILON = 1e-009;

    double normb = sqrt(dotProduct(b, b, sizesPerThreads));
    double dotRR;

    double res = 1;
    double prevRes = 1;
    bool diverge = false;
    int divergeCount = 0;
    int rightAnswerRepeat = 0;
    int iterCount = 0;
    while (res > EPSILON || rightAnswerRepeat < 5) {
        if (res < EPSILON) {
            ++rightAnswerRepeat;
        } else {
            rightAnswerRepeat = 0;
        }

        /// rn = b - A * xn
        matVecMul(A, prevSolution, sizesPerThreads, dispositions, N, Atmp);
        subVec(b, Atmp, sizesPerThreads, r);
        /// zn = rn
        for (int i = 0; i < sizesPerThreads[rank]; ++i) {
            z[i] = r[i];
        }
        /// alpha
        matVecMul(A, z, sizesPerThreads, dispositions, N, Atmp);
        dotRR = dotProduct(r, r, sizesPerThreads);
        alpha = dotRR / dotProduct(Atmp, z, sizesPerThreads);
        /// xn+1 = xn + alpha * zn
        mulByConst(z, alpha, sizesPerThreads, alphaZ);
        sumVec(prevSolution, alphaZ, sizesPerThreads, solution);
        /// rn+1 = rn - alpha * A * zn
        matVecMul(A, alphaZ, sizesPerThreads, dispositions, N, Atmp);
        subVec(r, Atmp, sizesPerThreads, rNext);
        /// beta
        beta = dotProduct(rNext, rNext, sizesPerThreads) / dotRR;
        /// zn+1 = rn+1 + beta * zn
        mulByConst(z, beta, sizesPerThreads, betaZ);
        sumVec(rNext, betaZ, sizesPerThreads, zNext);

        res = sqrt(dotRR) / normb;
        if (prevRes < res || res == INFINITY || res == NAN) {
            ++divergeCount;
            if (divergeCount > 10 || res == INFINITY || res == NAN) {
                diverge = true;
                break;
            }
        } else {
            divergeCount = 0;
        }
        prevRes = res;
        for (long i = 0; i < sizesPerThreads[rank]; i++) {
            prevSolution[i] = solution[i];
            r[i] = rNext[i];
            z[i] = zNext[i];
        }
        ++iterCount;
    }
    delete[](prevSolution);
    delete[](Atmp);
    delete[](r);
    delete[](z);
    delete[](rNext);
    delete[](zNext);
    delete[](alphaZ);
    delete[](betaZ);

    std::cout << "iterCount: " << iterCount << std::endl;

    if (diverge) {
        delete[](solution);
        return nullptr;
    } else {
        auto *fullSolution = new double[N];
        MPI_Allgatherv(solution, sizesPerThreads[rank], MPI_DOUBLE, fullSolution,
                       sizesPerThreads, dispositions, MPI_DOUBLE, MPI_COMM_WORLD);
        return fullSolution;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Program needs 2 arguments: size, filename" << std::endl;
        return 0;
    }
    int N = atoi(argv[1]);
    std::string name(argv[2]);

    MPI_Init(&argc, &argv);
    int threadCount, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &threadCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string fileName = std::to_string(rank) + "rank-" + name;
    std::ofstream fileStream(fileName);
    if (!fileStream) {
        std::cout << "error with output file" << std::endl;
        MPI_Finalize();
        return 0;
    }

    char host[32];
    gethostname(host, 32);
    std::cout << "Host name: " << host << " rank: " << rank << std::endl;
    if (rank == 0) {
        fileStream << "Matrix size: " << N << " threadCount: " << threadCount << std::endl;
    }

    int sizesPerThreads[threadCount];
    int dispositions[threadCount];
    std::fill(sizesPerThreads, sizesPerThreads + threadCount, N / threadCount);
    sizesPerThreads[threadCount - 1] += N % threadCount;
    dispositions[0] = 0;
    for (int i = 1; i < threadCount; ++i) {
        dispositions[i] = dispositions[i - 1] + sizesPerThreads[i - 1];
    }

    auto *b = new double[sizesPerThreads[rank]];
    auto *u = new double[sizesPerThreads[rank]];
    auto *A = new double[sizesPerThreads[rank] * N];

    for (int i = 0; i < sizesPerThreads[rank]; i++) {
        for (int j = 0; j < N; j++) {
            if (i + dispositions[rank] == j) {
                A[i * N + j] = 2;
            } else {
                A[i * N + j] = 1;
            }
        }
        u[i] = sin(2 * M_PI * (i + dispositions[rank]) / double(N));
    }
    matVecMul(A, u, sizesPerThreads, dispositions, N, b);


    auto startTime = system_clock::now();
    double *solution = solveSLAE(A, b, sizesPerThreads, dispositions, N);
    auto endTime = system_clock::now();
    auto duration = duration_cast<nanoseconds>(endTime - startTime);

    auto *fullU = new double[N];
    MPI_Allgatherv(u, sizesPerThreads[rank], MPI_DOUBLE, fullU,
                   sizesPerThreads, dispositions, MPI_DOUBLE, MPI_COMM_WORLD);

    if (rank == 0 && solution != nullptr) {
        fileStream << "Answer" << std::endl;
        printMat(fullU, 1, N, fileStream);
        fileStream << "SLAE solution" << std::endl;
        printMat(solution, 1, N, fileStream);
        fileStream << "Time: " << duration.count() / double(1000000000) << "sec" << std::endl;
    } else if (solution == nullptr) {
        fileStream << "Does not converge" << std::endl;
    }

    delete[](solution);
    delete[](b);
    delete[](u);
    delete[](A);
    MPI_Finalize();
    return 0;
}
