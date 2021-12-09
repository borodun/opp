#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <winsock2.h>
#include "mpi.h"

using namespace std::chrono;

void
matVecMul(const double *mat, const double *vec, int *sizesPerThreads, int *dispositions, int N, double *newVec) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto *tmpVec = new double[sizesPerThreads[rank]];
    for (int i = 0; i < sizesPerThreads[rank]; i++) {
        tmpVec[i] = 0;
        for (int j = 0; j < N; j++) {
            tmpVec[i] += mat[i * N + j] * vec[j];
        }
    }

    MPI_Allgatherv(tmpVec, sizesPerThreads[rank], MPI_DOUBLE, newVec,
                   sizesPerThreads, dispositions, MPI_DOUBLE, MPI_COMM_WORLD);
    delete[](tmpVec);
}

void
mulByConst(const double *vec, double c, const int *sizesPerThreads, const int *dispositions, int N, double *newVec) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto *tmpVec = new double[N];
    std::fill(tmpVec, tmpVec + N, 0);
    for (int i = dispositions[rank]; i < dispositions[rank] + sizesPerThreads[rank]; i++) {
        tmpVec[i] = vec[i] * c;
    }
    MPI_Allreduce(tmpVec, newVec, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    delete[](tmpVec);
}

void
subVec(const double *vec1, const double *vec2, const int *sizesPerThreads, const int *dispositions, int N,
       double *newVec) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto *tmpVec = new double[N];
    std::fill(tmpVec, tmpVec + N, 0);
    for (int i = dispositions[rank]; i < dispositions[rank] + sizesPerThreads[rank]; i++) {
        tmpVec[i] = vec1[i] - vec2[i];
    }
    MPI_Allreduce(tmpVec, newVec, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    delete[](tmpVec);
}

void
sumVec(const double *vec1, const double *vec2, const int *sizesPerThreads, const int *dispositions, int N,
       double *newVec) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto *tmpVec = new double[N];
    std::fill(tmpVec, tmpVec + N, 0);
    for (int i = dispositions[rank]; i < dispositions[rank] + sizesPerThreads[rank]; i++) {
        tmpVec[i] = vec1[i] + vec2[i];
    }
    MPI_Allreduce(tmpVec, newVec, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    delete[](tmpVec);
}

double dotProduct(const double *vec1, const double *vec2, const int *sizesPerThreads, const int *dispositions) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double sum = 0;
    for (int i = dispositions[rank]; i < dispositions[rank] + sizesPerThreads[rank]; i++) {
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

double *solveSLAE(const double *A, double *b, int *sizesPerThreads, int *dispositions, int N) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto *solution = new double[N]; // xn+1
    std::fill(solution, solution + N, 0);
    auto *prevSolution = new double[N]; // xn
    std::fill(prevSolution, prevSolution + N, 0);

    auto *Atmp = new double[N];
    auto *r = new double[N];
    auto *z = new double[N];
    auto *rNext = new double[N];
    auto *zNext = new double[N];
    auto *alphaZ = new double[N];
    auto *betaZ = new double[N];

    double alpha;
    double beta;

    const double EPSILON = 1e-009;

    double normb = sqrt(dotProduct(b, b, sizesPerThreads, dispositions));
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
        subVec(b, Atmp, sizesPerThreads, dispositions, N, r);
        /// zn = rn
        for (int i = 0; i < N; ++i) {
            z[i] = r[i];
        }
        /// alpha
        matVecMul(A, z, sizesPerThreads, dispositions, N, Atmp);
        dotRR = dotProduct(r, r, sizesPerThreads, dispositions);
        alpha = dotRR / dotProduct(Atmp, z, sizesPerThreads, dispositions);
        /// xn+1 = xn + alpha * zn
        mulByConst(z, alpha, sizesPerThreads, dispositions, N, alphaZ);
        sumVec(prevSolution, alphaZ, sizesPerThreads, dispositions, N, solution);
        /// rn+1 = rn - alpha * A * zn
        matVecMul(A, alphaZ, sizesPerThreads, dispositions, N, Atmp);
        subVec(r, Atmp, sizesPerThreads, dispositions, N, rNext);
        /// beta
        beta = dotProduct(rNext, rNext, sizesPerThreads, dispositions) / dotRR;
        /// zn+1 = rn+1 + beta * zn
        mulByConst(z, beta, sizesPerThreads, dispositions, N, betaZ);
        sumVec(rNext, betaZ, sizesPerThreads, dispositions, N, zNext);

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
        for (long i = 0; i < N; i++) {
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
        return solution;
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

    auto *b = new double[N];
    auto *u = new double[N];
    auto *A = new double[sizesPerThreads[rank] * N];

    for (int i = 0; i < sizesPerThreads[rank]; i++) {
        for (int j = 0; j < N; j++) {
            if (i + dispositions[rank] == j) {
                A[i * N + j] = 2;
            } else {
                A[i * N + j] = 1;
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        u[i] = sin(2 * M_PI * i / double(N));
    }
    matVecMul(A, u, sizesPerThreads, dispositions, N, b);

    auto startTime = system_clock::now();
    double *solution = solveSLAE(A, b, sizesPerThreads, dispositions, N);
    auto endTime = system_clock::now();
    auto duration = duration_cast<nanoseconds>(endTime - startTime);

    if (rank == 0 && solution != nullptr) {
        fileStream << "Answer:" << std::endl;
        printMat(u, 1, N, fileStream);
        fileStream << "SLAE solution:" << std::endl;
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
