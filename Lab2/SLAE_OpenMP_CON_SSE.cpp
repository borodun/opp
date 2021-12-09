#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <immintrin.h>

using namespace std::chrono;

void matVecMul(const double *mat, const double *vec, int N, double *newVec) {
    for (int i = 0; i < N; i++) {
        __m128d vA = _mm_setzero_pd();
        for (int j = 0; j < N; j += 2) {
            vA += _mm_loadu_pd(&mat[i * N + j]) * _mm_loadu_pd(&vec[j]);
        }
        newVec[i] = _mm_hadd_pd(vA, vA)[0];
    }
}

void mulByConst(const double *vec, double c, int size, double *newVec) {
    for (int i = 0; i < size; i += 2) {
        _mm_storeu_pd(&newVec[i], _mm_loadu_pd(&vec[i]) * _mm_set1_pd(c));
    }
}

void subVec(const double *vec1, const double *vec2, int size, double *newVec) {
    for (int i = 0; i < size; i += 2) {
        _mm_storeu_pd(&newVec[i], _mm_loadu_pd(&vec1[i]) - _mm_loadu_pd(&vec2[i]));
    }
}

void sumVec(const double *vec1, const double *vec2, int size, double *newVec) {
    for (int i = 0; i < size; i += 2) {
        _mm_storeu_pd(&newVec[i], _mm_loadu_pd(&vec1[i]) + _mm_loadu_pd(&vec2[i]));
    }
}

double dotProduct(const double *vec1, const double *vec2, int size) {
    __m128d vA = _mm_setzero_pd();
    for (int i = 0; i < size; i += 2) {
        vA += _mm_loadu_pd(&vec1[i]) * _mm_loadu_pd(&vec2[i]);
    }
    return _mm_hadd_pd(vA, vA)[0];
}

void printMat(double *mat, int rows, int columns, std::ostream &stream) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            stream << mat[i * columns + j] << " ";
        }
        stream << std::endl;
    }
}

double *solveSLAE(const double *A, double *b, int N) {
    auto *solution = new double[N]; //xn+1
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

    const double EPSILON = 1e-007;

    double normb = sqrt(dotProduct(b, b, N));
    double dotRR;

    double res = 1;
    double prevRes = 1;
    bool diverge = false;
    int divergeCount = 0;
    int rightAnswerRepeat = 0;
    int iterCount = 1;
    while (res > EPSILON || rightAnswerRepeat < 5) {
        if (res < EPSILON) {
            ++rightAnswerRepeat;
        } else {
            rightAnswerRepeat = 0;
        }

        /// rn = b - A * xn
        matVecMul(A, prevSolution, N, Atmp);
        subVec(b, Atmp, N, r);
        /// zn = rn
        for (int i = 0; i < N; i += 2) {
            _mm_storeu_pd(&z[i], _mm_loadu_pd(&r[i]));
        }
        /// alpha
        matVecMul(A, z, N, Atmp);
        dotRR = dotProduct(r, r, N);
        alpha = dotRR / dotProduct(Atmp, z, N);
        /// xn+1 = xn + alpha * zn
        mulByConst(z, alpha, N, alphaZ);
        sumVec(prevSolution, alphaZ, N, solution);
        /// rn+1 = rn - alpha * A * zn
        matVecMul(A, alphaZ, N, Atmp);
        subVec(r, Atmp, N, rNext);
        /// beta
        beta = dotProduct(rNext, rNext, N) / dotRR;
        /// zn+1 = rn+1 + beta * zn
        mulByConst(z, beta, N, betaZ);
        sumVec(rNext, betaZ, N, zNext);

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
        for (long i = 0; i < N; i += 2) {
            _mm_storeu_pd(&prevSolution[i], _mm_loadu_pd(&solution[i]));
            _mm_storeu_pd(&r[i], _mm_loadu_pd(&rNext[i]));
            _mm_storeu_pd(&z[i], _mm_loadu_pd(&zNext[i]));
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

    const std::string &fileName = argv[2];
    std::ofstream fileStream(fileName);
    if (!fileStream) {
        std::cout << "error with output file" << std::endl;
        return 0;
    }

    fileStream << "Matrix size: " << N << std::endl;

    auto *b = new double[N];
    auto *u = new double[N];
    auto *A = new double[N * N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                A[i * N + j] = 2;
            } else {
                A[i * N + j] = 1;
            }
        }
        u[i] = sin(2 * M_PI * i / double(N));
    }
    matVecMul(A, u, N, b);

    auto startTime = system_clock::now();
    double *solution = solveSLAE(A, b, N);
    auto endTime = system_clock::now();
    auto duration = duration_cast<nanoseconds>(endTime - startTime);

    if (solution != nullptr) {
        fileStream << "Answer:" << std::endl;
        printMat(u, 1, N, fileStream);
        fileStream << "SLAE solution:" << std::endl;
        printMat(solution, 1, N, fileStream);
        fileStream << "Time: " << duration.count() / double(1000000000) << "sec" << std::endl;
        std::cout << "Time: " << duration.count() / double(1000000000) << "sec" << std::endl;
    } else {
        fileStream << "Does not converge" << std::endl;
    }

    delete[](solution);
    delete[](b);
    delete[](u);
    delete[](A);
    return 0;
}
