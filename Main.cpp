#include "Common.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>
#include <omp.h>

// === DODANE: potrzebne do cudaFree (warm-up GPU)
#include <cuda_runtime.h>

using namespace std;

enum DataType {
    UNCORRELATED,
    WEAKLY_CORRELATED,
    STRONGLY_CORRELATED
};

// Funkcja do generowania danych
ProblemData generateProblem(int n, DataType type, int seed = 42) {
    ProblemData data;
    data.n = n;
    data.items.resize(n);

    const int MIN_VAL = 1000;
    const int MAX_VAL = 2000;
    mt19937 gen(seed);
    uniform_int_distribution<> distWeight(MIN_VAL, MAX_VAL);
    uniform_int_distribution<> distValue(MIN_VAL, MAX_VAL);
    uniform_int_distribution<> distNoise(-MAX_VAL / 10, MAX_VAL / 10);

    long long totalWeight = 0;

    for (int i = 0; i < n; i++) {
        int w = distWeight(gen);
        int v = 0;

        switch (type) {
        case UNCORRELATED:
            v = distValue(gen);
            break;
        case WEAKLY_CORRELATED:
            v = w + distNoise(gen);
            if (v < 1) v = 1;
            break;
        case STRONGLY_CORRELATED:
            v = w + 10;
            break;
        }

        data.items[i].weight = w;
        data.items[i].value = v;
        data.items[i].ratio = (double)v / (double)w;
        totalWeight += (long long)w;
    }

    data.C = totalWeight / 2;

    // Sortowanie malejaco po ratio
    sort(data.items.begin(), data.items.end(),
        [](const Item& a, const Item& b) {
            return a.ratio > b.ratio;
        });

    return data;
}

// Funkcja mierzaca czas wykonania
template <typename Func>
double measureTime(Func func) {
    auto start = chrono::high_resolution_clock::now();
    func();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;
    return elapsed.count();
}

static string typeToString(DataType t) {
    switch (t) {
        case UNCORRELATED: return "uncorrelated";
        case WEAKLY_CORRELATED: return "weakly_correlated";
        case STRONGLY_CORRELATED: return "strongly_correlated";
        default: return "unknown";
    }
}

struct TestConfig {
    int n;
    int instances;
    int repeats;
};

// === DODANE: rozgrzewka GPU (inicjalizacja CUDA, JIT, cache)
void warmUpGPU() {
    cudaFree(0);

    ProblemData warmData = generateProblem(20, STRONGLY_CORRELATED, 123);

    for (int i = 0; i < 5; i++) {
        solveGPU(warmData);
    }
}

// === DODANE: rozgrzewka wersji równoleg³ej CPU (cache, branch predictor)
void warmUpParallelCPU() {
    ProblemData warmData = generateProblem(20, STRONGLY_CORRELATED, 321);

    omp_set_num_threads(4);
    for (int i = 0; i < 3; i++) {
        solveParallel(warmData);
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // === DODANE: rozgrzewki (NIE S¥ MIAR¥ CZASU)
    warmUpGPU();
    warmUpParallelCPU();

    // Otwarcie pliku do zapisu wynikow
    ofstream csvFile("results.csv");
    csvFile << "Size,Type,InstanceID,Repetition,Algorithm,Threads,TimeMS\n";

    // parametry badania
    vector<TestConfig> configs = {
        {25, 50, 10}, {30, 50, 10},
        {32, 20, 5},  {35, 20, 5},
        {38, 10, 2},  {40, 10, 2},
        {45, 10, 1},  {50, 10, 1},
    };

    vector<DataType> types = { STRONGLY_CORRELATED };
    vector<int> threadCounts = { 1, 2, 4, 6, 8, 12 };

    cout << fixed << setprecision(5);

    for (auto type : types) {
        string typeStr = typeToString(type);
        cout << "typ danych: " << typeStr << "\n";

        for (const auto& config : configs) {
            int n = config.n;
            int NUM_INSTANCES = config.instances;
            int REPEAT_COUNT = config.repeats;

            cout << "\n[Rozmiar N = " << n << " | Instancji: " << NUM_INSTANCES << " | Powtorzen: " << REPEAT_COUNT << "]\n";

            double globalSeqSum = 0;
            double globalOptSum = 0;
            vector<double> globalParSums(threadCounts.size(), 0.0);
            double globalGPUSum = 0;

            for (int i = 0; i < NUM_INSTANCES; i++) {
                ProblemData data = generateProblem(n, type, i + (n * 100));

                double instSeqTime = 0;
                double instOptTime = 0;
                vector<double> instParTimes(threadCounts.size(), 0.0);
                double instGPUTime = 0;

                for (int r = 0; r < REPEAT_COUNT; r++) {

                    // Sekwencyjny
                    double tSeq = measureTime([&]() { solveSequential(data); });
                    instSeqTime += tSeq;
                    csvFile << n << "," << typeStr << "," << i << "," << r << ",Sequential,1," << tSeq << "\n";

                    // Sekwencyjny Zoptymalizowany
                    double tOpt = measureTime([&]() { solveSequentialOptimized(data); });
                    instOptTime += tOpt;
                    csvFile << n << "," << typeStr << "," << i << "," << r << ",SequentialOptimized,1," << tOpt << "\n";

                    // Równolegly
                    for (size_t t_idx = 0; t_idx < threadCounts.size(); t_idx++) {
                        int th = threadCounts[t_idx];
                        omp_set_num_threads(th);

                        double tPar = measureTime([&]() { solveParallel(data); });
                        instParTimes[t_idx] += tPar;
                        csvFile << n << "," << typeStr << "," << i << "," << r << ",Parallel," << th << "," << tPar << "\n";
                    }

                    // GPU
                    double tGPU = measureTime([&]() { solveGPU(data); });
                    instGPUTime += tGPU;
                    csvFile << n << "," << typeStr << "," << i << "," << r << ",GPU,0," << tGPU << "\n";
                }

                // Wypisanie wyników dla pojedynczej instancji
                globalSeqSum += instSeqTime / REPEAT_COUNT;
                globalOptSum += instOptTime / REPEAT_COUNT;
                cout << i + 1 << ": Seq=" << instSeqTime / REPEAT_COUNT << "ms | Opt=" << instOptTime / REPEAT_COUNT << "ms";
                for (size_t t_idx = 0; t_idx < threadCounts.size(); t_idx++) {
                    globalParSums[t_idx] += instParTimes[t_idx] / REPEAT_COUNT;
                    cout << " | Par(" << threadCounts[t_idx] << ")=" << instParTimes[t_idx] / REPEAT_COUNT << "ms";
                }
                globalGPUSum += instGPUTime / REPEAT_COUNT;
                cout << " | GPU =" << (instGPUTime / REPEAT_COUNT) << "ms\n";
            }
            cout << "SREDNIA:\n";
            cout << "Seq:      " << globalSeqSum / NUM_INSTANCES << " ms\n";
            cout << "SeqOpt:   " << globalOptSum / NUM_INSTANCES << " ms\n";
            for (size_t t_idx = 0; t_idx < threadCounts.size(); t_idx++) {
                cout << "Par(" << threadCounts[t_idx] << "):  "
                    << globalParSums[t_idx] / NUM_INSTANCES << " ms\n";
            }
            cout << "GPU:      " << globalGPUSum / NUM_INSTANCES << " ms\n";

        }
        cout << "\n";
    }

    csvFile.close();

    return 0;
}
