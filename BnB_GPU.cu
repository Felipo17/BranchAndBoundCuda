#include "Common.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

// Makro do sprawdzania błędów CUDA.
#define CHECK_CUDA(err) do { \
    cudaError_t _e = (err); \
    if (_e != cudaSuccess) { \
        printf("CUDA Error: %s (line %d)\n", cudaGetErrorString(_e), __LINE__); \
        std::exit(1); \
    } \
} while(0)

// Struktura opisująca stan początkowy przeszukiwania.
struct GPUState {
    int idx;     // Aktualny indeks przedmiotu
    int value;   // Aktualna wartość rozwiązania
    int weight;  // Aktualna waga rozwiązania
};

// Atomowy odczyt wartości z pamięci globalnej.
__device__ __forceinline__ int atomicLoad(const int* addr) {
    return atomicAdd((int*)addr, 0);
}

// Atomowa aktualizacja maksimum.
__device__ __forceinline__ void atomicMaxInt(int* addr, int val) {
    int old = *addr;
    int assumed;

    while (val > old) {
        assumed = old;
        old = atomicCAS(addr, assumed, val);
        if (old == assumed) break;
    }
}

// Prosty binary search na tablicy prefWeight.
// Szuka największego indeksu k w [left, right], że prefWeight[k] <= target.
// Zakładamy, że prefWeight jest niemalejące.
__device__ __forceinline__
int binarySearchPref(const int* prefWeight, int left, int right, int target) {
    int l = left;
    int r = right;
    while (l < r) {
        int mid = (l + r + 1) >> 1;
        if (prefWeight[mid] <= target) l = mid;
        else r = mid - 1;
    }
    return l;
}

// Upper Bound jak w wersji CPU (prefiksy + binary search + ułamek ostatniego)
__device__ __forceinline__
double upperBoundBinary(
    int idx,
    int currValue,
    int currWeight,
    int C,
    const int* prefWeight,
    const int* prefValue,
    const double* ratios,
    int n
) {
    int remaining = C - currWeight;
    if (remaining <= 0) return (double)currValue;
    if (idx >= n) return (double)currValue;

    // target w "skali prefiksów": prefWeight[idx] + remaining
    int target = prefWeight[idx] + remaining;

    // breakIndex = max k w [idx, n] s.t. prefWeight[k] <= target
    // Uwaga: pref arrays mają rozmiar n+1, więc right = n.
    int breakIndex = binarySearchPref(prefWeight, idx, n, target);

    // Całkowita suma wartości pełnych przedmiotów: prefValue[breakIndex] - prefValue[idx]
    double bound = (double)currValue + (double)(prefValue[breakIndex] - prefValue[idx]);

    // Ułamek kolejnego przedmiotu jeśli breakIndex < n
    if (breakIndex < n) {
        int usedWeight = prefWeight[breakIndex] - prefWeight[idx];
        int spaceLeft = remaining - usedWeight;
        if (spaceLeft > 0) {
            bound += (double)spaceLeft * ratios[breakIndex];
        }
    }

    return bound;
}

// Kernel realizujący algorytm Branch and Bound.
__global__ void bnbKernel(
    const GPUState* states,
    int numStates,
    int C,
    const int* weights,
    const int* values,
    const double* ratios,
    const int* prefWeight,
    const int* prefValue,
    int n,
    int* globalBest
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numStates) return;

    GPUState root = states[tid];
    if (root.weight > C) return;

    struct Node {
        int idx;
        int value;
        int weight;
    };

    // Lokalny stos DFS.
    // Uwaga: przy n~40 to zwykle starczy, ale to nadal ograniczenie.
    Node stack[64];
    int sp = 0;

    stack[sp++] = { root.idx, root.value, root.weight };

    while (sp > 0) {
        Node node = stack[--sp];

        if (node.weight > C) continue;

        // Aktualizacja globalnego najlepszego rozwiązania.
        int bestSnapshot = atomicLoad(globalBest);
        if (node.value > bestSnapshot) {
            atomicMaxInt(globalBest, node.value);
            bestSnapshot = node.value;
        }

        if (node.idx >= n) continue;

        // Upper bound w wersji "binarnej" (jak CPU).
        double ub = upperBoundBinary(
            node.idx,
            node.value,
            node.weight,
            C,
            prefWeight,
            prefValue,
            ratios,
            n
        );

        bestSnapshot = atomicLoad(globalBest);
        if (ub <= (double)bestSnapshot) continue;

        // Rozwinięcie: nie bierz / bierz.
        // Preferencja kolejności jak u Ciebie (nie bierz, potem bierz) jest OK.
        if (sp + 2 <= 64) {
            // Don't take
            stack[sp++] = { node.idx + 1, node.value, node.weight };

            // Take (nie filtrujemy tu po C, bo i tak odfiltrujemy na górze pętli)
            stack[sp++] = {
                node.idx + 1,
                node.value + values[node.idx],
                node.weight + weights[node.idx]
            };
        }
    }
}

// Funkcja uruchamiająca algorytm na GPU.
int solveGPU(const ProblemData& data) {
    const int n = data.n;
    if (n <= 0) return 0;

    const int C = (int)data.C;

    // Bez sensu trzymać stałe 12, gdy n jest mniejsze.
    const int prefixDepth = (n < 12) ? n : 12;

    // Spłaszczenie danych wejściowych.
    std::vector<int> h_weights(n), h_values(n);
    std::vector<double> h_ratios(n);

    for (int i = 0; i < n; i++) {
        h_weights[i] = (int)data.items[i].weight;
        h_values[i] = (int)data.items[i].value;
        h_ratios[i] = (double)data.items[i].ratio;
    }

    // Prefiksy jak w CPU: rozmiar n+1
    std::vector<int> h_prefWeight(n + 1, 0);
    std::vector<int> h_prefValue(n + 1, 0);

    for (int i = 0; i < n; i++) {
        h_prefWeight[i + 1] = h_prefWeight[i] + h_weights[i];
        h_prefValue[i + 1] = h_prefValue[i] + h_values[i];
    }

    // Generowanie stanów początkowych dla GPU.
    std::vector<GPUState> h_states;
    const int totalMasks = 1 << prefixDepth;

    h_states.reserve((size_t)totalMasks);

    for (int mask = 0; mask < totalMasks; mask++) {
        int w = 0;
        int v = 0;

        for (int i = 0; i < prefixDepth; i++) {
            if (mask & (1 << i)) {
                w += h_weights[i];
                v += h_values[i];
            }
        }

        if (w <= C) {
            h_states.push_back({ prefixDepth, v, w });
        }
    }

    if (h_states.empty()) return 0;

    // Alokacja pamięci na GPU.
    int* d_weights = nullptr;
    int* d_values = nullptr;
    double* d_ratios = nullptr;

    int* d_prefWeight = nullptr;
    int* d_prefValue = nullptr;

    GPUState* d_states = nullptr;
    int* d_best = nullptr;

    CHECK_CUDA(cudaMalloc(&d_weights, sizeof(int) * n));
    CHECK_CUDA(cudaMalloc(&d_values, sizeof(int) * n));
    CHECK_CUDA(cudaMalloc(&d_ratios, sizeof(double) * n));

    CHECK_CUDA(cudaMalloc(&d_prefWeight, sizeof(int) * (n + 1)));
    CHECK_CUDA(cudaMalloc(&d_prefValue, sizeof(int) * (n + 1)));

    CHECK_CUDA(cudaMalloc(&d_states, sizeof(GPUState) * h_states.size()));
    CHECK_CUDA(cudaMalloc(&d_best, sizeof(int)));

    // Kopiowanie danych na GPU.
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ratios, h_ratios.data(), sizeof(double) * n, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_prefWeight, h_prefWeight.data(), sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_prefValue, h_prefValue.data(), sizeof(int) * (n + 1), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_states, h_states.data(), sizeof(GPUState) * h_states.size(), cudaMemcpyHostToDevice));

    // Inicjalizacja najlepszego wyniku.
    int zero = 0;
    CHECK_CUDA(cudaMemcpy(d_best, &zero, sizeof(int), cudaMemcpyHostToDevice));

    // Uruchomienie kernela.
    int threads = 256;
    int blocks = ((int)h_states.size() + threads - 1) / threads;

    bnbKernel << <blocks, threads >> > (
        d_states,
        (int)h_states.size(),
        C,
        d_weights,
        d_values,
        d_ratios,
        d_prefWeight,
        d_prefValue,
        n,
        d_best
        );
    CHECK_CUDA(cudaDeviceSynchronize());

    // Pobranie wyniku z GPU.
    int result = 0;
    CHECK_CUDA(cudaMemcpy(&result, d_best, sizeof(int), cudaMemcpyDeviceToHost));

    // Zwolnienie pamięci GPU.
    cudaFree(d_weights);
    cudaFree(d_values);
    cudaFree(d_ratios);
    cudaFree(d_prefWeight);
    cudaFree(d_prefValue);
    cudaFree(d_states);
    cudaFree(d_best);

    return result;
}
