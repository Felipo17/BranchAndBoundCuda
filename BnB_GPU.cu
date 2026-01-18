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
// Każdy taki stan odpowiada jednemu poddrzewu drzewa decyzyjnego.
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

// Funkcja obliczająca górne oszacowanie (upper bound).
// Zakładane jest, że kolejne przedmioty mogą być dodawane w całości, a ostatni przedmiot może zostać dodany ułamkowo.
__device__ double upperBoundFrac(
    int idx,
    int currValue,
    int currWeight,
    int C,
    const int* weights,
    const int* values,
    const double* ratios,
    int n
) {
    if (currWeight > C) return 0.0;

    int remaining = C - currWeight;
    double bound = (double)currValue;

    for (int i = idx; i < n; i++) {
        if (weights[i] <= remaining) {
            remaining -= weights[i];
            bound += values[i];
        }
        else {
            bound += ratios[i] * remaining;
            break;
        }
    }
    return bound;
}

// Kernel realizujący algorytm Branch and Bound.
// Każdy wątek:
// - rozpoczyna przeszukiwanie od innego stanu początkowego,
// - wykonuje lokalne przeszukiwanie drzewa decyzyjnego w głąb,
// - oblicza upper bound i odcina niepotrzebne gałęzie,
// - aktualizuje globalnie najlepsze znalezione rozwiązanie.
__global__ void bnbKernel(
    const GPUState* states,
    int numStates,
    int C,
    const int* weights,
    const int* values,
    const double* ratios,
    int n,
    int* globalBest
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numStates) return;

    GPUState root = states[tid];

    if (root.weight > C) return;

    // Lokalna struktura węzła używana na stosie DFS.
    struct Node {
        int idx;
        int value;
        int weight;
    };

    // Lokalny stos do przeszukiwania drzewa w głąb.
    Node stack[64];
    int sp = 0;

    // Umieszczenie stanu początkowego na stosie.
    stack[sp++] = { root.idx, root.value, root.weight };

    while (sp > 0) {
        Node node = stack[--sp];

        // Odrzucenie węzłów przekraczających pojemność plecaka.
        if (node.weight > C) continue;

        // Aktualizacja globalnego najlepszego rozwiązania.
        int bestSnapshot = atomicLoad(globalBest);
        if (node.value > bestSnapshot) {
            atomicMaxInt(globalBest, node.value);
            bestSnapshot = node.value;
        }

        // Jeśli wszystkie przedmioty zostały rozpatrzone, to nie kontynuujemy.
        if (node.idx >= n) continue;

        // Obliczenie górnego oszacowania dla bieżącego węzła.
        double ub = upperBoundFrac(
            node.idx,
            node.value,
            node.weight,
            C,
            weights,
            values,
            ratios,
            n
        );

        // Odcinanie gałęzi, które nie mogą poprawić aktualnego wyniku.
        bestSnapshot = atomicLoad(globalBest);
        if (ub <= (double)bestSnapshot) continue;

        // Rozwinięcie węzła węzły są dodawane na stos w celu dalszego przeszukiwania.
        if (sp + 2 <= 64) {
            // Gałąź gdzie przedmiot nie jest brany.
            stack[sp++] = { node.idx + 1, node.value, node.weight };

            // Gałąź gdzie przedmiot jest brany.
            stack[sp++] = {
                node.idx + 1,
                node.value + values[node.idx],
                node.weight + weights[node.idx]
            };
        }
    }
}

// Funkcja uruchamiająca algorytm, drzewo przeszukiwania jest wstępnie dzielone na poddrzewa, które następnie są przeszukiwane równolegle przez wątki GPU.
int solveGPU(const ProblemData& data) {
    const int n = data.n;
    if (n <= 0) return 0;

    const int C = (int)data.C;

    // Głębokość prefiksu decyzyjnego generowanego na CPU gdzie każdy prefiks odpowiada jednemu poddrzewu dla GPU.
    const int prefixDepth = 12;

    // Spłaszczenie danych wejściowych do tablic.
    std::vector<int> h_weights(n), h_values(n);
    std::vector<double> h_ratios(n);

    for (int i = 0; i < n; i++) {
        h_weights[i] = data.items[i].weight;
        h_values[i] = data.items[i].value;
        h_ratios[i] = data.items[i].ratio;
    }

    // Generowanie stanów początkowych dla GPU.
    std::vector<GPUState> h_states;
    const int totalMasks = 1 << prefixDepth;

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

    // Alokacja pamięci na GPU.
    int* d_weights, * d_values, * d_best;
    double* d_ratios;
    GPUState* d_states;

    CHECK_CUDA(cudaMalloc(&d_weights, sizeof(int) * n));
    CHECK_CUDA(cudaMalloc(&d_values, sizeof(int) * n));
    CHECK_CUDA(cudaMalloc(&d_ratios, sizeof(double) * n));
    CHECK_CUDA(cudaMalloc(&d_states, sizeof(GPUState) * h_states.size()));
    CHECK_CUDA(cudaMalloc(&d_best, sizeof(int)));

    // Kopiowanie danych na GPU.
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ratios, h_ratios.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
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
        n,
        d_best
        );
    CHECK_CUDA(cudaDeviceSynchronize());

    // Pobranie wyniku z GPU.
    int result;
    CHECK_CUDA(cudaMemcpy(&result, d_best, sizeof(int), cudaMemcpyDeviceToHost));

    // Zwolnienie pamięci GPU.
    cudaFree(d_weights);
    cudaFree(d_values);
    cudaFree(d_ratios);
    cudaFree(d_states);
    cudaFree(d_best);

    return result;
}
