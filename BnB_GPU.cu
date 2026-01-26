#include "Common.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

// Makro do sprawdzania błędów CUDA
#define CHECK_CUDA(err) do { \
    cudaError_t _e = (err); \
    if (_e != cudaSuccess) { \
        printf("CUDA Error: %s (line %d)\n", cudaGetErrorString(_e), __LINE__); \
        std::exit(1); \
    } \
} while(0)

// Stan początkowy poddrzewa, od którego wątek zaczyna DFS
struct GPUState {
    int idx;
    int value;
    int weight;
};

// Atomowy update maksimum
__device__ __forceinline__ int atomicMaxInt(int* addr, int val) {
    int old = *addr;
    while (val > old) {
        int assumed = old;
        old = atomicCAS(addr, assumed, val);
        if (old == assumed) break;
    }
    return old;
}

// Górne oszacowanie (relaksacja ułamkowa)
__device__ __forceinline__ int upperBound(
    int idx, int currValue, int currWeight, int C,
    const int* weights, const int* values, const double* ratios, int n
) {
    int remaining = C - currWeight;
    if (remaining <= 0) return currValue;
    if (idx >= n) return currValue;

    int bound = currValue;

    for (int i = idx; i < n && remaining > 0; i++) {
        int w = weights[i];
        if (w <= remaining) {
            remaining -= w;
            bound += values[i];
        }
        else {
            bound += (int)((double)remaining * ratios[i]);
            break;
        }
    }
    return bound;
}

// Kernel Branch and Bound
// Każdy wątek przeszukuje swoje poddrzewo metodą DFS na lokalnym stosie
__global__ void bnbKernel(
    const GPUState* states, int numStates, int C,
    const int* weights, const int* values, const double* ratios, int n,
    int* globalBest
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numStates) return;

    GPUState root = states[tid];
    if (root.weight > C) return;

    // Węzeł na stosie DFS
    struct Node {
        short idx;
        int value;
        int weight;
    };

    // Lokalny stos DFS
    Node stack[32];
    int sp = 0;

    stack[sp++] = { (short)root.idx, root.value, root.weight };

    // Buforujemy najlepszy wynik lokalnie
    int localBest = atomicAdd(globalBest, 0);

    while (sp > 0) {
        Node node = stack[--sp];

        if (node.weight > C) continue;

        // Liść: aktualizujemy wynik
        if (node.idx >= n) {
            if (node.value > localBest) {
                atomicMaxInt(globalBest, node.value);
                localBest = atomicAdd(globalBest, 0);
            }
            continue;
        }

        // Co jakiś czas odświeżamy lokalny best, żeby uwzględniać postępy innych wątków bez ciągłych odczytów funkcji atomowych
        if ((sp & 7) == 0) {
            localBest = atomicAdd(globalBest, 0);
        }

        int ub = upperBound(node.idx, node.value, node.weight, C, weights, values, ratios, n);
        if (ub <= localBest) continue;

        // Ochrona przed przepełnieniem stosu
        if (sp + 2 > 32) continue;

        // Rozwinięcie węzła 1) gałąź nie bierzemy wrzucamy jako pierwszą, 2) gałąź bierzemy jako drugą
        stack[sp++] = { (short)(node.idx + 1), node.value, node.weight };

        int newWeight = node.weight + weights[node.idx];
        if (newWeight <= C) {
            stack[sp++] = {
                (short)(node.idx + 1),
                node.value + values[node.idx],
                newWeight
            };
        }
    }
}

int solveGPU(const ProblemData& data) {
    const int n = data.n;
    if (n <= 0) return 0;

    const int C = (int)data.C;

    // Głębokość prefiksu generowanego na CPU (liczba stanów startowych = 2^prefixDepth)
    int prefixDepth = 9;
    if (n >= 40) prefixDepth = 10;
    if (n < 30)  prefixDepth = std::min(n, 9);

    // Spłaszczenie danych
    std::vector<int> h_weights(n), h_values(n);
    std::vector<double> h_ratios(n);
    for (int i = 0; i < n; i++) {
        h_weights[i] = data.items[i].weight;
        h_values[i] = data.items[i].value;
        h_ratios[i] = data.items[i].ratio;
    }

    // Inicjalizacja zachłanna jako dolne ograniczenie
    int greedyBest = 0, greedyWeight = 0;
    for (int i = 0; i < n; i++) {
        if (greedyWeight + h_weights[i] <= C) {
            greedyWeight += h_weights[i];
            greedyBest += h_values[i];
        }
    }

    // Generowanie stanów startowych + odcinanie już na CPU
    std::vector<GPUState> h_states;
    const int totalMasks = 1 << prefixDepth;
    h_states.reserve((size_t)totalMasks);

    for (int mask = 0; mask < totalMasks; mask++) {
        int w = 0, v = 0;

        for (int i = 0; i < prefixDepth; i++) {
            if (mask & (1 << i)) {
                w += h_weights[i];
                v += h_values[i];
            }
        }
        if (w > C) continue;

        // Upper bound dla stanu startowego liczony na CPU
        int remaining = C - w;
        int ub = v;
        for (int i = prefixDepth; i < n && remaining > 0; i++) {
            if (h_weights[i] <= remaining) {
                remaining -= h_weights[i];
                ub += h_values[i];
            }
            else {
                ub += (int)((double)remaining * h_ratios[i]);
                break;
            }
        }

        // Jeśli  bound nie przebija greedy, nie wysyła tego na GPU
        if (ub > greedyBest) {
            h_states.push_back({ prefixDepth, v, w });
        }
    }

    if (h_states.empty()) return greedyBest;

    // Alokacja GPU.
    int* d_weights = nullptr;
    int* d_values = nullptr;
    double* d_ratios = nullptr;
    GPUState* d_states = nullptr;
    int* d_best = nullptr;

    CHECK_CUDA(cudaMalloc(&d_weights, sizeof(int) * n));
    CHECK_CUDA(cudaMalloc(&d_values, sizeof(int) * n));
    CHECK_CUDA(cudaMalloc(&d_ratios, sizeof(double) * n));
    CHECK_CUDA(cudaMalloc(&d_states, sizeof(GPUState) * h_states.size()));
    CHECK_CUDA(cudaMalloc(&d_best, sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_weights, h_weights.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ratios, h_ratios.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_states, h_states.data(), sizeof(GPUState) * h_states.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_best, &greedyBest, sizeof(int), cudaMemcpyHostToDevice));

    // Uruchamianie kernela w batchach zamiast ucinać liczbę bloków przetwarzamy wszystkie stany porcjami
    const int threads = 256;
    const int MAX_BLOCKS = 512;
    const int MAX_STATES_PER_LAUNCH = MAX_BLOCKS * threads;

    for (int offset = 0; offset < (int)h_states.size(); offset += MAX_STATES_PER_LAUNCH) {
        int batch = std::min(MAX_STATES_PER_LAUNCH, (int)h_states.size() - offset);
        int blocks = (batch + threads - 1) / threads;

        bnbKernel << <blocks, threads >> > (
            d_states + offset, batch, C,
            d_weights, d_values, d_ratios, n, d_best
            );

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    int result = 0;
    CHECK_CUDA(cudaMemcpy(&result, d_best, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_weights);
    cudaFree(d_values);
    cudaFree(d_ratios);
    cudaFree(d_states);
    cudaFree(d_best);

    return result;
}
