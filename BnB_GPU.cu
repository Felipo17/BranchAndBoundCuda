#include "Common.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// Sprawdzanie błędów CUDA
#define CHECK_CUDA(err) do { \
    cudaError_t _e = (err); \
    if (_e != cudaSuccess) { \
        printf("CUDA Error: %s (line %d)\n", cudaGetErrorString(_e), __LINE__); \
        std::exit(1); \
    } \
} while(0)

/*
Kernel GPU generujący wszystkie możliwe podzbiory dla danej połowy problemu, każdy wątek odpowiada dokładnie jednemu podzbiorowi (idx wątku traktowany jako maska bitowa)
Liczy sumę wag i wartości na podstawie ustawionych bitów. 
*/ 
__global__ void genSubsetsKernel(
    const int* weights,
    const int* values,
    int m,
    int* outW,
    int* outV
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Łączna liczba podzbiorów = 2^m
    int total = 1 << m;
    if (tid >= total) return;

    int w = 0;
    int v = 0;

    // Iteracja po bitach maski
    // Jeśli dany bit jest ustawiony, element należy do podzbioru
    for (int i = 0; i < m; i++) {
        if (tid & (1 << i)) {
            w += weights[i];
            v += values[i];
        }
    }

    // Zapis wyników do pamięci globalnej
    outW[tid] = w;
    outV[tid] = v;
}

// Pomocnicza struktura do sortowania prawej połowy
struct WV {
    int w;
    int v;
};


// Wyszukiwanie binarne największego indeksu, używane do dobrania najlepszego kompatybilnego podzbioru prawej połowy przy zadanym limicie wagi


static int upperBoundWeightIndex(const std::vector<WV>& arr, int maxW) {
    int lo = 0, hi = (int)arr.size() - 1;
    int ans = -1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (arr[mid].w <= maxW) {
            ans = mid;
            lo = mid + 1;
        }
        else {
            hi = mid - 1;
        }
    }
    return ans;
}
/*
Główna funkcja rozwiązujaca problem
GPU generuje wszystkie podzbiory lewej i prawej połowy problemu oraz wykonuje niezależne obliczenia sum
CPU filtruje i sortuje wyniki, wykonuje wyszukiwanie binarne i łączenie wyników
*/
int solveGPU(const ProblemData& data) {
    const int n = data.n;
    if (n <= 0) return 0;

    // Podział na połowy
    const int nL = n / 2;
    const int nR = n - nL;

    const int sizeL = 1 << nL;
    const int sizeR = 1 << nR;

    // Przygotowanie danych wejściowych dla GPU
    std::vector<int> h_wL(nL), h_vL(nL);
    std::vector<int> h_wR(nR), h_vR(nR);

    for (int i = 0; i < nL; i++) {
        h_wL[i] = data.items[i].weight;
        h_vL[i] = data.items[i].value;
    }
    for (int i = 0; i < nR; i++) {
        h_wR[i] = data.items[nL + i].weight;
        h_vR[i] = data.items[nL + i].value;
    }

    // Bufory dla GPU
    int* d_w = nullptr, * d_v = nullptr;
    int* d_outWL = nullptr, * d_outVL = nullptr;
    int* d_outWR = nullptr, * d_outVR = nullptr;

    CHECK_CUDA(cudaMalloc(&d_w, sizeof(int) * std::max(nL, nR)));
    CHECK_CUDA(cudaMalloc(&d_v, sizeof(int) * std::max(nL, nR)));

    CHECK_CUDA(cudaMalloc(&d_outWL, sizeof(int) * sizeL));
    CHECK_CUDA(cudaMalloc(&d_outVL, sizeof(int) * sizeL));
    CHECK_CUDA(cudaMalloc(&d_outWR, sizeof(int) * sizeR));
    CHECK_CUDA(cudaMalloc(&d_outVR, sizeof(int) * sizeR));

    // Generowanie podzbiorów lewej połowy
    CHECK_CUDA(cudaMemcpy(d_w, h_wL.data(), sizeof(int) * nL, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v, h_vL.data(), sizeof(int) * nL, cudaMemcpyHostToDevice));

    {
        int threads = 256;
        int blocks = (sizeL + threads - 1) / threads;
        genSubsetsKernel << <blocks, threads >> > (d_w, d_v, nL, d_outWL, d_outVL);
        CHECK_CUDA(cudaGetLastError());
    }

    // Generowanie podzbiorów prawej połowy
    CHECK_CUDA(cudaMemcpy(d_w, h_wR.data(), sizeof(int) * nR, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v, h_vR.data(), sizeof(int) * nR, cudaMemcpyHostToDevice));

    {
        int threads = 256;
        int blocks = (sizeR + threads - 1) / threads;
        genSubsetsKernel << <blocks, threads >> > (d_w, d_v, nR, d_outWR, d_outVR);
        CHECK_CUDA(cudaGetLastError());
    }

    // Synchronizacja, gdzie GPU kończy generowanie danych
    CHECK_CUDA(cudaDeviceSynchronize());

    // Kopiowanie wyników z GPU na CPU
    std::vector<int> h_outWL(sizeL), h_outVL(sizeL);
    std::vector<int> h_outWR(sizeR), h_outVR(sizeR);

    CHECK_CUDA(cudaMemcpy(h_outWL.data(), d_outWL, sizeof(int) * sizeL, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_outVL.data(), d_outVL, sizeof(int) * sizeL, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_outWR.data(), d_outWR, sizeof(int) * sizeR, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_outVR.data(), d_outVR, sizeof(int) * sizeR, cudaMemcpyDeviceToHost));

    // Zwonienie pamięci GPU
    cudaFree(d_w);
    cudaFree(d_v);
    cudaFree(d_outWL);
    cudaFree(d_outVL);
    cudaFree(d_outWR);
    cudaFree(d_outVR);

    const int C = (int)data.C;

    // Budowanie listy podzbiorów prawej połowy, które nie przekraczają pojemności
    std::vector<WV> right;
    right.reserve(sizeR);
    for (int i = 0; i < sizeR; i++) {
        int w = h_outWR[i];
        if (w <= C) {
            right.push_back({ w, h_outVR[i] });
        
    }

    // Sortuj po wadze rosnąco
    std::sort(right.begin(), right.end(), [](const WV& a, const WV& b) {
        if (a.w != b.w) return a.w < b.w;
        return a.v > b.v;
        });

    // Kompresja: dla rosnącej wagi zostawia tylko najlepsze wartości
    std::vector<WV> rightComp;
    rightComp.reserve(right.size());

    int bestV = -1;
    for (size_t i = 0; i < right.size(); i++) {
        if (right[i].v > bestV) {
            bestV = right[i].v;
            rightComp.push_back(right[i]);
        }
    }

    // Łączenie wyników lewej i prawej połowy
    int best = 0;
    for (int i = 0; i < sizeL; i++) {
        int wL = h_outWL[i];
        if (wL > C) continue;

        int vL = h_outVL[i];
        int remaining = C - wL;

        int idx = upperBoundWeightIndex(rightComp, remaining);
        if (idx >= 0) {
            int cand = vL + rightComp[idx].v;
            if (cand > best) best = cand;
        }
        else {
            if (vL > best) best = vL;
        }
    }

    return best;
}
