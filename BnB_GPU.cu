#include "Common.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define CHECK_CUDA(err) if (err != cudaSuccess) { \
	printf("CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
	exit(1); \
}

struct GPUState {
	int idx;
	int value;
	int weight;
};

__device__ int upperBound(int idx, int currentValue, int currentWeight, int C, const int* weights, const int* values, const double* ratios, int n) {
	int remainingCap = C - currentWeight;
	double bound = currentValue;

	for (int i = idx; i < n; i++) {
		if (weights[i] <= remainingCap) {
			remainingCap -= weights[i];
			bound += values[i];
		}
		else {
			bound += ratios[i] * remainingCap;
			break;
		}
	}

	return (int)bound;
}

__global__ void gpuKernel(GPUState* partialStates, int numStates, int C, const int* weights, const int* values, const double* ratios, int n, int* results) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numStates) return;

	GPUState state = partialStates[tid];
	int best = 0;

	const int maxDepth = 1 << (n - state.idx);

	for (int mask = 0; mask < maxDepth; mask++) {
		int currVal = state.value;
		int currW = state.weight;

		for (int i = 0; i < (n - state.idx); i++) {
			if (mask & (1 << i)) {
				int idx = state.idx + i;
				currW += weights[idx];
				if (currW > C) break;
				currVal += values[idx];
			}
		}

		if (currW <= C && currVal > best) {
			best = currVal;
		}
	}

	results[tid] = best;
}

int solveGPU(const ProblemData& data) {
	const int prefixDepth = 10;
	if (data.n - prefixDepth > 22) {
		printf("Za duży problem dla GPU (max n - prefixDepth = 22). \n");
		return -1;
	}

	int numStates = 1 << prefixDepth;
	std::vector<GPUState> states(numStates);

	for (int i = 0; i < numStates; ++i) {
		int weight = 0;
		int value = 0; 
		for (int j = 0; j < prefixDepth; ++j) {
			if (i & (1 << j)) {
				weight += data.items[j].weight;
				value += data.items[j].value;
			}
		}
		states[i] = { prefixDepth, value, weight };
	}

	// Dane do GPU
	std::vector<int> h_weights(data.n), h_values(data.n);
	std::vector<double> h_ratios(data.n);
	for (int i = 0; i < data.n; i++) {
		h_weights[i] = data.items[i].weight;
		h_values[i] = data.items[i].value;
		h_ratios[i] = data.items[i].ratio;
	}

	// Alokacja GPU
	GPUState* d_states;
	int* d_weights;
	int* d_values;
	double* d_ratios;
	int* d_results;

	CHECK_CUDA(cudaMalloc(&d_states, numStates * sizeof(GPUState)));
	CHECK_CUDA(cudaMalloc(&d_weights, data.n * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&d_values, data.n * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&d_ratios, data.n * sizeof(double)));
	CHECK_CUDA(cudaMalloc(&d_results, numStates * sizeof(int)));

	// Kopiowanie danych
	CHECK_CUDA(cudaMemcpy(d_states, states.data(), numStates * sizeof(GPUState), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_weights, h_weights.data(), data.n * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_values, h_values.data(), data.n * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_ratios, h_ratios.data(), data.n * sizeof(double), cudaMemcpyHostToDevice));

	// Launch kernela
	int threadsPerBlock = 256;
	int blocks = (numStates + threadsPerBlock - 1) / threadsPerBlock;

	gpuKernel<<<blocks, threadsPerBlock>>>(d_states, numStates, (int)data.C, d_weights, d_values, d_ratios, data.n, d_results);

	CHECK_CUDA(cudaDeviceSynchronize());

	// Wyniki
	std::vector<int> h_results(numStates);
	CHECK_CUDA(cudaMemcpy(h_results.data(), d_results, numStates * sizeof(int), cudaMemcpyDeviceToHost));

	cudaFree(d_states);
	cudaFree(d_weights);
	cudaFree(d_values);
	cudaFree(d_ratios);
	cudaFree(d_results);

	// zwracanie najlepszego wyniku
	return *std::max_element(h_results.begin(), h_results.end());
}
