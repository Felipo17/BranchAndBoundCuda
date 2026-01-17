#pragma once
#include <vector>
#include <iostream>

struct Item {
    int value;
    int weight;
    double ratio;
};

// Struktura trzymaj¹ca dane wejœciowe
struct ProblemData {
    int n;
    double C; // Pojemnoœæ plecaka
    std::vector<Item> items;
};

double solveSequential(const ProblemData& data);

int solveSequentialOptimized(const ProblemData& data);

int solveParallel(const ProblemData& data);

int solveGPU(const ProblemData& data);
