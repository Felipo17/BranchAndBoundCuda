#pragma once
#include <vector>
#include <iostream>

struct Item {
    int value;
    int weight;
    double ratio;
};

// Struktura trzymajaca dane wejsciowe
struct ProblemData {
    int n;
    int C; // Pojemnosc plecaka
    std::vector<Item> items;
};

int solveSequential(const ProblemData& data);

int solveSequentialOptimized(const ProblemData& data);

int solveParallel(const ProblemData& data);

int solveGPU(const ProblemData& data);
