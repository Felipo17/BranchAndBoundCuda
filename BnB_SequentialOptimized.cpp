#include "Common.h"
#include <algorithm>
#include <vector>

using namespace std;

namespace {

    // Struktura reprezentujaca pojedynczy wezel na stosie
    struct Node {
        int idx;
        int currentValue, currentWeight;
        // konstruktor potrzebny do uzywania emplace_back zamiast push_back 
        Node(int i, int v, int w) : idx(i), currentValue(v), currentWeight(w) {}
    };

    struct Solver {
        const ProblemData& data;
        int globalBestValue;

        // Tablice do optymalizacji obliczania ub
        vector<int> prefWeight, prefValue;

        Solver(const ProblemData& d) : data(d), globalBestValue(0) {
            // Prekomputacja sum prefiksowych do obliczania ub
            int n = data.n;
            prefWeight.resize((size_t)n + 1, 0);
            prefValue.resize((size_t)n + 1, 0);

            for (int i = 0; i < n; i++) {
                prefWeight[i + 1] = prefWeight[i] + data.items[i].weight;
                prefValue[i + 1] = prefValue[i] + data.items[i].value;
            }

            // Greedy Initialization
            // Szybkie znalezienie pierwszego sensownego rozwiazania algorytmem zachlannym
            int cV = 0, cW = 0;
            for (const auto& item : data.items) {
                if (cW + item.weight <= data.C) {
                    cV += item.value;
                    cW += item.weight;
                }
                else break;
            }
            globalBestValue = cV;
        }
        __forceinline 
        //__declspec(noinline)
        int upperBound(int idx, int currentValue, int currentWeight) {
            int remainingCap = data.C - currentWeight;

            if (remainingCap <= 0) return (double)currentValue;
            if (idx >= data.n) return (double)currentValue;

            auto it = upper_bound(prefWeight.begin() + idx, prefWeight.end(), remainingCap + prefWeight[idx]);

            int breakIndex = (int)distance(prefWeight.begin(), it) - 1;

            int bound = (currentValue + prefValue[breakIndex] - prefValue[idx]);

            if (breakIndex < data.n) {
                int weightTakenSoFar = prefWeight[breakIndex] - prefWeight[idx];
                bound += (int)((double)(remainingCap - weightTakenSoFar) * data.items[breakIndex].ratio);
            }

            return bound;
        }

        // Glowna petla iteracyjna
        void solve() {
            vector<Node> stack;
            // Rezerwujemy pamiec na n+1 wezlow w stosie
            stack.reserve((size_t)data.n + 1);

            // Wrzucamy korzen drzewa
            stack.emplace_back(0, 0, 0);

            while (!stack.empty()) {
                // Pobieramy wezel ze szczytu stosu
                Node node = stack.back();
                stack.pop_back();

                // Sprawdzamy czy to lisc
                if (node.idx == data.n) {
                    if (node.currentValue > globalBestValue) {
                        globalBestValue = node.currentValue;
                    }
                    continue; // Wracamy do petli, bierzemy kolejny element ze stosu
                }

                // Odcinanie
                if (upperBound(node.idx, node.currentValue, node.currentWeight) <= globalBestValue) {
                    continue;
                }

                // Chcemy najpierw sprawdzic wariant bierzemy
                // Zeby zostal zdjety ze stosu jako pierwszy, musimy go wrzucic jako drugiego

                // Nie bierzemy przedmiotu
                stack.emplace_back(node.idx + 1, node.currentValue, node.currentWeight);

                // Bierzemy przedmiot (o ile sie zmiesci)
                int nextWeight = node.currentWeight + data.items[node.idx].weight;
                if (nextWeight <= data.C) {
                    stack.emplace_back(node.idx + 1,
                        node.currentValue + data.items[node.idx].value,
                        nextWeight);
                }
            }
        }
    };
}

int solveSequentialOptimized(const ProblemData& data) {
    Solver solver(data);
    solver.solve();
    return solver.globalBestValue;
}
