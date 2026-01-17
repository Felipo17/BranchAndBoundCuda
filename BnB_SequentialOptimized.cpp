#include "Common.h"
#include <algorithm>
#include <vector>

using namespace std;

namespace {

    // Struktura reprezentuj¹ca pojedynczy wêze³ na stosie
    // Zastêpuje argumenty przekazywane wczeœniej w rekurencji
    struct Node {
        int idx;
        int currentValue, currentWeight;
        // konstruktor potrzebny do u¿ywania emplace_back zamiast push_back 
        Node(int i, int v, int w) : idx(i), currentValue(v), currentWeight(w) {}
    };

    struct Solver {
        const ProblemData& data;
        int globalBestValue;

        // Tablice do optymalizacji obliczania Upper Bound
        vector<int> prefWeight, prefValue;

        Solver(const ProblemData& d) : data(d), globalBestValue(0) {
            // Prekomputacja sum prefiksowych (do obliczania ub)
            int n = data.n;
            prefWeight.resize((size_t)n + 1, 0.0);
            prefValue.resize((size_t)n + 1, 0.0);

            for (int i = 0; i < n; i++) {
                prefWeight[i + 1] = prefWeight[i] + data.items[i].weight;
                prefValue[i + 1] = prefValue[i] + data.items[i].value;
            }

            // Greedy Initialization
            // Szybkie znalezienie pierwszego sensownego rozwi¹zania algorytmem zach³annym
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
        //__forceinline 
        __declspec(noinline)
            double upperBound(int idx, int currentValue, int currentWeight) {
            int remainingCap = data.C - currentWeight;

            if (remainingCap <= 0) return currentValue;
            if (idx >= data.n) return currentValue;

            auto it = upper_bound(prefWeight.begin() + idx, prefWeight.end(), remainingCap + prefWeight[idx]);

            int breakIndex = (int)distance(prefWeight.begin(), it) - 1;

            double bound = (double)currentValue + (double)(prefValue[breakIndex] - prefValue[idx]);

            if (breakIndex < data.n) {
                int weightTakenSoFar = prefWeight[breakIndex] - prefWeight[idx];
                int spaceLeft = remainingCap - weightTakenSoFar;
                bound += double(spaceLeft) * data.items[breakIndex].ratio;
            }

            return bound;
        }

        // G³ówna pêtla iteracyjna zamiast rekurencji
        void solve() {
            vector<Node> stack;
            // Rezerwujemy pamiêæ na n+1 wêz³ów w stosie
            stack.reserve((size_t)data.n + 1);

            // Wrzucamy korzeñ drzewa
            stack.emplace_back(0, 0.0, 0.0);

            while (!stack.empty()) {
                // Pobieramy wêze³ ze szczytu stosu
                Node node = stack.back();
                stack.pop_back();

                // Sprawdzamy czy to liœæ
                if (node.idx == data.n) {
                    if (node.currentValue > globalBestValue) {
                        globalBestValue = node.currentValue;
                    }
                    continue; // Wracamy do pêtli, bierzemy kolejny element ze stosu
                }

                // Odcinanie
                if (upperBound(node.idx, node.currentValue, node.currentWeight) <= globalBestValue) {
                    continue;
                }

                // Chcemy najpierw sprawdziæ wariant bierzemy
                // ¯eby zosta³ zdjêty ze stosu jako pierwszy, musimy go wrzuciæ jako drugiego

                // Nie bierzemy przedmiotu
                stack.emplace_back(node.idx + 1, node.currentValue, node.currentWeight);

                // Bierzemy przedmiot (o ile siê zmieœci)
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