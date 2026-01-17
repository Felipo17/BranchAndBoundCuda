#include "Common.h"
#include <algorithm>
#include <vector>

using namespace std;

namespace {

    // Struktura reprezentuj¹ca pojedynczy wêze³ na stosie
    // Zastêpuje argumenty przekazywane wczeœniej w rekurencji
    struct Node {
        int idx;
        double currValue, currWeight;
    };

    // Obliczanie górnego oszacowania
    //__declspec(noinline)
    double upperBound(int idx, double currValue, double currWeight, const ProblemData& data) {
        if (currWeight > data.C) return 0; // warunek sprawdzaj¹cy czy przekroczyliœmy pojemnoœæ plecaka

        double bound = currValue;
        double remaining = data.C - currWeight;

        for (int i = idx; i < data.n; i++) {
            // jesli przedmiot siê mieœci to go bierzemy w pe³ni
            if (data.items[i].weight <= remaining) {
                remaining -= data.items[i].weight;
                bound += data.items[i].value;
            }
            // jeœli siê nie mieœci to bierzemy u³amek
            else {
                bound += data.items[i].ratio * remaining;
                break;
            }
        }
        return bound;
    }
}

// Funkcja DFS
double solveSequential(const ProblemData& data) {
    double bestValue = 0;

    // Jawny stos zamiast rekurencji
    vector<Node> stack;

    // Wrzucamy stan pocz¹tkowy
    stack.push_back({ 0, 0.0, 0.0 });

    while (!stack.empty()) {
        // Pobieramy wêze³ ze szczytu stosu
        Node node = stack.back();
        stack.pop_back();

        // Sprawdzenie przepe³nienia
        if (node.currWeight > data.C) continue;

        // Sprawdzamy czy to liœæ
        if (node.idx == data.n) {
            if (node.currValue > bestValue) {
                bestValue = node.currValue;
            }
            continue;
        }

        // górne oszacowanie
        double ub = upperBound(node.idx, node.currValue, node.currWeight, data);

        // jeœli oszacowanie jest gorsze ni¿ to co mamy to porzucamy rozwijanie tej ga³êzi drzewa
        if (ub <= bestValue) continue;

        // 1 ga³¹Ÿ - pominiêcie przedmiotu
        stack.push_back({ node.idx + 1, node.currValue, node.currWeight });

        // 2 ga³¹Ÿ - wrzucamy przedmiot do plecaka
        stack.push_back({ node.idx + 1,
            node.currValue + data.items[node.idx].value,
            node.currWeight + data.items[node.idx].weight });
    }

    return bestValue;
}