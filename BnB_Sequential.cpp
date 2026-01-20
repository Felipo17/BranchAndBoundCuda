#include "Common.h"
#include <algorithm>
#include <vector>

using namespace std;

namespace {

    // Struktura reprezentujaca pojedynczy wezel na stosie
    struct Node {
        int idx;
        int currValue, currWeight;
    };

    // Obliczanie gornego oszacowania
    //__declspec(noinline)
    int upperBound(int idx, int currValue, int currWeight, const ProblemData& data) {
        if (currWeight > data.C) return 0; // warunek sprawdzajacy czy przekroczylismy pojemnosc plecaka

        int bound = currValue;
        int remaining = data.C - currWeight;

        for (int i = idx; i < data.n; i++) {
            // jesli przedmiot sie miesci to go bierzemy w pelni
            if (data.items[i].weight <= remaining) {
                remaining -= data.items[i].weight;
                bound += data.items[i].value;
            }
            // jezli sie nie miesci to bierzemy ulamek
            else {
                bound += (int)((double)data.items[i].ratio * remaining);
                break;
            }
        }
        return bound;
    }
}

// Funkcja DFS
int solveSequential(const ProblemData& data) {
    int bestValue = 0;

    // Jawny stos zamiast rekurencji
    vector<Node> stack;

    // Wrzucamy stan poczatkowy
    stack.push_back({ 0, 0, 0 });

    while (!stack.empty()) {
        // Pobieramy wezel ze szczytu stosu
        Node node = stack.back();
        stack.pop_back();

        // Sprawdzenie przepelnienia
        if (node.currWeight > data.C) continue;

        // Sprawdzamy czy to lisc
        if (node.idx == data.n) {
            if (node.currValue > bestValue) {
                bestValue = node.currValue;
            }
            continue;
        }

        // jesli oszacowanie jest gorsze niz to co mamy to porzucamy rozwijanie tej galezi drzewa
        if (upperBound(node.idx, node.currValue, node.currWeight, data) <= bestValue) continue;

        // 1 galaz - pominiecie przedmiotu
        stack.push_back({ node.idx + 1, node.currValue, node.currWeight });

        // 2 galaz - wrzucamy przedmiot do plecaka
        stack.push_back({ node.idx + 1,
            node.currValue + data.items[node.idx].value,
            node.currWeight + data.items[node.idx].weight });
    }

    return bestValue;
}
