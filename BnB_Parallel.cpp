#include "Common.h"
#include <algorithm>
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <thread>
#include <omp.h>
#include <random>
#include <memory>

using namespace std;

namespace {

    // Struktura reprezentujaca pojedynczy wezel na stosie
    struct Node {
        int idx;
        int currentValue, currentWeight;
        // konstruktor potrzebny do uzywania emplace_back zamiast push_back 
        Node(int i, int v, int w) : idx(i), currentValue(v), currentWeight(w) {}
    };

    class Solver {
        const ProblemData& data;
        // zmienne atomowe, alignas(64) zapewnia ze sa one trzymane w ramie w roznych liniach cache
        // dzieki czemu ograniczamy przeladowania pamieci watkow
        alignas(64) atomic<int> globalBestValue;
        alignas(64) atomic<long long> tasksInFlight;
        alignas(64) atomic<bool> finished;

        // Tablice do optymalizacji obliczania ub
        vector<int> prefWeight, prefValue;

        // Stosy dla każdego watku
        vector<unique_ptr<deque<Node>>> queues;
        vector<unique_ptr<mutex>> queueMutexes;
        int numThreads;

        // Cutoff oparty na pozostalej liczbie elementow
        const int SEQUENTIAL_CUTOFF_THRESHOLD = 20;

    public:
        Solver(const ProblemData& d)
            : data(d), globalBestValue(0), tasksInFlight(0), finished(false) {

            numThreads = omp_get_max_threads();
            if (numThreads < 1) numThreads = 1;

            queues.resize(numThreads);
            queueMutexes.resize(numThreads);

            for (int i = 0; i < numThreads; i++) {
                queues[i] = make_unique<deque<Node>>();
                queueMutexes[i] = make_unique<mutex>();
            }

            // Prekomputacja sum prefiksowych do obliczania ub
            int n = data.n;
            prefWeight.resize((size_t)n + 1, 0);
            prefValue.resize((size_t)n + 1, 0);

            for (int i = 0; i < n; i++) {
                prefWeight[i + 1] = prefWeight[i] + (int)data.items[i].weight;
                prefValue[i + 1] = prefValue[i] + (int)data.items[i].value;
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
            globalBestValue.store(cV, memory_order_relaxed);
        }

        int getResult() const {
            return globalBestValue.load();
        }

        __forceinline
        //__declspec(noinline)
        int upperBound(int idx, int currentValue, int currentWeight) {
            int remainingCap = data.C - currentWeight;

            if (remainingCap <= 0) return currentValue;
            if (idx >= data.n) return currentValue;

            auto it = upper_bound(prefWeight.begin() + idx, prefWeight.end(), remainingCap + prefWeight[idx]);

            int breakIndex = (int)distance(prefWeight.begin(), it) - 1;

            int bound = currentValue + prefValue[breakIndex] - prefValue[idx];

            if (breakIndex < data.n) {
                int weightTakenSoFar = prefWeight[breakIndex] - prefWeight[idx];
                bound += (int)((double)(remainingCap - weightTakenSoFar) * data.items[breakIndex].ratio);
            }
            return bound;
        }

        // aktualizowanie lock-free (petla Compare-And-Swap)
        void updateBestValue(int val) {
            int currentBest = globalBestValue.load(memory_order_relaxed);
            while (val > currentBest) {
                // compare_exchange_weak zwraca true jezeli globalBestValue == currentBest
                // inaczej false +  wtedy przypisuje currentBest = globalBestValue
                if (globalBestValue.compare_exchange_weak(currentBest, val, memory_order_relaxed)) {
                    break;
                }
            }
        }

        // Przetwarzanie sekwencyjne dla malych poddrzew
        void processSubtreeLocal(Node rootNode, vector<Node>& localStack) {
            localStack.clear();
            localStack.push_back(rootNode);

            // Pobieramy raz lokalnie
            int localBestCached = globalBestValue.load(memory_order_relaxed);

            while (!localStack.empty()) {
                Node node = localStack.back();
                localStack.pop_back();

                // Jezeli jestesmy na lisciu to sprawdzamy i aktualizujemy najelpsza wartosc
                if (node.idx == data.n) {
                    if (node.currentValue > localBestCached) {
                        updateBestValue(node.currentValue);
                        localBestCached = globalBestValue.load(memory_order_relaxed);
                    }
                    continue;
                }

                // Odcinanie
                if (upperBound(node.idx, node.currentValue, node.currentWeight) <= localBestCached) {
                    continue;
                }

                // Nie bierzemy przedmiotu
                localStack.emplace_back(node.idx + 1, node.currentValue, node.currentWeight);

                // Bierzemy przedmiot (o ile sie zmiesci)
                int nextW = node.currentWeight + (int)data.items[node.idx].weight;
                if (nextW <= data.C) {
                    localStack.emplace_back(
                        node.idx + 1,
                        node.currentValue + (int)data.items[node.idx].value,
                        nextW
                    );
                }
            }
        }

        // Glowna petla iteracyjna
        void solve() {
            queues[0]->emplace_back(0, 0, 0);
            tasksInFlight = 1;

            #pragma omp parallel
            {
                int myID = omp_get_thread_num();
                mt19937 rng(myID + 1337);
                uniform_int_distribution<int> dist(0, numThreads - 1);

                vector<Node> localStack;
                localStack.reserve((size_t)data.n + 1);

                Node currentNode(0, 0, 0);
                
                bool hasWork;
                while (!finished) {
                    hasWork = false;

                    // branie ze swojego stosu (LIFO)
                    if (queueMutexes[myID]->try_lock()) {
                        if (!queues[myID]->empty()) {
                            currentNode = queues[myID]->back();
                            queues[myID]->pop_back();
                            hasWork = true;
                        }
                        queueMutexes[myID]->unlock();
                    }

                    // kradziez od innego watku (FIFO)
                    if (!hasWork) {
                        if (tasksInFlight == 0) break;

                        int victimID = dist(rng);
                        if (victimID != myID) {
                            if (queueMutexes[victimID]->try_lock()) {
                                if (!queues[victimID]->empty()) {
                                    currentNode = queues[victimID]->front();
                                    queues[victimID]->pop_front();
                                    hasWork = true;
                                }
                                queueMutexes[victimID]->unlock();
                            }
                        }
                    }

                    if (hasWork) {
                        // Jesli poddrzewo jest male, robimy je lokalnie
                        if (data.n - currentNode.idx <= SEQUENTIAL_CUTOFF_THRESHOLD) {
                            processSubtreeLocal(currentNode, localStack);

                            long long prev = atomic_fetch_add(&tasksInFlight, -1);
                            if (prev == 1LL) finished = true;
                        }
                        // Jesli duze, dzielimy na czesci dla innych watkow
                        else {
                            int childrenCount = processNodeParallel(currentNode, myID);

                            long long diff = (long long)(childrenCount - 1);
                            if (diff != 0) {
                                long long prev = atomic_fetch_add(&tasksInFlight, diff);
                                if (prev + diff == 0LL) finished = true;
                            }
                        }
                    }
                    else {
                        this_thread::yield();
                    }
                }
            }
        }

        int processNodeParallel(const Node& node, int myID) {
            // Odcinanie
            if (upperBound(node.idx, node.currentValue, node.currentWeight) <= globalBestValue.load(memory_order_relaxed)) {
                return 0;
            }

            int childrenCount = 0;
            int nextW = node.currentWeight + (int)data.items[node.idx].weight;

            lock_guard<mutex> lock(*queueMutexes[myID]);

            // Nie bierzemy przedmiotu
            queues[myID]->emplace_back(
                node.idx + 1,
                node.currentValue,
                node.currentWeight
            );
            childrenCount++;

            // Bierzemy przedmiot (o ile sie zmiesci)
            if (nextW <= data.C) {
                queues[myID]->emplace_back(
                    node.idx + 1,
                    node.currentValue + (int)data.items[node.idx].value,
                    nextW
                );
                childrenCount++;
            }

            return childrenCount;
        }
    };
}

int solveParallel(const ProblemData& data) {
    Solver solver(data);
    solver.solve();
    return solver.getResult();
}
