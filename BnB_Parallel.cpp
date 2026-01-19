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

    struct Node {
        int idx;
        int currentValue, currentWeight;
        Node(int i, int v, int w) : idx(i), currentValue(v), currentWeight(w) {}
    };

    class Solver {
        const ProblemData& data;

        atomic<int> globalBestValue;

        alignas(64) atomic<long long> tasksInFlight;
        alignas(64) atomic<bool> finished;

        vector<int> prefWeight, prefValue;

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

            int n = data.n;
            prefWeight.resize((size_t)n + 1, 0);
            prefValue.resize((size_t)n + 1, 0);

            for (int i = 0; i < n; i++) {
                prefWeight[i + 1] = prefWeight[i] + (int)data.items[i].weight;
                prefValue[i + 1] = prefValue[i] + (int)data.items[i].value;
            }

            // Greedy Initialization
            int cV = 0, cW = 0;
            for (const auto& item : data.items) {
                if (cW + item.weight <= data.C) {
                    cV += (int)item.value;
                    cW += (int)item.weight;
                }
                else break;
            }
            globalBestValue.store(cV, memory_order_relaxed);
        }

        int getResult() const {
            return globalBestValue.load();
        }

        // __forceinline
        __declspec(noinline)
            double upperBound(int idx, int currentValue, int currentWeight) {
            int remainingCap = (int)data.C - currentWeight;

            if (remainingCap <= 0) return (double)currentValue;
            if (idx >= data.n) return (double)currentValue;

            auto it = upper_bound(prefWeight.begin() + idx, prefWeight.end(), remainingCap + prefWeight[idx]);

            int breakIndex = (int)distance(prefWeight.begin(), it) - 1;

            double bound = (double)currentValue + (double)(prefValue[breakIndex] - prefValue[idx]);

            if (breakIndex < data.n) {
                int weightTakenSoFar = prefWeight[breakIndex] - prefWeight[idx];
                bound += (double)(remainingCap - weightTakenSoFar) * data.items[breakIndex].ratio;
            }
            return bound;
        }

        // Lock-free update (CAS loop)
        void updateBestValue(int val) {
            int currentBest = globalBestValue.load(memory_order_relaxed);
            while (val > currentBest) {
                if (globalBestValue.compare_exchange_weak(currentBest, val, memory_order_relaxed)) {
                    break;
                }
                // Jesli sie nie udalo (ktos inny zmienil w miedzyczasie), currentBest jest automatycznie aktualizowany
            }
        }

        // Przetwarzanie sekwencyjne dla malych poddrzew
        void processSubtreeLocal(Node rootNode, vector<Node>& localStack) {
            localStack.clear();
            localStack.push_back(rootNode);

            // Pobieramy raz lokalnie, zeby nie obciazac cache atomicami w kazdej iteracji,
            // ale odswiezamy jesli nasza wartosc spadnie ponizej globalnej (rzadko).
            int localBestCached = globalBestValue.load(memory_order_relaxed);

            while (!localStack.empty()) {
                Node node = localStack.back();
                localStack.pop_back();

                if (node.idx == data.n) {
                    if (node.currentValue > localBestCached) {
                        updateBestValue(node.currentValue);
                        localBestCached = globalBestValue.load(memory_order_relaxed);
                    }
                    continue;
                }

                // Wazne: Tu odczytujemy atomica (lub jego cache), zeby wiedziec o postepach innych watkow
                if (upperBound(node.idx, node.currentValue, node.currentWeight) <= localBestCached) {
                    // Od czasu do czasu warto odswiezyc cache z globala, jesli dlugo mielimy
                    // W prostej wersji po prostu sprawdzamy upper bound vs localBestCached
                    // Dla wiekszej precyzji mozna co X iteracji robic reload globalBestValue
                    continue;
                }

                // Synchronizacja "lazy" - jesli upper bound jest bardzo blisko localBest, sprawdzmy czy global sie nie zmienil
                // (opcjonalna optymalizacja)

                localStack.emplace_back(node.idx + 1, node.currentValue, node.currentWeight);

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

        void solve() {
            queues[0]->emplace_back(0, 0, 0);
            tasksInFlight = 1;

#pragma omp parallel
            {
                int myID = omp_get_thread_num();
                mt19937 rng(myID + 1337);
                uniform_int_distribution<int> dist(0, numThreads - 1);

                vector<Node> localStack;
                localStack.reserve(200); // Wystarczy na glebokosc rekurencji

                Node currentNode(0, 0, 0);
                bool hasWork = false;

                while (!finished) {
                    hasWork = false;

                    // Queue Access (LIFO)
                    if (queueMutexes[myID]->try_lock()) {
                        if (!queues[myID]->empty()) {
                            currentNode = queues[myID]->back();
                            queues[myID]->pop_back();
                            hasWork = true;
                        }
                        queueMutexes[myID]->unlock();
                    }

                    // Stealing (FIFO)
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
                        // Warunek odciecia zalezy od tego ile zostalo do konca (n - idx)
                        // Jesli poddrzewo jest male (np. < 20 elementow), robimy je lokalnie.
                        int itemsLeft = data.n - currentNode.idx;

                        if (itemsLeft <= SEQUENTIAL_CUTOFF_THRESHOLD) {
                            processSubtreeLocal(currentNode, localStack);

                            long long prev = atomic_fetch_add(&tasksInFlight, -1);
                            if (prev - 1 == 0) finished = true;
                        }
                        else {
                            // Jesli duze, dzielimy na czesci dla innych watkow
                            int childrenCount = processNodeParallel(currentNode, myID);

                            long long diff = childrenCount - 1;
                            if (diff != 0) {
                                long long prev = atomic_fetch_add(&tasksInFlight, diff);
                                if (prev + diff == 0) finished = true;
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
            // Sprawdzamy bound na atomiku
            if (upperBound(node.idx, node.currentValue, node.currentWeight) <= globalBestValue.load(memory_order_relaxed)) {
                return 0;
            }

            int childrenCount = 0;
            int nextW = node.currentWeight + (int)data.items[node.idx].weight;

            lock_guard<mutex> lock(*queueMutexes[myID]);

            // Preferujemy wrzucenie "Take" pozniej, zeby LIFO (back) wzielo je pierwsze
            // (Heurystyka: przedmioty o wysokim ratio warto brac)

            // 1. Don't take
            queues[myID]->emplace_back(
                node.idx + 1,
                node.currentValue,
                node.currentWeight
            );
            childrenCount++;

            // 2. Take
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