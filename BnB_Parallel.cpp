#include "Common.h"
#include <algorithm>
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <thread>
#include <omp.h>
#include <random>
#include <memory> // Dla unique_ptr

using namespace std;

namespace {

    struct Node {
        int idx;
        int currentValue, currentWeight;
        Node(int i, int v, int w) : idx(i), currentValue(v), currentWeight(w) {}
    };

    class Solver {
        const ProblemData& data;

        int globalBestValue;
        mutex bestValueMutex;
        alignas(64) atomic<long long> tasksInFlight; // licznik zadañ, do wykrycia momentu zakoñczenia
        alignas(64) atomic<bool> finished; // flaga zatrzymania

        // Tablice do optymalizacji obliczania Upper Bound
        vector<int> prefWeight, prefValue;

        // struktury Work Stealing
        vector<unique_ptr<deque<Node>>> queues;
        vector<unique_ptr<mutex>> queueMutexes;
        int numThreads;

    public:
        Solver(const ProblemData& d)
            : data(d), globalBestValue(0), tasksInFlight(0), finished(false) {

            numThreads = omp_get_max_threads(); // ustalenie liczby w¹tków
            if (numThreads < 1) numThreads = 1; // zabezpieczenie

            // rezerwacja pamiêci
            queues.resize(numThreads);
            queueMutexes.resize(numThreads);

            // Inicjalizacja kolejek i mutexów
            for (int i = 0; i < numThreads; i++) {
                queues[i] = make_unique<deque<Node>>();
                queueMutexes[i] = make_unique<mutex>();
            }

            // Prekomputacja sum prefiksowych (do obliczania ub)
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
            globalBestValue = cV;
        }

        // getter do wyniku
        int getResult() const {
            return globalBestValue;
        }

        //__forceinline 
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

        void updateBestValue(int val) {
            if (val > globalBestValue) {
                lock_guard<mutex> lock(bestValueMutex);
                if (val > globalBestValue) {
                    globalBestValue = val;
                }
            }
        }

        void solve() {
            // Wrzucamy korzeñ do kolejki w¹tku 0
            queues[0]->emplace_back(0, 0, 0);
            tasksInFlight = 1;

#pragma omp parallel
            {
                int myID = omp_get_thread_num();

                // rng do wyboru w¹tku od którego kradniemy
                mt19937 rng(myID + 1337);
                uniform_int_distribution<int> dist(0, numThreads - 1);

                Node currentNode(0, 0, 0);
                bool hasWork = false;

                while (!finished) {
                    hasWork = false;

                    // Próba pobrania ze swojej kolejki (LIFO)
                    if (queueMutexes[myID]->try_lock()) {
                        if (!queues[myID]->empty()) {
                            currentNode = queues[myID]->back();
                            queues[myID]->pop_back();
                            hasWork = true;
                        }
                        queueMutexes[myID]->unlock();
                    }

                    // Sprawdzenie czy jest co kraœæ jeszcze
                    if (!hasWork) {
                        if (tasksInFlight == 0) {
                            break;
                        }

                        // Je¿eli jest co kraœæ, to losuje od kogo i próbuje ukraœæ (FIFO)
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

                    // Przetwarzanie
                    if (hasWork) {
                        int children = processNode(currentNode, myID);
                        long long diff = children - 1;

                        // Zmnieniamy licznik
                        if (diff != 0) {
                            long long prev = atomic_fetch_add(&tasksInFlight, diff);
                            if (prev + diff == 0) {
                                finished = true; // Je¿eli to by³o ostatnie zadanie to zakoñcz algorytm
                            }
                        }
                    }
                    else {
                        // Odpoczynek dla procesora, krótka pauza
                        this_thread::yield();
                    }
                }
            }
        }

        int processNode(const Node& node, int myID) {
            // Sprawdzamy czy to liœæ
            if (node.idx == data.n) {
                updateBestValue(node.currentValue);
                return 0;
            }

            // Odcinanie
            if (upperBound(node.idx, node.currentValue, node.currentWeight) <= globalBestValue) {
                return 0;
            }

            int childrenCount = 1;
            int nextW = node.currentWeight + (int)data.items[node.idx].weight;
            if (nextW <= data.C) {
                lock_guard<mutex> lock(*queueMutexes[myID]);
                queues[myID]->emplace_back(
                    node.idx + 1,
                    node.currentValue + (int)data.items[node.idx].value,
                    nextW
                );
                childrenCount++;
                queues[myID]->emplace_back(
                    node.idx + 1,
                    node.currentValue,
                    node.currentWeight
                );
            }
            else {
                lock_guard<mutex> lock(*queueMutexes[myID]);
                queues[myID]->emplace_back(
                    node.idx + 1,
                    node.currentValue,
                    node.currentWeight
                );
            }

            return childrenCount;
        }
    };
}

// Funkcja dostêpna publicznie
int solveParallel(const ProblemData& data) {
    Solver solver(data);
    solver.solve();
    return solver.getResult();
}