#include "aios/navi_sched/service/AbortMetrics.h"

#include <cstdlib>
#include <thread>
#include <vector>

void check(bool condition)
{
    if (!condition) {
        std::abort();
    }
}

int main()
{
    navi_sched::AbortMetrics metrics;
    for (size_t i = 0; i < metrics.STAGE_COUNT; ++i) {
        check(metrics.count(i) == 0);
        check(metrics.record(metrics.stages()[i], i + 1));
        check(metrics.count(i) == i + 1);
    }
    check(!metrics.record("request-id-must-not-be-a-stage", 10));
    std::vector<std::thread> threads;
    for (size_t i = 0; i < 8; ++i) {
        threads.emplace_back([&metrics]() {
            for (size_t j = 0; j < 10000; ++j) {
                check(metrics.record("received", 1));
            }
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    check(metrics.count(0) == 80001);
    for (size_t i = 1; i < metrics.STAGE_COUNT; ++i) {
        check(metrics.count(i) == i + 1);
    }
}
