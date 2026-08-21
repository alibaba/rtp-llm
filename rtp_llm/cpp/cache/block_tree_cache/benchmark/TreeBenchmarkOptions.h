#pragma once

#include <cstdint>
#include <string>

namespace rtp_llm::benchmark {

struct TreeOptions {
    // The only remaining Tree workload knob: sizes the shared
    // load/evict/store task pool. Default 4; the same online case can be run
    // with 8/32 for a task-pool comparison. It never changes logical
    // concurrency, scheduler thread count or the fixed forward sleep.
    size_t task_pool_size{4};

    static TreeOptions parse(int& argc, char**& argv);
    static void        printHelp();
};

}  // namespace rtp_llm::benchmark
