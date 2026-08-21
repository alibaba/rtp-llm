#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeBenchmarkOptions.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkArgumentParser.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace rtp_llm::benchmark {

TreeOptions TreeOptions::parse(int& argc, char**& argv) {
    TreeOptions opts;
    consumeOptions(argc, argv, [&](const std::string& key, const NextArgumentValue& next) {
        if (key == "task-pool-size") {
            opts.task_pool_size = parseUnsigned(key, next);
            if (opts.task_pool_size == 0) {
                throw std::runtime_error("--task-pool-size must be positive");
            }
        } else if (key == "help") {
            printHelp();
            std::exit(0);
        } else
            return false;
        return true;
    });
    return opts;
}

void TreeOptions::printHelp() {
    std::cout << "Tree benchmark options (online scheduler lifecycle workload):\n"
              << "  --task-pool-size=N        BlockTreeCache shared load/evict/store task pool (default: 4, > 0)\n"
              << "  --help                    Show this help\n"
              << "\n"
              << "All workload constants (C32 logical concurrency, 20k-node initial cache,\n"
              << "32,768-block device/host pools, 20 length buckets, 13 hit-rate buckets,\n"
              << "100ms forward sleep, 15s warmup, 60s measured) are fixed and recorded in\n"
              << "the resolved config; they are not configurable from the CLI.\n";
}

}  // namespace rtp_llm::benchmark
