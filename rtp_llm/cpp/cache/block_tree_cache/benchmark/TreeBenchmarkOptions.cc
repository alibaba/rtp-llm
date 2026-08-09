#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeBenchmarkOptions.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace rtp_llm::benchmark {

namespace {

std::pair<std::string, std::string> parseArg(const char* arg) {
    if (arg[0] != '-' || arg[1] != '-')
        return {};
    std::string full = arg + 2;
    auto        eq   = full.find('=');
    if (eq == std::string::npos)
        return {full, ""};
    return {full.substr(0, eq), full.substr(eq + 1)};
}

}  // anonymous namespace

TreeOptions TreeOptions::parse(int& argc, char**& argv) {
    TreeOptions opts;
    int         write_idx = 1;

    for (int i = 1; i < argc; ++i) {
        auto [key, value] = parseArg(argv[i]);
        if (key.empty()) {
            argv[write_idx++] = argv[i];
            continue;
        }

        auto next = [&]() -> std::string {
            if (!value.empty())
                return value;
            if (i + 1 < argc) {
                ++i;
                return argv[i];
            }
            throw std::runtime_error("Missing value for --" + key);
        };

        auto parseInt = [&]() -> size_t {
            const auto text   = next();
            size_t     parsed = 0;
            const auto result = std::stoull(text, &parsed);
            if (text.empty() || text.front() == '-' || parsed != text.size())
                throw std::runtime_error("Invalid integer for --" + key + ": " + text);
            return result;
        };
        auto parseDouble = [&]() -> double {
            const auto text   = next();
            size_t     parsed = 0;
            const auto result = std::stod(text, &parsed);
            if (parsed != text.size())
                throw std::runtime_error("Invalid number for --" + key + ": " + text);
            return result;
        };

        if (key == "payload-mode")
            opts.payload_mode = next();
        else if (key == "tree-node-count")
            opts.tree_node_count = parseInt();
        else if (key == "max-path-length")
            opts.max_path_length = parseInt();
        else if (key == "tree-branching-factor")
            opts.tree_branching_factor = parseInt();
        else if (key == "initial-min-path-length")
            opts.initial_min_path_length = parseInt();
        else if (key == "initial-max-path-length")
            opts.initial_max_path_length = parseInt();
        else if (key == "continuation-ratio")
            opts.continuation_ratio = parseDouble();
        else if (key == "fork-ratio")
            opts.fork_ratio = parseDouble();
        else if (key == "fork-reuse-min-ratio")
            opts.fork_reuse_min_ratio = parseDouble();
        else if (key == "fork-reuse-max-ratio")
            opts.fork_reuse_max_ratio = parseDouble();
        else if (key == "hot-path-ratio")
            opts.hot_path_ratio = parseDouble();
        else if (key == "active-path-limit")
            opts.active_path_limit = parseInt();
        else if (key == "append-length")
            opts.append_length = parseInt();
        else if (key == "inserts-per-match")
            opts.inserts_per_match = parseInt();
        else if (key == "operation-trace-count")
            opts.operation_trace_count = parseInt();
        else if (key == "steady-threads")
            opts.steady_threads = parseInt();
        else if (key == "warmup-seconds")
            opts.warmup_seconds = parseInt();
        else if (key == "min-measured-seconds")
            opts.min_measured_seconds = parseInt();
        else if (key == "help") {
            printHelp();
            std::exit(0);
        } else {
            argv[write_idx++] = argv[i];
        }
    }

    argc = write_idx;
    return opts;
}

void TreeOptions::printHelp() {
    std::cout << "Tree benchmark options (stateful match-then-insert workload):\n"
              << "  --payload-mode=MODE           model_sized (default) | scaled\n"
              << "  --tree-node-count=N           Steady-state target tree nodes (default: 100000)\n"
              << "  --max-path-length=N           Maximum full path length (default: 1000)\n"
              << "  --tree-branching-factor=N     Maximum children per node (default: 16)\n"
              << "  --initial-min-path-length=N   Minimum initial/request path length (default: 128)\n"
              << "  --initial-max-path-length=N   Maximum initial/request path length (default: 768)\n"
              << "  --continuation-ratio=RATIO    Continue an existing path (default: 0.7)\n"
              << "  --fork-ratio=RATIO            Fork from an existing prefix (default: 0.2)\n"
              << "  --fork-reuse-min-ratio=RATIO  Minimum reused prefix fraction (default: 0.25)\n"
              << "  --fork-reuse-max-ratio=RATIO  Maximum reused prefix fraction (default: 0.9)\n"
              << "  --hot-path-ratio=RATIO        Select from the hot subset (default: 0.2)\n"
              << "  --active-path-limit=N         Per-worker candidate pool bound (default: 4096)\n"
              << "  --append-length=N             New nodes per incremental insert (default: 32)\n"
              << "  --inserts-per-match=N         Incremental inserts after each match (default: 4)\n"
              << "  --operation-trace-count=N     Pre-generated transactions per phase (default: 20000)\n"
              << "  --steady-threads=N            Worker threads (default: 8, 1=baseline)\n"
              << "  --warmup-seconds=N            Warmup duration (default: 10)\n"
              << "  --min-measured-seconds=N      Measured duration floor (default: 30)\n"
              << "  --help                        Show this help\n";
}

}  // namespace rtp_llm::benchmark
