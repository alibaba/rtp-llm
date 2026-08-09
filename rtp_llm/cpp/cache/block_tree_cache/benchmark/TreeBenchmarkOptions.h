#pragma once

#include <cstdint>
#include <string>

namespace rtp_llm::benchmark {

struct TreeOptions {
    // Payload mode
    std::string payload_mode{"model_sized"};  // "model_sized" or "scaled"

    // Tree topology (steady-state target node count).
    size_t tree_node_count{100000};
    size_t max_path_length{1000};
    size_t tree_branching_factor{16};

    // Initial tree paths and steady-state request prefixes.
    size_t initial_min_path_length{128};
    size_t initial_max_path_length{768};

    // Stateful request mix: continue a cached path, fork from a cached prefix,
    // or use a completely new path (the remainder probability).
    double continuation_ratio{0.7};
    double fork_ratio{0.2};
    double fork_reuse_min_ratio{0.25};
    double fork_reuse_max_ratio{0.9};
    double hot_path_ratio{0.2};
    size_t active_path_limit{4096};

    // One transaction matches a request path, then commits this many
    // incremental extensions. Each insert passes the complete path to the
    // cache but allocates resources only for the new suffix.
    size_t append_length{32};
    size_t inserts_per_match{4};

    // Immutable operations pre-generated outside the measured interval. The
    // total is partitioned across workers and is not replayed on exhaustion.
    size_t operation_trace_count{20000};

    // Worker threads running the steady loop (1 = single-thread baseline)
    size_t steady_threads{8};
    // Duration floors (seconds)
    size_t warmup_seconds{10};
    size_t min_measured_seconds{30};

    static TreeOptions parse(int& argc, char**& argv);
    static void        printHelp();
};

}  // namespace rtp_llm::benchmark
