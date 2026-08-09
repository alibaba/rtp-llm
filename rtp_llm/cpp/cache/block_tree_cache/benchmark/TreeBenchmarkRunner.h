#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkFixture.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkJsonWriter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/ModelProfile.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeBenchmarkOptions.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeWorkloadGenerator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"

namespace rtp_llm::benchmark {

// Aggregated counters across steady-state worker threads.
struct SteadyCounters {
    std::atomic<size_t>                insert_calls{0};
    std::atomic<size_t>                insert_path_keys{0};
    std::atomic<size_t>                insert_new_nodes{0};
    std::atomic<size_t>                match_requests{0};
    std::atomic<size_t>                match_keys{0};
    std::atomic<size_t>                match_device_blocks{0};
    std::atomic<size_t>                match_host_blocks{0};
    std::atomic<size_t>                trace_exhaustions{0};
    std::atomic<size_t>                loads_committed{0};
    std::atomic<size_t>                loads_succeeded{0};
    std::atomic<size_t>                loads_failed{0};
    std::atomic<size_t>                loads_cancelled{0};
    std::atomic<size_t>                load_target_allocation_retries{0};
    std::atomic<size_t>                load_target_allocation_failed{0};
    std::atomic<size_t>                load_commit_failed{0};
    std::atomic<size_t>                loads_pending_at_measurement_end{0};
    std::array<std::atomic<size_t>, 3> scenario_requests{};
    std::array<std::atomic<size_t>, 3> scenario_match_keys{};
    std::array<std::atomic<size_t>, 3> scenario_matched_depth{};
    std::array<std::atomic<size_t>, 3> scenario_insert_calls{};
    std::array<std::atomic<size_t>, 3> scenario_insert_path_keys{};
    std::array<std::atomic<size_t>, 3> scenario_insert_new_nodes{};
    std::array<std::atomic<size_t>, 3> scenario_device_hits{};
    std::array<std::atomic<size_t>, 3> scenario_host_hits{};
    std::array<std::atomic<size_t>, 3> scenario_disk_hits{};
    std::array<std::atomic<size_t>, 3> scenario_misses{};
};

// Per-call latency sample set collected by workers, aggregated by the runner.
struct LatencySamples {
    std::vector<int64_t> insert_ns;
    std::vector<int64_t> match_ns;
    std::vector<int64_t> load_ns;
};

// Steady-state mixed workload runner: build tree -> warmup -> measured window
// of interleaved insert/match. Eviction is event-driven (insert commits trigger
// the cache's watermark check); after a bounded admission wait, workers may use
// the cache's request-admission reclaim fallback. Match misses fire asynchronous
// lower-tier loads.
class TreeBenchmarkRunner {
public:
    TreeBenchmarkRunner(const ModelProfile& profile,
                        const TreeOptions&  options,
                        uint64_t            seed,
                        const std::string&  output_json_path);

    bool run();

private:
    const ModelProfile& profile_;
    TreeOptions         options_;
    uint64_t            seed_;
    std::string         output_json_path_;
    BenchmarkJsonWriter writer_;

    bool runSteadyStateMeasurement();

    // One worker's steady loop for `seconds`, consuming its immutable trace.
    // On pool exhaustion workers wait briefly for watermark eviction (triggered
    // by concurrent commits), then use the cache's request-admission reclaim
    // fallback and retry. Match misses allocate device targets and commit an
    // async load.
    void workerLoop(BlockTreeCache&                                 cache,
                    const std::vector<StatefulPathOperation>&       trace,
                    double                                          seconds,
                    SteadyCounters&                                 counters,
                    LatencySamples&                                 latencies,
                    std::mutex&                                     merge_mutex,
                    std::vector<std::shared_ptr<LoadAsyncContext>>& pending_loads,
                    size_t&                                         executed_transactions);

    // Run num_workers steady loops for `seconds`; aggregates into `counters`.
    void runSteadyWorkers(BlockTreeCache&                                        cache,
                          const std::vector<std::vector<StatefulPathOperation>>& traces,
                          size_t                                                 num_workers,
                          double                                                 seconds,
                          SteadyCounters&                                        counters,
                          LatencySamples&                                        latencies,
                          std::vector<size_t>&                                   node_samples,
                          std::vector<std::shared_ptr<LoadAsyncContext>>&        pending_loads,
                          std::vector<size_t>&                                   executed_transactions);

    void drainLoads(std::vector<std::shared_ptr<LoadAsyncContext>>& pending_loads, SteadyCounters& counters);

    // Helper to build a fixture for tree operations. The shared async task pool
    // is fixed across worker-count cases; watermark ratios arm event-driven
    // eviction (0.0 = tier disabled).
    std::unique_ptr<BlockTreeCache> buildTreeCache(size_t             node_count,
                                                   const std::string& payload_mode,
                                                   bool               enable_host            = true,
                                                   double             device_watermark_ratio = 0.0,
                                                   double             host_watermark_ratio   = 0.0);

    // Insert one complete path while allocating resources only for the suffix
    // after `existing_prefix_length`.
    bool insertPathFromPrefix(BlockTreeCache& cache, const PathKeys& path, size_t existing_prefix_length);

    // Materialize every generated topology path; returns newly inserted nodes.
    size_t insertTopology(BlockTreeCache& cache, const std::vector<PathKeys>& paths);
};

}  // namespace rtp_llm::benchmark
