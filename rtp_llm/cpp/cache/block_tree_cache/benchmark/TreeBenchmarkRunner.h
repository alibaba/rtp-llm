#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkJsonWriter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/ModelProfile.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/OnlineTreeScheduler.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeBenchmarkOptions.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeWorkloadGenerator.h"

namespace rtp_llm {

class BlockTreeCache;
struct BlockTreePoolMetricsSnapshot;

namespace benchmark {

// Test seam for exercising the same adapter used by the online benchmark.
std::unique_ptr<OnlineCacheApi> makeBlockTreeCacheAdapterForTest(BlockTreeCache& cache);

// Online workload runner: fixed ~20k-node initial cache, 32,768-block
// device/host pools, one foreground scheduler thread driving 32 logical
// request contexts, load-before-forward admission, one 100ms sleep per READY
// batch and request refs held across forward. warmup 15s -> pressure check ->
// measured 60s -> finalize. The cache task pool is the only tunable dimension
// (--task-pool-size).
class TreeBenchmarkRunner {
public:
    TreeBenchmarkRunner(const ModelProfile& profile,
                        const TreeOptions&  options,
                        uint64_t            seed,
                        uint64_t            repetition_id,
                        int                 cuda_device,
                        double              max_device_memory_fraction,
                        const std::string&  output_json_path);

    bool run();

private:
    bool                            runOnlineBenchmark(const OnlineTreeWorkloadConfig& config);
    std::unique_ptr<BlockTreeCache> buildTreeCache(const OnlineTreeWorkloadConfig& config);
    size_t                          insertTopology(BlockTreeCache& cache, const std::vector<PathKeys>& paths);
    void                            addLatencyMetrics(const std::string& prefix, const std::vector<int64_t>& samples);
    void addDistributionMetrics(const std::string& prefix, const std::vector<int64_t>& samples);
    void writePressureMetrics(bool pressure_ready, const std::vector<BlockTreePoolMetricsSnapshot>& snapshots);
    void writeFinalZeroMetrics(const std::vector<BlockTreePoolMetricsSnapshot>& snapshots);

    const ModelProfile& profile_;
    TreeOptions         options_;
    uint64_t            seed_;
    uint64_t            repetition_id_;
    int                 cuda_device_;
    double              max_device_memory_fraction_;
    std::string         output_json_path_;
    BenchmarkJsonWriter writer_;
};

}  // namespace benchmark

}  // namespace rtp_llm
