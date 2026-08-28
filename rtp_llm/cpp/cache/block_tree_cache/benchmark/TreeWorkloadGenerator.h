#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rtp_llm::benchmark {

using PathKeys = std::vector<int64_t>;

// Fixed online workload constants. The formal benchmark never exposes these as
// CLI options; everything is recorded into the resolved config so results stay
// self-describing. A test-only small config is available for the GPU smoke and
// is injected through an internal seam, never as a public benchmark option.
struct OnlineTreeWorkloadConfig {
    size_t tokens_per_block{256};
    size_t logical_concurrency{32};
    size_t active_token_budget{1'048'576};
    size_t forward_sleep_ms{100};
    size_t request_lifecycle_timeout_ms{60'000};

    size_t shared_base_nodes{3'711};
    size_t background_tree_nodes{16'289};
    size_t device_pool_blocks{32'768};
    size_t host_pool_blocks{32'768};
    double device_watermark_ratio{0.8};
    double host_watermark_ratio{0.9};

    size_t operation_trace_count{20'000};
    size_t warmup_seconds{15};
    size_t measured_seconds{60};
    // Warmup must complete at least this many request transactions for the
    // pressure check to pass.
    size_t warmup_completed_requests_min{256};
    // Bounded retries when admission allocation hits a temporarily exhausted
    // pool; the scheduler only waits for the cache's event-driven watermark
    // eviction to free capacity, it never evicts directly.
    size_t admission_allocation_retry_limit{500};

    // 20 fixed length buckets (tokens) with weights and 13 equal-probability
    // hit-rate buckets.
    std::vector<size_t> length_buckets_tokens{8'000,   14'000,  32'000,  48'000,  92'000,  96'000,  117'000,
                                              120'000, 128'000, 135'000, 141'000, 150'000, 165'000, 200'000,
                                              235'000, 320'000, 480'000, 640'000, 800'000, 950'000};
    std::vector<size_t> length_weights{2, 3, 5, 10, 5, 20, 5, 20, 5, 10, 5, 5, 3, 1, 1, 4, 3, 3, 2, 1};
    std::vector<size_t> hit_rates_percent{0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99};

    // Test-only small config for the online lifecycle smoke. It keeps the same
    // request shape (shared base + unique suffix, load-before-forward, fixed
    // batch sleep) but runs in seconds instead of minutes.
    static OnlineTreeWorkloadConfig smokeTestConfig();
};

// One deterministic trace entry. Paths are pre-generated before setup; the
// timed region never performs RNG or path generation. Entries carry dependency
// metadata for the family-based continuation model: BASE requests start a new
// epoch, CONTINUATION requests inherit the parent path and append a unique tail.
struct OnlineRequestDescriptor {
    size_t   request_id{0};
    PathKeys path;  // full request path
    size_t   planned_reuse_blocks{0};
    size_t   input_blocks{0};
    size_t   target_tokens{0};

    // Family-based dependency tracking for append-only continuation.
    size_t  family_id{0};        // 0..31, groups requests into a logical session
    size_t  epoch_id{0};         // monotonically increasing within one family
    size_t  generation{0};       // 0 = BASE (new epoch), 1+ = CONTINUATION
    int64_t predecessor_id{-1};  // trace index of the parent, or -1 for BASE
    bool    is_continuation{false};
};

struct OnlineWorkloadMetadata {
    size_t actual_node_count{0};
    size_t shared_base_nodes{0};
    size_t background_tree_nodes{0};
    size_t base_request_count{0};
    size_t continuation_request_count{0};
    // workload_definition_hash covers the fixed workload protocol but not
    // task-pool size, so tp4/tp8 can be mechanically paired.
    uint64_t workload_definition_hash{0};
};

// Online workload generator: one 3,711-block shared base plus a background
// tree (together ~20k nodes) and a deterministic request trace whose paths
// reuse a planned shared-base prefix and diverge into a request-unique suffix
// key space. The same seed reproduces the exact same topology and trace.
class TreeWorkloadGenerator {
public:
    explicit TreeWorkloadGenerator(uint64_t seed, const OnlineTreeWorkloadConfig& config = OnlineTreeWorkloadConfig());

    // Deterministic topology: shared base path + background branches. Returns
    // the total node count and fills topologyPaths().
    OnlineWorkloadMetadata       generateTopology();
    const std::vector<PathKeys>& topologyPaths() const {
        return topology_paths_;
    }

    // Deterministic trace of config_.operation_trace_count entries. Idempotent
    // for the same seed; trace entries only describe workload paths and never
    // occupy cache blocks/refs.
    void                                        generateTrace();
    const std::vector<OnlineRequestDescriptor>& trace() const {
        return trace_;
    }

    uint64_t traceHash() const {
        return trace_hash_;
    }
    // workload_definition_hash covers the fixed workload protocol (distributions,
    // capacities, seed-independent topology) but not task-pool size. tp4/tp8
    // with the same workload_definition_hash are mechanically paired.
    uint64_t workloadDefinitionHash() const {
        return workload_definition_hash_;
    }
    static std::string hashHex(uint64_t hash);

    const OnlineTreeWorkloadConfig& config() const {
        return config_;
    }

    // BASE/CONTINUATION breakdown after generateTrace().
    size_t baseRequestCount() const {
        return base_request_count_;
    }
    size_t continuationRequestCount() const {
        return continuation_request_count_;
    }

private:
    int64_t nextTopologyKey();
    int64_t nextSuffixKey(size_t request_index, size_t suffix_index);

    uint64_t                             seed_;
    OnlineTreeWorkloadConfig             config_;
    std::vector<PathKeys>                topology_paths_;
    std::vector<OnlineRequestDescriptor> trace_;
    uint64_t                             trace_hash_{0};
    uint64_t                             workload_definition_hash_{0};
    size_t                               base_request_count_{0};
    size_t                               continuation_request_count_{0};
    int64_t                              next_topology_key_{0};
};

}  // namespace rtp_llm::benchmark
