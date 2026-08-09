#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace rtp_llm::benchmark {

using PathKeys   = std::vector<int64_t>;
using SharedPath = std::shared_ptr<const PathKeys>;

struct WorkloadMetadata {
    size_t actual_node_count{0};
    size_t max_depth{0};
    size_t leaf_count{0};
};

struct StatefulPathConfig {
    size_t max_path_length{1000};
    size_t initial_min_path_length{128};
    size_t initial_max_path_length{768};
    size_t append_length{32};
    size_t inserts_per_match{4};
    size_t active_path_limit{4096};
    double continuation_ratio{0.7};
    double fork_ratio{0.2};
    double fork_reuse_min_ratio{0.25};
    double fork_reuse_max_ratio{0.9};
    double hot_path_ratio{0.2};
};

enum class PathScenario : size_t {
    CONTINUATION = 0,
    FORK         = 1,
    COLD         = 2,
};

struct StatefulPathOperation {
    PathScenario            scenario{PathScenario::COLD};
    SharedPath              match_path;
    std::vector<SharedPath> insert_paths;
    size_t                  planned_reuse_prefix_length{0};
    size_t                  planned_new_node_count{0};
};

// Per-worker logical session state. It is used before a benchmark phase to
// generate an immutable operation trace, so candidate selection and RNG never
// add a benchmark-side lock to the measured cache operations.
class StatefulPathSession {
public:
    StatefulPathSession(uint64_t                     seed,
                        uint64_t                     key_space_id,
                        const std::vector<PathKeys>& initial_paths,
                        size_t                       worker_id,
                        size_t                       num_workers,
                        const StatefulPathConfig&    config);

    StatefulPathOperation nextOperation();

private:
    SharedPath selectCandidate();
    SharedPath createPath(const SharedPath& prefix, size_t prefix_length, size_t total_length);
    void       addCandidate(const SharedPath& path, const SharedPath& parent, PathScenario scenario);
    int64_t    nextUniqueKey();

    StatefulPathConfig      config_;
    std::mt19937_64         rng_;
    int64_t                 next_unique_key_{0};
    std::vector<SharedPath> active_paths_;
    std::vector<SharedPath> epoch_candidates_;
    std::vector<SharedPath> hot_paths_;
};

class TreeWorkloadGenerator {
public:
    TreeWorkloadGenerator(uint64_t                  seed,
                          size_t                    tree_node_count,
                          size_t                    tree_branching_factor,
                          const StatefulPathConfig& config);

    // Generate a bounded trie whose paths have varied lengths and share
    // prefixes through continuation/fork construction.
    WorkloadMetadata generateTopology();

    const std::vector<PathKeys>& treePaths() const {
        return tree_paths_;
    }

private:
    PathKeys createTopologyPath(const PathKeys* prefix, size_t prefix_length, size_t total_length);
    int64_t  nextTopologyKey();

    uint64_t              seed_;
    size_t                tree_node_count_;
    size_t                tree_branching_factor_;
    StatefulPathConfig    config_;
    WorkloadMetadata      metadata_;
    std::vector<PathKeys> tree_paths_;
    std::mt19937_64       rng_;
    int64_t               next_topology_key_{1};
};

}  // namespace rtp_llm::benchmark
