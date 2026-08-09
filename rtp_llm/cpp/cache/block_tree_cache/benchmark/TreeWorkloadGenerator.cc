#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeWorkloadGenerator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rtp_llm::benchmark {
namespace {

void validateConfig(const StatefulPathConfig& config) {
    if (config.max_path_length == 0 || config.initial_min_path_length == 0
        || config.initial_min_path_length > config.initial_max_path_length
        || config.initial_max_path_length > config.max_path_length || config.append_length == 0
        || config.inserts_per_match == 0 || config.active_path_limit == 0
        || config.append_length > config.max_path_length / config.inserts_per_match) {
        throw std::invalid_argument("invalid stateful path length configuration");
    }
    if (!std::isfinite(config.continuation_ratio) || !std::isfinite(config.fork_ratio)
        || !std::isfinite(config.fork_reuse_min_ratio) || !std::isfinite(config.fork_reuse_max_ratio)
        || !std::isfinite(config.hot_path_ratio) || config.continuation_ratio < 0.0 || config.fork_ratio < 0.0
        || config.continuation_ratio + config.fork_ratio > 1.0 || config.fork_reuse_min_ratio < 0.0
        || config.fork_reuse_min_ratio > config.fork_reuse_max_ratio || config.fork_reuse_max_ratio > 1.0
        || config.hot_path_ratio < 0.0 || config.hot_path_ratio > 1.0) {
        throw std::invalid_argument("invalid stateful path probability configuration");
    }
}

size_t randomSize(std::mt19937_64& rng, size_t min_value, size_t max_value) {
    if (min_value > max_value) {
        throw std::invalid_argument("invalid random range");
    }
    return std::uniform_int_distribution<size_t>(min_value, max_value)(rng);
}

double randomRatio(std::mt19937_64& rng) {
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}

}  // namespace

StatefulPathSession::StatefulPathSession(uint64_t                     seed,
                                         uint64_t                     key_space_id,
                                         const std::vector<PathKeys>& initial_paths,
                                         size_t                       worker_id,
                                         size_t                       num_workers,
                                         const StatefulPathConfig&    config):
    config_(config),
    rng_(seed ^ (key_space_id * 0x9e3779b97f4a7c15ULL) ^ (worker_id * 0xbf58476d1ce4e5b9ULL)),
    next_unique_key_(
        static_cast<int64_t>(1000000000000ULL + key_space_id * 10000000000ULL + worker_id * 100000000ULL)) {
    validateConfig(config_);
    if (num_workers == 0 || worker_id >= num_workers) {
        throw std::invalid_argument("invalid worker partition");
    }

    active_paths_.reserve(std::min(config_.active_path_limit, initial_paths.size()));
    for (size_t i = worker_id; i < initial_paths.size() && active_paths_.size() < config_.active_path_limit;
         i += num_workers) {
        active_paths_.push_back(std::make_shared<const PathKeys>(initial_paths[i]));
    }
    // Small trees may have fewer leaves than workers. Give those workers a
    // deterministic seed path instead of turning their workload into all-cold traffic.
    if (active_paths_.empty() && !initial_paths.empty()) {
        active_paths_.push_back(std::make_shared<const PathKeys>(initial_paths[worker_id % initial_paths.size()]));
    }

    const size_t hot_count = std::min<size_t>(active_paths_.size(), std::max<size_t>(1, active_paths_.size() / 10));
    hot_paths_.assign(active_paths_.begin(), active_paths_.begin() + hot_count);
}

int64_t StatefulPathSession::nextUniqueKey() {
    if (next_unique_key_ == std::numeric_limits<int64_t>::max()) {
        throw std::overflow_error("stateful workload key space exhausted");
    }
    return next_unique_key_++;
}

SharedPath StatefulPathSession::createPath(const SharedPath& prefix, size_t prefix_length, size_t total_length) {
    if (prefix_length > total_length || (prefix && prefix_length > prefix->size())) {
        throw std::invalid_argument("invalid path prefix length");
    }

    PathKeys path;
    path.reserve(total_length);
    if (prefix && prefix_length > 0) {
        path.insert(path.end(), prefix->begin(), prefix->begin() + prefix_length);
    }
    while (path.size() < total_length) {
        path.push_back(nextUniqueKey());
    }
    return std::make_shared<const PathKeys>(std::move(path));
}

SharedPath StatefulPathSession::selectCandidate() {
    if (active_paths_.empty()) {
        return nullptr;
    }

    if (!hot_paths_.empty() && randomRatio(rng_) < config_.hot_path_ratio) {
        return hot_paths_[randomSize(rng_, 0, hot_paths_.size() - 1)];
    }

    if (epoch_candidates_.empty()) {
        epoch_candidates_ = active_paths_;
        std::shuffle(epoch_candidates_.begin(), epoch_candidates_.end(), rng_);
    }
    const auto selected = epoch_candidates_.back();
    epoch_candidates_.pop_back();
    return selected;
}

void StatefulPathSession::addCandidate(const SharedPath& path, const SharedPath& parent, PathScenario scenario) {
    if (scenario == PathScenario::CONTINUATION && parent) {
        const auto active_it = std::find(active_paths_.begin(), active_paths_.end(), parent);
        if (active_it != active_paths_.end()) {
            *active_it        = path;
            const auto hot_it = std::find(hot_paths_.begin(), hot_paths_.end(), parent);
            if (hot_it != hot_paths_.end()) {
                *hot_it = path;
            }
            return;
        }
    }

    if (active_paths_.size() < config_.active_path_limit) {
        active_paths_.push_back(path);
        return;
    }

    // A bounded reservoir prevents a long benchmark from accumulating an
    // unbounded path catalog. New paths become visible at the next epoch.
    const size_t replacement   = randomSize(rng_, 0, active_paths_.size() - 1);
    const auto   evicted       = active_paths_[replacement];
    active_paths_[replacement] = path;
    const auto hot_it          = std::find(hot_paths_.begin(), hot_paths_.end(), evicted);
    if (hot_it != hot_paths_.end()) {
        *hot_it = path;
    }
}

StatefulPathOperation StatefulPathSession::nextOperation() {
    StatefulPathOperation operation;
    const size_t max_request_length = config_.max_path_length - config_.append_length * config_.inserts_per_match;
    if (max_request_length == 0) {
        throw std::invalid_argument("inserts leave no room for a match path");
    }

    const double roll = randomRatio(rng_);
    if (roll < config_.continuation_ratio) {
        operation.scenario = PathScenario::CONTINUATION;
    } else if (roll < config_.continuation_ratio + config_.fork_ratio) {
        operation.scenario = PathScenario::FORK;
    } else {
        operation.scenario = PathScenario::COLD;
    }

    SharedPath base;
    if (operation.scenario != PathScenario::COLD) {
        base = selectCandidate();
        if (!base) {
            operation.scenario = PathScenario::COLD;
        }
    }

    if (operation.scenario == PathScenario::CONTINUATION && base->size() > max_request_length) {
        // A session at the length ceiling can still branch from an earlier
        // prefix; it cannot be extended without violating max_path_length.
        operation.scenario = PathScenario::FORK;
    }

    if (operation.scenario == PathScenario::CONTINUATION) {
        operation.match_path                  = base;
        operation.planned_reuse_prefix_length = base->size();
    } else {
        const size_t request_min = std::min(config_.initial_min_path_length, max_request_length);
        const size_t request_max = std::min(config_.initial_max_path_length, max_request_length);
        const size_t request_len = randomSize(rng_, request_min, request_max);

        if (operation.scenario == PathScenario::FORK && request_len > 1 && !base->empty()) {
            const double reuse_ratio = std::uniform_real_distribution<double>(config_.fork_reuse_min_ratio,
                                                                              config_.fork_reuse_max_ratio)(rng_);
            const size_t raw_reuse   = static_cast<size_t>(std::floor(static_cast<double>(base->size()) * reuse_ratio));
            operation.planned_reuse_prefix_length =
                std::clamp<size_t>(raw_reuse, 1, std::min(base->size(), request_len - 1));
            operation.match_path = createPath(base, operation.planned_reuse_prefix_length, request_len);
        } else {
            operation.scenario                    = PathScenario::COLD;
            operation.planned_reuse_prefix_length = 0;
            operation.match_path                  = createPath(nullptr, 0, request_len);
        }
    }

    SharedPath previous              = operation.match_path;
    operation.planned_new_node_count = operation.match_path->size() - operation.planned_reuse_prefix_length;
    operation.insert_paths.reserve(config_.inserts_per_match);
    for (size_t i = 0; i < config_.inserts_per_match; ++i) {
        previous = createPath(previous, previous->size(), previous->size() + config_.append_length);
        operation.planned_new_node_count += config_.append_length;
        operation.insert_paths.push_back(previous);
    }
    addCandidate(previous, base, operation.scenario);
    return operation;
}

TreeWorkloadGenerator::TreeWorkloadGenerator(uint64_t                  seed,
                                             size_t                    tree_node_count,
                                             size_t                    tree_branching_factor,
                                             const StatefulPathConfig& config):
    seed_(seed),
    tree_node_count_(tree_node_count),
    tree_branching_factor_(tree_branching_factor),
    config_(config),
    rng_(seed) {
    validateConfig(config_);
    if (tree_node_count_ == 0 || tree_branching_factor_ == 0) {
        throw std::invalid_argument("tree topology dimensions must be positive");
    }
}

int64_t TreeWorkloadGenerator::nextTopologyKey() {
    if (next_topology_key_ == std::numeric_limits<int64_t>::max()) {
        throw std::overflow_error("tree topology key space exhausted");
    }
    return next_topology_key_++;
}

PathKeys TreeWorkloadGenerator::createTopologyPath(const PathKeys* prefix, size_t prefix_length, size_t total_length) {
    if (prefix_length > total_length || (prefix && prefix_length > prefix->size())) {
        throw std::invalid_argument("invalid topology prefix length");
    }
    PathKeys path;
    path.reserve(total_length);
    if (prefix && prefix_length > 0) {
        path.insert(path.end(), prefix->begin(), prefix->begin() + prefix_length);
    }
    while (path.size() < total_length) {
        path.push_back(nextTopologyKey());
    }
    return path;
}

WorkloadMetadata TreeWorkloadGenerator::generateTopology() {
    tree_paths_.clear();
    metadata_ = {};
    rng_.seed(seed_);
    next_topology_key_ = 1;

    // Node IDs let us enforce fanout without storing every long key prefix in
    // an ordered map. A generated key is globally unique, so each divergent
    // suffix maps one-to-one to newly assigned node IDs.
    std::vector<std::vector<size_t>> path_node_ids;
    std::vector<size_t>              child_counts;
    size_t                           root_child_count = 0;
    size_t                           node_count       = 0;

    const size_t min_length = std::min(config_.initial_min_path_length, tree_node_count_);
    const size_t max_length = std::min(config_.initial_max_path_length, config_.max_path_length);

    auto parentHasCapacity = [&](size_t path_index, size_t prefix_length) {
        if (prefix_length == 0) {
            return root_child_count < tree_branching_factor_;
        }
        return child_counts[path_node_ids[path_index][prefix_length - 1]] < tree_branching_factor_;
    };

    while (node_count < tree_node_count_) {
        const size_t remaining  = tree_node_count_ - node_count;
        size_t       base_index = 0;
        size_t       prefix_len = 0;
        size_t       target_len = 0;
        bool         found      = false;

        for (size_t attempt = 0; attempt < 128 && !found; ++attempt) {
            PathScenario scenario = PathScenario::COLD;
            const double roll     = randomRatio(rng_);
            if (!tree_paths_.empty() && roll < config_.continuation_ratio) {
                scenario = PathScenario::CONTINUATION;
            } else if (!tree_paths_.empty() && roll < config_.continuation_ratio + config_.fork_ratio) {
                scenario = PathScenario::FORK;
            }

            if (scenario != PathScenario::COLD) {
                base_index = randomSize(rng_, 0, tree_paths_.size() - 1);
            }
            if (scenario == PathScenario::CONTINUATION) {
                prefix_len = tree_paths_[base_index].size();
            } else if (scenario == PathScenario::FORK) {
                const double reuse_ratio = std::uniform_real_distribution<double>(config_.fork_reuse_min_ratio,
                                                                                  config_.fork_reuse_max_ratio)(rng_);
                prefix_len =
                    std::max<size_t>(1, static_cast<size_t>(std::floor(tree_paths_[base_index].size() * reuse_ratio)));
                prefix_len = std::min(prefix_len, tree_paths_[base_index].size());
            } else {
                prefix_len = 0;
            }

            if (prefix_len >= max_length || (prefix_len > 0 && !parentHasCapacity(base_index, prefix_len))
                || (prefix_len == 0 && root_child_count >= tree_branching_factor_)) {
                continue;
            }

            const size_t lower = std::max(min_length, prefix_len + 1);
            if (lower > max_length) {
                continue;
            }
            target_len        = randomSize(rng_, lower, max_length);
            size_t suffix_len = std::min(remaining, target_len - prefix_len);
            if (prefix_len + suffix_len < min_length) {
                suffix_len = min_length - prefix_len;
            }
            if (suffix_len > remaining || prefix_len + suffix_len > max_length) {
                continue;
            }
            target_len = prefix_len + suffix_len;
            found      = true;
        }

        if (!found) {
            // Deterministic fallback is important for the final short suffix
            // and for small branching factors.
            for (size_t i = 0; i < tree_paths_.size() && !found; ++i) {
                const size_t prefix_begin = min_length > remaining ? min_length - remaining : 0;
                const size_t prefix_end   = std::min(tree_paths_[i].size(), max_length - 1);
                for (size_t prefix = prefix_begin; prefix <= prefix_end; ++prefix) {
                    if (!parentHasCapacity(i, prefix)) {
                        continue;
                    }
                    const size_t suffix_len = std::min(remaining, max_length - prefix);
                    if (prefix + suffix_len < min_length) {
                        continue;
                    }
                    base_index = i;
                    prefix_len = prefix;
                    target_len = prefix + suffix_len;
                    found      = true;
                    break;
                }
            }
        }
        if (!found) {
            if (tree_paths_.empty() && root_child_count < tree_branching_factor_) {
                prefix_len = 0;
                target_len = std::min(remaining, max_length);
                found      = true;
            } else {
                throw std::runtime_error("unable to generate topology within path/fanout bounds");
            }
        }

        const PathKeys*     prefix = prefix_len == 0 ? nullptr : &tree_paths_[base_index];
        auto                path   = createTopologyPath(prefix, prefix_len, target_len);
        std::vector<size_t> node_ids;
        node_ids.reserve(target_len);
        if (prefix_len > 0) {
            node_ids.insert(
                node_ids.end(), path_node_ids[base_index].begin(), path_node_ids[base_index].begin() + prefix_len);
            ++child_counts[node_ids.back()];
        } else {
            ++root_child_count;
        }

        for (size_t depth = prefix_len; depth < target_len; ++depth) {
            const size_t node_id = child_counts.size();
            child_counts.push_back(depth + 1 < target_len ? 1 : 0);
            node_ids.push_back(node_id);
            ++node_count;
        }
        tree_paths_.push_back(std::move(path));
        path_node_ids.push_back(std::move(node_ids));
    }

    metadata_.actual_node_count = node_count;
    metadata_.max_depth         = 0;
    for (const auto& path : tree_paths_) {
        metadata_.max_depth = std::max(metadata_.max_depth, path.size());
    }
    for (const size_t child_count : child_counts) {
        if (child_count == 0) {
            ++metadata_.leaf_count;
        }
    }
    return metadata_;
}

}  // namespace rtp_llm::benchmark
