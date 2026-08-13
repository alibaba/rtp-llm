#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeWorkloadGenerator.h"

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace rtp_llm::benchmark {

namespace {

// Key spaces are disjoint by construction: shared base, background tree and
// request-unique suffixes can never collide, so a request can never match
// beyond its planned reuse prefix unless the key space layout is broken.
constexpr int64_t kSharedBaseKeyBase    = 1;
constexpr int64_t kBackgroundKeyBase    = 1'000'000;
constexpr int64_t kRequestSuffixKeyBase = 1'000'000'000'000;
// ceil(950000 / 256) = 3711 maximum input blocks; per-request stride leaves
// slack so every request gets its own contiguous suffix key range.
constexpr int64_t kRequestSuffixStride   = 4'096;
constexpr size_t  kBackgroundBranchCount = 8;

uint64_t fnv1a64(const void* data, size_t length, uint64_t hash = 1469598103934665603ULL) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < length; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t fnv1a64Int(uint64_t value, uint64_t hash) {
    return fnv1a64(&value, sizeof(value), hash);
}

uint64_t fnv1a64Double(double value, uint64_t hash) {
    return fnv1a64(&value, sizeof(value), hash);
}

uint64_t hashVector(const std::vector<size_t>& values, uint64_t hash) {
    hash = fnv1a64Int(values.size(), hash);
    for (const size_t value : values) {
        hash = fnv1a64Int(value, hash);
    }
    return hash;
}

size_t inputBlocksForTokens(size_t tokens, size_t tokens_per_block) {
    return (tokens + tokens_per_block - 1) / tokens_per_block;
}

void validateConfig(const OnlineTreeWorkloadConfig& config) {
    if (config.tokens_per_block == 0 || config.logical_concurrency == 0 || config.active_token_budget == 0
        || config.forward_sleep_ms == 0 || config.request_lifecycle_timeout_ms == 0 || config.shared_base_nodes == 0
        || config.background_tree_nodes == 0 || config.device_pool_blocks == 0 || config.host_pool_blocks == 0
        || config.operation_trace_count == 0) {
        throw std::invalid_argument("online tree workload constants must be positive");
    }
    if (config.length_buckets_tokens.size() != config.length_weights.size() || config.length_buckets_tokens.empty()
        || config.hit_rates_percent.empty()) {
        throw std::invalid_argument("online tree workload distributions are malformed");
    }
    for (size_t i = 0; i < config.length_buckets_tokens.size(); ++i) {
        if (config.length_weights[i] == 0) {
            throw std::invalid_argument("online tree length weights must be positive");
        }
    }
}

}  // anonymous namespace

OnlineTreeWorkloadConfig OnlineTreeWorkloadConfig::smokeTestConfig() {
    OnlineTreeWorkloadConfig config;
    config.logical_concurrency           = 8;
    config.active_token_budget           = 2'000'000;
    config.shared_base_nodes             = 640;
    config.background_tree_nodes         = 640;
    config.device_pool_blocks            = 16'384;
    config.host_pool_blocks              = 16'384;
    config.operation_trace_count         = 2'000;
    config.warmup_seconds                = 2;
    config.measured_seconds              = 5;
    config.warmup_completed_requests_min = 8;
    config.length_buckets_tokens         = {
        32'000, 48'000, 92'000, 96'000, 117'000, 120'000, 128'000, 135'000, 141'000, 150'000};
    config.length_weights    = {2, 3, 5, 10, 5, 20, 5, 10, 5, 5};
    config.hit_rates_percent = {0, 10, 30, 50, 70, 90, 99};
    return config;
}

TreeWorkloadGenerator::TreeWorkloadGenerator(uint64_t seed, const OnlineTreeWorkloadConfig& config):
    seed_(seed), config_(config) {
    validateConfig(config_);
}

int64_t TreeWorkloadGenerator::nextTopologyKey() {
    return kBackgroundKeyBase + next_topology_key_++;
}

int64_t TreeWorkloadGenerator::nextSuffixKey(size_t request_index, size_t suffix_index) {
    return kRequestSuffixKeyBase + static_cast<int64_t>(request_index) * kRequestSuffixStride
           + static_cast<int64_t>(suffix_index);
}

OnlineWorkloadMetadata TreeWorkloadGenerator::generateTopology() {
    topology_paths_.clear();
    next_topology_key_ = 0;

    // Shared base: one path of shared_base_nodes blocks. Requests reuse its
    // prefix and never pin it, so watermark eviction can partially demote it
    // and actual reuse may drop below planned reuse.
    PathKeys shared_base;
    shared_base.reserve(config_.shared_base_nodes);
    for (size_t i = 0; i < config_.shared_base_nodes; ++i) {
        shared_base.push_back(kSharedBaseKeyBase + static_cast<int64_t>(i));
    }
    topology_paths_.push_back(std::move(shared_base));

    // Background tree: kBackgroundBranchCount deterministic branches that
    // share only the implicit tree root and fill the remaining node budget.
    // They are never matched by requests and only supply churn/space pressure.
    size_t remaining = config_.background_tree_nodes;
    for (size_t branch = 0; branch < kBackgroundBranchCount; ++branch) {
        const size_t branch_length = remaining / (kBackgroundBranchCount - branch);
        PathKeys     branch_path;
        branch_path.reserve(branch_length);
        for (size_t i = 0; i < branch_length; ++i) {
            branch_path.push_back(nextTopologyKey());
        }
        topology_paths_.push_back(std::move(branch_path));
        remaining -= branch_length;
    }

    OnlineWorkloadMetadata metadata;
    metadata.actual_node_count     = config_.shared_base_nodes + config_.background_tree_nodes;
    metadata.shared_base_nodes     = config_.shared_base_nodes;
    metadata.background_tree_nodes = config_.background_tree_nodes;
    return metadata;
}

void TreeWorkloadGenerator::generateTrace() {
    if (topology_paths_.empty()) {
        throw std::logic_error("generateTopology() must be called before generateTrace()");
    }
    trace_.clear();
    trace_hash_                 = fnv1a64Int(seed_, 1469598103934665603ULL);
    base_request_count_         = 0;
    continuation_request_count_ = 0;

    // workload_definition_hash covers the fixed workload protocol (distributions,
    // capacities, etc.) but not task-pool size, so tp4/tp8 share the same hash.
    {
        workload_definition_hash_ = fnv1a64Int(config_.tokens_per_block, 1469598103934665603ULL);
        workload_definition_hash_ = fnv1a64Int(config_.logical_concurrency, workload_definition_hash_);
        workload_definition_hash_ = fnv1a64Int(config_.active_token_budget, workload_definition_hash_);
        workload_definition_hash_ = fnv1a64Int(config_.forward_sleep_ms, workload_definition_hash_);
        workload_definition_hash_ = fnv1a64Int(config_.request_lifecycle_timeout_ms, workload_definition_hash_);
        workload_definition_hash_ =
            fnv1a64Int(config_.shared_base_nodes + config_.background_tree_nodes, workload_definition_hash_);
        workload_definition_hash_ = fnv1a64Int(config_.shared_base_nodes, workload_definition_hash_);
        workload_definition_hash_ = fnv1a64Int(config_.background_tree_nodes, workload_definition_hash_);
        workload_definition_hash_ = fnv1a64Int(config_.device_pool_blocks, workload_definition_hash_);
        workload_definition_hash_ = fnv1a64Int(config_.host_pool_blocks, workload_definition_hash_);
        workload_definition_hash_ = fnv1a64Double(config_.device_watermark_ratio, workload_definition_hash_);
        workload_definition_hash_ = fnv1a64Double(config_.host_watermark_ratio, workload_definition_hash_);
        workload_definition_hash_ = fnv1a64Int(config_.operation_trace_count, workload_definition_hash_);
        workload_definition_hash_ = fnv1a64Int(config_.warmup_seconds, workload_definition_hash_);
        workload_definition_hash_ = fnv1a64Int(config_.measured_seconds, workload_definition_hash_);
        workload_definition_hash_ = fnv1a64Int(config_.warmup_completed_requests_min, workload_definition_hash_);
        workload_definition_hash_ = fnv1a64Int(config_.admission_allocation_retry_limit, workload_definition_hash_);
        workload_definition_hash_ = hashVector(config_.length_buckets_tokens, workload_definition_hash_);
        workload_definition_hash_ = hashVector(config_.length_weights, workload_definition_hash_);
        workload_definition_hash_ = hashVector(config_.hit_rates_percent, workload_definition_hash_);
    }

    std::mt19937_64     rng(seed_);
    std::vector<size_t> cumulative_weights(config_.length_weights.size());
    std::partial_sum(config_.length_weights.begin(), config_.length_weights.end(), cumulative_weights.begin());
    const size_t total_weight = cumulative_weights.back();

    const size_t family_count = config_.logical_concurrency;

    // Per-family generation state. Tracks the last published ancestor path so
    // continuations can inherit it.
    struct FamilyGenState {
        std::vector<int64_t> last_path;
        size_t               leaf_blocks{0};
        size_t               last_request_id{0};
        size_t               epoch_id{0};
        size_t               generation{0};
        bool                 has_epoch{false};
    };
    std::vector<FamilyGenState> families(family_count);

    // Round-robin interleave: each "round" gives each family one entry.
    const size_t rounds = (config_.operation_trace_count + family_count - 1) / family_count;
    trace_.reserve(config_.operation_trace_count);
    size_t request_index = 0;
    size_t suffix_offset = 0;  // globally unique suffix key base offset

    auto sampleLength = [&]() -> std::pair<size_t, size_t> {
        const size_t roll       = static_cast<size_t>(rng() % total_weight);
        const size_t bucket_idx = static_cast<size_t>(
            std::upper_bound(cumulative_weights.begin(), cumulative_weights.end(), roll) - cumulative_weights.begin());
        const size_t tokens = config_.length_buckets_tokens[bucket_idx];
        const size_t blocks = inputBlocksForTokens(tokens, config_.tokens_per_block);
        return {tokens, blocks};
    };

    for (size_t round = 0; round < rounds && request_index < config_.operation_trace_count; ++round) {
        for (size_t f = 0; f < family_count && request_index < config_.operation_trace_count; ++f) {
            OnlineRequestDescriptor descriptor;
            descriptor.request_id = request_index;
            descriptor.family_id  = f;

            const auto [tokens, blocks] = sampleLength();
            descriptor.target_tokens    = tokens;
            descriptor.input_blocks     = blocks;

            FamilyGenState& fam = families[f];
            if (fam.last_path.empty() || blocks <= fam.leaf_blocks) {
                // BASE — new epoch (no parent or sampled length ≤ current leaf).
                const size_t hit_percent        = config_.hit_rates_percent[rng() % config_.hit_rates_percent.size()];
                descriptor.planned_reuse_blocks = std::min(blocks * hit_percent / 100, blocks - 1);
                descriptor.generation           = 0;
                descriptor.predecessor_id       = -1;
                descriptor.is_continuation      = false;

                if (fam.has_epoch) {
                    ++fam.epoch_id;
                }
                fam.has_epoch       = true;
                fam.generation      = 0;
                descriptor.epoch_id = fam.epoch_id;

                const size_t suffix_blocks = blocks - descriptor.planned_reuse_blocks;
                descriptor.path.reserve(blocks);
                descriptor.path.insert(descriptor.path.end(),
                                       topology_paths_.front().begin(),
                                       topology_paths_.front().begin()
                                           + static_cast<ptrdiff_t>(descriptor.planned_reuse_blocks));
                for (size_t s = 0; s < suffix_blocks; ++s) {
                    descriptor.path.push_back(nextSuffixKey(suffix_offset, s));
                }
                ++suffix_offset;

                fam.last_path       = descriptor.path;
                fam.leaf_blocks     = blocks;
                fam.last_request_id = descriptor.request_id;
                ++base_request_count_;
            } else {
                // CONTINUATION — sampled length > leaf, append to parent path.
                const size_t append_blocks      = blocks - fam.leaf_blocks;
                descriptor.planned_reuse_blocks = fam.leaf_blocks;
                descriptor.epoch_id             = fam.epoch_id;
                descriptor.generation           = fam.generation + 1;
                descriptor.predecessor_id       = static_cast<int64_t>(fam.last_request_id);
                descriptor.is_continuation      = true;

                descriptor.path = fam.last_path;
                for (size_t s = 0; s < append_blocks; ++s) {
                    descriptor.path.push_back(nextSuffixKey(suffix_offset, s));
                }
                ++suffix_offset;

                fam.last_path       = descriptor.path;
                fam.leaf_blocks     = blocks;
                fam.last_request_id = descriptor.request_id;
                fam.generation      = descriptor.generation;
                ++continuation_request_count_;
            }

            // Hash all descriptor fields into trace_hash_.
            trace_hash_ = fnv1a64Int(descriptor.request_id, trace_hash_);
            trace_hash_ = fnv1a64Int(static_cast<uint64_t>(descriptor.planned_reuse_blocks), trace_hash_);
            trace_hash_ = fnv1a64Int(static_cast<uint64_t>(descriptor.input_blocks), trace_hash_);
            trace_hash_ = fnv1a64Int(static_cast<uint64_t>(descriptor.target_tokens), trace_hash_);
            trace_hash_ = fnv1a64(descriptor.path.data(), descriptor.path.size() * sizeof(int64_t), trace_hash_);
            trace_hash_ = fnv1a64Int(descriptor.family_id, trace_hash_);
            trace_hash_ = fnv1a64Int(descriptor.epoch_id, trace_hash_);
            trace_hash_ = fnv1a64Int(descriptor.is_continuation ? 1ULL : 0ULL, trace_hash_);
            trace_hash_ = fnv1a64Int(descriptor.generation, trace_hash_);
            trace_hash_ = fnv1a64Int(static_cast<uint64_t>(descriptor.predecessor_id), trace_hash_);

            trace_.push_back(std::move(descriptor));
            ++request_index;
        }
    }
}

std::string TreeWorkloadGenerator::hashHex(uint64_t hash) {
    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << hash;
    return stream.str();
}

}  // namespace rtp_llm::benchmark
