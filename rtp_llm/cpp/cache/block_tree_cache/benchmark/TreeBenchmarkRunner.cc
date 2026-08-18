#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeBenchmarkRunner.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

#include "rtp_llm/cpp/cache/BlockReleaseBatch.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkFixture.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::benchmark {

namespace {

using Clock = std::chrono::steady_clock;

int64_t elapsedNs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

std::string joinSizeValues(const std::vector<size_t>& values) {
    std::ostringstream output;
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << values[index];
    }
    return output.str();
}

size_t hardFailureCount(const OnlineSchedulerMetrics& metrics) {
    return metrics.loads_failed + metrics.loads_cancelled + metrics.load_target_allocation_failed
           + metrics.suffix_allocation_failed + metrics.load_commit_failed + metrics.cancel_request_failed
           + metrics.lifecycle_timeouts + metrics.dependency_failed_descendants;
}

// Test-only injection seam: the GPU smoke suite sets this env var so the
// runner executes a small online lifecycle config. It is never a public CLI
// option and never affects the formal workload.
constexpr const char* kTestConfigEnv = "BLOCK_TREE_CACHE_BENCHMARK_TEST_CONFIG";

OnlineTreeWorkloadConfig resolvedConfig() {
    const char* env = std::getenv(kTestConfigEnv);
    if (env != nullptr && std::string(env) == "1") {
        std::cerr << "[tree] using test-only small online workload config (env " << kTestConfigEnv << "=1)\n";
        return OnlineTreeWorkloadConfig::smokeTestConfig();
    }
    return OnlineTreeWorkloadConfig{};
}

// Adapts BlockTreeCache to the scheduler's single-threaded cache surface.
class BlockTreeCacheAdapter: public OnlineCacheApi {
public:
    explicit BlockTreeCacheAdapter(BlockTreeCache& cache): cache_(cache) {}

    MatchOutcome match(const PathKeys& path) override {
        auto         result = cache_.match(path);
        MatchOutcome outcome;
        outcome.matched_device_blocks = result.matched_device_blocks;
        outcome.actual_matched_depth =
            result.async_context != nullptr ? result.async_context->matchedBlocks() : result.matched_device_blocks;
        outcome.host_matched_blocks =
            result.async_context != nullptr ? result.async_context->matchedBlocks(Tier::HOST) : 0;
        outcome.matched_device_resources = std::move(result.matched_device_resources);
        if (result.async_context != nullptr) {
            outcome.load_ticket = std::move(result.async_context);
        }
        return outcome;
    }

    void materializeRequestBlocks(MatchOutcome& outcome) override {
        for (const MultiNodeResource& resource : outcome.matched_device_resources) {
            for (const auto& [_, blocks] : resource.node_blocks) {
                appendRequestBlocks(cache_.groupSets()[resource.group_set_id], blocks, outcome.request_blocks);
            }
        }
        outcome.matched_device_resources.clear();
        if (outcome.load_ticket != nullptr) {
            const auto& descs  = outcome.load_ticket->loadDescs();
            const auto& joined = outcome.load_ticket->joinedLoads();
            for (size_t desc_index = 0; desc_index < descs.size(); ++desc_index) {
                const TransferDescriptor& desc = descs[desc_index];
                if (joined[desc_index]) {
                    appendRequestBlocks(
                        cache_.groupSets()[desc.group_set_id], desc.target_blocks, outcome.request_blocks);
                    outcome.joined_target_block_count += desc.target_blocks.size();
                } else if (desc.source_tier == Tier::DEVICE) {
                    appendRequestBlocks(
                        cache_.groupSets()[desc.group_set_id], desc.source_blocks, outcome.request_blocks);
                }
            }
        }
    }

    bool allocateLoadTargets(const MatchOutcome& outcome, PreparedRequestResources& out) override {
        if (outcome.load_ticket == nullptr) {
            return false;
        }
        const auto& context    = outcome.load_ticket;
        const auto& group_sets = cache_.groupSets();
        out.load_target_blocks.resize(group_sets.size());
        if (context->empty()) {
            return true;
        }
        const auto&         descs  = context->loadDescs();
        const auto&         joined = context->joinedLoads();
        std::vector<size_t> required(group_sets.size(), 0);
        for (size_t d = 0; d < descs.size(); ++d) {
            if (descs[d].source_tier == Tier::DEVICE || joined[d]) {
                continue;
            }
            ++required[descs[d].group_set_id];
        }
        for (size_t gs = 0; gs < group_sets.size(); ++gs) {
            if (required[gs] == 0) {
                continue;
            }
            const auto& pools = group_sets[gs]->devicePools();
            if (pools.empty()) {
                releasePrepared(out);
                return false;
            }
            auto blocks = pools[0]->malloc(required[gs]);
            if (!blocks.has_value()) {
                releasePrepared(out);
                return false;
            }
            pools[0]->incRef(blocks.value(), BlockRefType::REQUEST);
            out.load_target_blocks[gs] = std::move(blocks.value());
        }
        out.load_targets_allocated = true;
        return true;
    }

    bool allocateSuffixBlocks(size_t suffix_block_count, PreparedRequestResources& out) override {
        const auto& group_sets = cache_.groupSets();
        if (out.suffix_blocks.empty()) {
            out.suffix_blocks.resize(group_sets.size());
        }
        for (size_t gs = 0; gs < group_sets.size(); ++gs) {
            const auto& pools = group_sets[gs]->devicePools();
            if (pools.empty() || suffix_block_count == 0) {
                continue;
            }
            auto blocks = pools[0]->malloc(suffix_block_count);
            if (!blocks.has_value()) {
                releasePrepared(out);
                return false;
            }
            pools[0]->incRef(blocks.value(), BlockRefType::REQUEST);
            out.suffix_blocks[gs] = std::move(blocks.value());
        }
        out.suffix_allocated = true;
        return true;
    }

    bool commitLoad(const std::shared_ptr<LoadAsyncContext>& ticket, PreparedRequestResources& out) override {
        if (ticket == nullptr) {
            return false;
        }
        const auto& context    = ticket;
        const auto& group_sets = cache_.groupSets();
        if (!context->empty()) {
            const auto&         descs  = context->loadDescs();
            const auto&         joined = context->joinedLoads();
            std::vector<size_t> next(group_sets.size(), 0);
            for (size_t d = 0; d < descs.size(); ++d) {
                if (descs[d].source_tier == Tier::DEVICE || joined[d]) {
                    continue;
                }
                const size_t gs = descs[d].group_set_id;
                context->setTargetBlocks(d, {out.load_target_blocks[gs][next[gs]++]});
            }
        }
        if (!context->commit()) {
            releasePrepared(out);
            return false;
        }
        return true;
    }

    void publishInsert(const PathKeys&                path,
                       size_t                         actual_matched_depth,
                       PreparedRequestResources&      out,
                       std::vector<BlockIndicesType>& request_blocks) override {
        const auto&                                group_sets = cache_.groupSets();
        std::vector<std::vector<GroupSetResource>> resources(path.size(),
                                                             std::vector<GroupSetResource>(group_sets.size()));
        for (size_t gs = 0; gs < group_sets.size(); ++gs) {
            if (out.suffix_blocks[gs].empty()) {
                continue;
            }
            for (size_t j = 0; j < out.suffix_blocks[gs].size(); ++j) {
                resources[actual_matched_depth + j][gs].device_blocks = {out.suffix_blocks[gs][j]};
            }
        }
        cache_.insert(path, resources, Tier::DEVICE);
        // Publish REQUEST-holder transitions: blocks accepted by the tree keep
        // BLOCK_CACHE ownership, rejected ones return to the pool.
        releaseRequestBlocks(request_blocks);
        releasePrepared(out);
    }

    void releaseRequestBlocks(std::vector<BlockIndicesType>& blocks) override {
        const auto&       group_sets = cache_.groupSets();
        BlockReleaseBatch releases;
        for (const GroupSetPtr& group_set : group_sets) {
            const auto& group_ids = group_set->groupIds();
            const auto& pools     = group_set->devicePools();
            RTP_LLM_CHECK(group_ids.size() == pools.size());
            for (size_t member_index = 0; member_index < group_ids.size(); ++member_index) {
                const size_t group_id = group_ids[member_index];
                if (group_id >= blocks.size() || blocks[group_id].empty()) {
                    continue;
                }
                releases.append(group_id,
                                pools[member_index]->decRefWithResult(blocks[group_id], BlockRefType::REQUEST));
            }
        }
        blocks.clear();
        const auto receipts = releases.finish();
        if (!receipts.empty()) {
            cache_.onBlocksReleased(receipts);
        }
    }

    void rollback(PreparedRequestResources& out, std::vector<BlockIndicesType>& request_blocks) override {
        releasePrepared(out);
        releaseRequestBlocks(request_blocks);
    }

    // Bounded drain for setup/warmup/measured/finalize with a configurable deadline.
    bool boundedDrain(std::chrono::milliseconds budget) {
        return boundedPendingTaskDrain(
            [this]() { return static_cast<size_t>(cache_.task_pool_->pending_tasks_.load()); }, budget);
    }

    // Access pending task count for final-zero checks.
    size_t pendingTaskCount() const {
        return static_cast<size_t>(cache_.task_pool_->pending_tasks_.load());
    }

private:
    static void appendRequestBlocks(const GroupSetPtr&             group_set,
                                    const BlockIndicesType&        blocks,
                                    std::vector<BlockIndicesType>& request_blocks) {
        const auto& group_ids = group_set->groupIds();
        RTP_LLM_CHECK(group_ids.size() == blocks.size());
        for (size_t member_index = 0; member_index < group_ids.size(); ++member_index) {
            const size_t group_id = group_ids[member_index];
            if (request_blocks.size() <= group_id) {
                request_blocks.resize(group_id + 1);
            }
            request_blocks[group_id].push_back(blocks[member_index]);
        }
    }

    static BlockIndicesType uniqueHeldBlocks(const PreparedRequestResources& out, size_t group_set_id) {
        std::unordered_set<BlockIdxType> unique;
        for (const auto* groups : {&out.load_target_blocks, &out.suffix_blocks}) {
            if (group_set_id < groups->size()) {
                unique.insert((*groups)[group_set_id].begin(), (*groups)[group_set_id].end());
            }
        }
        return BlockIndicesType(unique.begin(), unique.end());
    }

    void releasePrepared(PreparedRequestResources& out) {
        const auto&       group_sets = cache_.groupSets();
        BlockReleaseBatch releases;
        for (size_t gs = 0; gs < group_sets.size(); ++gs) {
            if (group_sets[gs]->devicePools().empty()) {
                continue;
            }
            BlockIndicesType held = uniqueHeldBlocks(out, gs);
            if (held.empty()) {
                continue;
            }
            releases.append(group_sets[gs]->groupIds().front(),
                            group_sets[gs]->devicePools()[0]->decRefWithResult(held, BlockRefType::REQUEST));
        }
        const auto receipts = releases.finish();
        if (!receipts.empty()) {
            cache_.onBlocksReleased(receipts);
        }
        out = PreparedRequestResources{};
    }

    BlockTreeCache& cache_;
};

// Insert one complete path while allocating resources only for the suffix
// after `existing_prefix_length`. Used for the initial topology only.
bool insertPathFromPrefix(BlockTreeCache& cache, const PathKeys& path, size_t existing_prefix_length) {
    if (existing_prefix_length > path.size()) {
        throw std::invalid_argument("insert prefix exceeds full path length");
    }
    const auto&                                group_sets = cache.groupSets();
    std::vector<std::vector<GroupSetResource>> resources(path.size(), std::vector<GroupSetResource>(group_sets.size()));
    bool                                       allocated = true;
    for (size_t i = existing_prefix_length; i < path.size(); ++i) {
        for (size_t gs = 0; gs < group_sets.size(); ++gs) {
            const auto& pools = group_sets[gs]->devicePools();
            if (pools.empty()) {
                continue;
            }
            auto block = pools[0]->malloc();
            if (!block.has_value()) {
                allocated = false;
                break;
            }
            pools[0]->incRef(block.value(), BlockRefType::REQUEST);
            resources[i][gs].device_blocks = {block.value()};
        }
        if (!allocated) {
            break;
        }
    }
    if (!allocated) {
        for (size_t i = existing_prefix_length; i < path.size(); ++i) {
            for (size_t gs = 0; gs < group_sets.size(); ++gs) {
                if (group_sets[gs]->devicePools().empty()) {
                    continue;
                }
                for (const BlockIdxType block : resources[i][gs].device_blocks) {
                    group_sets[gs]->devicePools()[0]->decRef(block, BlockRefType::REQUEST);
                }
            }
        }
        return false;
    }
    cache.insert(path, resources, Tier::DEVICE);
    BlockReleaseBatch releases;
    for (size_t gs = 0; gs < group_sets.size(); ++gs) {
        if (group_sets[gs]->devicePools().empty()) {
            continue;
        }
        BlockIndicesType inserted_blocks;
        inserted_blocks.reserve(path.size() - existing_prefix_length);
        for (size_t i = existing_prefix_length; i < path.size(); ++i) {
            inserted_blocks.insert(
                inserted_blocks.end(), resources[i][gs].device_blocks.begin(), resources[i][gs].device_blocks.end());
        }
        releases.append(group_sets[gs]->groupIds().front(),
                        group_sets[gs]->devicePools()[0]->decRefWithResult(inserted_blocks, BlockRefType::REQUEST));
    }
    const auto receipts = releases.finish();
    if (!receipts.empty()) {
        cache.onBlocksReleased(receipts);
    }
    return true;
}

}  // anonymous namespace

std::unique_ptr<OnlineCacheApi> makeBlockTreeCacheAdapterForTest(BlockTreeCache& cache) {
    return std::make_unique<BlockTreeCacheAdapter>(cache);
}

TreeBenchmarkRunner::TreeBenchmarkRunner(const ModelProfile& profile,
                                         const TreeOptions&  options,
                                         uint64_t            seed,
                                         uint64_t            repetition_id,
                                         int                 cuda_device,
                                         double              max_device_memory_fraction,
                                         const std::string&  output_json_path):
    profile_(profile),
    options_(options),
    seed_(seed),
    repetition_id_(repetition_id),
    cuda_device_(cuda_device),
    max_device_memory_fraction_(max_device_memory_fraction),
    output_json_path_(output_json_path) {
    writer_.setRunner("tree");
    writer_.setModelProfile(profile_.profile_id, profile_.sha256_hex);
    writer_.setPayloadMode("scaled", profile_.computeGroupSetPayloadBytes("full_context"));
}

bool TreeBenchmarkRunner::run() {
    const OnlineTreeWorkloadConfig config       = resolvedConfig();
    const bool                     benchmark_ok = runOnlineBenchmark(config);
    writer_.setStatus(benchmark_ok ? "completed" : "failed");

    bool output_ok = true;
    if (!output_json_path_.empty()) {
        std::ofstream output(output_json_path_, std::ios::trunc);
        output_ok = output.is_open();
        if (output_ok) {
            output << writer_.toJson() << '\n';
            output_ok = output.good();
        }
        if (!output_ok) {
            std::cerr << "Failed to write result JSON: " << output_json_path_ << std::endl;
        }
    }
    return benchmark_ok && output_ok;
}

std::unique_ptr<BlockTreeCache> TreeBenchmarkRunner::buildTreeCache(const OnlineTreeWorkloadConfig& config) {
    // Real group-set fixture: reads the profile's actual group type (FULL or
    // SWA) and constructs the corresponding GroupSet. No longer flattens SWA
    // to FULL. Fixed device/host pools of 32,768 blocks (or the test config's
    // smaller pools) with watermark eviction.
    std::vector<std::pair<std::string, rtp_llm::CacheGroupType>> group_specs;
    std::vector<size_t>                                          group_payloads;
    std::vector<size_t>                                          sliding_windows;
    for (const auto& gs_info : profile_.group_sets) {
        // Convert benchmark CacheGroupType to production rtp_llm::CacheGroupType.
        const auto prod_type = gs_info.group_type == benchmark::CacheGroupType::SWA ? rtp_llm::CacheGroupType::SWA :
                                                                                      rtp_llm::CacheGroupType::FULL;
        group_specs.emplace_back(gs_info.name, prod_type);
        group_payloads.push_back(BenchmarkFixture::computeScaledPayload(gs_info.payload_bytes));
        sliding_windows.push_back(gs_info.sliding_window_size);
    }
    auto topology = BenchmarkFixture::createTopology(group_specs, group_payloads, {}, sliding_windows);

    std::vector<GroupSetPtr> group_sets;
    for (size_t gs_idx = 0; gs_idx < profile_.group_sets.size(); ++gs_idx) {
        auto device_pool = BenchmarkFixture::createDevicePool(
            group_payloads[gs_idx], 1, config.device_pool_blocks, "device_" + profile_.group_sets[gs_idx].name);
        auto host_pool = BenchmarkFixture::createHostPool(
            group_payloads[gs_idx], config.host_pool_blocks, true, "host_" + profile_.group_sets[gs_idx].name);
        const std::vector<size_t> group_ids = {gs_idx};

        if (profile_.group_sets[gs_idx].group_type == benchmark::CacheGroupType::SWA) {
            group_sets.push_back(BenchmarkFixture::createSWAGroupSet({device_pool},
                                                                     host_pool,
                                                                     nullptr,
                                                                     gs_idx,
                                                                     topology,
                                                                     group_ids,
                                                                     profile_.group_sets[gs_idx].sliding_window_size));
        } else {
            group_sets.push_back(
                BenchmarkFixture::createFullGroupSet({device_pool}, host_pool, nullptr, gs_idx, topology, group_ids));
        }
    }

    return BenchmarkFixture::createCache(group_sets,
                                         /*enable_host=*/true,
                                         /*enable_disk=*/false,
                                         options_.task_pool_size,
                                         config.device_watermark_ratio,
                                         config.host_watermark_ratio);
}

size_t TreeBenchmarkRunner::insertTopology(BlockTreeCache& cache, const std::vector<PathKeys>& paths) {
    size_t inserted = 0;
    for (const auto& path : paths) {
        const size_t existing_prefix_length = cache.tree()->findNode(path).size();
        if (!insertPathFromPrefix(cache, path, existing_prefix_length)) {
            throw std::runtime_error("device pool exhausted while building initial tree topology");
        }
        inserted += path.size() - existing_prefix_length;
    }
    return inserted;
}

void TreeBenchmarkRunner::addLatencyMetrics(const std::string& prefix, const std::vector<int64_t>& samples) {
    if (samples.empty()) {
        return;
    }
    std::vector<int64_t> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    auto percentile = [&](double q) {
        const size_t idx = static_cast<size_t>(q * static_cast<double>(sorted.size() - 1));
        return sorted[idx];
    };
    const double avg =
        static_cast<double>(std::accumulate(sorted.begin(), sorted.end(), 0LL)) / static_cast<double>(sorted.size());
    writer_.addMetric(prefix + "_latency_ns_min", static_cast<double>(sorted.front()));
    writer_.addMetric(prefix + "_latency_ns_p50", static_cast<double>(percentile(0.5)));
    writer_.addMetric(prefix + "_latency_ns_p99", static_cast<double>(percentile(0.99)));
    writer_.addMetric(prefix + "_latency_ns_max", static_cast<double>(sorted.back()));
    writer_.addMetric(prefix + "_latency_ns_avg", avg);
    writer_.addMetric(prefix + "_calls", static_cast<double>(sorted.size()));
}

void TreeBenchmarkRunner::addDistributionMetrics(const std::string& prefix, const std::vector<int64_t>& samples) {
    if (samples.empty()) {
        return;
    }
    std::vector<int64_t> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    auto percentile = [&](double q) {
        const size_t idx = static_cast<size_t>(q * static_cast<double>(sorted.size() - 1));
        return sorted[idx];
    };
    const double avg =
        static_cast<double>(std::accumulate(sorted.begin(), sorted.end(), 0LL)) / static_cast<double>(sorted.size());
    writer_.addMetric(prefix + "_min", static_cast<double>(sorted.front()));
    writer_.addMetric(prefix + "_p50", static_cast<double>(percentile(0.5)));
    writer_.addMetric(prefix + "_p99", static_cast<double>(percentile(0.99)));
    writer_.addMetric(prefix + "_max", static_cast<double>(sorted.back()));
    writer_.addMetric(prefix + "_avg", avg);
    writer_.addMetric(prefix + "_samples", static_cast<double>(sorted.size()));
}

void TreeBenchmarkRunner::writePressureMetrics(bool                                             pressure_ready,
                                               const std::vector<BlockTreePoolMetricsSnapshot>& snapshots) {
    writer_.addMetric("pressure_ready", pressure_ready ? 1.0 : 0.0);
    double min_device_ratio = std::numeric_limits<double>::max();
    size_t host_pools_used  = 0;
    size_t host_pools_total = 0;
    for (const auto& snapshot : snapshots) {
        const std::string pool_key = "pool." + snapshot.pool_name + ".";
        writer_.addMetric(pool_key + "used_blocks", static_cast<double>(snapshot.used_blocks));
        writer_.addMetric(pool_key + "total_blocks", static_cast<double>(snapshot.total_blocks));
        if (snapshot.tier == Tier::DEVICE && snapshot.total_blocks > 0) {
            const double ratio = static_cast<double>(snapshot.used_blocks) / static_cast<double>(snapshot.total_blocks);
            writer_.addMetric(pool_key + "used_ratio", ratio);
            min_device_ratio = std::min(min_device_ratio, ratio);
        } else if (snapshot.tier == Tier::HOST) {
            ++host_pools_total;
            if (snapshot.used_blocks > 0) {
                ++host_pools_used;
            }
        }
    }
    if (min_device_ratio != std::numeric_limits<double>::max()) {
        writer_.addMetric("warmup.device_pool_used_ratio_min", min_device_ratio);
    }
    writer_.addMetric("warmup.host_pools_used", static_cast<double>(host_pools_used));
    writer_.addMetric("warmup.host_pools_total", static_cast<double>(host_pools_total));
}

void TreeBenchmarkRunner::writeFinalZeroMetrics(const std::vector<BlockTreePoolMetricsSnapshot>& snapshots) {
    size_t request_ref_blocks = 0;
    for (const auto& snapshot : snapshots) {
        request_ref_blocks += snapshot.request_ref_blocks;
    }
    writer_.addMetric("final.request_ref_blocks", static_cast<double>(request_ref_blocks));
    for (const auto& snapshot : snapshots) {
        const std::string pool_key = "final.pool." + snapshot.pool_name + ".";
        writer_.addMetric(pool_key + "used_blocks", static_cast<double>(snapshot.used_blocks));
        writer_.addMetric(pool_key + "request_ref_blocks", static_cast<double>(snapshot.request_ref_blocks));
    }
}

bool TreeBenchmarkRunner::runOnlineBenchmark(const OnlineTreeWorkloadConfig& config) {
    writer_.setMeasurement("online_lifecycle");

    // Resolved config: every fixed workload value, the trace hash, the model
    // profile and the task pool. Written before preflight so even a failed
    // preflight result is self-describing.
    writer_.addResolvedConfigInt("tokens_per_block", static_cast<int64_t>(config.tokens_per_block));
    writer_.addResolvedConfigInt("logical_concurrency", static_cast<int64_t>(config.logical_concurrency));
    writer_.addResolvedConfigInt("active_token_budget", static_cast<int64_t>(config.active_token_budget));
    writer_.addResolvedConfigInt("forward_sleep_ms", static_cast<int64_t>(config.forward_sleep_ms));
    writer_.addResolvedConfigInt("request_lifecycle_timeout_ms",
                                 static_cast<int64_t>(config.request_lifecycle_timeout_ms));
    writer_.addResolvedConfigInt("initial_cache_node_count",
                                 static_cast<int64_t>(config.shared_base_nodes + config.background_tree_nodes));
    writer_.addResolvedConfigInt("shared_base_nodes", static_cast<int64_t>(config.shared_base_nodes));
    writer_.addResolvedConfigInt("background_tree_nodes", static_cast<int64_t>(config.background_tree_nodes));
    writer_.addResolvedConfigInt("device_pool_blocks", static_cast<int64_t>(config.device_pool_blocks));
    writer_.addResolvedConfigInt("host_pool_blocks", static_cast<int64_t>(config.host_pool_blocks));
    writer_.addResolvedConfig("device_watermark_ratio", std::to_string(config.device_watermark_ratio));
    writer_.addResolvedConfig("host_watermark_ratio", std::to_string(config.host_watermark_ratio));
    writer_.addResolvedConfigInt("operation_trace_count", static_cast<int64_t>(config.operation_trace_count));
    writer_.addResolvedConfig("length_buckets_tokens", joinSizeValues(config.length_buckets_tokens));
    writer_.addResolvedConfig("length_weights", joinSizeValues(config.length_weights));
    writer_.addResolvedConfig("hit_rates_percent", joinSizeValues(config.hit_rates_percent));
    writer_.addResolvedConfigInt("warmup_seconds", static_cast<int64_t>(config.warmup_seconds));
    writer_.addResolvedConfigInt("measured_seconds", static_cast<int64_t>(config.measured_seconds));
    writer_.addResolvedConfigInt("task_pool_size_resolved", static_cast<int64_t>(options_.task_pool_size));
    writer_.addResolvedConfigInt("foreground_scheduler_threads", 1);
    writer_.addResolvedConfigInt("repetition_identity", static_cast<int64_t>(repetition_id_));
    writer_.addResolvedConfigInt("cuda_device_resolved", static_cast<int64_t>(cuda_device_));
    writer_.addResolvedConfig("max_device_memory_fraction_resolved", std::to_string(max_device_memory_fraction_));
    writer_.addResolvedConfig("fixture_layout", "scaled_group_set");
    for (const auto& gs_info : profile_.group_sets) {
        writer_.addResolvedConfig("group_set_" + gs_info.name + "_type",
                                  gs_info.group_type == benchmark::CacheGroupType::SWA ? "SWA" : "FULL");
        writer_.addResolvedConfigInt("group_set_" + gs_info.name + "_device_pool_blocks",
                                     static_cast<int64_t>(config.device_pool_blocks));
        writer_.addResolvedConfigInt("group_set_" + gs_info.name + "_host_pool_blocks",
                                     static_cast<int64_t>(config.host_pool_blocks));
        writer_.addResolvedConfigInt(
            "group_set_" + gs_info.name + "_scaled_payload_bytes",
            static_cast<int64_t>(BenchmarkFixture::computeScaledPayload(gs_info.payload_bytes)));
        if (gs_info.group_type == benchmark::CacheGroupType::SWA) {
            writer_.addResolvedConfigInt("group_set_" + gs_info.name + "_swa_window",
                                         static_cast<int64_t>(gs_info.sliding_window_size));
        }
    }

    // Preflight before any large allocation: fixed pools plus the admission
    // preparation peak, never derived from the initial node count.
    const ResourceBudget budget =
        BenchmarkFixture::preflightTreeResources(profile_, config, cuda_device_, max_device_memory_fraction_);
    writer_.addResourceBudget("estimated_device_bytes", budget.estimated_device_bytes);
    writer_.addResourceBudget("estimated_host_bytes", budget.estimated_host_bytes);
    writer_.addResourceBudget("available_device_bytes", budget.available_device_bytes);
    writer_.addResourceBudget("raw_available_device_bytes", budget.raw_available_device_bytes);
    writer_.addResourceBudget("available_host_or_cgroup_bytes", budget.available_host_or_cgroup_bytes);
    writer_.addResourceBudget("sufficient", budget.sufficient ? 1 : 0);
    if (!budget.sufficient) {
        std::cerr << "[tree] resource preflight failed: estimated_device=" << budget.estimated_device_bytes
                  << " available_device=" << budget.available_device_bytes
                  << " estimated_host=" << budget.estimated_host_bytes
                  << " available_host=" << budget.available_host_or_cgroup_bytes << std::endl;
        return false;
    }

    auto cache = buildTreeCache(config);
    if (!cache) {
        return false;
    }

    TreeWorkloadGenerator generator(seed_, config);
    const auto            setup_start = Clock::now();
    const auto            metadata    = generator.generateTopology();
    generator.generateTrace();
    std::cout << "[tree] setup: inserting " << metadata.actual_node_count << " nodes..." << std::endl;
    const size_t          built_nodes = insertTopology(*cache, generator.topologyPaths());
    BlockTreeCacheAdapter adapter(*cache);
    size_t                drain_timeouts = 0;
    bool                  all_drains_ok  = true;
    auto                  drain          = [&](const std::string& phase) {
        const bool ok = adapter.boundedDrain(std::chrono::milliseconds(30'000));
        writer_.addMetric("drain." + phase + ".pending_tasks_after", static_cast<double>(adapter.pendingTaskCount()));
        if (!ok) {
            ++drain_timeouts;
            all_drains_ok = false;
            std::cerr << "[tree] " << phase << " drain timed out with " << adapter.pendingTaskCount()
                      << " pending task(s)" << std::endl;
        }
        return ok;
    };
    drain("setup");
    const int64_t setup_ns = elapsedNs(setup_start, Clock::now());
    writer_.addPhaseNs("setup", setup_ns);
    const size_t setup_node_count = cache->getStats().tree_node_count;
    writer_.addMetric("tree_node_count_setup", static_cast<double>(setup_node_count));
    if (built_nodes != metadata.actual_node_count || setup_node_count != metadata.actual_node_count) {
        std::cerr << "[tree] topology build mismatch: generated=" << metadata.actual_node_count
                  << " inserted=" << built_nodes << " cache=" << setup_node_count << std::endl;
        return false;
    }

    writer_.addResolvedConfig("trace_hash", TreeWorkloadGenerator::hashHex(generator.traceHash()));
    writer_.addResolvedConfig("workload_definition_hash",
                              TreeWorkloadGenerator::hashHex(generator.workloadDefinitionHash()));
    writer_.addResolvedConfigInt("initial_topology_path_count", static_cast<int64_t>(generator.topologyPaths().size()));
    writer_.addResolvedConfigInt("base_request_count", static_cast<int64_t>(generator.baseRequestCount()));
    writer_.addResolvedConfigInt("continuation_request_count",
                                 static_cast<int64_t>(generator.continuationRequestCount()));

    OnlineTreeScheduler scheduler(adapter, config);

    // Warmup: full request lifecycle, then quiesce (tree/host cache is kept).
    std::cout << "[tree] warmup " << config.warmup_seconds << "s..." << std::endl;
    size_t     trace_offset = 0;
    int64_t    warmup_ns    = 0;
    const bool warmup_ok =
        scheduler.runPhase(generator.trace(), trace_offset, std::chrono::seconds(config.warmup_seconds), warmup_ns);
    writer_.addPhaseNs("warmup", warmup_ns);
    drain("warmup");
    const OnlineSchedulerMetrics warmup_metrics   = scheduler.takeMetrics();
    const auto                   warmup_snapshots = cache->poolMetricsSnapshots();
    const auto                   warmup_stats     = cache->getStats();
    const bool                   pressure_ready   = warmup_ok
                                && std::all_of(warmup_snapshots.begin(),
                                               warmup_snapshots.end(),
                                               [](const auto& snapshot) {
                                                   if (snapshot.tier == Tier::DEVICE && snapshot.total_blocks > 0) {
                                                       return static_cast<double>(snapshot.used_blocks)
                                                                  / static_cast<double>(snapshot.total_blocks)
                                                              >= 0.75;
                                                   }
                                                   if (snapshot.tier == Tier::HOST) {
                                                       return snapshot.used_blocks > 0;
                                                   }
                                                   return true;
                                               })
                                && warmup_stats.device_heap_total_size > 0
                                && warmup_metrics.completed_transactions >= config.warmup_completed_requests_min
                                && scheduler.activeContexts() == 0;
    writePressureMetrics(pressure_ready, warmup_snapshots);
    writer_.addMetric("warmup.completed_request_transactions",
                      static_cast<double>(warmup_metrics.completed_transactions));
    writer_.addMetric("warmup.device_heap_total_size", static_cast<double>(warmup_stats.device_heap_total_size));
    writer_.addMetric("warmup.failed_requests", static_cast<double>(hardFailureCount(warmup_metrics)));
    writer_.addMetric("warmup.trace_exhaustions", static_cast<double>(warmup_metrics.trace_exhaustions));
    writer_.addMetric("warmup.dependency_failed_descendants",
                      static_cast<double>(warmup_metrics.dependency_failed_descendants));
    std::cout << "[tree] warmup done: completed=" << warmup_metrics.completed_transactions
              << " pressure_ready=" << (pressure_ready ? "true" : "false") << std::endl;

    // Measured: dual profiler markers; the 2s attach window is outside the
    // measured timer. measured_ns spans MEASURE_START to the completion of the
    // last admitted request (deadline drain may slightly exceed 60s).
    std::cout << "PROFILE_ATTACH_READY" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "MEASURE_START" << std::endl;
    int64_t    measured_ns = 0;
    const bool measured_ok =
        scheduler.runPhase(generator.trace(), trace_offset, std::chrono::seconds(config.measured_seconds), measured_ns);
    writer_.addPhaseNs("measured", measured_ns);
    drain("measured");

    // Finalize: verify contexts, tickets, task pool and REQUEST refs are zero.
    const auto finalize_start = Clock::now();
    drain("finalize");
    const auto final_snapshots = cache->poolMetricsSnapshots();
    const auto final_stats     = cache->getStats();
    const auto finalize_ns     = elapsedNs(finalize_start, Clock::now());
    writer_.addPhaseNs("finalize", finalize_ns);
    const size_t final_active_requests      = scheduler.activeContexts();
    const size_t final_pending_load_tickets = scheduler.pendingLoadTickets();
    const size_t final_pending_tasks        = adapter.pendingTaskCount();
    const size_t final_request_ref_blocks   = std::accumulate(
        final_snapshots.begin(), final_snapshots.end(), size_t{0}, [](size_t sum, const auto& snapshot) {
            return sum + snapshot.request_ref_blocks;
        });
    writeFinalZeroMetrics(final_snapshots);
    writer_.addMetric("final.active_requests", static_cast<double>(final_active_requests));
    writer_.addMetric("final.pending_load_tickets", static_cast<double>(final_pending_load_tickets));
    writer_.addMetric("final.pending_tasks", static_cast<double>(final_pending_tasks));
    writer_.addMetric("drain_timeouts", static_cast<double>(drain_timeouts));
    writer_.addMetric("tree_node_count_final", static_cast<double>(final_stats.tree_node_count));
    const bool final_zero = final_active_requests == 0 && final_pending_load_tickets == 0 && final_pending_tasks == 0
                            && final_request_ref_blocks == 0;

    const auto& m = scheduler.metrics();
    addLatencyMetrics("match", m.match_ns);
    addLatencyMetrics("insert", m.insert_ns);
    addLatencyMetrics("load_commit", m.load_commit_ns);
    addLatencyMetrics("match_to_ready", m.match_to_ready_ns);

    const size_t measured_failed_requests = hardFailureCount(m);
    const size_t failed_requests          = hardFailureCount(warmup_metrics) + measured_failed_requests;
    const size_t completed_transactions   = warmup_metrics.completed_transactions + m.completed_transactions;
    const size_t attempted_transactions   = completed_transactions + failed_requests;

    writer_.setWorkload(seed_, attempted_transactions, attempted_transactions, completed_transactions, failed_requests);
    const size_t  lifecycle_forward_batches  = warmup_metrics.forward_batches + m.forward_batches;
    const size_t  lifecycle_forward_requests = warmup_metrics.forward_requests + m.forward_requests;
    const int64_t simulated_forward_sleep_ns =
        static_cast<int64_t>(lifecycle_forward_batches) * static_cast<int64_t>(config.forward_sleep_ms) * 1'000'000;
    writer_.addMetric("logical_concurrency_resolved", static_cast<double>(config.logical_concurrency));
    writer_.addMetric("foreground_scheduler_threads", 1.0);
    writer_.addMetric("task_pool_size_resolved", static_cast<double>(options_.task_pool_size));
    writer_.addMetric("active_requests_peak", static_cast<double>(m.active_requests_peak));
    writer_.addMetric("waiting_requests_peak", static_cast<double>(m.waiting_requests_peak));
    writer_.addMetric("loading_requests_peak", static_cast<double>(m.loading_requests_peak));
    writer_.addMetric("load_tickets_pending_peak", static_cast<double>(m.load_tickets_pending_peak));
    writer_.addMetric("ready_batch_size_avg",
                      m.ready_batch_sizes.empty() ? 0.0 :
                                                    static_cast<double>(std::accumulate(
                                                        m.ready_batch_sizes.begin(), m.ready_batch_sizes.end(), 0ULL))
                                                        / static_cast<double>(m.ready_batch_sizes.size()));
    writer_.addMetric("ready_batch_size_max", static_cast<double>(m.ready_batch_max));
    writer_.addMetric("scheduler_no_ready_wait_ns", static_cast<double>(m.scheduler_no_ready_wait_ns));
    writer_.addMetric("forward_batches", static_cast<double>(m.forward_batches));
    writer_.addMetric("forward_requests", static_cast<double>(m.forward_requests));
    writer_.addMetric("simulated_forward_sleep_ns", static_cast<double>(simulated_forward_sleep_ns));
    writer_.addMetric("held_request_blocks_peak", static_cast<double>(m.held_request_blocks_peak));
    writer_.addMetric("completed_request_transactions", static_cast<double>(m.completed_transactions));
    writer_.addMetric("loads_committed", static_cast<double>(m.loads_committed));
    writer_.addMetric("loads_succeeded", static_cast<double>(m.loads_succeeded));
    writer_.addMetric("loads_failed", static_cast<double>(m.loads_failed));
    writer_.addMetric("loads_cancelled", static_cast<double>(m.loads_cancelled));
    writer_.addMetric("cancel_request_failed", static_cast<double>(m.cancel_request_failed));
    writer_.addMetric("load_target_allocation_failed", static_cast<double>(m.load_target_allocation_failed));
    writer_.addMetric("suffix_allocation_failed", static_cast<double>(m.suffix_allocation_failed));
    writer_.addMetric("load_commit_failed", static_cast<double>(m.load_commit_failed));
    writer_.addMetric("joined_target_blocks_total", static_cast<double>(m.joined_target_blocks_total));
    writer_.addMetric("admission_allocation_retries", static_cast<double>(m.admission_allocation_retries));
    writer_.addMetric("unexpected_extra_match_count", static_cast<double>(m.unexpected_extra_match_count));
    writer_.addMetric("trace_exhaustions", static_cast<double>(m.trace_exhaustions));
    writer_.addMetric("lifecycle_timeouts", static_cast<double>(m.lifecycle_timeouts));
    writer_.addMetric("dropped_waiting_at_deadline", static_cast<double>(m.dropped_waiting_at_deadline));

    // Dependency metrics
    writer_.addMetric("dependency_skip_count", static_cast<double>(m.dependency_skip_count));
    writer_.addMetric("dependency_waiting_peak", static_cast<double>(m.dependency_waiting_peak));
    writer_.addMetric("dependency_failed_descendants", static_cast<double>(m.dependency_failed_descendants));
    writer_.addMetric("completed_base_transactions", static_cast<double>(m.completed_base_transactions));
    writer_.addMetric("completed_continuation_transactions",
                      static_cast<double>(m.completed_continuation_transactions));
    writer_.addMetric("completed_continuation_family_count",
                      static_cast<double>(m.completed_continuation_families.size()));
    writer_.addMetric("completed_family_epoch_count", static_cast<double>(m.completed_family_epochs.size()));
    writer_.addMetric("max_completed_generation", static_cast<double>(m.max_completed_generation));

    std::vector<int64_t> completed_by_family(config.logical_concurrency, 0);
    for (const auto& [family_id, count] : m.completed_requests_by_family) {
        if (family_id < completed_by_family.size()) {
            completed_by_family[family_id] = static_cast<int64_t>(count);
        }
    }
    std::vector<int64_t> epochs_by_family(config.logical_concurrency, 0);
    for (const auto& [family_id, epoch_id] : m.completed_family_epochs) {
        (void)epoch_id;
        if (family_id < epochs_by_family.size()) {
            ++epochs_by_family[family_id];
        }
    }
    addDistributionMetrics("completed_requests_per_family", completed_by_family);
    addDistributionMetrics("completed_epochs_per_family", epochs_by_family);
    addDistributionMetrics("completed_generation", m.completed_generation_samples);
    addDistributionMetrics("planned_reuse_blocks", m.planned_reuse_blocks_samples);
    addDistributionMetrics("actual_matched_depth_blocks", m.actual_matched_depth_samples);
    addDistributionMetrics("actual_minus_planned_reuse_blocks", m.reuse_delta_blocks_samples);

    writer_.addMetric("planned_reuse_blocks_per_request",
                      m.completed_transactions ? static_cast<double>(m.planned_reuse_blocks_total)
                                                     / static_cast<double>(m.completed_transactions) :
                                                 0.0);
    writer_.addMetric("actual_matched_depth_per_request",
                      m.completed_transactions ? static_cast<double>(m.actual_matched_depth_total)
                                                     / static_cast<double>(m.completed_transactions) :
                                                 0.0);
    writer_.addMetric("device_matched_blocks_per_request",
                      m.completed_transactions ? static_cast<double>(m.device_matched_blocks_total)
                                                     / static_cast<double>(m.completed_transactions) :
                                                 0.0);
    writer_.addMetric("host_matched_blocks_per_request",
                      m.completed_transactions ? static_cast<double>(m.host_matched_blocks_total)
                                                     / static_cast<double>(m.completed_transactions) :
                                                 0.0);
    writer_.addMetric("insert_path_keys_per_request",
                      m.completed_transactions ? static_cast<double>(m.insert_path_keys_total)
                                                     / static_cast<double>(m.completed_transactions) :
                                                 0.0);
    writer_.addMetric("insert_new_nodes_per_request",
                      m.completed_transactions ? static_cast<double>(m.insert_new_nodes_total)
                                                     / static_cast<double>(m.completed_transactions) :
                                                 0.0);
    writer_.addMetric("benchmark_request_transactions_per_second",
                      measured_ns > 0 ?
                          static_cast<double>(m.completed_transactions) / (static_cast<double>(measured_ns) / 1e9) :
                          0.0);

    writer_.setTreeLifecycle(completed_transactions,
                             failed_requests,
                             lifecycle_forward_batches,
                             lifecycle_forward_requests,
                             simulated_forward_sleep_ns,
                             warmup_metrics.unexpected_extra_match_count + m.unexpected_extra_match_count,
                             pressure_ready,
                             final_active_requests,
                             final_pending_load_tickets,
                             final_pending_tasks,
                             drain_timeouts,
                             final_request_ref_blocks);

    std::cout << "[tree] measured " << static_cast<double>(measured_ns) / 1e9
              << "s: completed=" << m.completed_transactions << " transactions, batches=" << m.forward_batches
              << " (avg ready "
              << (m.ready_batch_sizes.empty() ?
                      0.0 :
                      static_cast<double>(std::accumulate(m.ready_batch_sizes.begin(), m.ready_batch_sizes.end(), 0ULL))
                          / static_cast<double>(m.ready_batch_sizes.size()))
              << "), loads=" << m.loads_committed << " committed/" << m.loads_succeeded << " succeeded,"
              << " held_peak=" << m.held_request_blocks_peak
              << ", pressure_ready=" << (pressure_ready ? "true" : "false")
              << ", deps_skipped=" << m.dependency_skip_count << ", deps_failed=" << m.dependency_failed_descendants
              << std::endl;

    const bool continuation_coverage_ok = m.completed_base_transactions > 0 && m.completed_continuation_transactions > 0
                                          && m.completed_continuation_families.size() == config.logical_concurrency;
    const bool measured_duration_ok = measured_ns >= static_cast<int64_t>(config.measured_seconds) * 1'000'000'000LL;
    return warmup_ok && measured_ok && all_drains_ok && final_zero && failed_requests == 0
           && warmup_metrics.trace_exhaustions == 0 && m.trace_exhaustions == 0 && m.completed_transactions > 0
           && measured_duration_ok && continuation_coverage_ok && lifecycle_forward_requests == completed_transactions;
}

}  // namespace rtp_llm::benchmark
