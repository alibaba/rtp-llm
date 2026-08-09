#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeBenchmarkRunner.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>

#include "rtp_llm/cpp/cache/BlockReleaseBatch.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkFixture.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkJsonWriter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeWorkloadGenerator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/ModelProfile.h"

namespace rtp_llm::benchmark {

namespace {

using Clock = std::chrono::steady_clock;

int64_t elapsedNs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

StatefulPathConfig pathConfig(const TreeOptions& options) {
    StatefulPathConfig config;
    config.max_path_length         = options.max_path_length;
    config.initial_min_path_length = options.initial_min_path_length;
    config.initial_max_path_length = options.initial_max_path_length;
    config.append_length           = options.append_length;
    config.inserts_per_match       = options.inserts_per_match;
    config.active_path_limit       = options.active_path_limit;
    config.continuation_ratio      = options.continuation_ratio;
    config.fork_ratio              = options.fork_ratio;
    config.fork_reuse_min_ratio    = options.fork_reuse_min_ratio;
    config.fork_reuse_max_ratio    = options.fork_reuse_max_ratio;
    config.hot_path_ratio          = options.hot_path_ratio;
    return config;
}

std::vector<std::vector<StatefulPathOperation>> generateTraces(uint64_t                     seed,
                                                               const std::vector<PathKeys>& initial_paths,
                                                               size_t                       num_workers,
                                                               size_t                       total_operations,
                                                               const StatefulPathConfig&    config,
                                                               const std::vector<size_t>&   advance_counts = {}) {
    if (!advance_counts.empty() && advance_counts.size() != num_workers) {
        throw std::invalid_argument("advance count does not match worker count");
    }
    std::vector<std::vector<StatefulPathOperation>> traces(num_workers);
    for (size_t worker = 0; worker < num_workers; ++worker) {
        StatefulPathSession session(seed, 1, initial_paths, worker, num_workers, config);
        if (!advance_counts.empty()) {
            for (size_t i = 0; i < advance_counts[worker]; ++i) {
                session.nextOperation();
            }
        }
        const size_t operation_count =
            total_operations / num_workers + (worker < total_operations % num_workers ? 1 : 0);
        traces[worker].reserve(operation_count);
        for (size_t i = 0; i < operation_count; ++i) {
            traces[worker].push_back(session.nextOperation());
        }
    }
    return traces;
}

}  // anonymous namespace

TreeBenchmarkRunner::TreeBenchmarkRunner(const ModelProfile& profile,
                                         const TreeOptions&  options,
                                         uint64_t            seed,
                                         const std::string&  output_json_path):
    profile_(profile), options_(options), seed_(seed), output_json_path_(output_json_path) {
    writer_.setRunner("tree");
    writer_.setModelProfile(profile_.profile_id, profile_.sha256_hex);
    writer_.setPayloadMode(options_.payload_mode, profile_.computeGroupSetPayloadBytes("full_context"));
}

bool TreeBenchmarkRunner::run() {
    writer_.setMeasurement("steady_mixed");
    const bool benchmark_ok = runSteadyStateMeasurement();
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

bool TreeBenchmarkRunner::runSteadyStateMeasurement() {
    const size_t target      = options_.tree_node_count;
    const size_t num_workers = std::max<size_t>(1, options_.steady_threads);
    // Production-style eviction: inserts commit through
    // BlockTreeStorer::publishDeviceLocked -> settled(true,true) ->
    // checkWatermark, which demotes overflow down to the device watermark.
    // No benchmark-side eviction thread exists; workers use the cache's explicit
    // reclaim API only as a bounded request-admission fallback.
    // For profile-sized trees the watermark sits just below the 1.25x pool
    // capacity, so the excess after each insert commit is bounded and
    // event-driven eviction keeps up. Small smoke trees get extra admission
    // headroom in buildTreeCache() so one in-flight transaction can commit.
    const double device_ratio = 0.8;
    const double host_ratio   = 0.9;

    auto cache = buildTreeCache(target, options_.payload_mode, true, device_ratio, host_ratio);
    if (!cache)
        return false;

    const auto            path_config = pathConfig(options_);
    TreeWorkloadGenerator gen(seed_, target, options_.tree_branching_factor, path_config);
    const auto            topology = gen.generateTopology();

    // Phase 1: build the tree up to pool capacity (the cache is full, like a
    // production cache that has been running for a while).
    std::cout << "[tree] building tree up to " << target << " nodes..." << std::endl;
    const size_t built_nodes = insertTopology(*cache, gen.treePaths());
    cache->waitForPendingTasks();
    std::cout << "[tree] built, node_count=" << cache->getStats().tree_node_count << std::endl;
    if (built_nodes != topology.actual_node_count || cache->getStats().tree_node_count != topology.actual_node_count) {
        std::cerr << "[tree] topology build mismatch: generated=" << topology.actual_node_count
                  << " inserted=" << built_nodes << " cache=" << cache->getStats().tree_node_count << std::endl;
        return false;
    }

    // Phase 2: warmup — let the mixed workload reach dynamic balance.
    std::cout << "[tree] warmup " << options_.warmup_seconds << "s with " << num_workers << " worker(s)..."
              << std::endl;
    std::vector<size_t>                            warmup_samples;
    SteadyCounters                                 warmup_counters;
    LatencySamples                                 warmup_latencies;
    std::vector<std::shared_ptr<LoadAsyncContext>> warmup_pending_loads;
    std::vector<size_t>                            warmup_executed;
    auto                                           warmup_traces =
        generateTraces(seed_, gen.treePaths(), num_workers, options_.operation_trace_count, path_config);
    runSteadyWorkers(*cache,
                     warmup_traces,
                     num_workers,
                     static_cast<double>(options_.warmup_seconds),
                     warmup_counters,
                     warmup_latencies,
                     warmup_samples,
                     warmup_pending_loads,
                     warmup_executed);
    warmup_traces.clear();
    drainLoads(warmup_pending_loads, warmup_counters);
    cache->waitForPendingTasks();
    writer_.addMetric("warmup.load_target_allocation_retries",
                      static_cast<double>(warmup_counters.load_target_allocation_retries.load()));
    if (warmup_counters.loads_failed.load() != 0 || warmup_counters.loads_cancelled.load() != 0
        || warmup_counters.load_target_allocation_failed.load() != 0 || warmup_counters.load_commit_failed.load() != 0
        || warmup_counters.loads_succeeded.load() != warmup_counters.loads_committed.load()) {
        writer_.addMetric("warmup.loads_committed", static_cast<double>(warmup_counters.loads_committed.load()));
        writer_.addMetric("warmup.loads_succeeded", static_cast<double>(warmup_counters.loads_succeeded.load()));
        writer_.addMetric("warmup.loads_failed", static_cast<double>(warmup_counters.loads_failed.load()));
        writer_.addMetric("warmup.loads_cancelled", static_cast<double>(warmup_counters.loads_cancelled.load()));
        writer_.addMetric("warmup.load_target_allocation_failed",
                          static_cast<double>(warmup_counters.load_target_allocation_failed.load()));
        writer_.addMetric("warmup.load_commit_failed", static_cast<double>(warmup_counters.load_commit_failed.load()));
        std::cerr << "[tree] warmup load invariant failed: committed=" << warmup_counters.loads_committed.load()
                  << " succeeded=" << warmup_counters.loads_succeeded.load()
                  << " failed=" << warmup_counters.loads_failed.load()
                  << " cancelled=" << warmup_counters.loads_cancelled.load()
                  << " target_allocation_retries=" << warmup_counters.load_target_allocation_retries.load()
                  << " target_allocation_failed=" << warmup_counters.load_target_allocation_failed.load()
                  << " commit_failed=" << warmup_counters.load_commit_failed.load() << std::endl;
        return false;
    }

    if (warmup_counters.trace_exhaustions.load() != 0) {
        std::cerr << "[tree] warmup operation trace exhausted; increase --operation-trace-count" << std::endl;
        return false;
    }

    // Recreate the deterministic worker sessions and advance them by exactly
    // the operations that warmup committed. This keeps the measured trace
    // logically continuous without generating paths inside the timed region.
    auto measured_traces = generateTraces(
        seed_, gen.treePaths(), num_workers, options_.operation_trace_count, path_config, warmup_executed);

    // Phase 3: measured window. Announce MEASURE_START so the driver can attach
    // perf to this process, then wait briefly for the attach before timing.
    std::cout << "MEASURE_START" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::vector<size_t>                            node_samples;
    SteadyCounters                                 counters;
    LatencySamples                                 latencies;
    std::vector<std::shared_ptr<LoadAsyncContext>> pending_loads;
    std::vector<size_t>                            measured_executed;
    auto                                           start = Clock::now();
    runSteadyWorkers(*cache,
                     measured_traces,
                     num_workers,
                     static_cast<double>(options_.min_measured_seconds),
                     counters,
                     latencies,
                     node_samples,
                     pending_loads,
                     measured_executed);
    auto          end         = Clock::now();
    const int64_t measured_ns = elapsedNs(start, end);
    counters.loads_pending_at_measurement_end.store(pending_loads.size());
    const auto drain_start = Clock::now();
    drainLoads(pending_loads, counters);
    cache->waitForPendingTasks();
    const int64_t drain_ns = elapsedNs(drain_start, Clock::now());

    const size_t insert_calls                   = counters.insert_calls.load();
    const size_t insert_path_keys               = counters.insert_path_keys.load();
    const size_t insert_new_nodes               = counters.insert_new_nodes.load();
    const size_t match_requests                 = counters.match_requests.load();
    const size_t match_keys                     = counters.match_keys.load();
    const size_t match_device_blocks            = counters.match_device_blocks.load();
    const size_t match_host_blocks              = counters.match_host_blocks.load();
    const size_t loads_committed                = counters.loads_committed.load();
    const size_t loads_succeeded                = counters.loads_succeeded.load();
    const size_t loads_failed                   = counters.loads_failed.load();
    const size_t loads_cancelled                = counters.loads_cancelled.load();
    const size_t load_target_allocation_retries = counters.load_target_allocation_retries.load();
    const size_t load_target_allocation_failed  = counters.load_target_allocation_failed.load();
    const size_t load_commit_failed             = counters.load_commit_failed.load();
    const size_t requested_ops =
        insert_calls + match_requests + loads_committed + load_target_allocation_failed + load_commit_failed;
    const size_t failed_ops    = loads_failed + loads_cancelled + load_target_allocation_failed + load_commit_failed;
    const size_t succeeded_ops = requested_ops - failed_ops;

    // Node count stats
    double node_avg = 0;
    size_t node_min = std::numeric_limits<size_t>::max(), node_max = 0;
    if (!node_samples.empty()) {
        node_avg =
            static_cast<double>(std::accumulate(node_samples.begin(), node_samples.end(), 0ULL)) / node_samples.size();
        for (auto n : node_samples) {
            node_min = std::min(node_min, n);
            node_max = std::max(node_max, n);
        }
    } else {
        node_min = 0;
    }

    writer_.setWorkload(seed_, requested_ops, requested_ops, succeeded_ops, failed_ops);
    writer_.addPhaseNs("measured", measured_ns);
    writer_.addPhaseNs("sync_drain", drain_ns);
    writer_.addResolvedConfigInt("target_node_count", target);
    writer_.addResolvedConfig("topology", "stateful_shared_prefix_trie");
    writer_.addResolvedConfigInt("initial_topology_path_count", gen.treePaths().size());
    writer_.addResolvedConfigInt("initial_topology_max_depth", topology.max_depth);
    writer_.addResolvedConfigInt("initial_topology_leaf_count", topology.leaf_count);
    writer_.addResolvedConfigInt("max_path_length", options_.max_path_length);
    writer_.addResolvedConfigInt("tree_branching_factor", options_.tree_branching_factor);
    writer_.addResolvedConfigInt("initial_min_path_length", options_.initial_min_path_length);
    writer_.addResolvedConfigInt("initial_max_path_length", options_.initial_max_path_length);
    writer_.addResolvedConfig("continuation_ratio", std::to_string(options_.continuation_ratio));
    writer_.addResolvedConfig("fork_ratio", std::to_string(options_.fork_ratio));
    writer_.addResolvedConfig("cold_ratio", std::to_string(1.0 - options_.continuation_ratio - options_.fork_ratio));
    writer_.addResolvedConfig("fork_reuse_min_ratio", std::to_string(options_.fork_reuse_min_ratio));
    writer_.addResolvedConfig("fork_reuse_max_ratio", std::to_string(options_.fork_reuse_max_ratio));
    writer_.addResolvedConfig("hot_path_ratio", std::to_string(options_.hot_path_ratio));
    writer_.addResolvedConfigInt("active_path_limit", options_.active_path_limit);
    writer_.addResolvedConfigInt("append_length", options_.append_length);
    writer_.addResolvedConfigInt("inserts_per_match", options_.inserts_per_match);
    writer_.addResolvedConfigInt("operation_trace_count", options_.operation_trace_count);
    writer_.addResolvedConfigInt("device_watermark_ratio", static_cast<int>(device_ratio * 100));
    writer_.addResolvedConfigInt("steady_threads", num_workers);
    writer_.addResolvedConfigInt("warmup_seconds", options_.warmup_seconds);
    writer_.addResolvedConfigInt("min_measured_seconds", options_.min_measured_seconds);

    // Per-call latency stats: min / p50 / p99 / max / avg (ns).
    auto add_latency_metrics = [&](const std::string& prefix, const std::vector<int64_t>& samples) {
        if (samples.empty())
            return;
        std::vector<int64_t> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        auto percentile = [&](double q) {
            const size_t idx = static_cast<size_t>(q * static_cast<double>(sorted.size() - 1));
            return sorted[idx];
        };
        const double avg = static_cast<double>(std::accumulate(sorted.begin(), sorted.end(), 0LL))
                           / static_cast<double>(sorted.size());
        writer_.addMetric(prefix + "_latency_ns_min", static_cast<double>(sorted.front()));
        writer_.addMetric(prefix + "_latency_ns_p50", static_cast<double>(percentile(0.5)));
        writer_.addMetric(prefix + "_latency_ns_p99", static_cast<double>(percentile(0.99)));
        writer_.addMetric(prefix + "_latency_ns_max", static_cast<double>(sorted.back()));
        writer_.addMetric(prefix + "_latency_ns_avg", avg);
        writer_.addMetric(prefix + "_calls", static_cast<double>(sorted.size()));
    };
    add_latency_metrics("insert", latencies.insert_ns);
    add_latency_metrics("match", latencies.match_ns);
    add_latency_metrics("load", latencies.load_ns);

    writer_.addMetric("insert_path_keys_per_call",
                      insert_calls ? static_cast<double>(insert_path_keys) / insert_calls : 0.0);
    writer_.addMetric("insert_new_nodes_per_call",
                      insert_calls ? static_cast<double>(insert_new_nodes) / insert_calls : 0.0);
    writer_.addMetric("match_keys_per_call", match_requests ? static_cast<double>(match_keys) / match_requests : 0.0);
    writer_.addMetric("match_device_matched_blocks_per_request",
                      match_requests ? static_cast<double>(match_device_blocks) / match_requests : 0.0);
    writer_.addMetric("match_host_matched_blocks_per_request",
                      match_requests ? static_cast<double>(match_host_blocks) / match_requests : 0.0);
    writer_.addMetric("trace_exhaustions", static_cast<double>(counters.trace_exhaustions.load()));
    writer_.addMetric("loads_committed", static_cast<double>(loads_committed));
    writer_.addMetric("loads_succeeded", static_cast<double>(loads_succeeded));
    writer_.addMetric("loads_failed", static_cast<double>(loads_failed));
    writer_.addMetric("loads_cancelled", static_cast<double>(loads_cancelled));
    writer_.addMetric("load_target_allocation_retries", static_cast<double>(load_target_allocation_retries));
    writer_.addMetric("load_target_allocation_failed", static_cast<double>(load_target_allocation_failed));
    writer_.addMetric("load_commit_failed", static_cast<double>(load_commit_failed));
    writer_.addMetric("loads_pending_at_measurement_end",
                      static_cast<double>(counters.loads_pending_at_measurement_end.load()));
    writer_.addMetric("steady_state_node_count_avg", node_avg);
    writer_.addMetric("steady_state_node_count_min", static_cast<double>(node_min));
    writer_.addMetric("steady_state_node_count_max", static_cast<double>(node_max));
    writer_.addMetric("node_samples", static_cast<double>(node_samples.size()));

    static const std::array<std::string, 3> kScenarios = {"continuation", "fork", "cold"};
    for (size_t kind = 0; kind < kScenarios.size(); ++kind) {
        const double      requests              = static_cast<double>(counters.scenario_requests[kind].load());
        const double      scenario_insert_calls = static_cast<double>(counters.scenario_insert_calls[kind].load());
        const std::string prefix                = "scenario." + kScenarios[kind] + ".";
        writer_.addMetric(prefix + "requests", requests);
        if (requests > 0) {
            writer_.addMetric(prefix + "average_match_key_count",
                              static_cast<double>(counters.scenario_match_keys[kind].load()) / requests);
            writer_.addMetric(prefix + "average_matched_depth",
                              static_cast<double>(counters.scenario_matched_depth[kind].load()) / requests);
        }
        writer_.addMetric(prefix + "insert_calls", scenario_insert_calls);
        if (scenario_insert_calls > 0) {
            writer_.addMetric(prefix + "average_insert_path_length",
                              static_cast<double>(counters.scenario_insert_path_keys[kind].load())
                                  / scenario_insert_calls);
            writer_.addMetric(prefix + "average_new_nodes_per_insert",
                              static_cast<double>(counters.scenario_insert_new_nodes[kind].load())
                                  / scenario_insert_calls);
        }
        writer_.addMetric(prefix + "device_hits", static_cast<double>(counters.scenario_device_hits[kind].load()));
        writer_.addMetric(prefix + "host_hits", static_cast<double>(counters.scenario_host_hits[kind].load()));
        writer_.addMetric(prefix + "disk_hits", static_cast<double>(counters.scenario_disk_hits[kind].load()));
        writer_.addMetric(prefix + "misses", static_cast<double>(counters.scenario_misses[kind].load()));
    }

    std::cout << "[tree] measured " << static_cast<double>(measured_ns) / 1e9 << "s with " << num_workers
              << " worker(s): insert=" << insert_calls << " calls/" << insert_new_nodes << " new nodes"
              << " (avg path " << (insert_calls ? insert_path_keys / insert_calls : 0) << ", new "
              << (insert_calls ? insert_new_nodes / insert_calls : 0) << " nodes/call), "
              << "match=" << match_requests << " (avg "
              << (match_requests ? static_cast<double>(match_device_blocks) / match_requests : 0) << " device/"
              << (match_requests ? static_cast<double>(match_host_blocks) / match_requests : 0)
              << " host matched blocks/request, " << (match_requests ? match_keys / match_requests : 0)
              << " keys/call), "
              << "loads=" << loads_committed << " committed/" << loads_succeeded << " succeeded/" << loads_failed
              << " failed, "
              << "nodes avg=" << node_avg << " [" << node_min << "," << node_max << "]" << std::endl;

    if (!latencies.insert_ns.empty()) {
        std::vector<int64_t> v = latencies.insert_ns;
        std::sort(v.begin(), v.end());
        std::cout << "[tree] insert latency ns: min=" << v.front() << " p50=" << v[v.size() / 2]
                  << " p99=" << v[v.size() * 99 / 100] << " max=" << v.back() << std::endl;
    }
    if (!latencies.match_ns.empty()) {
        std::vector<int64_t> v = latencies.match_ns;
        std::sort(v.begin(), v.end());
        std::cout << "[tree] match latency ns: min=" << v.front() << " p50=" << v[v.size() / 2]
                  << " p99=" << v[v.size() * 99 / 100] << " max=" << v.back() << std::endl;
    }

    return requested_ops > 0 && failed_ops == 0 && loads_succeeded == loads_committed
           && counters.trace_exhaustions.load() == 0;
}

void TreeBenchmarkRunner::workerLoop(BlockTreeCache&                                 cache,
                                     const std::vector<StatefulPathOperation>&       trace,
                                     double                                          seconds,
                                     SteadyCounters&                                 counters,
                                     LatencySamples&                                 latencies,
                                     std::mutex&                                     merge_mutex,
                                     std::vector<std::shared_ptr<LoadAsyncContext>>& shared_pending_loads,
                                     size_t&                                         executed_transactions) {
    std::vector<std::shared_ptr<LoadAsyncContext>> pending_loads;
    std::vector<int64_t>                           insert_lat_ns, match_lat_ns, load_lat_ns;

    const auto start    = Clock::now();
    const auto deadline = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    size_t     trace_index = 0;
    while (Clock::now() < deadline && trace_index < trace.size()) {
        const auto&  operation = trace[trace_index];
        const size_t scenario  = static_cast<size_t>(operation.scenario);

        // A previous asynchronous load must settle before the next match can
        // select the same lower-tier nodes again.
        drainLoads(pending_loads, counters);
        const auto match_start = Clock::now();
        auto       result      = cache.match(*operation.match_path);
        match_lat_ns.push_back(elapsedNs(match_start, Clock::now()));

        const size_t matched_depth =
            result.async_context != nullptr ? result.async_context->matchedBlocks() : result.matched_device_blocks;
        counters.match_requests.fetch_add(1);
        counters.match_keys.fetch_add(operation.match_path->size());
        counters.match_device_blocks.fetch_add(result.matched_device_blocks);
        counters.scenario_requests[scenario].fetch_add(1);
        counters.scenario_match_keys[scenario].fetch_add(operation.match_path->size());
        counters.scenario_matched_depth[scenario].fetch_add(matched_depth);
        if (result.matched_device_blocks > 0) {
            counters.scenario_device_hits[scenario].fetch_add(1);
        }
        if (result.async_context != nullptr) {
            const size_t host_matched_blocks = result.async_context->matchedBlocks(Tier::HOST);
            counters.match_host_blocks.fetch_add(host_matched_blocks);
            if (host_matched_blocks > 0) {
                counters.scenario_host_hits[scenario].fetch_add(1);
            }
            if (result.async_context->matchedBlocks(Tier::DISK) > 0) {
                counters.scenario_disk_hits[scenario].fetch_add(1);
            }
        }
        if (matched_depth == 0) {
            counters.scenario_misses[scenario].fetch_add(1);
        }

        // Match misses carry an async lower-tier load: allocate device targets,
        // commit the transfer, and keep its context alive until completion.
        if (result.async_context != nullptr && !result.async_context->empty()) {
            using OwnedTargets                   = std::pair<DeviceBlockPoolPtr, BlockIdList>;
            const auto                load_start = Clock::now();
            bool                      targets_ok = true;
            const auto&               descs      = result.async_context->loadDescs();
            std::vector<size_t>       required_targets(cache.groupSets().size(), 0);
            std::vector<size_t>       next_target(cache.groupSets().size(), 0);
            std::vector<OwnedTargets> owned_targets(cache.groupSets().size());
            for (size_t d = 0; d < descs.size(); ++d) {
                if (descs[d].source_tier == Tier::DEVICE || result.async_context->joinedLoads()[d]) {
                    continue;
                }
                ++required_targets[descs[d].group_set_id];
            }
            for (size_t gs = 0; gs < required_targets.size(); ++gs) {
                if (required_targets[gs] == 0) {
                    continue;
                }
                const auto& pools = cache.groupSets()[gs]->devicePools();
                if (pools.empty()) {
                    targets_ok = false;
                    break;
                }
                auto   blocks             = pools[0]->malloc(required_targets[gs]);
                size_t allocation_retries = 0;
                while (!blocks.has_value() && allocation_retries < 250) {
                    ++allocation_retries;
                    counters.load_target_allocation_retries.fetch_add(1);
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    if (allocation_retries % 25 == 0 && !cache.groupSets()[gs]->groupIds().empty()) {
                        cache.evictForGroup(cache.groupSets()[gs]->groupIds().front(), required_targets[gs]);
                    }
                    blocks = pools[0]->malloc(required_targets[gs]);
                }
                if (!blocks.has_value()) {
                    targets_ok = false;
                    break;
                }
                pools[0]->incRef(blocks.value(), BlockRefType::REQUEST);
                owned_targets[gs] = {pools[0], std::move(blocks.value())};
            }
            if (targets_ok) {
                for (size_t d = 0; d < descs.size(); ++d) {
                    if (descs[d].source_tier == Tier::DEVICE || result.async_context->joinedLoads()[d]) {
                        continue;
                    }
                    const size_t gs = descs[d].group_set_id;
                    result.async_context->setTargetBlocks(d, {owned_targets[gs].second[next_target[gs]++]});
                }
            }
            const bool committed = targets_ok && result.async_context->commit();
            for (const auto& [pool, blocks] : owned_targets) {
                if (pool != nullptr && !blocks.empty()) {
                    pool->decRef(blocks, BlockRefType::REQUEST);
                }
            }
            if (committed) {
                counters.loads_committed.fetch_add(1);
                load_lat_ns.push_back(elapsedNs(load_start, Clock::now()));
                pending_loads.push_back(std::move(result.async_context));
            } else {
                if (targets_ok) {
                    counters.load_commit_failed.fetch_add(1);
                } else {
                    counters.load_target_allocation_failed.fetch_add(1);
                }
                result.async_context.reset();
            }
        }

        // The first insert creates the part of the matched request that was not
        // already cached, plus one append. Later inserts extend the path made by
        // the preceding commit. Complete a started transaction even if it
        // crosses the phase deadline so session reconstruction remains exact.
        size_t existing_prefix_length = matched_depth;
        for (const auto& insert_path : operation.insert_paths) {
            size_t       failed_retries = 0;
            const size_t new_nodes      = insert_path->size() - existing_prefix_length;
            const auto   insert_start   = Clock::now();
            while (!insertPathFromPrefix(cache, *insert_path, existing_prefix_length)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                if (++failed_retries >= 50) {
                    failed_retries = 0;
                    for (const auto& group_set : cache.groupSets()) {
                        if (!group_set->groupIds().empty()) {
                            cache.evictForGroup(group_set->groupIds().front(), new_nodes);
                        }
                    }
                }
            }
            insert_lat_ns.push_back(elapsedNs(insert_start, Clock::now()));
            counters.insert_calls.fetch_add(1);
            counters.insert_path_keys.fetch_add(insert_path->size());
            counters.insert_new_nodes.fetch_add(new_nodes);
            counters.scenario_insert_calls[scenario].fetch_add(1);
            counters.scenario_insert_path_keys[scenario].fetch_add(insert_path->size());
            counters.scenario_insert_new_nodes[scenario].fetch_add(new_nodes);
            existing_prefix_length = insert_path->size();
        }
        cache.releaseMatchedResources(result.matched_device_resources);
        ++trace_index;

        for (const auto& context : pending_loads) {
            if (context->done()) {
                if (context->success()) {
                    counters.loads_succeeded.fetch_add(1);
                } else if (context->isRequestCanceled()) {
                    counters.loads_cancelled.fetch_add(1);
                } else {
                    counters.loads_failed.fetch_add(1);
                }
            }
        }
        pending_loads.erase(
            std::remove_if(pending_loads.begin(),
                           pending_loads.end(),
                           [](const std::shared_ptr<LoadAsyncContext>& context) { return context->done(); }),
            pending_loads.end());
    }

    if (trace_index == trace.size() && Clock::now() < deadline) {
        counters.trace_exhaustions.fetch_add(1);
    }
    executed_transactions = trace_index;
    {
        std::lock_guard<std::mutex> lock(merge_mutex);
        latencies.insert_ns.insert(latencies.insert_ns.end(), insert_lat_ns.begin(), insert_lat_ns.end());
        latencies.match_ns.insert(latencies.match_ns.end(), match_lat_ns.begin(), match_lat_ns.end());
        latencies.load_ns.insert(latencies.load_ns.end(), load_lat_ns.begin(), load_lat_ns.end());
        shared_pending_loads.insert(shared_pending_loads.end(), pending_loads.begin(), pending_loads.end());
    }
}

void TreeBenchmarkRunner::runSteadyWorkers(BlockTreeCache&                                        cache,
                                           const std::vector<std::vector<StatefulPathOperation>>& traces,
                                           size_t                                                 num_workers,
                                           double                                                 seconds,
                                           SteadyCounters&                                        counters,
                                           LatencySamples&                                        latencies,
                                           std::vector<size_t>&                                   node_samples,
                                           std::vector<std::shared_ptr<LoadAsyncContext>>&        pending_loads,
                                           std::vector<size_t>& executed_transactions) {
    if (traces.size() != num_workers) {
        throw std::invalid_argument("operation trace count does not match worker count");
    }
    executed_transactions.assign(num_workers, 0);
    std::mutex              merge_mutex;
    std::mutex              sampler_mutex;
    std::condition_variable sampler_cv;
    bool                    workers_done = false;
    std::thread             sampler([&]() {
        std::unique_lock<std::mutex> lock(sampler_mutex);
        while (!workers_done) {
            node_samples.push_back(cache.getStats().tree_node_count);
            sampler_cv.wait_for(lock, std::chrono::milliseconds(100), [&]() { return workers_done; });
        }
    });

    std::vector<std::thread> threads;
    threads.reserve(num_workers);
    for (size_t w = 0; w < num_workers; ++w) {
        threads.emplace_back([&, w]() {
            workerLoop(
                cache, traces[w], seconds, counters, latencies, merge_mutex, pending_loads, executed_transactions[w]);
        });
    }
    for (auto& t : threads) {
        if (t.joinable())
            t.join();
    }
    {
        std::lock_guard<std::mutex> lock(sampler_mutex);
        workers_done = true;
    }
    sampler_cv.notify_one();
    sampler.join();
}

void TreeBenchmarkRunner::drainLoads(std::vector<std::shared_ptr<LoadAsyncContext>>& pending_loads,
                                     SteadyCounters&                                 counters) {
    for (const auto& context : pending_loads) {
        context->waitDone();
        if (context->success())
            counters.loads_succeeded.fetch_add(1);
        else if (context->isRequestCanceled())
            counters.loads_cancelled.fetch_add(1);
        else
            counters.loads_failed.fetch_add(1);
    }
    pending_loads.clear();
}

std::unique_ptr<BlockTreeCache> TreeBenchmarkRunner::buildTreeCache(size_t             node_count,
                                                                    const std::string& payload_mode,
                                                                    bool               enable_host,
                                                                    double             device_watermark_ratio,
                                                                    double             host_watermark_ratio) {
    // Pool sized just above the watermark so a commit's excess (and thus the
    // number of async demote tasks submitted under the cache mutex) stays
    // small; a 2.5x pool makes every insert commit flood the task pool. For a
    // small smoke tree, however, 25% headroom can be smaller than one cold
    // request. Reserve one maximum path per worker so allocation can reach the
    // insert commit that triggers event-driven eviction.
    const size_t watermark_capacity = static_cast<size_t>(static_cast<double>(node_count) * 1.25);
    const size_t admission_headroom = options_.max_path_length * std::max<size_t>(1, options_.steady_threads);
    const size_t device_block_count = std::max(watermark_capacity, node_count + admission_headroom);

    // Shared topology: one group per group set with a globally unique group id.
    std::vector<std::pair<std::string, rtp_llm::CacheGroupType>> group_specs;
    std::vector<size_t>                                          group_payloads;
    for (const auto& gs_info : profile_.group_sets) {
        bool is_swa = false;
        for (const auto& tag : gs_info.member_tags) {
            const auto* group = profile_.findGroup(tag);
            if (group && group->type == benchmark::CacheGroupType::SWA) {
                is_swa = true;
                break;
            }
        }
        group_specs.emplace_back(gs_info.name, is_swa ? rtp_llm::CacheGroupType::SWA : rtp_llm::CacheGroupType::FULL);
        group_payloads.push_back(payload_mode == "scaled" ?
                                     BenchmarkFixture::computeScaledPayload(gs_info.payload_bytes) :
                                     gs_info.payload_bytes);
    }
    auto topology = BenchmarkFixture::createTopology(group_specs, group_payloads);

    std::vector<GroupSetPtr> group_sets;
    for (size_t gs_idx = 0; gs_idx < profile_.group_sets.size(); ++gs_idx) {
        const auto& gs_info    = profile_.group_sets[gs_idx];
        size_t      gs_payload = group_payloads[gs_idx];

        auto device_pool =
            BenchmarkFixture::createDevicePool(gs_payload, 1, device_block_count, "device_" + gs_info.name);
        std::shared_ptr<HostBlockPool> host_pool = nullptr;
        BlockTreeDiskBlockPoolPtr      disk_pool = nullptr;

        if (enable_host) {
            // Host must hold the demoted steady-state working set; a half-size
            // host fills up and drops data before match can load it back.
            host_pool = BenchmarkFixture::createHostPool(gs_payload, device_block_count, true);
        }

        const std::vector<size_t> group_ids = {gs_idx};

        GroupSetPtr gs;
        if (group_specs[gs_idx].second == rtp_llm::CacheGroupType::SWA) {
            gs = BenchmarkFixture::createSWAGroupSet(
                {device_pool}, host_pool, disk_pool, gs_idx, topology, group_ids, 128);
        } else {
            gs = BenchmarkFixture::createFullGroupSet({device_pool}, host_pool, disk_pool, gs_idx, topology, group_ids);
        }
        group_sets.push_back(gs);
    }

    // Keep the async demote/load execution capacity identical across the
    // single- and multi-worker cases, avoiding a second concurrency variable.
    constexpr size_t task_pool_size = 32;
    return BenchmarkFixture::createCache(
        group_sets, enable_host, false, task_pool_size, device_watermark_ratio, host_watermark_ratio);
}

bool TreeBenchmarkRunner::insertPathFromPrefix(BlockTreeCache& cache,
                                               const PathKeys& path,
                                               size_t          existing_prefix_length) {
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
            // The tree takes BLOCK_CACHE ownership only when it accepts this
            // incoming resource. Hold a temporary request reference across
            // insert so rejected/adopted-race resources are released instead
            // of leaking pool capacity.
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
    // Publish REQUEST-holder transitions through the cache so blocks accepted
    // by the tree become eviction candidates as soon as their temporary holder
    // is gone. Rejected resources have no reverse-index entry and are simply
    // returned to the pool by the same release batch.
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

}  // namespace rtp_llm::benchmark
