#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/KVCMStorageBackend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <iterator>
#include <mutex>
#include <unordered_set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <unistd.h>

#include "autil/EnvUtil.h"
#include "autil/legacy/jsonizable.h"
#include "kvcm_client/HostCacheStateCompat.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientWrapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/GroupPolicy.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

namespace rtp_llm {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::array<const char*, 4> kQuarantineReasons{
    "sdk_timeout", "rpc_timeout", "rpc_error", "invalid_response"};

int64_t elapsedTimeUs(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
}

size_t hashString(const std::string& value) {
    return std::hash<std::string>{}(value);
}

std::string nextTraceId(const char* operation, std::atomic<uint64_t>& sequence) {
    return std::string("block_tree_") + operation + "_" + std::to_string(sequence.fetch_add(1));
}

struct KVCMMatchMeta final: StorageBackendMatchMeta {
    kv_cache_manager::Locations locations;
    bool                       filtered = false;
};

const StorageBlockHandle* findHandle(const std::vector<StorageBlockHandle>& handles, std::string_view tag) {
    const auto it = std::find_if(handles.begin(), handles.end(), [tag](const StorageBlockHandle& handle) {
        return handle.tag == tag && !isNullBlockIdx(handle.block);
    });
    return it == handles.end() ? nullptr : &*it;
}

template<typename Match>
void appendHostCacheMatch(const Match& host, RemoteHostCacheMatchPB& output) {
    output.set_host_ip_port(host.host_ip_port);
    output.set_local(host.local);
    if constexpr (kvcm::HasGlobalHostCacheMatch<Match>::value) {
        output.set_global(host.global);
    } else {
        output.set_p2p_1_fetch(host.p2p_1_fetch);
        output.set_p2p_1_total_match(host.p2p_1_total_match);
    }
}

}  // namespace

class KVCMStorageBackend::Impl {
public:
    using ActualUriGather = std::vector<std::vector<kv_cache_manager::LocationSpecUnit*>>;

    enum class TransferResult {
        SUCCESS,
        FAILED,
        TIMEOUT
    };

    Impl(const StorageBackend& backend,
         const CacheConfig&                         cache_config,
         const KVCacheConfig&                       kv_cache_config,
         const RuntimeConfig&                       runtime_config,
         const ParallelismConfig&                   parallelism_config,
         const SpeculativeExecutionConfig&          sp_config,
         std::shared_ptr<BroadcastManager>          broadcast_manager,
         std::shared_ptr<kvcm::ClientWrapper>       client_wrapper,
         std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter,
         bool gdr_enabled):
        backend_(backend),
        cache_config_(cache_config),
        kv_cache_config_(kv_cache_config),
        runtime_config_(runtime_config),
        parallelism_config_(parallelism_config),
        sp_config_(sp_config),
        broadcast_manager_(std::move(broadcast_manager)),
        metrics_reporter_(std::move(metrics_reporter)),
        client_wrapper_(std::move(client_wrapper)),
        gdr_enabled_(gdr_enabled),
        sdk_check_enabled_(autil::EnvUtil::getEnv("KVCM_SDK_CHECK", autil::EnvUtil::getEnv("RECO_SDK_CHECK", false))) {
    }

    std::optional<ControlConfig> controlConfig() const {
        return control_config_;
    }

    bool init(const CacheTopology&                                                topology,
              StorageBackend::BufferResolver                                      buffer_resolver,
              const std::function<const DeviceBlockPoolPtr&(const std::string&)>& pool_resolver) {
        RTP_LLM_LOG_INFO("start init BlockTree KVCM storage backend");
        const auto quarantine_abort_ratio = remoteCacheQuarantineAbortRatio();
        if (!quarantine_abort_ratio.has_value()) {
            RTP_LLM_LOG_ERROR("RTP_LLM_REMOTE_CACHE_QUARANTINE_ABORT_RATIO must be a finite number in [0, 1]");
            return false;
        }
        quarantine_abort_ratio_ = *quarantine_abort_ratio;
        if (parallelism_config_.tp_rank == 0 && parallelism_config_.tp_size > 1 && !broadcast_manager_) {
            RTP_LLM_LOG_ERROR("BlockTree KVCM rank 0 requires a broadcast manager for tp_size=%ld",
                              parallelism_config_.tp_size);
            return false;
        }
        std::vector<std::string> full_group_tags;
        std::vector<std::string> other_group_tags;
        // CacheConfig::blockSizeBytesForGroup resolves MTP child-owned physical
        // strides; do not replace this with topology-derived geometry.
        std::unordered_map<std::string, size_t> group_block_size_bytes;
        const std::vector<GroupBase>&           groups = topology.groups();
        topology_ = &topology;
        group_block_size_bytes.reserve(groups.size());
        for (const auto& group : groups) {
            if (!group.policy.enable_prefix_reuse) {
                continue;
            }
            has_swa_ = has_swa_ || group.policy.group_type == CacheGroupType::SWA;
            if (group.policy.group_type == CacheGroupType::FULL) {
                full_group_tags.push_back(group.tag);
            } else {
                other_group_tags.push_back(group.tag);
            }
            group_block_size_bytes.emplace(group.tag, cache_config_.blockSizeBytesForGroup(group.tag));
        }
        if (other_group_tags.empty()) {
            group_policy_ = std::make_unique<kvcm::FullLayerGroupPolicy>(
                topology, buffer_resolver, full_group_tags, other_group_tags, std::move(group_block_size_bytes));
        } else {
            group_policy_ = std::make_unique<kvcm::FullLinearLayerGroupPolicy>(
                topology, buffer_resolver, full_group_tags, other_group_tags, std::move(group_block_size_bytes));
        }
        if (!group_policy_->init()) {
            RTP_LLM_LOG_ERROR("BlockTree KVCM group policy init failed");
            return false;
        }
        kvcm::ClientWrapper::ConfigMap client_config_map;
        try {
            if (!kv_cache_config_.kvcm_client_config.empty()) {
                auto [location_infos, location_groups] = genLocationSpecs();
                autil::legacy::FromJsonString(client_config_map, kv_cache_config_.kvcm_client_config);
                if (client_config_map.size() == 1 && !client_config_map.begin()->first.empty()) {
                    auto config = client_config_map.begin()->second;
                    client_config_map.clear();
                    client_config_map.emplace("", std::move(config));
                }
                if (group_policy_->groups().size() != topology.groups().size()) {
                    // Keep custom settings and aliases, but only advertise
                    // payloads that the runtime can actually reuse and transfer.
                    auto config = client_config_map.at("");
                    auto fields = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(autil::legacy::ToJson(config));
                    kvcm::KVCMConfig::LocationSpecGroups custom_groups;
                    if (autil::legacy::AnyCast<autil::legacy::json::JsonMap>(&fields.at("location_spec_groups"))) {
                        autil::legacy::FromJson(custom_groups, fields.at("location_spec_groups"));
                    }
                    for (auto it = custom_groups.begin(); it != custom_groups.end();) {
                        if (std::any_of(it->second.begin(), it->second.end(), [&](const auto& name) {
                                return location_infos->count(name) == 0;
                            })) {
                            it = custom_groups.erase(it);
                        } else {
                            ++it;
                        }
                    }
                    for (const auto& [name, specs] : *location_groups) {
                        custom_groups[name] = specs;
                    }
                    fields["location_spec_infos"] = autil::legacy::ToJson(location_infos);
                    fields["location_spec_groups"] = autil::legacy::ToJson(custom_groups);
                    autil::legacy::FromJson(*config, fields);
                }
                const auto configured = client_config_map.at("");
                if (!configured) {
                    RTP_LLM_LOG_ERROR("KVCM_CLIENT_CONFIG has no default instance config");
                    return false;
                }
                const auto fields =
                    autil::legacy::AnyCast<autil::legacy::json::JsonMap>(autil::legacy::ToJson(configured));
                kvcm::KVCMConfig::LocationSpecInfoMap configured_infos;
                kvcm::KVCMConfig::LocationSpecGroups configured_groups;
                autil::legacy::FromJson(configured_infos, fields.at("location_spec_infos"));
                autil::legacy::FromJson(configured_groups, fields.at("location_spec_groups"));
                for (const auto& [group_id, group] : group_policy_->groups()) {
                    (void)group_id;
                    for (int rank = 0; rank < parallelism_config_.tp_size; ++rank) {
                        const std::string legacy_name = "tp" + std::to_string(rank) + "_" + group.group_name;
                        const std::string expected_name =
                            kvcm::genLocationSpecName(rank, group.group_name, runtime_config_.model_name);
                        if (legacy_name != expected_name && configured_infos.count(legacy_name) != 0) {
                            RTP_LLM_LOG_ERROR("KVCM_CLIENT_CONFIG must replace location spec [%s] with [%s]",
                                              legacy_name.c_str(), expected_name.c_str());
                            return false;
                        }
                    }
                }
                for (const auto& [name, size] : *location_infos) {
                    (void)size;
                    if (configured_infos.count(name) == 0) {
                        RTP_LLM_LOG_ERROR("KVCM_CLIENT_CONFIG is missing location spec [%s]", name.c_str());
                        return false;
                    }
                }
                for (const auto& [name, size] : configured_infos) {
                    (void)size;
                    if (location_infos->count(name) == 0) {
                        RTP_LLM_LOG_ERROR("KVCM_CLIENT_CONFIG has unexpected location spec [%s]", name.c_str());
                        return false;
                    }
                }
                for (const auto& [name, specs] : *location_groups) {
                    const auto it = configured_groups.find(name);
                    if (it == configured_groups.end() || it->second != specs) {
                        RTP_LLM_LOG_ERROR("KVCM_CLIENT_CONFIG location group [%s] does not match runtime specs",
                                          name.c_str());
                        return false;
                    }
                }
                for (const auto& [group_name, specs] : configured_groups) {
                    for (const auto& name : specs) {
                        if (location_infos->count(name) == 0) {
                            RTP_LLM_LOG_ERROR("KVCM_CLIENT_CONFIG location group [%s] has unknown spec [%s]",
                                              group_name.c_str(), name.c_str());
                            return false;
                        }
                    }
                }
            } else {
                client_config_map = genClientConfig(full_group_tags);
            }
        } catch (const autil::legacy::ExceptionBase& error) {
            RTP_LLM_LOG_ERROR("parse KVCM_CLIENT_CONFIG failed: %s", error.what());
            return false;
        } catch (const std::exception& error) {
            RTP_LLM_LOG_ERROR("initialize BlockTree KVCM client config failed: %s", error.what());
            return false;
        }
        if (client_config_map.size() != 1 || client_config_map.count("") != 1 || !client_config_map.at("")) {
            RTP_LLM_LOG_ERROR("BlockTree KVCM requires one default instance config");
            return false;
        }
        const auto& config = client_config_map.at("");
        // Use prefix queries for the instance; payload reuse is filtered locally.
        config->set_default_query_type(default_query_type_);
        ControlConfig control_config{config->instance_group(), config->instance_id(), config->registerInstanceRequest(),
                                     config->enable_vipserver(), config->vipserver_domain(), config->addresses()};
        if (kv_cache_config_.kvcm_min_replica_count < 0
            || (kv_cache_config_.kvcm_read_backend_type != 0
                && !isPayloadBackend(kv_cache_config_.kvcm_read_backend_type))) {
            RTP_LLM_LOG_ERROR("invalid KVCM replica/backend config");
            return false;
        }

        const auto registrations = makePoolRegistrations(pool_resolver);
        if (registrations.empty()) {
            return false;
        }
        const auto role =
            parallelism_config_.tp_rank == 0 ? kv_cache_manager::RoleType::HYBRID : kv_cache_manager::RoleType::WORKER;
        if (!client_wrapper_) {
            client_wrapper_ = std::make_shared<kvcm::ClientWrapper>();
        }
        const bool initialized =
            client_wrapper_->initForPools(client_config_map, role, registrations, registration_tags_);
        if (!initialized) {
            client_wrapper_->shutdown();
            RTP_LLM_LOG_ERROR("create BlockTree KVCM clients failed");
            return false;
        }
        control_config_ = std::move(control_config);
        const auto tp_rank = parallelism_config_.tp_rank;
        RTP_LLM_LOG_INFO("BlockTree KVCM storage backend initialized, tp_rank=%ld tp_size=%ld policy={%s}",
                         tp_rank,
                         parallelism_config_.tp_size,
                         group_policy_->debugString().c_str());
        RTP_LLM_LOG_INFO("BlockTree KVCM quarantine guard, tp_rank=%ld abort_ratio=%.6f total_device_blocks=%zu",
                         tp_rank,
                         quarantine_abort_ratio_,
                         total_device_block_count_);
        reportQuarantineMetrics(/*generation=*/0, /*lease_count=*/0, /*block_count=*/0, {});
        return true;
    }

    void reportRemoteCacheSkipped(const char* stage) noexcept {
        try {
            RtpLLMRemoteCacheOutcomeMetricsCollector collector;
            collector.stage = stage;
            collector.outcome = "disabled";
            reportRemoteMetrics<RtpLLMRemoteCacheOutcomeMetrics>(collector);
        } catch (...) {
            RTP_LLM_LOG_WARNING("failed to report disabled remote cache operation");
        }
    }

    StorageMatchResult match(const StorageRequest& request) {
        RTP_LLM_CHECK_WITH_INFO(parallelism_config_.tp_rank == 0,
                                "KVCM metadata match must run on tp rank 0, got %ld",
                                parallelism_config_.tp_rank);
        RTP_LLM_CHECK(request.keys != nullptr && request.keys->size() == request.handles.size());
        // The allocator already caps this sequence at the final reusable full
        // block. Dropping another key here would exclude two tail blocks.
        const CacheKeysType& keys = *request.keys;
        if (request.local_matched_blocks_num >= keys.size()) {
            return {request.local_matched_blocks_num, nullptr};
        }
        RtpLLMRemoteCacheMatchMetricsCollector collector;
        const auto                            start = Clock::now();
        block_tree_cache_detail::ScopeRollback report([&] {
            collector.remote_match_time_us = elapsedTimeUs(start);
            reportRemoteMetrics<RtpLLMRemoteCacheMatchMetrics>(collector);
        });
        const auto trace_id       = nextTraceId("match", match_trace_sequence_);
        auto query_type = static_cast<kv_cache_manager::QueryType>(default_query_type_);
        if (has_swa_) {
            // Fetch the FULL prefix as well as SWA locations, then apply each
            // group's model-defined reuse window locally.
            query_type = kv_cache_manager::QueryType::QT_BATCH_GET;
        }
        (void)remainingTimeout(request, "before match");
        bool success = false;
        kv_cache_manager::Locations locations;
        bool positional = query_type == kv_cache_manager::QueryType::QT_BATCH_GET;
        if (kv_cache_config_.kvcm_read_backend_type != 0) {
            auto result = client_wrapper_->getCacheLocationsByBackend(
                "", trace_id, keys, {}, request.local_matched_blocks_num, {},
                static_cast<kv_cache_manager::StorageType>(kv_cache_config_.kvcm_read_backend_type));
            success = result.first;
            if (success) {
                locations = selectBackendLocations(result.second);
            }
            positional = true;
        } else {
            auto result = client_wrapper_->match(
                "", trace_id, query_type, keys, request.local_matched_blocks_num, {});
            success = result.first;
            locations = std::move(result.second);
        }
        if (!success) {
            return {request.local_matched_blocks_num, nullptr};
        }
        if (positional) {
            auto meta = std::make_shared<KVCMMatchMeta>();
            meta->locations = reusableLocations(request, std::move(locations));
            meta->filtered = true;
            collector.remote_match_fail_qps        = false;
            collector.remote_match_reuse_block_num = static_cast<int64_t>(meta->locations.size());
            if (!meta->locations.empty()) {
                collector.remote_valid_hit_ratio =
                    static_cast<double>(request.local_matched_blocks_num + meta->locations.size()) / keys.size();
            }
            return {request.local_matched_blocks_num + meta->locations.size(), std::move(meta)};
        }
        kvcm::LocationsView locations_view;
        if (!group_policy_->filterNeedLoadLocations(locations, locations_view, /*block_mask=*/0)) {
            throw std::runtime_error("KVCM returned an invalid location shape");
        }
        if (locations_view.size() > keys.size() - request.local_matched_blocks_num) {
            throw std::runtime_error("KVCM prefix match exceeds the requested key range");
        }
        auto meta       = std::make_shared<KVCMMatchMeta>();
        meta->locations = std::move(locations);
        collector.remote_match_fail_qps        = false;
        collector.remote_match_reuse_block_num = static_cast<int64_t>(locations_view.size());
        if (!locations_view.empty()) {
            collector.remote_valid_hit_ratio =
                static_cast<double>(request.local_matched_blocks_num + locations_view.size()) / keys.size();
        }
        return {request.local_matched_blocks_num + locations_view.size(), std::move(meta)};
    }

    void read(const StorageRequest& request, const std::shared_ptr<StorageBackendMatchMeta>& match_meta) {
        RTP_LLM_CHECK_WITH_INFO(parallelism_config_.tp_rank == 0,
                                "KVCM metadata read must run on tp rank 0, got %ld",
                                parallelism_config_.tp_rank);
        RtpLLMRemoteCacheReadMetricsCollector collector;
        bool sdk_failed = false;
        const auto                           start = Clock::now();
        block_tree_cache_detail::ScopeRollback report([&] {
            collector.remote_read_task_cost_time_us = elapsedTimeUs(start);
            reportRemoteMetrics<RtpLLMRemoteCacheReadMetrics>(collector);
            if (collector.remote_read_fail_qps && sdk_failed) reportSdkFailure("read");
        });
        const auto meta = std::dynamic_pointer_cast<KVCMMatchMeta>(match_meta);
        RTP_LLM_CHECK_WITH_INFO(meta != nullptr, "KVCM read received invalid match metadata");
        kvcm::LocationsView locations_view;
        if (meta->filtered) {
            locations_view.resize(meta->locations.size());
            for (size_t i = 0; i < meta->locations.size(); ++i) {
                for (const auto& spec : meta->locations[i]) {
                    locations_view[i].emplace_back(spec);
                }
            }
        } else {
            RTP_LLM_CHECK_WITH_INFO(
                group_policy_->filterNeedLoadLocations(meta->locations, locations_view, /*block_mask=*/0),
                "KVCM read location filtering failed");
        }
        const size_t remote_blocks = request.handles.size() - request.local_matched_blocks_num;
        RTP_LLM_CHECK_WITH_INFO(locations_view.size() == remote_blocks,
                                "KVCM read shape mismatch: locations=%zu remote_blocks=%zu",
                                locations_view.size(),
                                remote_blocks);

        std::vector<FunctionRequestPB> requests(static_cast<size_t>(parallelism_config_.tp_size));
        const auto&                    spec_info = group_policy_->spec_info_map();
        const std::string              trace_id  = nextTraceId("read", read_trace_sequence_);
        initializeRequests(requests, REMOTE_OPERATION_READ, trace_id);
        for (size_t location_idx = 0; location_idx < locations_view.size(); ++location_idx) {
            const size_t key_idx = request.local_matched_blocks_num + location_idx;
            for (const auto& location_spec : locations_view[location_idx]) {
                const auto        info = spec_info.find(location_spec.spec_name);
                const std::string spec_name(location_spec.spec_name);
                RTP_LLM_CHECK_WITH_INFO(info != spec_info.end(), "KVCM read has unknown spec [%s]", spec_name.c_str());
                auto* remote = requests.at(static_cast<size_t>(info->second.tp_rank)).mutable_remote_request();
                remote->add_uris(std::string(location_spec.uri));
                if (merge_full_specs_) remote->add_uri_block_counts(info->second.group_ids.size());
                for (const auto group_id : info->second.group_ids) {
                    const auto& group = group_policy_->groups().at(group_id);
                    const auto* handle = findHandle(request.handles[key_idx], group.tag);
                    RTP_LLM_CHECK_WITH_INFO(handle != nullptr, "KVCM read has no destination for key=%zu tag=%s",
                                            key_idx, group.tag.c_str());
                    remote->add_group_tags(group.tag);
                    remote->add_block_ids(handle->block);
                }
            }
        }
        if (!backend_.remoteCacheEnabled()) {
            reportRemoteCacheSkipped("read");
            collector.remote_read_qps = false;
            collector.remote_read_fail_qps = false;
            throw std::runtime_error("remote cache disabled by KVCM before read submission");
        }
        (void)dispatchRequests(std::move(requests), request, sdk_failed);
        // A canonical CP key already covers every shard. Do not multiply this
        // logical token count by the number of TP ranks or payload groups.
        collector.remote_read_token_num =
            static_cast<int64_t>(remote_blocks * topology_->groups().front().cacheKeyTokenStride());
        collector.remote_read_fail_qps = false;
    }

    void write(const StorageRequest& request) {
        if (!backend_.remoteCacheEnabled()) {
            reportRemoteCacheSkipped("write");
            return;
        }
        RTP_LLM_CHECK_WITH_INFO(parallelism_config_.tp_rank == 0,
                                "KVCM metadata write must run on tp rank 0, got %ld",
                                parallelism_config_.tp_rank);
        RTP_LLM_CHECK(request.keys != nullptr && request.keys->size() == request.handles.size());
        const size_t valid_keys_size = request.keys->size();
        if (valid_keys_size == 0) {
            return;
        }
        RtpLLMRemoteCacheWriteMetricsCollector collector;
        bool sdk_failed = false;
        const auto                            start = Clock::now();
        block_tree_cache_detail::ScopeRollback report([&] {
            collector.remote_write_task_cost_time_us = elapsedTimeUs(start);
            reportRemoteMetrics<RtpLLMRemoteCacheWriteMetrics>(collector);
            if (collector.remote_write_fail_qps && sdk_failed) reportSdkFailure("write");
        });
        CacheKeysType            keys(request.keys->begin(), request.keys->begin() + valid_keys_size);
        std::vector<std::string> location_spec_group_names;
        RTP_LLM_CHECK_WITH_INFO(group_policy_->getNeedWriteGroups(request, valid_keys_size, location_spec_group_names),
                                "KVCM write group selection failed");
        const std::string trace_id     = nextTraceId("write", write_trace_sequence_);
        auto [success, write_location] = [&] {
            const auto stage_start = Clock::now();
            block_tree_cache_detail::ScopeRollback timer([&] {
                collector.remote_get_write_location_time_us = elapsedTimeUs(stage_start);
            });
            return client_wrapper_->getWriteLocation(
                "", trace_id, keys, /*tokens=*/{}, location_spec_group_names, /*write_timeout_seconds=*/40,
                kv_cache_config_.kvcm_min_replica_count);
        }();
        RTP_LLM_CHECK_WITH_INFO(success, "KVCM StartWrite failed");
        static const kv_cache_manager::Locations empty_locations;
        bool                                     finish_attempted = false;
        try {
            std::vector<FunctionRequestPB> requests(static_cast<size_t>(parallelism_config_.tp_size));
            ActualUriGather                actual_uri_gather(requests.size());
            initializeRequests(requests, REMOTE_OPERATION_WRITE, trace_id);
            const auto key_indices = unmaskedKeyIndices(write_location.block_mask, valid_keys_size);
            RTP_LLM_CHECK_WITH_INFO(key_indices.size() == write_location.locations.size(),
                                    "KVCM write mask/location mismatch: keys=%zu locations=%zu",
                                    key_indices.size(),
                                    write_location.locations.size());
            if (write_location.locations.empty()) {
                if (!write_location.write_session_id.empty()) {
                    // The server can create an empty, short-lived session when
                    // every key already has enough replicas. Close it normally.
                    (void)remainingTimeout(request, "before empty FinishWrite");
                    finish_attempted = true;
                    const auto stage_start = Clock::now();
                    block_tree_cache_detail::ScopeRollback timer([&] {
                        collector.remote_finish_write_time_us += elapsedTimeUs(stage_start);
                    });
                    try {
                        if (!client_wrapper_->finishWrite(
                            "", nextTraceId("finish_write", finish_write_trace_sequence_),
                            write_location.write_session_id, kv_cache_manager::BlockMaskOffset{0}, empty_locations)) {
                            RTP_LLM_LOG_WARNING("KVCM failed to close empty write session [%s]",
                                                write_location.write_session_id.c_str());
                        }
                    } catch (...) {
                        RTP_LLM_LOG_WARNING("KVCM failed to close empty write session [%s]",
                                            write_location.write_session_id.c_str());
                    }
                }
                // Keep the legacy task QPS, but report no payload blocks or
                // transfer time when all keys already have enough replicas.
                collector.remote_write_fail_qps = false;
                return;
            }
            const auto& spec_info = group_policy_->spec_info_map();
            for (size_t location_idx = 0; location_idx < write_location.locations.size(); ++location_idx) {
                const size_t key_idx = key_indices[location_idx];
                for (auto& location_spec : write_location.locations[location_idx]) {
                    const auto info = spec_info.find(location_spec.spec_name);
                    RTP_LLM_CHECK_WITH_INFO(
                        info != spec_info.end(), "KVCM write has unknown spec [%s]", location_spec.spec_name.c_str());
                    const size_t rank   = static_cast<size_t>(info->second.tp_rank);
                    auto*        remote = requests.at(rank).mutable_remote_request();
                    remote->add_uris(location_spec.uri);
                    if (merge_full_specs_)
                        remote->add_uri_block_counts(info->second.group_ids.size());
                    for (const auto group_id : info->second.group_ids) {
                        const auto& group = group_policy_->groups().at(group_id);
                        const StorageBlockHandle* handle = findHandle(request.handles[key_idx], group.tag);
                        RTP_LLM_CHECK_WITH_INFO(handle != nullptr,
                                                "KVCM write has no source handle for key=%zu tag=%s",
                                                key_idx,
                                                group.tag.c_str());
                        remote->add_group_tags(group.tag);
                        remote->add_block_ids(handle->block);
                    }
                    actual_uri_gather[rank].push_back(&location_spec);
                }
            }

            const auto responses = [&] {
                const auto                             stage_start = Clock::now();
                block_tree_cache_detail::ScopeRollback timer(
                    [&] { collector.remote_write_broadcast_time_us = elapsedTimeUs(stage_start); });
                if (!backend_.remoteCacheEnabled()) {
                    reportRemoteCacheSkipped("write");
                    throw std::runtime_error("remote cache disabled by KVCM before write submission");
                }
                return dispatchRequests(std::move(requests), request, sdk_failed);
            }();
            bool has_actual_uri = false;
            for (size_t rank = 0; rank < responses.size(); ++rank) {
                const auto& actual_uris = responses[rank].remote_response().actual_uris();
                RTP_LLM_CHECK_WITH_INFO(
                    actual_uris.empty() || static_cast<size_t>(actual_uris.size()) == actual_uri_gather[rank].size(),
                    "KVCM write returned a partial actual URI vector for rank=%zu",
                    rank);
                for (int uri_idx = 0; uri_idx < actual_uris.size(); ++uri_idx) {
                    if (!actual_uris[uri_idx].empty()) {
                        has_actual_uri                                             = true;
                        actual_uri_gather[rank][static_cast<size_t>(uri_idx)]->uri = actual_uris[uri_idx];
                    }
                }
            }
            const auto& actual_locations = has_actual_uri ? write_location.locations : empty_locations;
            finish_attempted             = true;
            {
                const auto                             stage_start = Clock::now();
                block_tree_cache_detail::ScopeRollback timer(
                    [&] { collector.remote_finish_write_time_us += elapsedTimeUs(stage_start); });
                RTP_LLM_CHECK_WITH_INFO(
                    client_wrapper_->finishWrite("",
                                                 nextTraceId("finish_write", finish_write_trace_sequence_),
                                                 write_location.write_session_id,
                                                 write_location.locations.size(),
                                                 actual_locations),
                    "KVCM FinishWrite failed");
            }
            collector.remote_write_cache_block_num = static_cast<int64_t>(write_location.locations.size());
            collector.remote_write_fail_qps        = false;
        } catch (...) {
            if (!finish_attempted) {
                const auto                             stage_start = Clock::now();
                block_tree_cache_detail::ScopeRollback timer(
                    [&] { collector.remote_finish_write_time_us += elapsedTimeUs(stage_start); });
                try {
                    if (!client_wrapper_->abortWrite("",
                                                      nextTraceId("abort_write", finish_write_trace_sequence_),
                                                      write_location.write_session_id)) {
                        RTP_LLM_LOG_WARNING("KVCM failed to abort write session [%s]",
                                            write_location.write_session_id.c_str());
                    }
                } catch (...) {
                    RTP_LLM_LOG_WARNING("KVCM abort write session threw, session=[%s]",
                                        write_location.write_session_id.c_str());
                }
            }
            throw;
        }
    }

    bool prepareBlockBuffers(const RemoteOperationRequestPB& request,
                             const std::vector<std::string>& tags,
                             const std::vector<int32_t>& blocks,
                             kv_cache_manager::BlockBuffers& buffers) {
        if (tags.size() != blocks.size()) {
            return false;
        }
        const bool grouped = !request.uri_block_counts().empty();
        if ((!grouped && blocks.size() != static_cast<size_t>(request.uris_size()))
            || (grouped && request.uri_block_counts_size() != request.uris_size())) {
            return false;
        }
        // Reject the complete request before resolving any physical block or
        // submitting I/O. Counts describe positions, including repeated URIs.
        for (const auto& tag : tags) {
            if (pools_by_tag_.find(tag) == pools_by_tag_.end()) {
                RTP_LLM_LOG_ERROR("KVCM transfer contains unregistered tag [%s]", tag.c_str());
                return false;
            }
        }
        if (grouped) {
            size_t begin = 0;
            const auto& groups = group_policy_->groups();
            const auto& infos = group_policy_->spec_info_map();
            for (int uri = 0; uri < request.uris_size(); ++uri) {
                const size_t count = request.uri_block_counts(uri);
                if (count == 0 || count > blocks.size() - begin) {
                    return false;
                }
                const auto group = std::find_if(groups.begin(), groups.end(), [&](const auto& entry) {
                    return entry.second.tag == tags[begin];
                });
                if (group == groups.end()) {
                    return false;
                }
                const auto name = kvcm::genLocationSpecName(parallelism_config_.tp_rank,
                                                            group->second.group_name,
                                                            runtime_config_.model_name,
                                                            merge_full_specs_);
                const auto info = infos.find(name);
                if (info == infos.end() || info->second.group_ids.size() != count) {
                    return false;
                }
                for (size_t member = 0; member < count; ++member) {
                    if (tags[begin + member] != groups.at(info->second.group_ids[member]).tag) {
                        return false;
                    }
                }
                begin += count;
            }
            if (begin != blocks.size()) {
                return false;
            }
            if (blocks.size() != static_cast<size_t>(request.uris_size())
                && !std::all_of(request.uris().begin(), request.uris().end(), [](const auto& uri) {
                       return uri.compare(0, 7, "pace://") == 0;
                   })) {
                return false;
            }
        }
        if (tags.empty()) {
            return true;
        }
        setCudaDevice();
        if (!group_policy_->genBlockBuffers(tags, blocks, buffers)) {
            return false;
        }
        if (!grouped) {
            return true;
        }
        size_t begin = 0;
        for (int uri = 0; uri < request.uris_size(); ++uri) {
            const size_t count = request.uri_block_counts(uri);
            auto& buffer = buffers[begin];
            if (count > 1) {
                size_t iov_count = 0;
                for (size_t member = begin; member < begin + count; ++member) {
                    iov_count += buffers[member].iovs.size();
                }
                buffer.iovs.reserve(iov_count);
                for (size_t member = begin + 1; member < begin + count; ++member) {
                    auto& iovs = buffers[member].iovs;
                    buffer.iovs.insert(buffer.iovs.end(), std::make_move_iterator(iovs.begin()),
                                       std::make_move_iterator(iovs.end()));
                }
                // SDK gather/scatter groups by size. Use the same stable wire
                // order for normal I/O; only pointer descriptors move here.
                for (size_t i = 1; i < buffer.iovs.size(); ++i) {
                    auto iov = std::move(buffer.iovs[i]);
                    size_t j = i;
                    while (j > 0 && buffer.iovs[j - 1].size > iov.size) {
                        buffer.iovs[j] = std::move(buffer.iovs[j - 1]);
                        --j;
                    }
                    buffer.iovs[j] = std::move(iov);
                }
            }
            if (static_cast<size_t>(uri) != begin) {
                buffers[uri] = std::move(buffer);
            }
            begin += count;
        }
        buffers.resize(request.uris_size());
        return true;
    }

    bool execute(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response,
                 const TransferTask* parent_task = nullptr) {
        if (request.op() != REMOTE_OPERATION_READ && request.op() != REMOTE_OPERATION_WRITE) {
            return executeMetadata(request, response);
        }
        response.set_sdk_failed(false);
        const TransferTask task = parent_task ? parent_task->subtask({}) :
            TransferTask({}, std::chrono::milliseconds(request.timeout_ms() > 0 ? request.timeout_ms() :
                                                       kv_cache_config_.kvcm_task_timeout_ms));
        if (request.timeout_ms() < 0 || task.expired()) {
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_TIMEOUT);
            return false;
        }
        const std::vector<std::string>    tags(request.group_tags().begin(), request.group_tags().end());
        const std::vector<int32_t>        blocks(request.block_ids().begin(), request.block_ids().end());
        const kv_cache_manager::UriStrVec uris(request.uris().begin(), request.uris().end());
        kv_cache_manager::BlockBuffers buffers;
        if (!prepareBlockBuffers(request, tags, blocks, buffers)) {
            RTP_LLM_LOG_WARNING("KVCM transfer has invalid block/URI layout");
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
            return false;
        }
        if (tags.empty()) {
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_SUCCESS);
            return true;
        }
        const auto result = executeTagTransfers(request.op(), tags, blocks, uris, buffers, response, task, &request);
        response.set_transfer_status(result == TransferResult::SUCCESS ? REMOTE_TRANSFER_STATUS_SUCCESS :
                                     result == TransferResult::TIMEOUT ? REMOTE_TRANSFER_STATUS_TIMEOUT :
                                                                         REMOTE_TRANSFER_STATUS_FAILED);
        return result == TransferResult::SUCCESS;
    }

    void shutdown() noexcept {
        if (client_wrapper_) {
            client_wrapper_->shutdown();
        }
    }

    void reportQuarantineEvent(bool is_write, StorageQuarantineReason reason, size_t block_count) const noexcept {
        RtpLLMRemoteCacheQuarantineEventMetricsCollector collector;
        collector.op          = is_write ? "write" : "read";
        collector.block_count = static_cast<int64_t>(block_count);
        switch (reason) {
            case StorageQuarantineReason::SDK_TIMEOUT:
                collector.reason = "sdk_timeout";
                break;
            case StorageQuarantineReason::RPC_TIMEOUT:
                collector.reason = "rpc_timeout";
                break;
            case StorageQuarantineReason::RPC_ERROR:
                collector.reason = "rpc_error";
                break;
            case StorageQuarantineReason::INVALID_RESPONSE:
                collector.reason = "invalid_response";
                break;
        }
        reportRemoteMetrics<RtpLLMRemoteCacheQuarantineMetrics>(collector);
    }

    void reportQuarantineState(uint64_t                          generation,
                               size_t                            task_count,
                               size_t                            block_count,
                               const StorageQuarantineBreakdown& breakdown) noexcept {
        reportQuarantineMetrics(generation, task_count, block_count, breakdown);
        if (task_count > 0 && parallelism_config_.tp_rank == 0 && quarantine_abort_ratio_ > 0.0
            && total_device_block_count_ > 0) {
            // Quarantine metrics count per-task pins; capacity counts each
            // physical block once, using the same FROZEN state as BTC.
            size_t frozen_block_count = 0;
            std::unordered_set<const DeviceBlockPool*> counted_pools;
            for (const auto& [tag, pool] : pools_by_tag_) {
                if (counted_pools.insert(pool.get()).second) {
                    frozen_block_count += pool->referencedBlocksNum(BlockTreeRefType::FROZEN);
                }
            }
            const long double threshold = static_cast<long double>(total_device_block_count_) * quarantine_abort_ratio_;
            if (static_cast<long double>(frozen_block_count) > threshold) {
                RTP_LLM_LOG_ERROR("KVCM quarantined block capacity exceeded on rank 0; aborting process: "
                                  "frozen_blocks=%zu total_device_blocks=%zu abort_ratio=%.6f "
                                  "threshold_blocks=%.3Lf",
                                  frozen_block_count,
                                  total_device_block_count_,
                                  quarantine_abort_ratio_,
                                  threshold);
                std::abort();
            }
        }
    }

private:
    int resolveQueryType(int query_type) const {
        return query_type == 0 ? default_query_type_ : query_type;
    }

    static bool isPayloadBackend(int type) {
        return type == 1 || type == 2 || type == 3 || type == 4 || type == 5;
    }

    kv_cache_manager::Locations selectBackendLocations(const kv_cache_manager::BackendLocations& result) const {
        kv_cache_manager::Locations locations;
        locations.reserve(result.size());
        for (const auto& key_locations : result) {
            RTP_LLM_CHECK_WITH_INFO(key_locations.size() <= 1, "KVCM returned multiple locations for one backend");
            locations.push_back(key_locations.empty() ? kv_cache_manager::Location{} :
                                                        key_locations.front().location_specs);
        }
        return locations;
    }

    kv_cache_manager::Locations reusableLocations(const StorageRequest& request,
                                                   kv_cache_manager::Locations locations) const {
        const size_t local = request.local_matched_blocks_num;
        const size_t key_count = request.keys->size();
        RTP_LLM_CHECK_WITH_INFO(locations.size() == key_count, "KVCM positional query shape mismatch");
        const auto& infos = group_policy_->spec_info_map();
        std::unordered_map<std::string, size_t> runs;
        size_t matched = local;
        for (size_t i = local; i < locations.size(); ++i) {
            std::unordered_map<std::string, const kv_cache_manager::LocationSpecUnit*> present;
            for (const auto& spec : locations[i]) {
                RTP_LLM_CHECK_WITH_INFO(infos.count(spec.spec_name) != 0, "KVCM returned an unknown spec");
                RTP_LLM_CHECK_WITH_INFO(present.emplace(spec.spec_name, &spec).second, "KVCM returned a duplicate spec");
            }
            bool complete = true;
            for (const auto& [id, group] : group_policy_->groups()) {
                (void)id;
                bool available = true;
                for (int rank = 0; rank < parallelism_config_.tp_size; ++rank) {
                    const auto name = kvcm::genLocationSpecName(
                        rank, group.group_name, runtime_config_.model_name, merge_full_specs_);
                    const auto found = present.find(name);
                    available = available && found != present.end() && !found->second->uri.empty();
                }
                auto& run = runs[group.tag];
                run = available ? run + 1 : 0;
                const size_t required = std::min(i + 1 - local, topology_->group(group.tag).reuseBlockCount(i + 1));
                complete = complete && run >= required;
            }
            if (complete) {
                matched = i + 1;
            }
        }
        locations.resize(matched);
        for (size_t i = local; i < matched; ++i) {
            auto& location = locations[i];
            location.erase(std::remove_if(location.begin(), location.end(), [&](const auto& spec) {
                const auto& group = topology_->group(infos.at(spec.spec_name).tag);
                return matched - i > group.reuseBlockCount(matched) || spec.uri.empty();
            }), location.end());
        }
        locations.erase(locations.begin(), locations.begin() + local);
        return locations;
    }

    bool executeMetadata(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response) {
        if (parallelism_config_.tp_rank != 0) {
            RTP_LLM_LOG_WARNING("KVCM metadata operations require the TP rank 0 endpoint, got rank=%ld",
                                parallelism_config_.tp_rank);
            return false;
        }
        if (!request.has_metadata()) {
            return false;
        }
        const auto& query = request.metadata();
        const int type = resolveQueryType(query.query_type());
        if (type < 1 || type > 2 || query.p2p_host_count() < 0) {
            return false;
        }
        const std::vector<int64_t> keys(query.block_keys().begin(), query.block_keys().end());
        const std::vector<int64_t> tokens(query.token_ids().begin(), query.token_ids().end());
        const std::vector<std::string> names(query.location_spec_names().begin(), query.location_spec_names().end());
        kv_cache_manager::BlockMask mask = kv_cache_manager::BlockMaskOffset{0};
        if (query.block_mask().has_bool_masks()) {
            mask = kv_cache_manager::BlockMaskVector(query.block_mask().bool_masks().values().begin(),
                                                      query.block_mask().bool_masks().values().end());
        } else if (query.block_mask().info_case() == RemoteBlockMaskPB::kOffset) {
            if (query.block_mask().offset() < 0) {
                return false;
            }
            mask = static_cast<size_t>(query.block_mask().offset());
        }
        const auto query_type = static_cast<kv_cache_manager::QueryType>(type);
        auto appendLocation = [](const kv_cache_manager::Location& location, RemoteCacheLocationPB* output) {
            for (const auto& spec : location) {
                auto* item = output->add_specs();
                item->set_name(spec.spec_name);
                item->set_uri(spec.uri);
            }
        };
        switch (request.op()) {
            case REMOTE_OPERATION_REMOVE_CACHE:
                return client_wrapper_->removeCache("", request.trace_id(), keys, tokens, mask);
            case REMOTE_OPERATION_GET_LOCATIONS_BY_BACKEND: {
                if (query.query_type() != 0 && query.query_type() != 1) {
                    return false;
                }
                const int backend = query.backend_type() == 0 ? kv_cache_config_.kvcm_read_backend_type :
                                                                query.backend_type();
                if (!isPayloadBackend(backend)) {
                    return false;
                }
                auto [success, locations] = client_wrapper_->getCacheLocationsByBackend(
                    "", request.trace_id(), keys, tokens, mask, names, static_cast<kv_cache_manager::StorageType>(backend));
                if (success) {
                    for (const auto& key_locations : locations) {
                        auto* output = response.add_backend_locations();
                        for (const auto& location : key_locations) {
                            auto* item = output->add_locations();
                            item->set_backend_type(static_cast<int32_t>(location.type));
                            item->set_spec_size(location.spec_size);
                            appendLocation(location.location_specs, item);
                        }
                    }
                }
                return success;
            }
            case REMOTE_OPERATION_GET_HOST_CACHE_STATE: {
                if (type != 2) {
                    RTP_LLM_LOG_WARNING("KVCM host-state queries require prefix mode, got %d", type);
                    return false;
                }
                const std::vector<std::string> medium(query.medium().begin(), query.medium().end());
                auto [success, hosts] = client_wrapper_->getHostCacheState(
                    "", request.trace_id(), query_type, keys, medium, query.p2p_host_count());
                if (success) {
                    for (const auto& host : hosts) {
                        appendHostCacheMatch(host, *response.add_hosts());
                    }
                }
                return success;
            }
            default:
                return false;
        }
    }

    std::pair<std::shared_ptr<kvcm::KVCMConfig::LocationSpecInfoMap>,
              std::shared_ptr<kvcm::KVCMConfig::LocationSpecGroups>>
    genLocationSpecs() {
        auto infos  = std::make_shared<kvcm::KVCMConfig::LocationSpecInfoMap>();
        auto groups = std::make_shared<kvcm::KVCMConfig::LocationSpecGroups>();
        RTP_LLM_CHECK_WITH_INFO(
            group_policy_->buildLocationSpecGroups(static_cast<int>(parallelism_config_.tp_size),
                                                    *groups,
                                                    runtime_config_.model_name, merge_full_specs_),
            "failed to build KVCM location spec groups");
        for (const auto& [group_id, group] : group_policy_->groups()) {
            for (int rank = 0; rank < parallelism_config_.tp_size; ++rank) {
                const std::string spec_name =
                    kvcm::genLocationSpecName(rank, group.group_name, runtime_config_.model_name, merge_full_specs_);
                auto& size = (*infos)[spec_name];
                RTP_LLM_CHECK_WITH_INFO(group.block_size_bytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max() - size),
                                        "KVCM location spec size overflow [%s]", spec_name.c_str());
                size += group.block_size_bytes;
            }
        }
        return {std::move(infos), std::move(groups)};
    }

    kvcm::ClientWrapper::ConfigMap genClientConfig(const std::vector<std::string>& full_group_tags) {
        std::vector<std::string> addresses;
        if (!kv_cache_config_.kvcm_server_address.empty()) {
            addresses.push_back(kv_cache_config_.kvcm_server_address);
        }
        auto channel = std::make_shared<kvcm::MetaChannelConfig>(kv_cache_config_.kvcm_meta_channel_retry_time,
                                                                 kv_cache_config_.kvcm_meta_channel_connection_timeout,
                                                                 kv_cache_config_.kvcm_meta_channel_call_timeout);
        auto sdk     = std::make_shared<kvcm::SdkWrapperConfig>(kv_cache_config_.kvcm_storage_thread_num,
                                                            kv_cache_config_.kvcm_storage_queue_size,
                                                            kv_cache_config_.kvcm_put_timeout_ms,
                                                            kv_cache_config_.kvcm_get_timeout_ms);
        sdk->parseBackendConfigs(kv_cache_config_.kvcm_model_sdk_config);
        const auto& backends = sdk->sdk_backend_configs();
        if (runtime_config_.model_name == "DeepSeekV41" && full_group_tags.size() == 3 && backends.size() == 1
            && std::dynamic_pointer_cast<kvcm::TairMempoolSdkConfig>(backends.front())) {
            const auto& first = topology_->group(full_group_tags.front());
            std::set<std::string> names;
            bool compatible = true;
            for (const auto& [id, group] : group_policy_->groups()) {
                (void)id;
                if (!group.is_full)
                    continue;
                names.insert(kvcm::genLocationSpecName(0, group.group_name, runtime_config_.model_name));
                const auto& member = topology_->group(group.tag);
                compatible = compatible && member.seqSizePerBlock() == first.seqSizePerBlock()
                             && member.cacheKeyTokenStride() == first.cacheKeyTokenStride()
                             && member.policy.cp_mapping == first.policy.cp_mapping
                             && member.policy.cp_slice == first.policy.cp_slice;
            }
            merge_full_specs_ = compatible && names == std::set<std::string>{"F0_TP0", "F1_TP0", "F2_TP0"};
        }
        auto [location_infos, location_groups] = genLocationSpecs();

        const std::string model_name = runtime_config_.model_name;
        const std::string dtype      = getDataTypeStr(cache_config_.dtype);
        std::string       extra      = kv_cache_config_.kvcm_model_extra_info;
        extra += '/' + autil::EnvUtil::getEnv("BIZ_NAME", std::string("")) + '/'
                 + std::to_string(hashString(autil::EnvUtil::getEnv("CHECKPOINT_PATH", std::string(""))));
        std::string draft_info;
        if (!cache_config_.mtp_sub_configs.empty()) {
            draft_info = '{' + sp_config_.to_string() + '}';
        }
        std::stringstream identity;
        identity << "instance_group: " << kv_cache_config_.kvcm_instance_group
                 << ";block_size:" << cache_config_.seq_size_per_block << ";model_name:" << model_name
                 << ";dtype_str:" << dtype << ";use_mla:" << cache_config_.use_mla
                 << ";fp8_kv_cache:" << kv_cache_config_.fp8_kv_cache << ";tp_size:" << parallelism_config_.tp_size
                 << ";dp_size:" << parallelism_config_.dp_size << ";extra_info:" << extra
                 << ";location_spec_info:" << autil::legacy::ToJsonString(location_infos, true)
                 << ";location_spec_groups:" << autil::legacy::ToJsonString(location_groups, true)
                 << ";default_query_type:" << default_query_type_
                 << ";draft_model_info:" << draft_info;
        std::string instance_id = kv_cache_config_.kvcm_instance_id_salt;
        if (!instance_id.empty()) {
            instance_id += '_';
        }
        instance_id += std::to_string(hashString(identity.str()));

        auto config =
            std::make_shared<kvcm::KVCMConfig>(kv_cache_config_.kvcm_enable_vipserver,
                                               kv_cache_config_.kvcm_vipserver_domain,
                                               static_cast<int32_t>(cache_config_.seq_size_per_block),
                                               kv_cache_config_.kvcm_instance_group,
                                               instance_id,
                                               addresses,
                                               location_infos,
                                               channel,
                                               sdk,
                                               location_groups,
                                               kvcm::ModelDeployment(model_name,
                                                                     dtype,
                                                                     cache_config_.use_mla,
                                                                     static_cast<int32_t>(parallelism_config_.tp_size),
                                                                     static_cast<int32_t>(parallelism_config_.dp_size),
                                                                     1,
                                                                     extra,
                                                                     kv_cache_config_.kvcm_model_user_data));
        return {{"", std::move(config)}};
    }

    std::vector<kvcm::ClientWrapper::PoolRegistration>
    makePoolRegistrations(const std::function<const DeviceBlockPoolPtr&(const std::string&)>& pool_resolver) {
        const auto&          groups = group_policy_->groups();
        std::vector<int32_t> ordered_groups;
        for (const auto& [id, group] : groups) {
            ordered_groups.push_back(id);
        }
        // Preserve the existing primary registration identity independently of map order.
        std::sort(ordered_groups.begin(), ordered_groups.end(), [&](int32_t left, int32_t right) {
            const auto& lhs = groups.at(left);
            const auto& rhs = groups.at(right);
            return std::make_pair(!lhs.is_full, lhs.group_name) < std::make_pair(!rhs.is_full, rhs.group_name);
        });
        registration_tags_.clear();
        pools_by_tag_.clear();
        total_device_block_count_ = 0;
        std::vector<kvcm::ClientWrapper::PoolRegistration> registrations;
        std::vector<std::string>                             registration_tags;
        std::unordered_map<std::string, DeviceBlockPoolPtr> pools_by_tag;
        std::unordered_set<const DeviceBlockPool*>           unique_pools;
        size_t                                                total_device_block_count = 0;
        registrations.reserve(ordered_groups.size());
        registration_tags.reserve(ordered_groups.size());
        pools_by_tag.reserve(ordered_groups.size());
        for (int32_t group_id : ordered_groups) {
            const auto& tag  = groups.at(group_id).tag;
            const auto& pool = pool_resolver(tag);
            if (!pool->getBaseAddress() || pool->getTotalSizeBytes() == 0) {
                RTP_LLM_LOG_ERROR("KVCM group %d has no valid registration span", group_id);
                return {};
            }
            if (unique_pools.emplace(pool.get()).second) {
                const size_t block_count = pool->totalBlocksNum();
                if (block_count > std::numeric_limits<size_t>::max() - total_device_block_count) {
                    RTP_LLM_LOG_ERROR("KVCM total device block capacity overflow");
                    return {};
                }
                total_device_block_count += block_count;
            }
            bool use_gdr = false;
#if USING_CUDA
            use_gdr = gdr_enabled_ && pool->where() == MemoryType::MEMORY_GPU && pool->usesDedicatedDeviceAllocation()
                      && pool->getAllocationSizeBytes() > 0 && pool->deviceIndex() >= 0;
#endif
            const size_t registration_size = use_gdr ? pool->getAllocationSizeBytes() : pool->getTotalSizeBytes();
            kvcm::ClientWrapper::PoolRegistration registration{
                {pool->getBaseAddress(), registration_size},
                kvcm::genLocationSpecName(static_cast<int>(parallelism_config_.tp_rank),
                                          groups.at(group_id).group_name,
                                          runtime_config_.model_name,
                                          merge_full_specs_)};
            if (use_gdr) {
#if USING_CUDA
                const long page_size = sysconf(_SC_PAGESIZE);
                if (page_size <= 0
                    || reinterpret_cast<uintptr_t>(pool->getBaseAddress()) % static_cast<size_t>(page_size) != 0
                    || pool->getAllocationSizeBytes() % static_cast<size_t>(page_size) != 0) {
                    RTP_LLM_LOG_ERROR("RemoteCache GDR registration span must be page aligned for group %d: "
                                      "base=%p size=%zu page_size=%ld",
                                      group_id,
                                      pool->getBaseAddress(),
                                      pool->getAllocationSizeBytes(),
                                      page_size);
                    return {};
                }
                kv_cache_manager::ClientMemoryRegistrations memory_registrations;
                memory_registrations.gpu.push_back(
                    {pool->getBaseAddress(), pool->getAllocationSizeBytes(), pool->deviceIndex()});
                registration.memory_registrations = std::move(memory_registrations);
#endif
            }
            if (gdr_enabled_ && !use_gdr) {
                RTP_LLM_LOG_WARNING("RemoteCache GDR unavailable for group %d tag=%s; falling back to standard "
                                    "registration, memory_type=%d dedicated=%d allocation_size=%zu device=%d",
                                    group_id,
                                    tag.c_str(),
                                    static_cast<int>(pool->where()),
                                    pool->usesDedicatedDeviceAllocation(),
                                    pool->getAllocationSizeBytes(),
                                    pool->deviceIndex());
            }
            RTP_LLM_LOG_INFO("KVCM pool registration group=%d tag=%s mode=%s span_size=%zu",
                             group_id,
                             tag.c_str(),
                             use_gdr ? "gdr" : "standard",
                             registration_size);
            pools_by_tag.emplace(tag, pool);
            registration_tags.push_back(tag);
            registrations.push_back(std::move(registration));
        }
        pools_by_tag_      = std::move(pools_by_tag);
        registration_tags_ = std::move(registration_tags);
        total_device_block_count_ = total_device_block_count;
        return registrations;
    }

    static int remainingTimeout(const StorageRequest& request, const char* phase) {
        RTP_LLM_CHECK(request.transfer_task.has_value());
        const auto remaining = request.transfer_task->remainingTimeout();
        if (!remaining) {
            RTP_LLM_LOG_WARNING("KVCM task deadline exceeded: %s", phase);
            throw StorageDeadlineExceeded(std::string("KVCM task deadline exceeded: ") + phase);
        }
        return static_cast<int>(remaining->count());
    }

    TransferResult executeClientTransfer(RemoteOpType                       operation,
                                         const std::string&                 client_tag,
                                         const std::vector<int32_t>&        blocks,
                                         const kv_cache_manager::UriStrVec& uris,
                                         kv_cache_manager::BlockBuffers&    buffers,
                                         kv_cache_manager::UriStrVec&       actual_uris,
                                         RemoteOperationResponsePB&         response,
                                         const TransferTask& task, const RemoteOperationRequestPB* request = nullptr) {
        const auto                           trace_info = makeTransferTraceInfo(blocks, request);
        auto completion = kvcm::ClientWrapper::TransferCompletion::NOT_SUBMITTED;
        RtpLLMRemoteCacheSDKMetricsCollector collector;
        collector.remote_sdk_block_num               = static_cast<int64_t>(uris.size());
        const auto                             start = Clock::now();
        block_tree_cache_detail::ScopeRollback report([&] {
            collector.remote_sdk_cost_time_us = elapsedTimeUs(start);
            if (completion != kvcm::ClientWrapper::TransferCompletion::NOT_SUBMITTED && collector.remote_sdk_fail_qps) {
                response.set_sdk_failed(true);
            }
            collector.remote_sdk_fail_qps = false;
            reportRemoteMetrics<RtpLLMRemoteCacheSDKMetrics>(collector,
                                                             operation == REMOTE_OPERATION_READ ? "read" : "write");
        });
        if (task.expired()) return TransferResult::TIMEOUT;
        if (operation == REMOTE_OPERATION_READ) {
            const auto ec = client_wrapper_->loadKvCachesForTagWithStatus(client_tag, uris, buffers, trace_info, &completion, task.deadline());
            if (ec == kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT) {
                return TransferResult::TIMEOUT;
            }
            if (ec != kv_cache_manager::ClientErrorCode::ER_OK) {
                return TransferResult::FAILED;
            }
        } else {
            auto [ec, result] = client_wrapper_->saveKvCachesForTagWithStatus(client_tag, uris, buffers, trace_info, &completion, task.deadline());
            if (ec == kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT) {
                return TransferResult::TIMEOUT;
            }
            if (ec != kv_cache_manager::ClientErrorCode::ER_OK || (!result.empty() && result.size() != uris.size())) {
                return TransferResult::FAILED;
            }
            actual_uris = std::move(result);
        }
        collector.remote_sdk_fail_qps = false;
        return task.expired() ? TransferResult::TIMEOUT : TransferResult::SUCCESS;
    }

    TransferResult executeTagTransfers(RemoteOpType                       operation,
                                       const std::vector<std::string>&    tags,
                                       const std::vector<int32_t>&        blocks,
                                       const kv_cache_manager::UriStrVec& uris,
                                       kv_cache_manager::BlockBuffers&    buffers,
                                       RemoteOperationResponsePB&         response,
                                       const TransferTask& task, const RemoteOperationRequestPB* request = nullptr) {
        if (operation != REMOTE_OPERATION_READ && operation != REMOTE_OPERATION_WRITE) {
            RTP_LLM_LOG_WARNING("KVCM transfer has invalid operation [%d]", operation);
            return TransferResult::FAILED;
        }
        kv_cache_manager::UriStrVec actual_uris;
        if (std::all_of(uris.begin(), uris.end(), [](const auto& uri) { return uri.compare(0, 7, "pace://") == 0; })) {
            // The first client owns all PACE GPU registrations. Forward the whole
            // request unchanged; the SDK selects normal or gather/scatter mode.
            const auto result =
                executeClientTransfer(operation, registration_tags_.front(), blocks, uris, buffers, actual_uris, response, task, request);
            if (result != TransferResult::SUCCESS) {
                return result;
            }
        } else {
            // Mooncake binds each client to its pool's span. Keep the original
            // tag routing for non-PACE requests, including mixed-backend tags.
            std::unordered_map<std::string, std::vector<size_t>> indices_by_tag;
            for (size_t index = 0; index < tags.size(); ++index) {
                indices_by_tag[tags[index]].push_back(index);
            }
            // Publish replacements only after every tag succeeds.
            actual_uris = uris;
            for (const auto& tag : registration_tags_) {
                const auto found = indices_by_tag.find(tag);
                if (found == indices_by_tag.end()) {
                    continue;
                }
                const auto&                    indices = found->second;
                kv_cache_manager::UriStrVec    tag_uris, tag_actual_uris;
                kv_cache_manager::BlockBuffers tag_buffers;
                std::vector<int32_t>           tag_blocks;
                for (size_t index : indices) {
                    tag_uris.push_back(uris[index]);
                    tag_buffers.push_back(std::move(buffers[index]));
                    tag_blocks.push_back(blocks[index]);
                }
                const auto result =
                    executeClientTransfer(operation, tag, tag_blocks, tag_uris, tag_buffers, tag_actual_uris, response, task, request);
                if (result != TransferResult::SUCCESS) {
                    return result;
                }
                for (size_t index = 0; index < tag_actual_uris.size(); ++index) {
                    actual_uris[indices[index]] = std::move(tag_actual_uris[index]);
                }
                indices_by_tag.erase(found);
            }
            RTP_LLM_CHECK(indices_by_tag.empty());
        }
        if (operation == REMOTE_OPERATION_WRITE && !actual_uris.empty() && actual_uris != uris) {
            for (auto& uri : actual_uris) {
                *response.add_actual_uris() = std::move(uri);
            }
        }
        return TransferResult::SUCCESS;
    }

    void reportSdkFailure(std::string_view mode) const noexcept {
        RtpLLMRemoteCacheSDKMetricsCollector collector;
        collector.report_details = false;
        reportRemoteMetrics<RtpLLMRemoteCacheSDKMetrics>(collector, mode);
    }

    template<typename Metrics, typename Collector>
    void reportRemoteMetrics(Collector& collector, std::string_view mode = {}) const noexcept {
        if (!metrics_reporter_) {
            return;
        }
        try {
            // Only the SDK metrics have this dimension. Cache the two tag sets
            // instead of allocating tags for every transfer batch.
            const kmonitor::MetricsTags* tags = nullptr;
            if (!mode.empty()) {
                static const kmonitor::MetricsTags read_tags("mode", "read");
                static const kmonitor::MetricsTags write_tags("mode", "write");
                tags = mode == "read" ? &read_tags : &write_tags;
            }
            metrics_reporter_->report<Metrics, Collector>(tags, &collector);
        } catch (...) {
            // Telemetry must not change the transfer outcome or mask an I/O
            // exception while a scope guard is unwinding.
            RTP_LLM_LOG_ERROR("failed to report KVCM remote cache metrics");
        }
    }

    void reportQuarantineMetrics(uint64_t                          generation,
                                 size_t                            lease_count,
                                 size_t                            block_count,
                                 const StorageQuarantineBreakdown& breakdown) const noexcept {
        if (!metrics_reporter_) {
            return;
        }
        // Reporting may call into external metric backends. Serialize it
        // independently from quarantine state updates, and discard snapshots
        // superseded by a later state transition.
        std::lock_guard<std::mutex> lock(quarantine_metrics_mutex_);
        if (quarantine_metrics_reported_ && generation <= last_quarantine_metrics_generation_) {
            return;
        }
        try {
            const auto report = [this](const StorageQuarantineCounts& counts, const kmonitor::MetricsTags* tags) {
                RtpLLMRemoteCacheQuarantineMetricsCollector collector;
                collector.quarantined_lease_count = static_cast<int64_t>(counts.task_count);
                collector.quarantined_block_count = static_cast<int64_t>(counts.block_count);
                if (parallelism_config_.tp_rank == 0 && total_device_block_count_ > 0) {
                    collector.quarantined_block_ratio = static_cast<double>(counts.block_count)
                                                        / static_cast<double>(total_device_block_count_);
                }
                metrics_reporter_->report<RtpLLMRemoteCacheQuarantineMetrics,
                                          RtpLLMRemoteCacheQuarantineMetricsCollector>(tags, &collector);
            };
            const auto report_bucket = [&report](const char* op,
                                                  const char* reason,
                                                  const StorageQuarantineCounts& counts) {
                kmonitor::MetricsTags tags("op", op);
                tags.AddTag("reason", reason);
                report(counts, &tags);
            };
            // Keep the original untagged totals for existing dashboards.
            report({lease_count, block_count}, nullptr);
            report_bucket("total", "total", {lease_count, block_count});
            std::array<StorageQuarantineCounts, 4> reason_totals{};
            for (size_t op = 0; op < breakdown.size(); ++op) {
                const char* operation = op == 0 ? "read" : "write";
                StorageQuarantineCounts operation_total;
                for (size_t reason = 0; reason < breakdown[op].size(); ++reason) {
                    const auto& counts = breakdown[op][reason];
                    operation_total.task_count += counts.task_count;
                    operation_total.block_count += counts.block_count;
                    reason_totals[reason].task_count += counts.task_count;
                    reason_totals[reason].block_count += counts.block_count;
                    report_bucket(operation, kQuarantineReasons[reason], counts);
                }
                report_bucket(operation, "total", operation_total);
            }
            for (size_t reason = 0; reason < reason_totals.size(); ++reason) {
                report_bucket("total", kQuarantineReasons[reason], reason_totals[reason]);
            }
            quarantine_metrics_reported_        = true;
            last_quarantine_metrics_generation_ = generation;
        } catch (...) {
            // Keep the same generation eligible for a later retry.
            quarantine_metrics_reported_ = false;
            RTP_LLM_LOG_ERROR("failed to report KVCM quarantine metrics");
        }
    }

    void initializeRequests(std::vector<FunctionRequestPB>& requests,
                            RemoteOpType                    operation,
                            const std::string&              trace_id) const {
        for (auto& request : requests) {
            request.mutable_remote_request()->set_op(operation);
            request.mutable_remote_request()->set_trace_id(trace_id);
        }
    }

    std::vector<size_t> unmaskedKeyIndices(const kv_cache_manager::BlockMask& mask, size_t key_count) const {
        std::vector<size_t> result;
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, kv_cache_manager::BlockMaskOffset>) {
                    RTP_LLM_CHECK_WITH_INFO(value <= key_count, "KVCM write offset exceeds key count");
                    for (size_t key = static_cast<size_t>(value); key < key_count; ++key) {
                        result.push_back(key);
                    }
                } else {
                    RTP_LLM_CHECK_WITH_INFO(value.size() == key_count, "KVCM write mask size mismatch");
                    for (size_t key = 0; key < value.size(); ++key) {
                        if (!value[key]) {
                            result.push_back(key);
                        }
                    }
                }
            },
            mask);
        return result;
    }

    std::vector<FunctionResponsePB> dispatchRequests(std::vector<FunctionRequestPB> requests,
                                                     const StorageRequest& storage_request, bool& sdk_failed) {
        const int timeout_ms = remainingTimeout(storage_request, "before payload dispatch");
        const auto request_id = storage_request.request_id;
        for (auto& request : requests) request.mutable_remote_request()->set_timeout_ms(timeout_ms);
        const auto&                     remote    = requests.front().remote_request();
        const std::string               direction = remote.op() == REMOTE_OPERATION_READ ? "REMOTE2D" : "D2REMOTE";
        std::vector<FunctionResponsePB> responses;
        if (!broadcast_manager_) {
            RTP_LLM_CHECK_WITH_INFO(
                requests.size() == 1, "KVCM local transfer requires exactly one request, got %zu", requests.size());
            responses.emplace_back();
            auto& local_response = *responses.back().mutable_remote_response();
            block_tree_cache_detail::ScopeRollback collect_sdk_failure([&] {
                sdk_failed = sdk_failed || (local_response.has_sdk_failed() && local_response.sdk_failed());
            });
            execute(requests.front().remote_request(), local_response, &*storage_request.transfer_task);
        } else {
            auto rpc_call = [&storage_request](const std::shared_ptr<RpcService::Stub>&    stub,
                               const std::shared_ptr<grpc::ClientContext>& context,
                               const FunctionRequestPB&                    request,
                               grpc::CompletionQueue*                      queue) {
                const auto remaining = storage_request.transfer_task->remainingTimeout();
                if (!remaining) throw StorageDeadlineExceeded("KVCM task expired before rank submission");
                context->set_deadline(std::chrono::system_clock::now() + *remaining);
                return stub->AsyncExecuteFunction(context.get(), request, queue);
            };
            auto result =
                broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(requests, timeout_ms, rpc_call,
                                                                                    request_id, direction, /*fatal_on_error=*/false);
            // BroadcastManager rejects before dispatching any rank. No worker
            // can access these buffers when admission fails.
            RTP_LLM_CHECK_WITH_INFO(result != nullptr, "KVCM broadcast dispatch failed");
            try {
                (void)result->waitDone(/*timeout_ms=*/0, /*fatal_on_error=*/false);
            } catch (const std::exception& exception) {
                if (result->deadlineExceeded()) {
                    throw StorageOperationTimeout(std::string("KVCM broadcast timed out: ") + exception.what(),
                                                  StorageQuarantineReason::RPC_TIMEOUT);
                }
                throw StorageOperationCompletionUnknown(std::string("KVCM broadcast completion is unknown: ")
                                                            + exception.what(),
                                                        StorageQuarantineReason::RPC_ERROR);
            } catch (...) {
                if (result->deadlineExceeded()) {
                    throw StorageOperationTimeout("KVCM broadcast timed out", StorageQuarantineReason::RPC_TIMEOUT);
                }
                throw StorageOperationCompletionUnknown("KVCM broadcast completion is unknown",
                                                        StorageQuarantineReason::RPC_ERROR);
            }
            responses = result->responses();
            for (size_t rank = 0; rank < responses.size(); ++rank) {
                const auto& response = responses[rank];
                if ((result->success() || result->rpcSucceeded(rank)) && response.has_remote_response()) {
                    const auto& payload = response.remote_response();
                    sdk_failed = sdk_failed || (payload.has_sdk_failed() && payload.sdk_failed());
                }
            }
            if (!result->success()) {
                if (result->deadlineExceeded()) {
                    throw StorageOperationTimeout("KVCM broadcast timed out", StorageQuarantineReason::RPC_TIMEOUT);
                }
                throw StorageOperationCompletionUnknown("KVCM broadcast completion is unknown",
                                                        StorageQuarantineReason::RPC_ERROR);
            }
        }

        if (responses.size() != requests.size()) {
            throw StorageOperationCompletionUnknown("KVCM worker response count mismatch",
                                                    StorageQuarantineReason::INVALID_RESPONSE);
        }
        bool failed = false;
        for (const auto& response : responses) {
            if (!response.has_remote_response()) {
                throw StorageOperationCompletionUnknown("KVCM worker returned no transfer response",
                                                        StorageQuarantineReason::INVALID_RESPONSE);
            }
            switch (response.remote_response().transfer_status()) {
                case REMOTE_TRANSFER_STATUS_UNSPECIFIED:  // successful legacy worker
                case REMOTE_TRANSFER_STATUS_SUCCESS:
                    break;
                case REMOTE_TRANSFER_STATUS_FAILED:
                    failed = true;
                    break;
                case REMOTE_TRANSFER_STATUS_TIMEOUT:
                    throw StorageOperationTimeout("KVCM transfer timed out", StorageQuarantineReason::SDK_TIMEOUT);
                default:
                    throw StorageOperationCompletionUnknown("KVCM transfer completion is unknown",
                                                            StorageQuarantineReason::INVALID_RESPONSE);
            }
        }
        if (failed) {
            throw std::runtime_error("KVCM transfer failed");
        }
        return responses;
    }

    void setCudaDevice() const {
        const int expected = static_cast<int>(parallelism_config_.local_rank);
        int       current  = -1;
        check_cuda_value(cudaGetDevice(&current));
        if (current != expected) {
            check_cuda_value(cudaSetDevice(expected));
        }
    }

    std::shared_ptr<kv_cache_manager::TransferTraceInfo>
    makeTransferTraceInfo(const std::vector<int32_t>& block_ids,
                           const RemoteOperationRequestPB* request = nullptr) const {
        if (!sdk_check_enabled_) {
            return nullptr;
        }
        auto trace_info        = std::make_shared<kv_cache_manager::TransferTraceInfo>();
        trace_info->need_print = true;
        if (request && !request->uri_block_counts().empty()) {
            trace_info->block_ids.reserve(request->uris_size());
            size_t begin = 0;
            for (const auto count : request->uri_block_counts()) {
                std::string ids;
                for (size_t member = 0; member < count; ++member) {
                    if (member)
                        ids += ',';
                    ids += std::to_string(block_ids[begin++]);
                }
                trace_info->block_ids.push_back(std::move(ids));
            }
        } else {
            trace_info->block_ids.reserve(block_ids.size());
            for (const auto block_id : block_ids) {
                trace_info->block_ids.push_back(std::to_string(block_id));
            }
        }
        return trace_info;
    }

private:
    const StorageBackend& backend_;
    CacheConfig                          cache_config_;
    KVCacheConfig                        kv_cache_config_;
    RuntimeConfig                        runtime_config_;
    ParallelismConfig                    parallelism_config_;
    SpeculativeExecutionConfig           sp_config_;
    std::shared_ptr<BroadcastManager>    broadcast_manager_;
    std::shared_ptr<kmonitor::MetricsReporter>          metrics_reporter_;
    std::unique_ptr<kvcm::GroupPolicy>   group_policy_;
    std::shared_ptr<kvcm::ClientWrapper> client_wrapper_;
    std::optional<ControlConfig> control_config_;
    const bool gdr_enabled_;
    std::vector<std::string>             registration_tags_;
    std::unordered_map<std::string, DeviceBlockPoolPtr> pools_by_tag_;
    size_t                                              total_device_block_count_{0};
    double                                              quarantine_abort_ratio_{0.0};
    mutable std::mutex                                  quarantine_metrics_mutex_;
    mutable bool                                        quarantine_metrics_reported_{false};
    mutable uint64_t                                    last_quarantine_metrics_generation_{0};
    // Preserve KVCM's operation-local, one-based request
    // order. Abort and finish share a sequence because both call FinishWrite.
    std::atomic<uint64_t> match_trace_sequence_{1};
    std::atomic<uint64_t> read_trace_sequence_{1};
    std::atomic<uint64_t> write_trace_sequence_{1};
    std::atomic<uint64_t> finish_write_trace_sequence_{1};
    const bool            sdk_check_enabled_;
    const CacheTopology*  topology_ = nullptr;
    bool                  has_swa_ = false;
    bool                  merge_full_specs_ = false;
    const int32_t         default_query_type_ = static_cast<int32_t>(kv_cache_manager::QueryType::QT_PREFIX_MATCH);
};

KVCMStorageBackend::KVCMStorageBackend(const CacheConfig&                         cache_config,
                                       const KVCacheConfig&                       kv_cache_config,
                                       const RuntimeConfig&                       runtime_config,
                                       const ParallelismConfig&                   parallelism_config,
                                       const SpeculativeExecutionConfig&          sp_config,
                                       std::shared_ptr<BroadcastManager>          broadcast_manager,
                                       bool                                       gdr_enabled,
                                       std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter,
                                       std::shared_ptr<kvcm::ClientWrapper>       client_wrapper):
    StorageBackend(makeStorageBackendExecutor(kv_cache_config.kvcm_asyncwrapper_thread_num,
                                              kv_cache_config.kvcm_asyncwrapper_queue_size),
                   std::make_shared<BlockTreeCacheMetricsReporter>(metrics_reporter),
                   parallelism_config.world_rank,
                   std::chrono::milliseconds(kv_cache_config.kvcm_task_timeout_ms)),
    impl_(std::make_unique<Impl>(*this,
                                 cache_config,
                                 kv_cache_config,
                                 runtime_config,
                                 parallelism_config,
                                 sp_config,
                                 std::move(broadcast_manager),
                                 std::move(client_wrapper),
                                 std::move(metrics_reporter), gdr_enabled)) {}

KVCMStorageBackend::KVCMStorageBackend(const CacheConfig& cache_config,
                                       const KVCacheConfig& kv_cache_config,
                                       const RuntimeConfig& runtime_config,
                                       const ParallelismConfig& parallelism_config,
                                       const SpeculativeExecutionConfig& sp_config,
                                       std::shared_ptr<BroadcastManager> broadcast_manager,
                                       std::shared_ptr<kvcm::ClientWrapper> client_wrapper,
                                       std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter):
    KVCMStorageBackend(cache_config, kv_cache_config, runtime_config, parallelism_config, sp_config,
                       std::move(broadcast_manager), remoteCacheGdrEnabled(), std::move(metrics_reporter),
                       std::move(client_wrapper)) {}

KVCMStorageBackend::~KVCMStorageBackend() = default;

std::optional<KVCMStorageBackend::ControlConfig> KVCMStorageBackend::controlConfig() const {
    return impl_->controlConfig();
}

bool KVCMStorageBackend::initImpl() {
    return impl_->init(
        topology(),
        [this](int layer_id, const std::string& tag, int block_id) {
            return convertIndexToBuffer(layer_id, tag, block_id);
        },
        [this](const std::string& tag) -> const DeviceBlockPoolPtr& { return devicePool(tag); });
}

StorageMatchResult KVCMStorageBackend::matchImpl(const StorageRequest& request) {
    return impl_->match(request);
}

void KVCMStorageBackend::readImpl(const StorageRequest&                           request,
                                  const std::shared_ptr<StorageBackendMatchMeta>& match_meta) {
    impl_->read(request, match_meta);
}

void KVCMStorageBackend::writeImpl(const StorageRequest& request) {
    impl_->write(request);
}

void KVCMStorageBackend::shutdownImpl() noexcept {
    impl_->shutdown();
}

void KVCMStorageBackend::onRemoteCacheSkipped(const char* stage) noexcept {
    impl_->reportRemoteCacheSkipped(stage);
}

void KVCMStorageBackend::onQuarantined(bool is_write, StorageQuarantineReason reason, size_t block_count) noexcept {
    impl_->reportQuarantineEvent(is_write, reason, block_count);
}

void KVCMStorageBackend::onQuarantineChanged(uint64_t                          generation,
                                             size_t                            task_count,
                                             size_t                            block_count,
                                             const StorageQuarantineBreakdown& breakdown) noexcept {
    impl_->reportQuarantineState(generation, task_count, block_count, breakdown);
}

bool KVCMStorageBackend::execute(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response) {
    if (request.op() != REMOTE_OPERATION_READ && request.op() != REMOTE_OPERATION_WRITE) {
        return impl_->execute(request, response);
    }
    try {
        return impl_->execute(request, response);
    } catch (...) {
        // Preserve an SDK failure already recorded before the payload threw.
        response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
        return false;
    }
}

}  // namespace rtp_llm
