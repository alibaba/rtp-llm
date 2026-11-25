#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/KVCMStorageBackend.h"

#include <algorithm>
#include <array>
#include <future>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <atomic>
#include <chrono>
#include <limits>
#include <list>
#include <iterator>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <unistd.h>

#include "autil/EnvUtil.h"
#include "autil/legacy/jsonizable.h"
#include <unordered_set>

#include "kvcm_client/HostCacheStateCompat.h"
#include "kvcm_client/TransferClientCompat.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientWrapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/GroupPolicy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/WorkerPayloadProof.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeReclaimer.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

namespace rtp_llm {
namespace {

using Clock = std::chrono::steady_clock;

constexpr auto kMaxReadSafetyQueryDuration = std::chrono::seconds(1);
constexpr std::array<const char*, 5> kCompletionUnknownReasons{
    "sdk_timeout", "rpc_timeout", "rpc_error", "invalid_response", "sdk_error"};

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
    bool                        filtered = false;
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

class KVCMStorageBackend::Impl: public std::enable_shared_from_this<Impl> {
public:
    using ActualUriGather = std::vector<std::vector<kv_cache_manager::LocationSpecUnit*>>;
    using PayloadResult   = BroadcastResult<FunctionRequestPB, FunctionResponsePB>;

    static std::exception_ptr withTargetProtection(const std::exception_ptr&    failure,
                                                   StorageTargetProtection protection) noexcept {
        // Throw/catch the requested subtype directly: allocation failure must
        // not be substituted for the UNKNOWN whose targets still need custody.
        try {
            try {
                std::rethrow_exception(failure);
            } catch (const StorageOperationTimeout& error) {
                throw StorageOperationTimeout(error.what(), error.reason(), protection);
            } catch (const StorageOperationCompletionUnknown& error) {
                throw StorageOperationCompletionUnknown(error.what(), error.reason(), protection);
            }
        } catch (const StorageOperationCompletionUnknown& error) {
            if (error.localTargetProtection() == protection)
                return std::current_exception();
        } catch (...) {}
        std::terminate();
    }

    std::vector<TransferTargetBlock> readTargets(const std::vector<FunctionRequestPB>& requests) const {
        std::vector<TransferTargetBlock> targets;
        for (const auto& request : requests) {
            const auto& remote = request.remote_request();
            RTP_LLM_CHECK(remote.group_tags_size() == remote.block_ids_size());
            for (int index = 0; index < remote.block_ids_size(); ++index) {
                targets.push_back({pools_by_tag_.at(remote.group_tags(index)), remote.block_ids(index)});
            }
        }
        return targets;
    }

    enum class TransferResult {
        SUCCESS,
        FAILED,
        TIMEOUT
    };

    Impl(const KVCMStorageBackend&                      backend,
         const CacheConfig&                         cache_config,
         const KVCacheConfig&                       kv_cache_config,
         const RuntimeConfig&                       runtime_config,
         const ParallelismConfig&                   parallelism_config,
         const SpeculativeExecutionConfig&          sp_config,
         std::shared_ptr<BroadcastManager>          broadcast_manager,
         std::shared_ptr<kvcm::ClientWrapper>       client_wrapper,
         std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter,
         bool                                       gdr_enabled):
        backend_(backend),
        cache_config_(cache_config),
        kv_cache_config_(kv_cache_config),
        runtime_config_(runtime_config),
        parallelism_config_(parallelism_config),
        sp_config_(sp_config),
        broadcast_manager_(std::move(broadcast_manager)),
        metrics_reporter_(std::move(metrics_reporter)),
        client_wrapper_(std::move(client_wrapper)),
        gdr_enabled_(gdr_enabled) {}

    void setReclaimer(std::shared_ptr<BlockTreeReclaimer> reclaimer) {
        RTP_LLM_CHECK(reclaimer && !reclaimer_);
        reclaimer->setRecoveryMaintenance([impl = weak_from_this()] {
            if (auto owner = impl.lock())
                owner->retryReadRecoveryDiscards();
        });
        reclaimer_ = std::move(reclaimer);
    }

    std::optional<ControlConfig> controlConfig() const {
        return control_config_;
    }

    bool init(std::shared_ptr<const CacheConfig>                                  cache_config,
              StorageBackend::StorageBufferResolver                               buffer_resolver,
              const std::function<const DeviceBlockPoolPtr&(const std::string&)>& pool_resolver) {
        if (!reclaimer_) {
            RTP_LLM_LOG_ERROR("BlockTree KVCM storage backend requires a reclaimer");
            return false;
        }
        RTP_LLM_CHECK_WITH_INFO(cache_config != nullptr, "KVCM storage backend requires a cache config");
        RTP_LLM_LOG_INFO("start init BlockTree KVCM storage backend");
        if (parallelism_config_.tp_rank == 0 && parallelism_config_.tp_size > 1 && !broadcast_manager_) {
            RTP_LLM_LOG_ERROR("BlockTree KVCM rank 0 requires a broadcast manager for tp_size=%ld",
                              parallelism_config_.tp_size);
            return false;
        }
        storage_config_ = cache_config;
        std::vector<std::string>               full_group_tags;
        std::vector<std::string>               other_group_tags;
        std::unordered_map<std::string, size_t> group_block_size_bytes;
        const auto group_count = static_cast<size_t>(cache_config->allGroupNums());
        group_block_size_bytes.reserve(group_count);
        for (size_t module_id = 0; module_id < cache_config->moduleCount(); ++module_id) {
            const auto& module_topology = cache_config->topology(module_id);
            for (const auto& group : module_topology.groups()) {
                if (!group.policy.enable_prefix_reuse || module_topology.layerIdsForGroup(group.tag).empty()) {
                    continue;
                }
                has_swa_ = has_swa_ || group.policy.group_type == CacheGroupType::SWA;
                if (group.policy.group_type == CacheGroupType::FULL) {
                    full_group_tags.push_back(group.tag);
                } else {
                    other_group_tags.push_back(group.tag);
                }
                group_block_size_bytes.emplace(group.tag, module_topology.blockSizeBytesForGroup(group.tag));
            }
        }
        if (other_group_tags.empty()) {
            group_policy_ = std::make_unique<kvcm::FullLayerGroupPolicy>(
                cache_config, buffer_resolver, full_group_tags, other_group_tags, std::move(group_block_size_bytes));
        } else {
            group_policy_ = std::make_unique<kvcm::FullLinearLayerGroupPolicy>(
                cache_config, buffer_resolver, full_group_tags, other_group_tags, std::move(group_block_size_bytes));
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
                if (group_policy_->groups().size() != group_count) {
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
                    fields["location_spec_infos"]  = autil::legacy::ToJson(location_infos);
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
                kvcm::KVCMConfig::LocationSpecGroups  configured_groups;
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
                                              legacy_name.c_str(),
                                              expected_name.c_str());
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
                                              group_name.c_str(),
                                              name.c_str());
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
        ControlConfig control_config{config->instance_group(),
                                     config->instance_id(),
                                     config->registerInstanceRequest(),
                                     config->enable_vipserver(),
                                     config->vipserver_domain(),
                                     config->addresses(),
                                     merge_full_specs_};
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
        control_config_     = std::move(control_config);
        const auto capacity = kv_cache_config_.kvcm_asyncwrapper_queue_size;
        if (parallelism_config_.tp_size > 1) {
            const auto workers  = kv_cache_config_.kvcm_storage_thread_num;
            local_payload_pool_ = std::make_unique<BlockTreeTaskPool>(workers, capacity, "remote_read");
            rank_write_pool_    = std::make_unique<BlockTreeTaskPool>(workers, capacity, "remote_write");
        } else if (parallelism_config_.tp_size == 1 && !broadcast_manager_) {
            local_payload_pool_ = std::make_unique<BlockTreeTaskPool>(
                kv_cache_config_.kvcm_asyncwrapper_thread_num, capacity, "remote_payload");
        }
        if ((local_payload_pool_ && !local_payload_pool_->start())
            || (rank_write_pool_ && !rank_write_pool_->start())) {
            RTP_LLM_LOG_ERROR("failed to start KVCM payload workers");
            return false;
        }
        const auto tp_rank = parallelism_config_.tp_rank;
        RTP_LLM_LOG_INFO("BlockTree KVCM storage backend initialized, tp_rank=%ld tp_size=%ld policy={%s}",
                         tp_rank,
                         parallelism_config_.tp_size,
                         group_policy_->debugString().c_str());
        return true;
    }

    void reportRemoteCacheSkipped(const char* stage) noexcept {
        try {
            RtpLLMRemoteCacheOutcomeMetricsCollector collector;
            collector.stage   = stage;
            collector.outcome = "disabled";
            reportRemoteMetrics<RtpLLMRemoteCacheOutcomeMetrics>(collector);
        } catch (...) {
            RTP_LLM_LOG_WARNING("failed to report disabled remote cache operation");
        }
    }

    bool requestsOnlyFullGroups(const StorageRequest& request) const {
        bool has_full_group = false;
        for (const auto& handles : request.handles) {
            for (const auto& handle : handles) {
                if (cache_config_.group(handle.tag).policy.group_type != CacheGroupType::FULL) {
                    return false;
                }
                has_full_group = true;
            }
        }
        return has_full_group;
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
        const auto                             start = Clock::now();
        block_tree_cache_detail::ScopeRollback report([&] {
            collector.remote_match_time_us = elapsedTimeUs(start);
            reportRemoteMetrics<RtpLLMRemoteCacheMatchMetrics>(collector);
        });
        const auto                             trace_id = nextTraceId("match", match_trace_sequence_);
        auto query_type                                 = static_cast<kv_cache_manager::QueryType>(default_query_type_);
        if (has_swa_) {
            // Fetch FULL and SWA locations, then apply each group's
            // model-defined reuse window locally.
            query_type = kv_cache_manager::QueryType::QT_BATCH_GET;
        }
        (void)remainingTimeout(request, "before match");
        bool                        success = false;
        kv_cache_manager::Locations locations;
        const bool                  positional = query_type == kv_cache_manager::QueryType::QT_BATCH_GET;
        if (positional && request.local_matched_blocks_num > 0) {
            const CacheKeysType query_keys(keys.begin() + request.local_matched_blocks_num, keys.end());
            auto                result =
                client_wrapper_->match("", trace_id, query_type, query_keys, kv_cache_manager::BlockMaskOffset{0}, {});
            success   = result.first;
            locations = std::move(result.second);
            if (success) {
                locations.insert(locations.begin(), request.local_matched_blocks_num, kv_cache_manager::Location{});
            }
        } else {
            auto result = client_wrapper_->match("", trace_id, query_type, keys, request.local_matched_blocks_num, {});
            success     = result.first;
            locations   = std::move(result.second);
        }
        if (!success) {
            return {request.local_matched_blocks_num, nullptr};
        }
        if (positional) {
            auto meta                              = std::make_shared<KVCMMatchMeta>();
            meta->locations                        = reusableLocations(request, std::move(locations));
            meta->filtered                         = true;
            collector.remote_match_fail_qps        = false;
            collector.remote_match_reuse_block_num = static_cast<int64_t>(meta->locations.size());
            if (!meta->locations.empty()) {
                collector.remote_valid_hit_ratio =
                    static_cast<double>(request.local_matched_blocks_num + meta->locations.size()) / keys.size();
            }
            return {request.local_matched_blocks_num + meta->locations.size(), std::move(meta)};
        }
        kvcm::LocationsView locations_view;
        if (!group_policy_->filterNeedLoadLocations(
                locations, locations_view, /*block_mask=*/0, requestsOnlyFullGroups(request))) {
            throw std::runtime_error("KVCM returned an invalid location shape");
        }
        if (locations_view.size() > keys.size() - request.local_matched_blocks_num) {
            throw std::runtime_error("KVCM prefix match exceeds the requested key range");
        }
        auto meta                              = std::make_shared<KVCMMatchMeta>();
        meta->locations                        = std::move(locations);
        collector.remote_match_fail_qps        = false;
        collector.remote_match_reuse_block_num = static_cast<int64_t>(locations_view.size());
        if (!locations_view.empty()) {
            collector.remote_valid_hit_ratio =
                static_cast<double>(request.local_matched_blocks_num + locations_view.size()) / keys.size();
        }
        return {request.local_matched_blocks_num + locations_view.size(), std::move(meta)};
    }

    bool usesAsyncPayload(const StorageRequest&, bool) const noexcept {
        return true;
    }

    bool readAsync(const StorageRequest&                           request,
                   const std::shared_ptr<StorageBackendMatchMeta>& match_meta,
                   ReadCompletion                                  done) {
        read(request, match_meta, std::move(done));
        return true;
    }

    void read(const StorageRequest&                           request,
              const std::shared_ptr<StorageBackendMatchMeta>& match_meta,
              ReadCompletion                                  done = {}) {
        bool               source_may_be_touched = false;
        std::string read_trace;
        try {
            readTransaction(request, match_meta, source_may_be_touched, std::move(done), read_trace);
        } catch (...) {
            rethrowReadPayloadFailure(source_may_be_touched, read_trace);
        }
    }

    void readTransaction(const StorageRequest&                           request,
                         const std::shared_ptr<StorageBackendMatchMeta>& match_meta,
                         bool&                                           source_may_be_touched,
                         ReadCompletion                                  done,
                         std::string&                                     read_trace) {
        RTP_LLM_CHECK_WITH_INFO(parallelism_config_.tp_rank == 0,
                                "KVCM metadata read must run on tp rank 0, got %ld",
                                parallelism_config_.tp_rank);
        RtpLLMRemoteCacheReadMetricsCollector  collector;
        bool                                   sdk_failed = false;
        const auto                             start      = Clock::now();
        block_tree_cache_detail::ScopeRollback report([&] {
            collector.remote_read_task_cost_time_us = elapsedTimeUs(start);
            reportRemoteMetrics<RtpLLMRemoteCacheReadMetrics>(collector);
            if (collector.remote_read_fail_qps && sdk_failed)
                reportSdkFailure("read");
        });
        const auto                             meta = std::dynamic_pointer_cast<KVCMMatchMeta>(match_meta);
        RTP_LLM_CHECK_WITH_INFO(meta != nullptr, "KVCM read received invalid match metadata");
        kvcm::LocationsView locations_view;
        const bool only_full_groups = requestsOnlyFullGroups(request);
        if (meta->filtered) {
            locations_view.resize(meta->locations.size());
            for (size_t i = 0; i < meta->locations.size(); ++i) {
                for (const auto& spec : meta->locations[i]) {
                    locations_view[i].emplace_back(spec);
                }
            }
        } else {
            RTP_LLM_CHECK_WITH_INFO(group_policy_->filterNeedLoadLocations(meta->locations,
                                                                          locations_view,
                                                                          /*block_mask=*/0,
                                                                          only_full_groups),
                                   "KVCM read location filtering failed");
        }
        const size_t remote_blocks = request.handles.size() - request.local_matched_blocks_num;
        if (only_full_groups && locations_view.size() > remote_blocks) {
            // Full-group-only filtering counts one view row per FULL-group
            // location, so the re-derived view can legitimately report more
            // rows than this request reserved handles for (match() counts the
            // same view the same way, while the allocator caps the handle
            // range at the final reusable full block). The view is
            // order-preserving and each row depends only on the rows before
            // it, so reading the first remote_blocks rows covers exactly the
            // keys with reserved destination handles; the excess tail rows
            // belong to keys this request never asked to transfer.
            locations_view.resize(remote_blocks);
        }
        RTP_LLM_CHECK_WITH_INFO(locations_view.size() == remote_blocks,
                                "KVCM read shape mismatch: locations=%zu remote_blocks=%zu",
                                locations_view.size(),
                                remote_blocks);

        std::vector<FunctionRequestPB> requests(static_cast<size_t>(parallelism_config_.tp_size));
        const auto&                    spec_info = group_policy_->spec_info_map();
        std::string                    trace_id  = nextTraceId("read", read_trace_sequence_);
        initializeRequests(requests, REMOTE_OPERATION_READ, trace_id);
        for (size_t location_idx = 0; location_idx < locations_view.size(); ++location_idx) {
            const size_t key_idx = request.local_matched_blocks_num + location_idx;
            for (const auto& location_spec : locations_view[location_idx]) {
                const auto        info = spec_info.find(location_spec.spec_name);
                const std::string spec_name(location_spec.spec_name);
                RTP_LLM_CHECK_WITH_INFO(info != spec_info.end(), "KVCM read has unknown spec [%s]", spec_name.c_str());
                auto* remote = requests.at(static_cast<size_t>(info->second.tp_rank)).mutable_remote_request();
                remote->add_uris(std::string(location_spec.uri));
                if (merge_full_specs_)
                    remote->add_uri_block_counts(info->second.group_ids.size());
                for (const auto group_id : info->second.group_ids) {
                    const auto& group  = group_policy_->groups().at(group_id);
                    const auto* handle = findHandle(request.handles[key_idx], group.tag);
                    RTP_LLM_CHECK_WITH_INFO(handle != nullptr,
                                            "KVCM read has no destination for key=%zu tag=%s",
                                            key_idx,
                                            group.tag.c_str());
                    remote->add_group_tags(group.tag);
                    remote->add_block_ids(handle->block);
                }
            }
        }
        if (!backend_.remoteCacheEnabled()) {
            reportRemoteCacheSkipped("read");
            collector.remote_read_qps      = false;
            collector.remote_read_fail_qps = false;
            throw std::runtime_error("remote cache disabled by KVCM before read submission");
        }
        // Capture the exact selected request targets, before any rank can
        // start. Ordinary StorageRequest pins remain held through settlement.
        const TransferCompletionKey key{trace_id, TransferCompletionKey::LegacyWireField::KVCM_TRACE_ID};
        const auto ranks = broadcast_manager_ ? broadcast_manager_->workerNum() : 1;
        auto targets = readTargets(requests);
        RTP_LLM_CHECK(reclaimer_->registerTask(key, {broadcast_manager_ != nullptr, ranks}, std::move(targets), "kvcm"));
        read_trace.swap(trace_id);
        if (done) {
            auto prepare_completion = [&](auto collect) {
                Task finish = [this,
                               collect = std::move(collect),
                               read_trace,
                               deadline = request.transfer_task->deadline(),
                               collector,
                               start,
                               tokens = remote_blocks * storage_config_->groups().front().cacheKeyTokenStride()]() mutable {
                    bool                                   sdk_failed = false;
                    block_tree_cache_detail::ScopeRollback report([&] {
                        collector.remote_read_task_cost_time_us = elapsedTimeUs(start);
                        reportRemoteMetrics<RtpLLMRemoteCacheReadMetrics>(collector);
                        if (collector.remote_read_fail_qps && sdk_failed)
                            reportSdkFailure("read");
                    });
                    bool                                   unknown = true;
                    try {
                        (void)collect(unknown, sdk_failed);
                        if (Clock::now() >= deadline) {
                            throw StorageOperationBufferSafeFailure("KVCM task deadline exceeded: after read");
                        }
                        collector.remote_read_token_num = static_cast<int64_t>(tokens);
                        collector.remote_read_fail_qps  = false;
                        reclaimer_->unregisterTask(read_trace);
                    } catch (...) {
                        rethrowReadPayloadFailure(unknown, read_trace);
                    }
                };
                return [done = std::move(done), finish = std::move(finish)]() mutable { done(std::move(finish)); };
            };
            dispatchPayloadAsync(requests, request, source_may_be_touched, prepare_completion, read_trace);
            report.dismiss();
            return;
        }
        (void)dispatchRequests(requests, request, source_may_be_touched, sdk_failed, nullptr, read_trace);
        source_may_be_touched = false;
        (void)remainingTimeout(request, "after read");
        // A canonical CP key already covers every shard. Do not multiply this
        // logical token count by the number of TP ranks or payload groups.
        collector.remote_read_token_num =
            static_cast<int64_t>(remote_blocks * storage_config_->groups().front().cacheKeyTokenStride());
        collector.remote_read_fail_qps = false;
        reclaimer_->unregisterTask(read_trace);
    }

    struct WritePublication {
        kv_cache_manager::WriteLocation        location;
        ActualUriGather                        actual_uris;
        RtpLLMRemoteCacheWriteMetricsCollector collector;
        bool                                   finish_attempted{false};
    };

    bool write(const StorageRequest& request, WriteCompletion done = {}, WritePinFilter retain_pins = {}) {
        bool source_may_be_touched = false;
        try {
            return writeTransaction(request, source_may_be_touched, std::move(done), std::move(retain_pins));
        } catch (...) {
            rethrowWritePayloadFailure(source_may_be_touched);
        }
    }

    [[noreturn]] void rethrowPayloadFailure(bool source_may_be_touched) const {
        try {
            throw;
        } catch (const StorageOperationBufferSafeFailure&) {
            throw;
        } catch (const StorageOperationCompletionUnknown&) {
            if (source_may_be_touched) {
                throw;
            }
            std::rethrow_exception(buffer_safe_failure_);
        } catch (...) {
            std::rethrow_exception(source_may_be_touched ? completion_unknown_failure_ : buffer_safe_failure_);
        }
    }

    [[noreturn]] void rethrowReadPayloadFailure(bool source_may_be_touched,
                                               const std::string& read_trace) const {
        try {
            rethrowPayloadFailure(source_may_be_touched);
        } catch (const StorageOperationCompletionUnknown&) {
            if (!read_trace.empty()) {
                reclaimer_->freezeTargetBlocksWithTraceIDAndStillIOWorkerID(read_trace, {});
                std::rethrow_exception(withTargetProtection(std::current_exception(),
                                                                      StorageTargetProtection::PROTECTED_BY_BACKEND));
            }
            reclaimer_->unregisterTask(read_trace);
            throw;
        } catch (...) {
            reclaimer_->unregisterTask(read_trace);
            throw;
        }
    }

    [[noreturn]] void rethrowWritePayloadFailure(bool source_may_be_touched) const {
        try {
            rethrowPayloadFailure(source_may_be_touched);
        } catch (const StorageOperationCompletionUnknown&) {
            std::rethrow_exception(
                withTargetProtection(std::current_exception(), StorageTargetProtection::NO_LOCAL_TARGETS));
        }
    }

    bool writeTransaction(const StorageRequest& request,
                          bool&                 source_may_be_touched,
                          WriteCompletion       done,
                          WritePinFilter        retain_pins) {
        if (!backend_.remoteCacheEnabled()) {
            reportRemoteCacheSkipped("write");
            return false;
        }
        RTP_LLM_CHECK_WITH_INFO(parallelism_config_.tp_rank == 0,
                                "KVCM metadata write must run on tp rank 0, got %ld",
                                parallelism_config_.tp_rank);
        RTP_LLM_CHECK(request.keys != nullptr && request.keys->size() == request.handles.size());
        const size_t valid_keys_size = request.keys->size();
        if (valid_keys_size == 0) {
            return false;
        }
        const bool                             defer       = static_cast<bool>(done);
        auto                                   async_write = defer ? std::make_shared<WritePublication>() : nullptr;
        WritePublication                       local;
        auto&                                  publication = defer ? *async_write : local;
        auto&                                  collector   = publication.collector;
        bool                                   sdk_failed  = false;
        const auto                             start       = Clock::now();
        block_tree_cache_detail::ScopeRollback report([&] {
            collector.remote_write_task_cost_time_us = elapsedTimeUs(start);
            reportRemoteMetrics<RtpLLMRemoteCacheWriteMetrics>(collector);
            if (collector.remote_write_fail_qps && sdk_failed)
                reportSdkFailure("write");
        });
        CacheKeysType                          keys(request.keys->begin(), request.keys->begin() + valid_keys_size);
        std::vector<std::string>               location_spec_group_names;
        RTP_LLM_CHECK_WITH_INFO(group_policy_->getNeedWriteGroups(request, valid_keys_size, location_spec_group_names),
                                "KVCM write group selection failed");
        const std::string trace_id = nextTraceId("write", write_trace_sequence_);
        (void)remainingTimeout(request, "before StartWrite");
        auto [success, allocation] = [&] {
            const auto                             stage_start = Clock::now();
            block_tree_cache_detail::ScopeRollback timer(
                [&] { collector.remote_get_write_location_time_us = elapsedTimeUs(stage_start); });
            // The server interprets 0 as one readable replica.
            return client_wrapper_->getWriteLocation("",
                                                     trace_id,
                                                     keys,
                                                     /*tokens=*/{},
                                                     location_spec_group_names,
                                                     /*write_timeout_seconds=*/300,
                                                     /*min_replica_count=*/0);
        }();
        RTP_LLM_CHECK_WITH_INFO(success, "KVCM StartWrite failed");
        auto& write_location = publication.location;
        write_location       = std::move(allocation);
        static const kv_cache_manager::Locations empty_locations;
        auto&                                    finish_attempted = publication.finish_attempted;
        try {
            std::vector<FunctionRequestPB> requests(static_cast<size_t>(parallelism_config_.tp_size));
            auto&                          actual_uri_gather = publication.actual_uris;
            actual_uri_gather.resize(requests.size());
            initializeRequests(requests, REMOTE_OPERATION_WRITE, trace_id);
            const auto key_indices = unmaskedKeyIndices(write_location.block_mask, valid_keys_size);
            RTP_LLM_CHECK_WITH_INFO(key_indices.size() == write_location.locations.size(),
                                    "KVCM write mask/location mismatch: keys=%zu locations=%zu",
                                    key_indices.size(),
                                    write_location.locations.size());
            if (retain_pins)
                retain_pins(key_indices);
            if (write_location.locations.empty()) {
                if (!write_location.write_session_id.empty()) {
                    // The server can create an empty, short-lived session when
                    // every key already has enough replicas. Close it normally.
                    (void)remainingTimeout(request, "before empty FinishWrite");
                    finish_attempted                                   = true;
                    const auto                             stage_start = Clock::now();
                    block_tree_cache_detail::ScopeRollback timer(
                        [&] { collector.remote_finish_write_time_us += elapsedTimeUs(stage_start); });
                    try {
                        if (!client_wrapper_->finishWrite("",
                                                          nextTraceId("finish_write", finish_write_trace_sequence_),
                                                          write_location.write_session_id,
                                                          kv_cache_manager::BlockMaskOffset{0},
                                                          empty_locations)) {
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
                return false;
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
                        const auto&               group  = group_policy_->groups().at(group_id);
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

            const auto                             broadcast_start = Clock::now();
            block_tree_cache_detail::ScopeRollback broadcast_timer(
                [&] { collector.remote_write_broadcast_time_us = elapsedTimeUs(broadcast_start); });
            if (!backend_.remoteCacheEnabled()) {
                reportRemoteCacheSkipped("write");
                throw StorageOperationBufferSafeFailure("remote cache disabled by KVCM before write submission");
            }
            (void)remainingTimeout(request, "before write dispatch");
            if (defer) {
                auto prepare_completion = [&](auto collect) {
                    WriteResult finish =
                        [this, async_write, collect = std::move(collect), &request, start](bool on_worker) {
                            auto&                                  write      = *async_write;
                            bool                                   sdk_failed = false;
                            block_tree_cache_detail::ScopeRollback report([&] {
                                write.collector.remote_write_task_cost_time_us = elapsedTimeUs(start);
                                reportRemoteMetrics<RtpLLMRemoteCacheWriteMetrics>(write.collector);
                                if (write.collector.remote_write_fail_qps && sdk_failed)
                                    reportSdkFailure("write");
                            });
                            bool                                   touched = true;
                            try {
                                auto responses = collect(touched, sdk_failed);
                                if (!on_worker)
                                    std::rethrow_exception(buffer_safe_failure_);
                                publishWrite(request, write, responses);
                            } catch (...) {
                                if (on_worker && !write.finish_attempted)
                                    abortUnfinishedWrite(write.location.write_session_id, touched, write.collector);
                                rethrowWritePayloadFailure(touched);
                            }
                        };
                    return
                        [async_write, broadcast_start, done = std::move(done), finish = std::move(finish)]() mutable {
                            async_write->collector.remote_write_broadcast_time_us = elapsedTimeUs(broadcast_start);
                            done(std::move(finish));
                        };
                };
                dispatchPayloadAsync(requests, request, source_may_be_touched, prepare_completion);
                broadcast_timer.dismiss();
                report.dismiss();
                return true;
            }
            auto responses        = dispatchRequests(requests, request, source_may_be_touched, sdk_failed);
            source_may_be_touched = false;
            publishWrite(request, publication, responses);
            return false;
        } catch (...) {
            if (!finish_attempted)
                abortUnfinishedWrite(write_location.write_session_id, source_may_be_touched, collector);
            throw;
        }
    }

    void publishWrite(const StorageRequest&                  request,
                      WritePublication&                      write,
                      const std::vector<FunctionResponsePB>& responses) {
        auto&                                    write_location    = write.location;
        auto&                                    actual_uri_gather = write.actual_uris;
        auto&                                    collector         = write.collector;
        static const kv_cache_manager::Locations empty_locations;
        bool                                     has_actual_uri = false;
        for (size_t rank = 0; rank < responses.size(); ++rank) {
            const auto& actual_uris = responses[rank].remote_response().actual_uris();
            RTP_LLM_CHECK_WITH_INFO(actual_uris.empty()
                                        || static_cast<size_t>(actual_uris.size()) == actual_uri_gather[rank].size(),
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
        (void)remainingTimeout(request, "before FinishWrite");
        write.finish_attempted = true;
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
    }

    void abortUnfinishedWrite(const std::string&                      write_session_id,
                              bool                                    source_may_be_touched,
                              RtpLLMRemoteCacheWriteMetricsCollector& collector) {
        if (source_may_be_touched) {
            return;
        }
        const auto                             stage_start = Clock::now();
        block_tree_cache_detail::ScopeRollback timer(
            [&] { collector.remote_finish_write_time_us += elapsedTimeUs(stage_start); });
        try {
            const auto abort_trace = nextTraceId("abort_write", finish_write_trace_sequence_);
            if (!client_wrapper_->abortWrite("", abort_trace, write_session_id)) {
                RTP_LLM_LOG_WARNING("KVCM failed to abort write session [%s]", write_session_id.c_str());
            }
        } catch (...) {
            RTP_LLM_LOG_WARNING("KVCM abort write session threw, session=[%s]", write_session_id.c_str());
        }
    }

    bool prepareBlockBuffers(const RemoteOperationRequestPB& request,
                             const std::vector<std::string>& tags,
                             const std::vector<int32_t>&     blocks,
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
            size_t      begin  = 0;
            const auto& groups = group_policy_->groups();
            const auto& infos  = group_policy_->spec_info_map();
            for (int uri = 0; uri < request.uris_size(); ++uri) {
                const size_t count = request.uri_block_counts(uri);
                if (count == 0 || count > blocks.size() - begin) {
                    return false;
                }
                const auto group = std::find_if(
                    groups.begin(), groups.end(), [&](const auto& entry) { return entry.second.tag == tags[begin]; });
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
            const size_t count  = request.uri_block_counts(uri);
            auto&        buffer = buffers[begin];
            if (count > 1) {
                size_t iov_count = 0;
                for (size_t member = begin; member < begin + count; ++member) {
                    iov_count += buffers[member].iovs.size();
                }
                buffer.iovs.reserve(iov_count);
                for (size_t member = begin + 1; member < begin + count; ++member) {
                    auto& iovs = buffers[member].iovs;
                    buffer.iovs.insert(
                        buffer.iovs.end(), std::make_move_iterator(iovs.begin()), std::make_move_iterator(iovs.end()));
                }
                // SDK gather/scatter groups by size. Use the same stable wire
                // order for normal I/O; only pointer descriptors move here.
                for (size_t i = 1; i < buffer.iovs.size(); ++i) {
                    auto   iov = std::move(buffer.iovs[i]);
                    size_t j   = i;
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

    struct ReadExecution {
        std::string                           trace_id;
        std::shared_ptr<TransferBufferSafety> proof;
        std::shared_ptr<std::atomic<bool>> recovery_cancelled;
    };
    using ReadExecutionPtr = std::shared_ptr<ReadExecution>;

    void retryReadRecoveryDiscards() {
        std::unordered_set<std::string> retries;
        {
            std::lock_guard<std::mutex> lock(recovery_discard_mutex_);
            retries.swap(pending_recovery_discards_);
        }
        for (const auto& trace_id : retries)
            discardReadRecovery(trace_id);
    }

    void discardReadRecovery(const std::string& trace_id) {
        try {
            if (client_wrapper_)
                client_wrapper_->discardReadRecovery(trace_id);
        } catch (...) {
            std::lock_guard<std::mutex> lock(recovery_discard_mutex_);
            pending_recovery_discards_.insert(trace_id);
            RTP_LLM_LOG_WARNING("KVCM READ recovery discard failed trace=%s", trace_id.c_str());
        }
    }

    ReadExecutionPtr beginReadExecution(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response) {
        response.set_sdk_failed(false);
        response.set_payload_buffer_safe(false);
        response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
        response.clear_actual_uris();
        const auto reject = [&](const char* reason) {
            RTP_LLM_LOG_WARNING("KVCM payload admission rejected op=READ rank=%ld trace=%s reason=%s",
                                static_cast<long>(parallelism_config_.world_rank),
                                request.trace_id().c_str(),
                                reason);
            // This rejected request never starts I/O. Its response does not
            // supply completion evidence for the already registered execution.
            response.set_payload_buffer_safe(true);
        };
        if (!reclaimer_)
            return nullptr;
        if (request.trace_id().empty()) {
            reject("empty_trace");
            return nullptr;
        }
        // Allocate the completion path before admission. It retains the registry
        // independently of business callbacks and never infers proof on destruction.
        auto execution      = std::make_shared<ReadExecution>();
        execution->trace_id = request.trace_id();
        execution->proof = std::make_shared<TransferBufferSafety>(
            [registry = reclaimer_, trace_id = execution->trace_id] { registry->ioFinishedOnWorkerWithTraceID(trace_id); });
        if (kvcm::kSupportsReadRecoveryDiscard &&
            parallelism_config_.role_type != RoleType::DECODE &&
            std::all_of(request.uris().begin(), request.uris().end(),
                        [](const auto& uri) { return uri.compare(0, 7, "pace://") == 0; })) {
            execution->recovery_cancelled = std::make_shared<std::atomic<bool>>(false);
        }
        BlockTreeReclaimer::QueryCallback query;
        const auto id = request.trace_id();
        if (client_wrapper_ && parallelism_config_.role_type != RoleType::DECODE) {
            query = [client = client_wrapper_, id](Clock::time_point deadline) {
                const auto now = Clock::now();
                deadline = std::min(deadline, now + kMaxReadSafetyQueryDuration);
                const auto deadline_ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline.time_since_epoch()).count();
                return now < deadline && client->queryReadBufferSafety(id, deadline_ms);
            };
        }
        auto discard = [self = weak_from_this(), cancelled = execution->recovery_cancelled, id](bool forced) {
            if (forced) {
                if (cancelled)
                    cancelled->store(true);
                if (auto owner = self.lock())
                    owner->discardReadRecovery(id);
            }
        };
        if (!reclaimer_->registerOnWorkerTaskWithTraceID(id, std::move(query), std::move(discard))) {
            reject("duplicate_trace");
            return nullptr;
        }
        return execution;
    }

    bool execute(const RemoteOperationRequestPB& request,
                 RemoteOperationResponsePB&      response,
                 ReadExecutionPtr                execution = nullptr) {
        const bool read = request.op() == REMOTE_OPERATION_READ;
        if (!read && request.op() != REMOTE_OPERATION_WRITE) {
            return executeMetadata(request, response);
        }
        if (read && !execution) {
            execution = beginReadExecution(request, response);
            if (!execution)
                return false;
        }
        block_tree_cache_detail::ScopeRollback unsubmitted([execution] {
            if (execution)
                execution->proof->report(true);
        });
        if (payload_stopped_.load(std::memory_order_acquire)) {
            response.set_sdk_failed(false);
            response.set_payload_buffer_safe(true);
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
            return false;
        }
        const auto timeout_ms = request.timeout_ms() > 0 ? request.timeout_ms() : kv_cache_config_.kvcm_task_timeout_ms;
        const TransferTask task({}, std::chrono::milliseconds(timeout_ms));
        if (!rank_write_pool_ || request.timeout_ms() < 0) {
            unsubmitted.dismiss();
            return executePayload(request, response, task, nullptr, std::move(execution));
        }

        // The synchronous RPC owns request and response until this job returns.
        // Start the budget at admission; an expired job never accesses buffers.
        auto&      pool       = read ? *local_payload_pool_ : *rank_write_pool_;
        const auto task_class = read ? BlockTreeTaskClass::LOAD : BlockTreeTaskClass::BACKGROUND;
        response.set_sdk_failed(false);
        response.set_payload_buffer_safe(true);
        try {
            auto work = std::make_shared<std::packaged_task<bool()>>(
                [&] { return executePayload(request, response, task, nullptr, execution); });
            auto                  done   = work->get_future();
            std::function<void()> run    = [work] { (*work)(); };
            std::function<void()> expire = run;
            unsubmitted.dismiss();
            bool accepted;
            try {
                accepted = pool.submit(task_class, std::move(run), task.deadline(), std::move(expire));
            } catch (...) {
                // BlockTreeTaskPool only throws before enqueueing. A worker may
                // otherwise begin before submit returns, so do not infer from timing.
                if (execution)
                    execution->proof->report(true);
                throw;
            }
            if (accepted)
                return done.get();
            if (execution)
                execution->proof->report(true);
        } catch (...) {
            // executePayload records buffer safety even when the SDK throws.
            RTP_LLM_LOG_WARNING("KVCM rank payload execution failed");
        }
        response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
        return false;
    }

    bool executePayload(const RemoteOperationRequestPB& request,
                        RemoteOperationResponsePB&      response,
                        const TransferTask&             task,
                        bool*                           source_may_be_touched = nullptr,
                        ReadExecutionPtr                execution             = nullptr) {
        if (request.op() == REMOTE_OPERATION_READ && !execution) {
            execution = beginReadExecution(request, response);
            if (!execution) {
                if (source_may_be_touched)
                    *source_may_be_touched = false;
                return false;
            }
        }
        // Declared before the existing payload reconciliation scopes: their
        // final explicit proof must be published first, including while unwinding.
        block_tree_cache_detail::ScopeRollback publish_completion([&] {
            if (!execution)
                return;
            if (response.payload_buffer_safe()) {
                execution->proof->report(true);
            }
            if constexpr (!kvcm::kSupportsReadRecoveryCancellation) {
                if (execution->recovery_cancelled && execution->recovery_cancelled->load())
                    discardReadRecovery(execution->trace_id);
            }
        });
        response.set_sdk_failed(false);
        response.set_payload_buffer_safe(true);
        if (request.timeout_ms() < 0 || task.expired()) {
            if (request.op() == REMOTE_OPERATION_READ) {
                RTP_LLM_LOG_WARNING("KVCM READ payload expired before I/O: trace=%s timeout_ms=%d",
                                    request.trace_id().c_str(),
                                    request.timeout_ms());
            }
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_TIMEOUT);
            return false;
        }
        const std::vector<std::string>    tags(request.group_tags().begin(), request.group_tags().end());
        const std::vector<int32_t>        blocks(request.block_ids().begin(), request.block_ids().end());
        const kv_cache_manager::UriStrVec uris(request.uris().begin(), request.uris().end());
        kv_cache_manager::BlockBuffers    buffers;
        if (!prepareBlockBuffers(request, tags, blocks, buffers)) {
            RTP_LLM_LOG_WARNING("KVCM transfer has invalid block/URI layout");
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
            return false;
        }
        if (tags.empty()) {
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_SUCCESS);
            return true;
        }
        bool                                   unknown = false;
        block_tree_cache_detail::ScopeRollback reconcile_completion([&] {
            response.set_payload_buffer_safe(!unknown);
            if (source_may_be_touched)
                *source_may_be_touched = unknown;
        });
        const auto result = executeTagTransfers(request.op(), tags, uris, buffers, response, unknown, nullptr, task, execution.get());
        response.set_transfer_status(result == TransferResult::SUCCESS ? REMOTE_TRANSFER_STATUS_SUCCESS :
                                     result == TransferResult::TIMEOUT ? REMOTE_TRANSFER_STATUS_TIMEOUT :
                                                                         REMOTE_TRANSFER_STATUS_FAILED);
        return result == TransferResult::SUCCESS;
    }

    BlockTreeQueueSizes localPayloadQueueSizes() const {
        auto sizes = local_payload_pool_ ? local_payload_pool_->queueSizes() : BlockTreeQueueSizes{};
        if (rank_write_pool_)
            sizes.background += rank_write_pool_->queueSizes().background;
        return sizes;
    }

    void shutdown() noexcept {
        payload_stopped_.store(true, std::memory_order_release);
        // Incoming rank RPCs can still be waiting on either payload queue.
        // Close both queues to new work, then drain them before closing the client.
        const std::array<BlockTreeTaskPool*, 2> pools{local_payload_pool_.get(), rank_write_pool_.get()};
        for (auto* pool : pools) {
            if (pool)
                pool->stopAdmission();
        }
        for (auto* pool : pools) {
            if (pool) {
                pool->waitForIdle();
                pool->shutdown();
            }
        }
        if (client_wrapper_) {
            client_wrapper_->shutdown();
        }
    }

private:
    int resolveQueryType(int query_type) const {
        return query_type == 0 ? default_query_type_ : query_type;
    }

    static bool isPayloadBackend(int type) {
        return type == 1 || type == 2 || type == 3 || type == 4 || type == 5;
    }

    kv_cache_manager::Locations reusableLocations(const StorageRequest&       request,
                                                  kv_cache_manager::Locations locations) const {
        const size_t local     = request.local_matched_blocks_num;
        const size_t key_count = request.keys->size();
        RTP_LLM_CHECK_WITH_INFO(locations.size() == key_count, "KVCM positional query shape mismatch");
        const auto&                             infos = group_policy_->spec_info_map();
        std::unordered_map<std::string, size_t> runs;
        size_t                                  matched = local;
        for (size_t i = local; i < locations.size(); ++i) {
            std::unordered_map<std::string, const kv_cache_manager::LocationSpecUnit*> present;
            for (const auto& spec : locations[i]) {
                RTP_LLM_CHECK_WITH_INFO(infos.count(spec.spec_name) != 0, "KVCM returned an unknown spec");
                RTP_LLM_CHECK_WITH_INFO(present.emplace(spec.spec_name, &spec).second,
                                        "KVCM returned a duplicate spec");
            }
            bool complete = true;
            for (const auto& [id, group] : group_policy_->groups()) {
                (void)id;
                bool available = true;
                for (int rank = 0; rank < parallelism_config_.tp_size; ++rank) {
                    const auto name = kvcm::genLocationSpecName(
                        rank, group.group_name, runtime_config_.model_name, merge_full_specs_);
                    const auto found = present.find(name);
                    available        = available && found != present.end() && !found->second->uri.empty();
                }
                auto& run             = runs[group.tag];
                run                   = available ? run + 1 : 0;
                const size_t required =
                    std::min(i + 1 - local, storage_config_->group(group.tag).reuseBlockCount(i + 1));
                complete              = complete && run >= required;
            }
            if (complete) {
                matched = i + 1;
            }
        }
        locations.resize(matched);
        for (size_t i = local; i < matched; ++i) {
            auto& location = locations[i];
            location.erase(std::remove_if(location.begin(),
                                          location.end(),
                                          [&](const auto& spec) {
                                              const auto& group = storage_config_->group(infos.at(spec.spec_name).tag);
                                              return matched - i > group.reuseBlockCount(matched) || spec.uri.empty();
                                          }),
                           location.end());
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
        const int   type  = resolveQueryType(query.query_type());
        if (type < 1 || type > 2 || query.p2p_host_count() < 0) {
            return false;
        }
        const std::vector<int64_t>     keys(query.block_keys().begin(), query.block_keys().end());
        const std::vector<int64_t>     tokens(query.token_ids().begin(), query.token_ids().end());
        const std::vector<std::string> names(query.location_spec_names().begin(), query.location_spec_names().end());
        kv_cache_manager::BlockMask    mask = kv_cache_manager::BlockMaskOffset{0};
        if (query.block_mask().has_bool_masks()) {
            mask = kv_cache_manager::BlockMaskVector(query.block_mask().bool_masks().values().begin(),
                                                     query.block_mask().bool_masks().values().end());
        } else if (query.block_mask().info_case() == RemoteBlockMaskPB::kOffset) {
            if (query.block_mask().offset() < 0) {
                return false;
            }
            mask = static_cast<size_t>(query.block_mask().offset());
        }
        const auto query_type     = static_cast<kv_cache_manager::QueryType>(type);
        auto       appendLocation = [](const kv_cache_manager::Location& location, RemoteCacheLocationPB* output) {
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
                const int backend = query.backend_type();
                if (!isPayloadBackend(backend)) {
                    return false;
                }
                auto [success, locations] =
                    client_wrapper_->getCacheLocationsByBackend("",
                                                                request.trace_id(),
                                                                keys,
                                                                tokens,
                                                                mask,
                                                                names,
                                                                static_cast<kv_cache_manager::StorageType>(backend));
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
            group_policy_->buildLocationSpecGroups(
                static_cast<int>(parallelism_config_.tp_size), *groups, runtime_config_.model_name, merge_full_specs_),
            "failed to build KVCM location spec groups");
        for (const auto& [group_id, group] : group_policy_->groups()) {
            for (int rank = 0; rank < parallelism_config_.tp_size; ++rank) {
                const std::string spec_name =
                    kvcm::genLocationSpecName(rank, group.group_name, runtime_config_.model_name, merge_full_specs_);
                auto& size = (*infos)[spec_name];
                RTP_LLM_CHECK_WITH_INFO(group.block_size_bytes
                                            <= static_cast<size_t>(std::numeric_limits<int64_t>::max() - size),
                                        "KVCM location spec size overflow [%s]",
                                        spec_name.c_str());
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
        const bool  legacy_dsv41_specs =
            runtime_config_.model_name == "DeepSeekV41" || runtime_config_.model_name == "DeepSeekV41DSpark";
        if (legacy_dsv41_specs && full_group_tags.size() == 3 && backends.size() == 1
            && std::dynamic_pointer_cast<kvcm::TairMempoolSdkConfig>(backends.front())) {
            const auto&           first = storage_config_->group(full_group_tags.front());
            std::set<std::string> names;
            bool                  compatible = true;
            for (const auto& [id, group] : group_policy_->groups()) {
                (void)id;
                if (!group.is_full)
                    continue;
                names.insert(kvcm::genLocationSpecName(0, group.group_name, runtime_config_.model_name));
                const auto& member = storage_config_->group(group.tag);
                compatible         = compatible && member.seqSizePerBlock() == first.seqSizePerBlock()
                             && member.cacheKeyTokenStride() == first.cacheKeyTokenStride()
                             && member.policy.cp_mapping == first.policy.cp_mapping
                             && member.policy.cp_slice == first.policy.cp_slice;
            }
            merge_full_specs_ = compatible && names == std::set<std::string>{"F0_TP0", "F1_TP0", "F2_TP0"};
        }
        auto [location_infos, location_groups] = genLocationSpecs();

        const std::string model_name = runtime_config_.model_name;
        const std::string dtype      = getDataTypeStr(cache_config_.dtype());
        std::string       extra      = kv_cache_config_.kvcm_model_extra_info;
        extra += '/' + autil::EnvUtil::getEnv("BIZ_NAME", std::string("")) + '/'
                 + std::to_string(hashString(autil::EnvUtil::getEnv("CHECKPOINT_PATH", std::string(""))));
        std::string draft_info;
        if (cache_config_.moduleCount() > 1) {
            draft_info = '{' + sp_config_.to_string() + '}';
        }
        std::stringstream identity;
        identity << "instance_group: " << kv_cache_config_.kvcm_instance_group
                 << ";block_size:" << cache_config_.seq_size_per_block << ";model_name:" << model_name
                 << ";dtype_str:" << dtype << ";use_mla:" << cache_config_.use_mla()
                 << ";fp8_kv_cache:" << kv_cache_config_.fp8_kv_cache << ";tp_size:" << parallelism_config_.tp_size
                 << ";dp_size:" << parallelism_config_.dp_size << ";extra_info:" << extra
                 << ";location_spec_info:" << autil::legacy::ToJsonString(location_infos, true)
                 << ";location_spec_groups:" << autil::legacy::ToJsonString(location_groups, true)
                 << ";default_query_type:" << default_query_type_ << ";draft_model_info:" << draft_info;
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
                                                                     cache_config_.use_mla(),
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
        std::vector<kvcm::ClientWrapper::PoolRegistration>  registrations;
        std::vector<std::string>                            registration_tags;
        std::unordered_map<std::string, DeviceBlockPoolPtr> pools_by_tag;
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
                                         const kv_cache_manager::UriStrVec& uris,
                                         kv_cache_manager::BlockBuffers&    buffers,
                                         kv_cache_manager::UriStrVec&       actual_uris,
                                         RemoteOperationResponsePB&         response,
                                         bool&                              source_may_be_touched,
                                         bool*                              transfer_attempted,
                                         const TransferTask&                task,
                                         const ReadExecution* execution = nullptr) {
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
        // One SDK group may fail while another is still running. An error code
        // alone cannot prove buffer safety, including on the ordinary path.
        block_tree_cache_detail::ScopeRollback reconcile_completion([&] {
            source_may_be_touched = completion == kvcm::ClientWrapper::TransferCompletion::UNKNOWN;
            if (transfer_attempted && completion != kvcm::ClientWrapper::TransferCompletion::NOT_SUBMITTED) {
                *transfer_attempted = true;
            }
        });
        if (task.expired()) {
            return TransferResult::TIMEOUT;
        }
        if (operation == REMOTE_OPERATION_READ) {
            std::shared_ptr<kv_cache_manager::TransferTraceInfo> trace;
            if (execution && execution->recovery_cancelled)
                kvcm::setSdkReadExecutionId(trace, execution->trace_id, execution->recovery_cancelled);
            const auto ec = client_wrapper_->loadKvCachesForTagWithStatus(
                client_tag, uris, buffers, trace, &completion, task.deadline());
            if (ec == kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT) {
                return TransferResult::TIMEOUT;
            }
            if (ec != kv_cache_manager::ClientErrorCode::ER_OK) {
                return TransferResult::FAILED;
            }
        } else {
            auto [ec, result] = client_wrapper_->saveKvCachesForTagWithStatus(
                client_tag, uris, buffers, nullptr, &completion, task.deadline());
            if (ec == kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT) {
                return TransferResult::TIMEOUT;
            }
            if (ec != kv_cache_manager::ClientErrorCode::ER_OK || (!result.empty() && result.size() != uris.size())) {
                return TransferResult::FAILED;
            }
            actual_uris = std::move(result);
        }
        collector.remote_sdk_fail_qps = false;
        if (task.expired()) {
            return TransferResult::TIMEOUT;
        }
        return TransferResult::SUCCESS;
    }

    TransferResult executeTagTransfers(RemoteOpType                       operation,
                                       const std::vector<std::string>&    tags,
                                       const kv_cache_manager::UriStrVec& uris,
                                       kv_cache_manager::BlockBuffers&    buffers,
                                       RemoteOperationResponsePB&         response,
                                       bool&                              source_may_be_touched,
                                       bool*                              transfer_attempted,
                                       const TransferTask&                task,
                                         const ReadExecution* execution = nullptr) {
        if (operation != REMOTE_OPERATION_READ && operation != REMOTE_OPERATION_WRITE) {
            RTP_LLM_LOG_WARNING("KVCM transfer has invalid operation [%d]", operation);
            return TransferResult::FAILED;
        }
        kv_cache_manager::UriStrVec actual_uris;
        if (std::all_of(uris.begin(), uris.end(), [](const auto& uri) { return uri.compare(0, 7, "pace://") == 0; })) {
            // The first client owns all PACE GPU registrations. Forward the whole
            // request unchanged; the SDK selects normal or gather/scatter mode.
            const auto result = executeClientTransfer(operation,
                                                      registration_tags_.front(),
                                                      uris,
                                                      buffers,
                                                      actual_uris,
                                                      response,
                                                      source_may_be_touched,
                                                      transfer_attempted,
                                                      task, execution);
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
                for (size_t index : indices) {
                    tag_uris.push_back(uris[index]);
                    tag_buffers.push_back(std::move(buffers[index]));
                }
                const auto result = executeClientTransfer(operation,
                                                          tag,
                                                          tag_uris,
                                                          tag_buffers,
                                                          tag_actual_uris,
                                                          response,
                                                          source_may_be_touched,
                                                          transfer_attempted,
                                                          task);
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

    template<typename Completion>
    static void onPayloadDone(const std::shared_ptr<PayloadResult>& result, Completion complete) {
        try {
            result->onDone(complete);
        } catch (...) {
            // Registration failed. Keep the submitting worker until the results are available.
            (void)result->waitDone(/*timeout_ms=*/0, /*fatal_on_error=*/false);
            complete();
        }
    }

    template<typename PrepareCompletion>
    void dispatchPayloadAsync(std::vector<FunctionRequestPB>& requests,
                              const StorageRequest&           request,
                              bool&                           source_may_be_touched,
                              PrepareCompletion               prepare_completion,
                              std::string                     read_trace = {}) {
        if (broadcast_manager_) {
            auto result = dispatchRemoteRequests(requests, request, source_may_be_touched);
            onPayloadDone(result, prepare_completion([this, result, read_trace](bool& touched, bool& sdk_failed) {
                              return collectResponses(result, touched, sdk_failed, read_trace);
                          }));
            return;
        }

        RTP_LLM_CHECK(local_payload_pool_ && requests.size() == 1);
        struct LocalResult {
            std::vector<FunctionResponsePB> responses;
            std::exception_ptr              error;
            bool                            touched{false};
            bool                            sdk_failed{false};
        };
        const auto       task_class = requests.front().remote_request().op() == REMOTE_OPERATION_READ ?
                                          BlockTreeTaskClass::LOAD :
                                          BlockTreeTaskClass::BACKGROUND;
        const auto       deadline   = request.transfer_task->deadline();
        auto             local      = std::make_shared<LocalResult>();
        ReadExecutionPtr execution;
        if (requests.front().remote_request().op() == REMOTE_OPERATION_READ) {
            RemoteOperationResponsePB admission_response;
            execution = beginReadExecution(requests.front().remote_request(), admission_response);
            if (!execution) {
                source_may_be_touched = false;
                std::rethrow_exception(buffer_safe_failure_);
            }
        }
        block_tree_cache_detail::ScopeRollback unsubmitted([execution] {
            if (execution)
                execution->proof->report(true);
        });
        auto                                   complete = prepare_completion([local](bool& touched, bool& sdk_failed) {
            touched    = local->touched;
            sdk_failed = local->sdk_failed;
            if (local->error)
                std::rethrow_exception(local->error);
            return std::move(local->responses);
        });
        // complete owns the StorageRequest and its pins through settlement.
        std::function<void()> run =
            [this, requests = std::move(requests), &request, local, complete, execution, read_trace]() mutable {
                try {
                    local->responses =
                        dispatchRequests(requests, request, local->touched, local->sdk_failed, execution, read_trace);
                } catch (...) {
                    local->error = std::current_exception();
                }
                complete();
            };
        std::function<void()> expire =
            [local, error = buffer_safe_failure_, complete = std::move(complete), execution]() mutable {
                if (execution)
                    execution->proof->report(true);
                local->error = error;  // Expired in the queue, before any buffer access.
                complete();
            };
        unsubmitted.dismiss();
        bool accepted;
        try {
            accepted = local_payload_pool_->submit(task_class, std::move(run), deadline, std::move(expire));
        } catch (...) {
            // submit can throw only before enqueueing; its callbacks were
            // already allocated above and may run before a successful return.
            if (execution)
                execution->proof->report(true);
            throw;
        }
        if (!accepted) {
            if (execution)
                execution->proof->report(true);
            throw StorageOperationBufferSafeFailure("KVCM local payload queue rejected transfer");
        }
    }

    std::shared_ptr<PayloadResult> dispatchRemoteRequests(std::vector<FunctionRequestPB>& requests,
                                                          const StorageRequest&           storage_request,
                                                          bool&                           source_may_be_touched) {
        const int timeout_ms = remainingTimeout(storage_request, "before payload dispatch");
        for (auto& request : requests) {
            request.mutable_remote_request()->set_timeout_ms(timeout_ms);
        }
        const auto&       remote    = requests.front().remote_request();
        const std::string direction = remote.op() == REMOTE_OPERATION_READ ? "REMOTE2D" : "D2REMOTE";
        auto              rpc_call  = [&](const std::shared_ptr<RpcService::Stub>&    stub,
                            const std::shared_ptr<grpc::ClientContext>& context,
                            const FunctionRequestPB&                    request,
                            grpc::CompletionQueue*                      queue) {
            const auto remaining = storage_request.transfer_task->remainingTimeout();
            if (!remaining) {
                throw StorageDeadlineExceeded("KVCM task expired before rank submission");
            }
            context->set_deadline(std::chrono::system_clock::now() + *remaining);
            return stub->AsyncExecuteFunction(context.get(), request, queue);
        };
        source_may_be_touched = true;
        auto result = broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
            requests, timeout_ms, rpc_call, storage_request.request_id, direction, /*fatal_on_error=*/false);
        if (result == nullptr) {
            // BroadcastManager returns no result only during validation or
            // connection preparation, before invoking any rank's RPC call.
            // Exceptions after dispatch begins remain conservatively UNKNOWN.
            source_may_be_touched = false;
            std::rethrow_exception(buffer_safe_failure_);
        }
        return result;
    }

    std::vector<FunctionResponsePB>
    collectResponses(const std::shared_ptr<PayloadResult>& result, bool& source_may_be_touched, bool& sdk_failed, const std::string& read_trace) {
        auto       responses = result->responses();
        const bool success   = result->success();
        const bool track_workers = !read_trace.empty() && responses.size() == broadcast_manager_->workerNum();
        std::vector<size_t> pending;
        for (size_t rank = 0; rank < responses.size(); ++rank) {
            const auto& response = responses[rank];
            const bool acknowledged = (success || result->rpcSucceeded(rank)) && response.has_remote_response();
            if (acknowledged) {
                const auto& payload = response.remote_response();
                sdk_failed = sdk_failed || payload.sdk_failed();
            }
            if (track_workers) {
                const auto& payload = response.remote_response();
                if (!acknowledged || !RemoteTransferStatusPB_IsValid(payload.transfer_status())
                    || payload.transfer_status() == REMOTE_TRANSFER_STATUS_UNSPECIFIED || !payload.payload_buffer_safe())
                    pending.push_back(rank);
            }
        }
        if (track_workers) {
            source_may_be_touched = !pending.empty();
            if (!pending.empty())
                reclaimer_->freezeTargetBlocksWithTraceIDAndStillIOWorkerID(read_trace, pending);
        }
        if (!success) {
            if (result->deadlineExceeded()) {
                throw StorageOperationTimeout("KVCM broadcast timed out: " + result->failureSummary(),
                                              StorageQuarantineReason::RPC_TIMEOUT);
            }
            throw StorageOperationCompletionUnknown("KVCM broadcast completion is unknown: " + result->failureSummary(),
                                                    StorageQuarantineReason::RPC_ERROR);
        }
        return validateResponses(std::move(responses), broadcast_manager_->workerNum(), source_may_be_touched);
    }

    std::vector<FunctionResponsePB>
    validateResponses(std::vector<FunctionResponsePB> responses, size_t expected, bool& source_may_be_touched) {
        if (responses.size() != expected) {
            throw StorageOperationCompletionUnknown("KVCM worker response count mismatch",
                                                    StorageQuarantineReason::INVALID_RESPONSE);
        }
        kvcm::WorkerPayloadProof proof;
        bool                     any_timeout = false;
        for (const auto& response : responses) {
            if (!response.has_remote_response()) {
                proof.observe(false, false, false);
                continue;
            }
            const auto& payload = response.remote_response();
            any_timeout         = any_timeout || payload.transfer_status() == REMOTE_TRANSFER_STATUS_TIMEOUT;
            proof.observe(payload.payload_buffer_safe(),
                          payload.transfer_status() == REMOTE_TRANSFER_STATUS_SUCCESS,
                          payload.transfer_status() == REMOTE_TRANSFER_STATUS_FAILED
                              || payload.transfer_status() == REMOTE_TRANSFER_STATUS_TIMEOUT);
        }
        const auto disposition = proof.result();
        if (disposition != kvcm::WorkerPayloadProof::Result::UNKNOWN) {
            source_may_be_touched = false;
        }
        if (const auto failure = proof.failure(
                any_timeout, buffer_safe_failure_, completion_unknown_failure_, completion_timeout_failure_)) {
            std::rethrow_exception(failure);
        }
        return responses;
    }

    std::vector<FunctionResponsePB> dispatchRequests(std::vector<FunctionRequestPB>& requests,
                                                     const StorageRequest&           storage_request,
                                                     bool&                           source_may_be_touched,
                                                     bool&                           sdk_failed,
                                                     ReadExecutionPtr                execution   = nullptr,
                                                     const std::string&              read_trace = {}) {
        if (broadcast_manager_) {
            auto result = dispatchRemoteRequests(requests, storage_request, source_may_be_touched);
            (void)result->waitDone(/*timeout_ms=*/0, /*fatal_on_error=*/false);
            return collectResponses(result, source_may_be_touched, sdk_failed, read_trace);
        }
        block_tree_cache_detail::ScopeRollback unsubmitted([&] {
            if (execution)
                execution->proof->report(true);
        });
        RTP_LLM_CHECK_WITH_INFO(
            requests.size() == 1, "KVCM local transfer requires exactly one request, got %zu", requests.size());
        std::vector<FunctionResponsePB> responses(1);
        {
            auto& local_response = *responses.front().mutable_remote_response();
            if (requests.front().remote_request().op() == REMOTE_OPERATION_READ && !execution) {
                execution = beginReadExecution(requests.front().remote_request(), local_response);
                if (!execution) {
                    return validateResponses(std::move(responses), 1, source_may_be_touched);
                }
            }
            const int timeout_ms = remainingTimeout(storage_request, "before payload dispatch");
            requests.front().mutable_remote_request()->set_timeout_ms(timeout_ms);
            block_tree_cache_detail::ScopeRollback collect_sdk_failure([&] {
                sdk_failed = sdk_failed || local_response.sdk_failed();
            });
            unsubmitted.dismiss();
            executePayload(requests.front().remote_request(),
                           local_response,
                           *storage_request.transfer_task,
                           &source_may_be_touched,
                           std::move(execution));
        }
        if (!read_trace.empty())
            source_may_be_touched = !responses.front().remote_response().payload_buffer_safe();
        return validateResponses(std::move(responses), 1, source_may_be_touched);
    }

    void setCudaDevice() const {
        const int expected = static_cast<int>(parallelism_config_.local_rank);
        int       current  = -1;
        check_cuda_value(cudaGetDevice(&current));
        if (current != expected) {
            check_cuda_value(cudaSetDevice(expected));
        }
    }

private:
    const KVCMStorageBackend&                      backend_;
    std::shared_ptr<BlockTreeReclaimer>            reclaimer_;
    CacheConfig                                    cache_config_;
    KVCacheConfig                                  kv_cache_config_;
    RuntimeConfig                                  runtime_config_;
    ParallelismConfig                              parallelism_config_;
    SpeculativeExecutionConfig                     sp_config_;
    std::shared_ptr<BroadcastManager>              broadcast_manager_;
    std::atomic<bool>                              payload_stopped_{false};
    std::mutex                                           recovery_discard_mutex_;
    std::unordered_set<std::string>                      pending_recovery_discards_;
    std::unique_ptr<BlockTreeTaskPool>             local_payload_pool_;
    std::unique_ptr<BlockTreeTaskPool>             rank_write_pool_;
    const std::exception_ptr                       buffer_safe_failure_ = std::make_exception_ptr(
        StorageOperationBufferSafeFailure("KVCM payload was not submitted or completed unsuccessfully"));
    const std::exception_ptr completion_unknown_failure_ = std::make_exception_ptr(
        StorageOperationCompletionUnknown("KVCM payload completion is unknown", StorageQuarantineReason::SDK_ERROR));
    const std::exception_ptr completion_timeout_failure_ = std::make_exception_ptr(
        StorageOperationTimeout("KVCM payload timed out", StorageQuarantineReason::SDK_TIMEOUT));
    std::shared_ptr<kmonitor::MetricsReporter>          metrics_reporter_;
    std::unique_ptr<kvcm::GroupPolicy>                  group_policy_;
    std::shared_ptr<kvcm::ClientWrapper>                client_wrapper_;
    std::optional<ControlConfig>                        control_config_;
    const bool                                          gdr_enabled_;
    std::vector<std::string>                            registration_tags_;
    std::unordered_map<std::string, DeviceBlockPoolPtr> pools_by_tag_;
    // Preserve KVCM's operation-local, one-based request
    // order. Abort and finish share a sequence because both call FinishWrite.
    std::atomic<uint64_t> match_trace_sequence_{1};
    std::atomic<uint64_t> read_trace_sequence_{1};
    std::atomic<uint64_t> write_trace_sequence_{1};
    std::atomic<uint64_t> finish_write_trace_sequence_{1};
    std::shared_ptr<const CacheConfig> storage_config_;
    bool                  has_swa_            = false;
    bool                  merge_full_specs_   = false;
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
    impl_(std::make_shared<Impl>(*this,
                                 cache_config,
                                 kv_cache_config,
                                 runtime_config,
                                 parallelism_config,
                                 sp_config,
                                 std::move(broadcast_manager),
                                 std::move(client_wrapper),
                                 std::move(metrics_reporter),
                                 gdr_enabled)) {}

KVCMStorageBackend::KVCMStorageBackend(const CacheConfig&                         cache_config,
                                       const KVCacheConfig&                       kv_cache_config,
                                       const RuntimeConfig&                       runtime_config,
                                       const ParallelismConfig&                   parallelism_config,
                                       const SpeculativeExecutionConfig&          sp_config,
                                       std::shared_ptr<BroadcastManager>          broadcast_manager,
                                       std::shared_ptr<kvcm::ClientWrapper>       client_wrapper,
                                       std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter):
    KVCMStorageBackend(cache_config,
                       kv_cache_config,
                       runtime_config,
                       parallelism_config,
                       sp_config,
                       std::move(broadcast_manager),
                       /*gdr_enabled=*/true,
                       std::move(metrics_reporter),
                       std::move(client_wrapper)) {}

KVCMStorageBackend::~KVCMStorageBackend() = default;

BlockTreeQueueSizes KVCMStorageBackend::queueSizes() const {
    auto       sizes = StorageBackend::queueSizes();
    const auto local = impl_->localPayloadQueueSizes();
    sizes.load += local.load;
    sizes.background += local.background;
    return sizes;
}

std::optional<KVCMStorageBackend::ControlConfig> KVCMStorageBackend::controlConfig() const {
    return impl_->controlConfig();
}

bool KVCMStorageBackend::initImpl() {
    // KVCM's protocol metadata is owned by the CP-adjusted CacheConfig the
    // factory passed to StorageBackend, not by the transitional legacy topology.
    return impl_->init(
        std::make_shared<CacheConfig>(cacheConfig()),
        [this](int layer_ordinal, const std::string& tag, int block_id) {
            return convertIndexToBuffer(layer_ordinal, tag, block_id);
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

bool KVCMStorageBackend::readAsyncImpl(const StorageRequest&                           request,
                                       const std::shared_ptr<StorageBackendMatchMeta>& match_meta,
                                       ReadCompletion                                  done) {
    return impl_->readAsync(request, match_meta, std::move(done));
}

bool KVCMStorageBackend::usesAsyncPayload(const StorageRequest& request, bool is_write) const noexcept {
    return impl_->usesAsyncPayload(request, is_write);
}

void KVCMStorageBackend::writeImpl(const StorageRequest& request) {
    impl_->write(request);
}

bool KVCMStorageBackend::startWriteImpl(const StorageRequest& request,
                                        WriteCompletion       done,
                                        WritePinFilter        retain_pins) {
    return impl_->write(request, std::move(done), std::move(retain_pins));
}

void KVCMStorageBackend::shutdownImpl() noexcept {
    impl_->shutdown();
}

void KVCMStorageBackend::onRemoteCacheSkipped(const char* stage) noexcept {
    impl_->reportRemoteCacheSkipped(stage);
}

bool KVCMStorageBackend::execute(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response) {
    Impl::ReadExecutionPtr execution;
    if (request.op() == REMOTE_OPERATION_READ) {
        try {
            execution = impl_->beginReadExecution(request, response);
        } catch (...) {
            // Admission has not completed. Preserve the conservative response
            // without manufacturing a proof for an existing execution.
            RTP_LLM_LOG_WARNING("KVCM READ registry admission failed trace=%s", request.trace_id().c_str());
            return false;
        }
        if (!execution)
            return false;
    }
    block_tree_cache_detail::ScopeRollback unsubmitted([&] {
        if (execution) {
            response.set_payload_buffer_safe(true);
            execution->proof->report(true);
        }
    });
    if (request.op() != REMOTE_OPERATION_READ && request.op() != REMOTE_OPERATION_WRITE) {
        return impl_->execute(request, response);
    }
    try {
        unsubmitted.dismiss();
        return impl_->execute(request, response, std::move(execution));
    } catch (...) {
        // Preserve an SDK failure already recorded before the payload threw.
        response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
        return false;
    }
}

void KVCMStorageBackend::setReclaimer(std::shared_ptr<BlockTreeReclaimer> reclaimer) {
    impl_->setReclaimer(std::move(reclaimer));
}

bool KVCMStorageBackend::executeRemoteRequest(const RemoteOperationRequestPB& request, FunctionResponsePB& response) {
    const bool success = execute(request, *response.mutable_remote_response());
    return request.op() == REMOTE_OPERATION_READ || request.op() == REMOTE_OPERATION_WRITE || success;
}

}  // namespace rtp_llm
