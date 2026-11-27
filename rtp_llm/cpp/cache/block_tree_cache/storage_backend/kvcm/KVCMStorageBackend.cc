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
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/PayloadDispatchProof.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/PublicationSettlement.h"
#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif
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

    BlockTreeReclaimer* writeSourceReclaimer() const noexcept { return reclaimer_.get(); }

    std::optional<ControlConfig> controlConfig() const {
        return control_config_;
    }

    bool init(const CacheTopology&                                                topology,
              StorageBackend::BufferResolver                                      buffer_resolver,
              const std::function<const DeviceBlockPoolPtr&(const std::string&)>& pool_resolver) {
        if (!reclaimer_) {
            RTP_LLM_LOG_ERROR("BlockTree KVCM storage backend requires a reclaimer");
            return false;
        }
        RTP_LLM_LOG_INFO("start init BlockTree KVCM storage backend");
        kvcm::ClientWrapper::ConfigMap client_config_map;
        kvcm::KVCMConfigPtr publication_config;
        kvcm::KVCMConfigPtr custom_config;
        try {
            if (!kv_cache_config_.kvcm_client_config.empty()) {
                autil::legacy::FromJsonString(client_config_map, kv_cache_config_.kvcm_client_config);
                if (parallelism_config_.role_type == RoleType::DECODE && client_config_map.size() > 1) {
                    auto projected = client_config_map.extract("decode_publication");
                    if (remoteCacheDecodePublicationEnabled() && !projected.empty()) {
                        publication_config = std::move(projected.mapped());
                        if (!publication_config) {
                            RTP_LLM_LOG_ERROR("KVCM_CLIENT_CONFIG has no Decode publication instance config");
                            return false;
                        }
                    }
                }
                if (client_config_map.size() != 1 || !client_config_map.begin()->second) {
                    RTP_LLM_LOG_ERROR("KVCM_CLIENT_CONFIG requires a default instance config");
                    return false;
                }
                auto default_config = client_config_map.extract(client_config_map.begin());
                default_config.key().clear();
                custom_config = default_config.mapped();
                client_config_map.insert(std::move(default_config));
            }
        } catch (const autil::legacy::ExceptionBase& error) {
            RTP_LLM_LOG_ERROR("parse KVCM_CLIENT_CONFIG failed: %s", error.what());
            return false;
        }
        if (!configureDecodePublication(topology, buffer_resolver, pool_resolver,
                                        publication_config ? publication_config : custom_config))
            return false;
        if (parallelism_config_.tp_rank == 0 && parallelism_config_.tp_size > 1 && !broadcast_manager_) {
            RTP_LLM_LOG_ERROR("BlockTree KVCM rank 0 requires a broadcast manager for tp_size=%ld",
                              parallelism_config_.tp_size);
            return false;
        }
        if (kv_cache_config_.kvcm_min_replica_count < 0
            || (kv_cache_config_.kvcm_read_backend_type != 0
                && !isPayloadBackend(kv_cache_config_.kvcm_read_backend_type))) {
            RTP_LLM_LOG_ERROR("invalid KVCM read backend or minimum replica count");
            return false;
        }
        std::vector<std::string> full_group_tags;
        std::vector<std::string> other_group_tags;
        // CacheConfig::blockSizeBytesForGroup resolves MTP child-owned physical
        // strides; do not replace this with topology-derived geometry.
        std::unordered_map<std::string, size_t> group_block_size_bytes;
        std::unordered_map<std::string, size_t> projected_block_size_bytes;
        const std::vector<GroupBase>&           groups = topology.groups();
        topology_                                      = &topology;
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
            if (decode_publication_enabled_) {
                projected_block_size_bytes.emplace(group.tag, publicationWireBytes(group.tag));
            }
        }
        const auto make_policy = [&](auto sizes) -> std::unique_ptr<kvcm::GroupPolicy> {
            if (other_group_tags.empty()) {
                return std::make_unique<kvcm::FullLayerGroupPolicy>(
                    topology, buffer_resolver, full_group_tags, other_group_tags, std::move(sizes));
            }
            return std::make_unique<kvcm::FullLinearLayerGroupPolicy>(
                topology, buffer_resolver, full_group_tags, other_group_tags, std::move(sizes));
        };
        group_policy_ = make_policy(std::move(group_block_size_bytes));
        if (decode_publication_enabled_) {
            publication_group_policy_ = make_policy(std::move(projected_block_size_bytes));
        }
        if (!group_policy_->init() || (publication_group_policy_ && !publication_group_policy_->init())) {
            RTP_LLM_LOG_ERROR("BlockTree KVCM group policy init failed");
            return false;
        }
        try {
            if (custom_config && publication_group_policy_ && !publication_config
                && target_prefill_tp_ != parallelism_config_.tp_size) {
                if (configurePublicationLayout(*custom_config, full_group_tags)) {
                    // A single projected config was accepted on the development
                    // line. Keep it for writing and derive the native read config
                    // using the same manager and SDK settings.
                    publication_config = custom_config;
                    client_config_map.at("") = genClientConfig(full_group_tags, *group_policy_,
                        parallelism_config_.tp_size, parallelism_config_.dp_size, merge_full_specs_, custom_config);
                    custom_config.reset();
                }
            }
            if (custom_config) {
                merge_full_specs_ = custom_config->hasLocationSpec("F_TP0");
                if (merge_full_specs_ && !canMergeFullSpecs(full_group_tags, custom_config->sdk_wrapper_config())) {
                    RTP_LLM_LOG_ERROR("KVCM_CLIENT_CONFIG has an incompatible merged FULL layout");
                    return false;
                }
                auto [location_infos, location_groups] =
                    genLocationSpecs(*group_policy_, parallelism_config_.tp_size, merge_full_specs_);
                if (publication_group_policy_ && !matchesPublicationSizes(*custom_config, *location_infos)) {
                    RTP_LLM_LOG_ERROR("KVCM_CLIENT_CONFIG has incompatible native Decode sizes");
                    return false;
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
                            kvcm::genLocationSpecName(rank, group.group_name, runtime_config_.model_name, merge_full_specs_);
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
            } else if (client_config_map.empty()) {
                client_config_map.emplace("", genClientConfig(full_group_tags, *group_policy_,
                    parallelism_config_.tp_size, parallelism_config_.dp_size, merge_full_specs_));
            }
            if (publication_group_policy_) {
                // The P topology belongs only to the end-of-request publication.
                // Match/read/control still use the Decode rank's original namespace.
                if (publication_config) {
                    if (!configurePublicationLayout(*publication_config, full_group_tags)) {
                        RTP_LLM_LOG_ERROR("KVCM_CLIENT_CONFIG Decode publication specs do not match target Prefill");
                        return false;
                    }
                } else if (custom_config && target_prefill_tp_ == parallelism_config_.tp_size) {
                    publication_config = client_config_map.at("");
                    if (!configurePublicationLayout(*publication_config, full_group_tags)) {
                        RTP_LLM_LOG_ERROR("KVCM_CLIENT_CONFIG has incompatible Decode publication specs");
                        return false;
                    }
                } else {
                    if (custom_config) {
                        RTP_LLM_LOG_ERROR("KVCM_CLIENT_CONFIG requires decode_publication specs when the P/D layouts differ");
                        return false;
                    }
                    publication_config = genClientConfig(full_group_tags, *publication_group_policy_,
                                                         target_prefill_tp_, /*wire_dp=*/1, publication_merge_full_specs_);
                }
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
        if (publication_config) {
            publication_config->set_default_query_type(default_query_type_);
        }
        if (publication_config) {
            publication_location_groups_ = *publication_config->locationSpecGroups();
            for (auto& [name, specs] : publication_location_groups_) {
                (void)name;
                std::sort(specs.begin(), specs.end());
            }
        }
        if (publication_config && publication_config->instance_id() == config->instance_id()
            && publication_config->instance_group() == config->instance_group()
            && publication_config->registerInstanceRequest() != config->registerInstanceRequest()) {
            RTP_LLM_LOG_ERROR("KVCM Decode publication cannot change specs in the native read namespace");
            return false;
        }
        if (publication_config
            && publicationBackendType(publication_config->sdk_wrapper_config())
                   != publicationBackendType(config->sdk_wrapper_config())) {
            RTP_LLM_LOG_ERROR("KVCM Decode publication requires the native payload client's PACE backend");
            return false;
        }
        const bool separate_publication_metadata =
            publication_config && autil::legacy::ToJsonString(publication_config) != autil::legacy::ToJsonString(config);
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
        if (separate_publication_metadata) {
            projected_metadata_client_ = client_wrapper_->createMetadataOnlyClient(publication_config);
            if (!projected_metadata_client_) {
                client_wrapper_->shutdown();
                RTP_LLM_LOG_ERROR("create Decode publication metadata client failed");
                return false;
            }
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
        } else if (positional && request.local_matched_blocks_num > 0) {
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
                               tokens = remote_blocks * topology_->groups().front().cacheKeyTokenStride()]() mutable {
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
            static_cast<int64_t>(remote_blocks * topology_->groups().front().cacheKeyTokenStride());
        collector.remote_read_fail_qps = false;
        reclaimer_->unregisterTask(read_trace);
    }

    struct PayloadBuffers {
        std::vector<std::string> tags;
        std::vector<int32_t> blocks;
        kv_cache_manager::UriStrVec uris;
        kv_cache_manager::BlockBuffers buffers;
    };

    struct WritePublication {
        explicit WritePublication(bool decode_write) {
            if (decode_write)
                dispatch_proof.emplace();
        }

        kv_cache_manager::WriteLocation location;
        // Pointers refer into location.locations, which stays in this object through settlement.
        ActualUriGather                        actual_uris;
        RtpLLMRemoteCacheWriteMetricsCollector collector;
        bool                                   finish_attempted{false};
        bool                                   settlement_started{false};
        bool                                   start_acknowledged{false};
        std::mutex                             settlement_mutex;
        std::optional<kvcm::PayloadDispatchProof> dispatch_proof;

        const std::string& sessionId() const noexcept {
            return location.write_session_id;
        }
    };

    bool allowsWrite(const StorageRequest& request) const noexcept {
        return kvcm::allowsPublication(decode_publication_enabled_, is_decode_role_, static_cast<bool>(request.decode_write));
    }

    bool write(const StorageRequest& request, WriteCompletion done = {}, WritePinFilter retain_pins = {}) {
        if (!allowsWrite(request)) {
            return false;
        }
        if (payload_stopped_.load()) {
            std::rethrow_exception(buffer_safe_failure_);
        }
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
            if (decode_publication_enabled_)
                throw;
            std::rethrow_exception(
                withTargetProtection(std::current_exception(), StorageTargetProtection::NO_LOCAL_TARGETS));
        }
    }

    bool writeTransaction(const StorageRequest& request, bool& source_may_be_touched, WriteCompletion done,
                          WritePinFilter retain_pins) {
        if (!backend_.remoteCacheEnabled()) {
            reportRemoteCacheSkipped("write");
            return false;
        }
        // Preserve ordinary Prefill publication. D publishes through the EOS
        // transaction only; no benchmark flag may disable P's normal cache path.
        if (!allowsWrite(request))
            return false;
        RTP_LLM_CHECK_WITH_INFO(parallelism_config_.tp_rank == 0,
                                "KVCM metadata write must run on tp rank 0, got %ld",
                                parallelism_config_.tp_rank);
        RTP_LLM_CHECK(request.keys != nullptr && request.keys->size() == request.handles.size());
        const size_t valid_keys_size = request.keys->size();
        if (valid_keys_size == 0) {
            return false;
        }
        const bool defer = static_cast<bool>(done);
        // Public D publication always supplies completion. The required legacy
        // synchronous adapter is Prefill-only; reject before opening a session.
        if (!defer && request.decode_write)
            throw StorageOperationBufferSafeFailure("synchronous Decode publication requires completion");
        auto async_write = defer ? std::make_shared<WritePublication>(static_cast<bool>(request.decode_write)) : nullptr;
        std::optional<WritePublication> local;
        if (!defer) local.emplace(false);
        auto&      publication = defer ? *async_write : *local;
        auto&      collector   = publication.collector;
        bool       sdk_failed  = false;
        const auto start       = Clock::now();
        block_tree_cache_detail::ScopeRollback report([&] {
            collector.remote_write_task_cost_time_us = elapsedTimeUs(start);
            reportRemoteMetrics<RtpLLMRemoteCacheWriteMetrics>(collector);
            if (collector.remote_write_fail_qps && sdk_failed)
                reportSdkFailure("write");
        });
        CacheKeysType                          keys(request.keys->begin(), request.keys->begin() + valid_keys_size);
        std::vector<std::string>               location_spec_group_names;
        const auto& write_policy = request.decode_write ? *publication_group_policy_ : *group_policy_;
        RTP_LLM_CHECK_WITH_INFO(write_policy.getNeedWriteGroups(request, valid_keys_size, location_spec_group_names),
                                "KVCM write group selection failed");
        const std::string trace_id     = nextTraceId("write", write_trace_sequence_);
        (void)remainingTimeout(request, "before StartWrite");
        auto [success, allocation] = [&] {
            const auto                             stage_start = Clock::now();
            block_tree_cache_detail::ScopeRollback timer(
                [&] { collector.remote_get_write_location_time_us = elapsedTimeUs(stage_start); });
            return writeMetadataClient(request).getWriteLocation("",
                                                     trace_id,
                                                     keys,
                                                     /*tokens=*/{},
                                                     location_spec_group_names,
                                                      request.decode_write ? 40 : 300,
                                                      kv_cache_config_.kvcm_min_replica_count);
        }();
        RTP_LLM_CHECK_WITH_INFO(success, "KVCM StartWrite failed");
        publication.start_acknowledged = true;
        auto& write_location = publication.location;
        write_location = std::move(allocation);
        const std::string& write_session_id = publication.sessionId();
        static const kv_cache_manager::Locations empty_locations;
        const auto check_write_admission = [&] {
            if (!backend_.remoteCacheEnabled()) {
                reportRemoteCacheSkipped("write");
                collector.remote_write_qps      = false;
                collector.remote_write_fail_qps = false;
                // No worker can touch this allocation yet. The existing catch
                // closes it with a zero success mask, without quarantining pins.
                throw StorageOperationBufferSafeFailure("remote cache disabled by KVCM before write submission");
            }
        };
        try {
            (void)remainingTimeout(request, "after StartWrite");
            check_write_admission();
            std::vector<FunctionRequestPB> requests(
                static_cast<size_t>(request.decode_write ? target_prefill_tp_ : parallelism_config_.tp_size));
            auto& actual_uri_gather = publication.actual_uris;
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
                if (!write_session_id.empty()) {
                    // The server can create an empty, short-lived session when
                    // every key already has enough replicas. Close it normally.
                    (void)remainingTimeout(request, "before empty FinishWrite");
                    const auto stage_start = Clock::now();
                    block_tree_cache_detail::ScopeRollback timer([&] {
                        collector.remote_finish_write_time_us += elapsedTimeUs(stage_start);
                    });
                    try {
                        if (!finishSession(request, publication, kv_cache_manager::BlockMaskOffset{0}, empty_locations, true)) {
                            RTP_LLM_LOG_WARNING("KVCM failed to close empty write session [%s]",
                                                write_session_id.c_str());
                        }
                    } catch (...) {
                        RTP_LLM_LOG_WARNING("KVCM failed to close empty write session [%s]", write_session_id.c_str());
                    }
                }
                // Keep the legacy task QPS, but report no payload blocks or
                // transfer time when all keys already have enough replicas.
                collector.remote_write_fail_qps = false;
                return false;
            }
            const bool merge_full = request.decode_write ? publication_merge_full_specs_ : merge_full_specs_;
            const auto& spec_info = write_policy.spec_info_map();
            for (size_t location_idx = 0; location_idx < write_location.locations.size(); ++location_idx) {
                const size_t key_idx = key_indices[location_idx];
                auto& location = write_location.locations[location_idx];
                if (request.decode_write) {
                    // Single FULL uses the Manager's implicit default group.
                    const auto& expected = location_spec_group_names.empty() ? write_policy.allLocationSpecs() :
                        publication_location_groups_.at(location_spec_group_names.at(key_idx));
                    RTP_LLM_CHECK_WITH_INFO(write_policy.validateWriteLocation(location, expected),
                                            "KVCM Decode write received invalid location specs at key=%zu", key_idx);
                }
                for (auto& location_spec : location) {
                    const auto info = spec_info.find(location_spec.spec_name);
                    RTP_LLM_CHECK_WITH_INFO(
                        info != spec_info.end(), "KVCM write has unknown spec [%s]", location_spec.spec_name.c_str());
                    const size_t rank   = static_cast<size_t>(info->second.tp_rank);
                    auto*        remote = requests.at(rank).mutable_remote_request();
                    remote->add_uris(location_spec.uri);

                    if (merge_full)
                        remote->add_uri_block_counts(info->second.group_ids.size());
                    for (const auto group_id : info->second.group_ids) {
                        const auto&               group  = write_policy.groups().at(group_id);
                        const StorageBlockHandle* handle = findHandle(request.handles[key_idx], group.tag);
                        RTP_LLM_CHECK_WITH_INFO(handle != nullptr,
                                                "KVCM write has no source handle for key=%zu tag=%s",
                                                key_idx,
                                                group.tag.c_str());
                        remote->add_group_tags(group.tag);
                        auto block = handle->block;
                        if (const auto& decode = request.decode_write) {
                            const auto found = decode->peer_blocks.find(group.tag);
                            if (found != decode->peer_blocks.end() && !found->second.at(key_idx).empty())
                                block = found->second.at(key_idx).at(rank);
                        }
                        remote->add_block_ids(block);
                    }
                    actual_uri_gather[rank].push_back(&location_spec);
                }
            }

            const auto                             broadcast_start = Clock::now();
            block_tree_cache_detail::ScopeRollback broadcast_timer(
                [&] { collector.remote_write_broadcast_time_us = elapsedTimeUs(broadcast_start); });
            check_write_admission();
            if (!request.decode_write)
                (void)remainingTimeout(request, "before write dispatch");

            if (defer) {
                // From this point a payload may be accepted before the submitter
                // returns. Only completion can prove source safety afterwards.
                auto prepare_completion = [&](auto collect) {
                    WriteResult finish = [this, async_write, collect = std::move(collect), &request, start](
                                             bool on_worker) {
                        auto& write = *async_write;
                        std::lock_guard<std::mutex> settlement_lock(write.settlement_mutex);
                        if (write.settlement_started)
                            return;
                        write.settlement_started = true;
                        bool  sdk_failed = false;
                        block_tree_cache_detail::ScopeRollback report([&] {
                            write.collector.remote_write_task_cost_time_us = elapsedTimeUs(start);
                            reportRemoteMetrics<RtpLLMRemoteCacheWriteMetrics>(write.collector);
                            if (write.collector.remote_write_fail_qps && sdk_failed) reportSdkFailure("write");
                        });
                        bool touched = true;
                        try {
                            auto responses = collect(touched, sdk_failed);
                            // Only the business worker may publish or abort the write.
                            if (!on_worker)
                                std::rethrow_exception(buffer_safe_failure_);
                            publishWrite(request, write, responses);
                        } catch (...) {
                            if (on_worker) {
                                // Starts true above; only the completed collect
                                // can establish the strict all-peer safe proof.
                                // A failure in metadata cleanup must not
                                // hide the payload's buffer-safety disposition.
                                try { settleWriteFailure(request, write, touched, !touched); } catch (...) {}
                            }
                            // Off-worker fallback never performs metadata I/O.
                            // The original Manager/SDK owns the unclosed session.
                            rethrowWritePayloadFailure(touched);
                        }
                    };
                    return [async_write, broadcast_start, done = std::move(done), finish = std::move(finish)]() mutable {
                        async_write->collector.remote_write_broadcast_time_us = elapsedTimeUs(broadcast_start);
                        done(std::move(finish));
                    };
                };
                if (request.decode_write) {
                    dispatchLocalPeers(std::move(requests), request, source_may_be_touched, prepare_completion,
                        std::shared_ptr<kvcm::PayloadDispatchProof>(async_write, &*publication.dispatch_proof));
                } else {
                    dispatchPayloadAsync(requests, request, source_may_be_touched, prepare_completion);
                }
                broadcast_timer.dismiss();
                report.dismiss();
                return true;
            }

            auto responses = dispatchRequests(requests, request, source_may_be_touched, sdk_failed);
            source_may_be_touched = false;
            broadcast_timer.run();
            publishWrite(request, publication, responses);
        } catch (...) {
            std::lock_guard<std::mutex> settlement_lock(publication.settlement_mutex);
            if (!publication.settlement_started) {
                settleWriteFailure(request, publication, source_may_be_touched);
            }
            throw;
        }
        return false;
    }

    void settleWriteFailure(const StorageRequest& request, WritePublication& write,
                            bool source_may_be_touched, bool payload_settled_safe = false) {
        // PACE's explicit SAFE failures retain no destination writer; raw
        // errors, incomplete replies and exceptions never supply this proof.
        const bool settled_safe = request.decode_write && pace_backend_ && payload_settled_safe;
        if (request.decode_write) {
            const auto action = kvcm::failedPublicationAction(
                !write.dispatch_proof->neverStarted(), write.finish_attempted, settled_safe);
            if (action != kvcm::FailedPublicationAction::ABORT_UNCOMMITTED) {
                if (!write.finish_attempted) {
                    RTP_LLM_LOG_WARNING("KVCM remote destination completion unproven; retaining pending session [%s] without abort",
                                         write.sessionId().c_str());
                    RtpLLMRemoteCacheOutcomeMetricsCollector result;
                    result.stage = "write_allocation";
                    result.outcome = "retained_unproven";
                    reportRemoteMetrics<RtpLLMRemoteCacheOutcomeMetrics>(result);
                }
                return;
            }
        } else {
            if (write.start_acknowledged && !write.finish_attempted && !write.sessionId().empty()) {
                write.finish_attempted = true;
                abortUnfinishedWrite(write.sessionId(), source_may_be_touched, write.collector);
            }
            return;
        }
        // Decode requires no submission or a terminal buffer-safety proof
        // before using the original zero-mask close.
        closeUnusedWrite(request, write, /*require_not_started=*/!settled_safe);
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
        {
            const auto                             stage_start = Clock::now();
            block_tree_cache_detail::ScopeRollback timer(
                [&] { collector.remote_finish_write_time_us += elapsedTimeUs(stage_start); });
            RTP_LLM_CHECK_WITH_INFO(
                finishSession(request, write, write_location.locations.size(), actual_locations, false),
                "KVCM FinishWrite failed");
        }
        collector.remote_write_cache_block_num = static_cast<int64_t>(write_location.locations.size());
        collector.remote_write_fail_qps        = false;
    }

    kvcm::ClientWrapper& writeMetadataClient(const StorageRequest& request) const {
        return request.decode_write && projected_metadata_client_ ? *projected_metadata_client_ : *client_wrapper_;
    }

    bool finishSession(const StorageRequest& request, WritePublication& write,
                       const kv_cache_manager::BlockMask& mask,
                       const kv_cache_manager::Locations& locations, bool unused_close) {
        const auto trace = nextTraceId(unused_close ? "close_write" : "finish_write", finish_write_trace_sequence_);
        RTP_LLM_CHECK_WITH_INFO(write.start_acknowledged && !write.finish_attempted,
                                "KVCM metadata action already settled");
        // Anchor before the call, including exceptions. A cleanup may retry
        // only a locally proved NOT_SENT result; a commit never retries here.
        // Keep the anchor set across retries so no later abort can replay it.
        write.finish_attempted = true;
        if (!request.decode_write) {
            return client_wrapper_->finishWrite("", trace, write.sessionId(), mask, locations);
        }
        using Result = kvcm::ClientWrapper::FinishWriteResult;
        constexpr int max_cleanup_attempts = 3;
        for (int attempt = 0; ; ++attempt) {
            auto& metadata = writeMetadataClient(request);
            const auto result = metadata.finishWriteWithResult("", trace, write.sessionId(), mask, locations);
            if (result == Result::ACKNOWLEDGED) return true;
            if (!unused_close || result != Result::NOT_SENT || attempt + 1 >= max_cleanup_attempts
                || !request.transfer_task || request.transfer_task->expired()) return false;
            const auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                request.transfer_task->deadline() - Clock::now()).count();
            if (!metadata.waitForRetry(static_cast<int>(std::clamp<int64_t>(remaining_ms, 0, 20)))
                || request.transfer_task->expired()) return false;
        }
    }

    void closeUnusedWrite(const StorageRequest& request, WritePublication& write, bool require_not_started = true) {
        if (!write.start_acknowledged || write.finish_attempted
            || (require_not_started && !write.dispatch_proof->neverStarted())) {
            return;
        }
        const auto                             stage_start = Clock::now();
        block_tree_cache_detail::ScopeRollback timer(
            [&] { write.collector.remote_finish_write_time_us += elapsedTimeUs(stage_start); });
        try {
            const bool aborted = finishSession(request, write, kv_cache_manager::BlockMaskOffset{0}, {}, true);
            if (!aborted) {
                RTP_LLM_LOG_WARNING("KVCM failed to abort write session [%s]",
                                    write.sessionId().c_str());
            }
        } catch (...) {
            RTP_LLM_LOG_WARNING("KVCM abort write session threw, session=[%s]",
                                write.sessionId().c_str());
        }
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
                             kv_cache_manager::BlockBuffers& buffers,
                             int                             peer = -1) {
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
        const auto& policy = peer >= 0 ? *publication_group_policy_ : *group_policy_;
        if (grouped) {
            size_t      begin  = 0;
            const auto& groups = policy.groups();
            const auto& infos  = policy.spec_info_map();
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
                const auto name = kvcm::genLocationSpecName(peer < 0 ? parallelism_config_.tp_rank : peer,
                                                            group->second.group_name,
                                                            runtime_config_.model_name,
                                                            peer >= 0 ? publication_merge_full_specs_ : merge_full_specs_);
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
        if (peer < 0) {
            if (!policy.genBlockBuffers(tags, blocks, buffers)) {
                return false;
            }
        } else {
            CPSlotMapper mapper(0, target_physical_shards_, cache_config_.seq_size_per_block);
            buffers.reserve(blocks.size());
            // One immutable geometry lookup per distinct tag, not per page.
            std::unordered_map<std::string, std::pair<std::vector<int>, size_t>> geometry;
            geometry.reserve(policy.groups().size());
            for (size_t member = 0; member < blocks.size(); ++member) {
                const auto& tag = tags[member];
                auto [cached, inserted] = geometry.try_emplace(tag);
                if (inserted) {
                    cached->second.first = cache_config_.layerIdsForGroup(tag);
                    const auto& groups = policy.groups();
                    const auto group = std::find_if(groups.begin(), groups.end(), [&](const auto& entry) {
                        return entry.second.tag == tag;
                    });
                    cached->second.second = group == groups.end() ? publicationWireBytes(tag) :
                                                                 group->second.block_size_bytes;
                }
                const auto& layer_ids = cached->second.first;
                buffers.emplace_back();
                auto& iovs = buffers.back().iovs;
                iovs.reserve(layer_ids.size() * 2);
                size_t bytes = 0;
                for (int layer : layer_ids) {
                    const auto parts =
                        mapper.sliceBlockForPeer(cache_config_, tag, file_resolver_(layer, tag, blocks[member]), peer);
                    for (const auto& part : parts) {
                        if (!part.is_cuda || !part.addr || !part.size_bytes
                            || part.size_bytes > std::numeric_limits<size_t>::max() - bytes) {
                            return false;
                        }
                        iovs.push_back({kv_cache_manager::MemoryType::GPU, part.addr, part.size_bytes, false});
                        bytes += part.size_bytes;
                    }
                }
                // Reuse the registered native/MTP wire size, as the P path does.
                if (bytes != cached->second.second) {
                    RTP_LLM_LOG_WARNING("Decode wire size mismatch tag=%s", tag.c_str());
                    return false;
                }
            }
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

#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DtopCompletionBridge.inc"

    BlockTreeQueueSizes localPayloadQueueSizes() const {
        auto sizes = local_payload_pool_ ? local_payload_pool_->queueSizes() : BlockTreeQueueSizes{};
        if (rank_write_pool_)
            sizes.background += rank_write_pool_->queueSizes().background;
        return sizes;
    }

    void shutdown() noexcept {
        payload_stopped_.store(true, std::memory_order_release);
        if (publication_budget_)
            publication_budget_->stopAdmission();
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
        if (projected_metadata_client_) {
            projected_metadata_client_->shutdown();
        }
        if (client_wrapper_) {
            client_wrapper_->shutdown();
        }
    }

private:
    int resolveQueryType(int query_type) const {
        return query_type == 0 ? default_query_type_ : query_type;
    }

    static kv_cache_manager::Locations selectBackendLocations(const kv_cache_manager::BackendLocations& result) {
        kv_cache_manager::Locations locations;
        locations.reserve(result.size());
        for (const auto& key_locations : result) {
            RTP_LLM_CHECK_WITH_INFO(key_locations.size() <= 1, "KVCM returned multiple locations for one backend");
            locations.push_back(key_locations.empty() ? kv_cache_manager::Location{} :
                                                        key_locations.front().location_specs);
        }
        return locations;
    }

    static bool isPayloadBackend(int type) {
        return type == 1 || type == 2 || type == 3 || type == 4 || type == 5 || type == 9;
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
                const bool valid = infos.count(spec.spec_name) && present.emplace(spec.spec_name, &spec).second;
                if (is_decode_role_ && !valid) {
                    RTP_LLM_LOG_WARNING("Ignoring malformed native Decode cache location at key=%zu", i);
                    present.clear();
                    locations[i].clear();
                    break;
                }
                RTP_LLM_CHECK_WITH_INFO(valid, "KVCM returned an unknown or duplicate spec");
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
                const size_t required = std::min(i + 1 - local, topology_->group(group.tag).reuseBlockCount(i + 1));
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
                                              const auto& group = topology_->group(infos.at(spec.spec_name).tag);
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
    genLocationSpecs(kvcm::GroupPolicy& policy, int tp_size, bool merge_full) const {
        auto infos  = std::make_shared<kvcm::KVCMConfig::LocationSpecInfoMap>();
        auto groups = std::make_shared<kvcm::KVCMConfig::LocationSpecGroups>();
        RTP_LLM_CHECK_WITH_INFO(
            policy.buildLocationSpecGroups(tp_size, *groups, runtime_config_.model_name, merge_full),
            "failed to build KVCM location spec groups");
        for (const auto& [group_id, group] : policy.groups()) {
            for (int rank = 0; rank < tp_size; ++rank) {
                const std::string spec_name =
                    kvcm::genLocationSpecName(rank, group.group_name, runtime_config_.model_name, merge_full);
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

    bool matchesPublicationSizes(const kvcm::KVCMConfig& config,
                                 const kvcm::KVCMConfig::LocationSpecInfoMap& infos) const {
        if (config.blockSize() != cache_config_.seq_size_per_block) {
            return false;
        }
        const auto& supplied = config.locationSpecSizes();
        return supplied && std::all_of(infos.begin(), infos.end(), [&](const auto& spec) {
            const auto it = supplied->find(spec.first);
            return it != supplied->end() && it->second == spec.second;
        });
    }

    bool configurePublicationLayout(const kvcm::KVCMConfig& config, const std::vector<std::string>& full_group_tags) {
        const bool merge_full = config.hasLocationSpec("F_TP0");
        if (config.blockSize() != cache_config_.seq_size_per_block
            || (merge_full && !canMergeFullSpecs(full_group_tags, config.sdk_wrapper_config()))) {
            return false;
        }
        auto [infos, groups] = genLocationSpecs(*publication_group_policy_, target_prefill_tp_, merge_full);
        if (!hasExactLocationSpecs(config, *infos, *groups)) {
            return false;
        }
        publication_merge_full_specs_ = merge_full;
        return true;
    }

    static bool hasExactLocationSpecs(const kvcm::KVCMConfig& config,
                                      const kvcm::KVCMConfig::LocationSpecInfoMap& infos,
                                      const kvcm::KVCMConfig::LocationSpecGroups& groups) {
        const auto fields = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(autil::legacy::ToJson(config));
        kvcm::KVCMConfig::LocationSpecInfoMap configured_infos;
        kvcm::KVCMConfig::LocationSpecGroups configured_groups;
        autil::legacy::FromJson(configured_infos, fields.at("location_spec_infos"));
        autil::legacy::FromJson(configured_groups, fields.at("location_spec_groups"));
        if (configured_infos != infos) {
            return false;
        }
        for (const auto& [name, specs] : groups) {
            const auto it = configured_groups.find(name);
            if (it == configured_groups.end() || it->second != specs) {
                return false;
            }
        }
        for (const auto& [name, specs] : configured_groups) {
            (void)name;
            if (std::any_of(specs.begin(), specs.end(), [&](const auto& spec) { return infos.count(spec) == 0; })) {
                return false;
            }
        }
        return true;
    }

    bool canMergeFullSpecs(const std::vector<std::string>& full_group_tags,
                           const std::shared_ptr<kvcm::SdkWrapperConfig>& sdk) const {
        if (!sdk) {
            return false;
        }
        const auto& backends = sdk->sdk_backend_configs();
        const bool  legacy_dsv41_specs =
            runtime_config_.model_name == "DeepSeekV41" || runtime_config_.model_name == "DeepSeekV41DSpark";
        if (legacy_dsv41_specs && full_group_tags.size() == 3 && backends.size() == 1
            && std::dynamic_pointer_cast<kvcm::TairMempoolSdkConfig>(backends.front())) {
            const auto&           first = topology_->group(full_group_tags.front());
            std::set<std::string> names;
            bool                  compatible = true;
            for (const auto& [id, group] : group_policy_->groups()) {
                (void)id;
                if (!group.is_full)
                    continue;
                names.insert(kvcm::genLocationSpecName(0, group.group_name, runtime_config_.model_name));
                const auto& member = topology_->group(group.tag);
                compatible         = compatible && member.seqSizePerBlock() == first.seqSizePerBlock()
                             && member.cacheKeyTokenStride() == first.cacheKeyTokenStride()
                             && member.policy.cp_mapping == first.policy.cp_mapping
                             && member.policy.cp_slice == first.policy.cp_slice;
            }
            return compatible && names == std::set<std::string>{"F0_TP0", "F1_TP0", "F2_TP0"};
        }
        return false;
    }

    kvcm::KVCMConfigPtr genClientConfig(const std::vector<std::string>& full_group_tags,
                                       kvcm::GroupPolicy& policy, int wire_tp, int wire_dp, bool& merge_full,
                                       const kvcm::KVCMConfigPtr& transport_base = nullptr) {
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
        if (transport_base) {
            addresses = transport_base->addresses();
            channel = transport_base->meta_channel_config();
            sdk = transport_base->sdk_wrapper_config();
            RTP_LLM_CHECK_WITH_INFO(channel && sdk, "Decode publication requires manager and SDK configuration");
        } else {
            sdk->parseBackendConfigs(kv_cache_config_.kvcm_model_sdk_config);
        }
        merge_full = canMergeFullSpecs(full_group_tags, sdk);
        auto [location_infos, location_groups] = genLocationSpecs(policy, wire_tp, merge_full);

        const std::string model_name = runtime_config_.model_name;
        const std::string dtype      = getDataTypeStr(cache_config_.dtype);
        std::string       extra      = kv_cache_config_.kvcm_model_extra_info;
        const auto checkpoint_path = kv_cache_config_.kvcm_model_checkpoint_path.empty()
                                         ? autil::EnvUtil::getEnv("CHECKPOINT_PATH", std::string(""))
                                         : kv_cache_config_.kvcm_model_checkpoint_path;
        extra += '/' + autil::EnvUtil::getEnv("BIZ_NAME", std::string("")) + '/'
                 + std::to_string(hashString(checkpoint_path));
        std::string draft_info;
        if (!cache_config_.mtp_sub_configs.empty()) {
            draft_info = '{' + sp_config_.to_string() + '}';
        }
        const auto& instance_group = transport_base ? transport_base->instance_group() : kv_cache_config_.kvcm_instance_group;
        std::stringstream identity;
        identity << "instance_group: " << instance_group
                 << ";block_size:" << cache_config_.seq_size_per_block << ";model_name:" << model_name
                 << ";dtype_str:" << dtype << ";use_mla:" << cache_config_.use_mla
                 << ";fp8_kv_cache:" << kv_cache_config_.fp8_kv_cache << ";tp_size:" << wire_tp
                 << ";dp_size:" << wire_dp << ";extra_info:" << extra
                 << ";location_spec_info:" << autil::legacy::ToJsonString(location_infos, true)
                 << ";location_spec_groups:" << autil::legacy::ToJsonString(location_groups, true)
                 << ";default_query_type:" << default_query_type_ << ";draft_model_info:" << draft_info;
        std::string instance_id = kv_cache_config_.kvcm_instance_id_salt;
        if (!instance_id.empty()) {
            instance_id += '_';
        }
        instance_id += std::to_string(hashString(identity.str()));

        auto config =
            std::make_shared<kvcm::KVCMConfig>(
                transport_base ? transport_base->enable_vipserver() : kv_cache_config_.kvcm_enable_vipserver,
                transport_base ? transport_base->vipserver_domain() : kv_cache_config_.kvcm_vipserver_domain,
                                               static_cast<int32_t>(cache_config_.seq_size_per_block),
                                               instance_group,
                                               instance_id,
                                               addresses,
                                               location_infos,
                                               channel,
                                               sdk,
                                               location_groups,
                                               kvcm::ModelDeployment(model_name,
                                                                     dtype,
                                                                     cache_config_.use_mla,
                                                                     wire_tp,
                                                                     wire_dp,
                                                                     1,
                                                                     extra,
                                                                     kv_cache_config_.kvcm_model_user_data));
        return config;
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

    template<typename PrepareCompletion>
    void dispatchLocalPeers(std::vector<FunctionRequestPB> requests, const StorageRequest& request,
        bool& touched, PrepareCompletion prepare_completion, std::shared_ptr<kvcm::PayloadDispatchProof> proof) {
        touched = false;
        RTP_LLM_CHECK(local_payload_pool_ && proof && !requests.empty());
        using Context = PayloadResult::WorkerRpcContext;
        std::vector<std::shared_ptr<Context>> contexts;
        auto operations = std::make_shared<std::vector<PayloadBuffers>>(requests.size());
        contexts.reserve(requests.size());
        // Resolve the D layout before submitting anything. The original P
        // result owns aggregation; no separate peer latch or business cycle.
        for (size_t peer = 0; peer < requests.size(); ++peer) {
            auto context = std::make_shared<Context>();
            context->request = std::move(requests[peer]);
            const auto& remote = context->request.remote_request();
            auto& operation = operations->at(peer);
            operation.uris.assign(remote.uris().begin(), remote.uris().end());
            operation.tags.assign(remote.group_tags().begin(), remote.group_tags().end());
            operation.blocks.assign(remote.block_ids().begin(), remote.block_ids().end());
            for (const auto& uri : remote.uris()) RTP_LLM_CHECK(uri.compare(0, 7, "pace://") == 0);
            RTP_LLM_CHECK_WITH_INFO(prepareBlockBuffers(remote, operation.tags, operation.blocks,
                operation.buffers, static_cast<int>(peer)), "invalid D source layout");
            auto* response = context->response.mutable_remote_response();
            response->set_payload_buffer_safe(true);
            response->set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
            contexts.push_back(std::move(context));
        }
        auto result = std::make_shared<PayloadResult>(contexts, request.request_id, "remote_write_local");
        auto complete = prepare_completion([this, result, count = requests.size()](bool& source_touched, bool& sdk_failed) {
            auto responses = result->responses();
            for (const auto& response : responses) sdk_failed |= response.remote_response().sdk_failed();
            return validateResponses(std::move(responses), count, source_touched);
        });
        auto notify = [proof, complete = std::move(complete)]() mutable {
            proof->notStarted(); // All terminal; never downgrades a possible SDK handoff.
            complete();
        };
        std::vector<Task> run, expire;
        run.reserve(requests.size()); expire.reserve(requests.size());
        for (size_t peer = 0; peer < requests.size(); ++peer) {
            const auto context = contexts[peer];
            run.push_back([this, context, operations, result, proof, peer, task = *request.transfer_task] {
                auto& response = *context->response.mutable_remote_response();
                bool source_touched = false;
                try {
                    setCudaDevice();
                    if (!backend_.remoteCacheEnabled()) response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
                    else executeProjectedWrite(operations->at(peer), response, task, source_touched, *proof);
                } catch (...) {
                    response.set_payload_buffer_safe(!source_touched);
                    response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
                }
                result->finishLocalRank(peer);
            });
            expire.push_back([context, result, peer] {
                context->response.mutable_remote_response()->set_transfer_status(REMOTE_TRANSFER_STATUS_TIMEOUT);
                result->finishLocalRank(peer);
            });
        }
        proof->pending(); touched = true;
        for (size_t peer = 0; peer < requests.size(); ++peer) {
            bool accepted = false;
            try {
                accepted = local_payload_pool_->submit(BlockTreeTaskClass::BACKGROUND, std::move(run[peer]),
                    request.transfer_task->deadline(), std::move(expire[peer]));
            } catch (...) {} // Pool throws only before enqueue; earlier peers must still finish.
            if (!accepted) {
                if (request.transfer_task->expired())
                    contexts[peer]->response.mutable_remote_response()->set_transfer_status(REMOTE_TRANSFER_STATUS_TIMEOUT);
                result->finishLocalRank(peer);
            }
        }
        // Register only after all submit attempts. onDone also handles peers
        // that expired/completed inline, without retiring the active driver.
        onPayloadDone(result, std::move(notify));
    }

    void executeProjectedWrite(PayloadBuffers& operation, RemoteOperationResponsePB& response,
                               const TransferTask& task, bool& touched, kvcm::PayloadDispatchProof& proof) {
        if (operation.uris.empty()) {
            response.set_transfer_status(task.expired() ? REMOTE_TRANSFER_STATUS_TIMEOUT : REMOTE_TRANSFER_STATUS_SUCCESS);
            return;
        }
        bool attempted = false;
        block_tree_cache_detail::ScopeRollback reconcile([&] {
            if (attempted)
                proof.started();
            response.set_payload_buffer_safe(!touched);
        });
        kv_cache_manager::UriStrVec actual;
        const auto result = executeClientTransfer(REMOTE_OPERATION_WRITE, registration_tags_.front(),
                                                  operation.uris, operation.buffers, actual, response,
                                                  touched, &attempted, task);
        if (result == TransferResult::SUCCESS && actual != operation.uris) {
            for (const auto& uri : actual)
                response.add_actual_uris(uri);
        }
        response.set_transfer_status(result == TransferResult::SUCCESS ? REMOTE_TRANSFER_STATUS_SUCCESS :
                                     result == TransferResult::TIMEOUT ? REMOTE_TRANSFER_STATUS_TIMEOUT :
                                                                         REMOTE_TRANSFER_STATUS_FAILED);
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
    int target_prefill_cp_{1}, target_prefill_tp_{1}, target_physical_shards_{1};
    bool decode_publication_enabled_{false}, pace_backend_{false}, is_decode_role_{false};
    StorageBackend::BufferResolver file_resolver_;
    std::shared_ptr<PublicationBudget> publication_budget_;
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
    std::unique_ptr<kvcm::GroupPolicy>                  publication_group_policy_;
    std::shared_ptr<kvcm::ClientWrapper>                projected_metadata_client_;
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
    const CacheTopology*  topology_           = nullptr;
    bool                  has_swa_            = false;
    bool                  merge_full_specs_   = false;
    bool                  publication_merge_full_specs_ = false;
    kvcm::KVCMConfig::LocationSpecGroups publication_location_groups_;
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

StorageWriteTask KVCMStorageBackend::prepareWrite(StorageRequest request) {
    // Gate only new remote writes before source pins or executor work. Tree
    // callers keep their local insertion; existing I/O still settles normally.
    if (!impl_->allowsWrite(request))
        return {};
    return StorageBackend::prepareWrite(std::move(request));
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

BlockTreeReclaimer* KVCMStorageBackend::writeSourceReclaimer() const noexcept {
    return impl_->writeSourceReclaimer();
}

std::shared_ptr<StorageCompletionAdmission> KVCMStorageBackend::reserveDecodePublication(size_t bytes) {
    return remoteCacheEnabled() ? impl_->reserveDecodePublication(bytes) : nullptr;
}

bool KVCMStorageBackend::canStartDecodePublication(const BatchKVCacheResource& resource) const {
    return remoteCacheEnabled() && Clock::now() < resource.decodePublicationDeadline(operationTimeout())
           && resource.mayAttemptDecodePublication();
}

StorageWriteTask KVCMStorageBackend::prepareCompletedDecodeWrite(
    const BatchKVCacheResource& source, size_t end, Clock::time_point deadline,
    std::shared_ptr<StorageCompletionAdmission> admission) {
    auto request = impl_->planCompletedDecode(source, end, deadline, std::move(admission));
    if (request.empty()) return {};
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now());
    if (remaining.count() <= 0) return {};
    request.transfer_task.emplace(std::vector<TransferDescriptor>{}, remaining, request.request_id);
    return prepareWrite(std::move(request));
}

bool KVCMStorageBackend::enqueueCompletedDecode(const BatchKVCacheResource& resource, size_t computed_end,
                                                DecodePublicationPrepare prepare_source) {
    if (!canStartDecodePublication(resource)) return false;
    const auto deadline = resource.decodePublicationDeadline(operationTimeout());
    return enqueuePlannedWrite(
        [this, prepare_source = std::move(prepare_source), computed_end, deadline] {
            const auto source = prepare_source();
            if (!source) return StorageWriteTask{};
            return prepareCompletedDecodeWrite(*source, computed_end, deadline, source->dspark_release_admission);
        },
        deadline);
}

bool KVCMStorageBackend::captureCompletedDecode(const BatchKVCacheResource& resource, size_t end, bool finished) {
    // A progress callback is intentionally a no-op: no GPU query, pin or copy.
    if (!finished)
        return true;
    if (!remoteCacheEnabled()) {
        onRemoteCacheSkipped("write");
        return true;
    }
    try {
        if (!resource.beginDecodePublication()) return true;
        block_tree_cache_detail::ScopeRollback end_preparing([&resource] { resource.endDecodePublication(); });
        const auto deadline = resource.decodePublicationDeadline(operationTimeout());
        bool admitted = false;
        if (resource.dspark_decode_publication_unit) {
            if (resource.dspark_release_provenance_lost || !resource.decode_endpoint
                || resource.batchSize() != 1) {
                return true;
            }
            // Reserve the FULL prefix and verified STATE/SWA pages before
            // handing their pins to the storage queue.
            const auto bytes = resource.decode_endpoint->retained_page_bytes;
            auto admission = resource.dspark_release_admission ? resource.dspark_release_admission
                                                              : reserveDecodePublication(bytes);
            if (!admission) {
                RTP_LLM_LOG_WARNING("DTOP_GPU admission=rejected retained_bytes=%zu",
                                    bytes);
                return true;
            }
            // The existing writer fence ran before EOS. Pass the retained
            // allocation owners and finalized keys, with no live stream/table
            // reference, so release can run as soon as we enqueue.
            auto snapshot = std::make_shared<BatchKVCacheResource>();
            snapshot->resetBatchSize(1);
            snapshot->decode_publication_keys = resource.decode_publication_keys;
            if (!snapshot->decode_publication_keys) {
                snapshot->decode_publication_keys = std::make_shared<const CacheKeysType>(resource.cacheKeys(0));
            }
            snapshot->decode_endpoint = resource.decode_endpoint;
            if (const auto handoff = resource.prefillComputedEnd()) snapshot->recordPrefillComputedEnd(*handoff);
            admitted = enqueuePlannedWrite(
                [this, snapshot = std::move(snapshot), admission = std::move(admission), end, deadline] {
                    return prepareCompletedDecodeWrite(*snapshot, end, deadline, admission);
                },
                deadline);
        } else {
            auto task = prepareCompletedDecodeWrite(resource, end, deadline);
            if (!task) return true;
            admitted = write(std::move(task));
        }
        if (admitted) resource.decodeCompletionSubmitted() = true; // Before fallible logging after admission.
        RTP_LLM_LOG_DEBUG("DTOP_GPU EOS_queued=%d computed_end=%zu", static_cast<int>(admitted), end);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("DTOP_GPU EOS admission failed; no publication claimed: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("DTOP_GPU EOS admission failed; unknown error");
    }
    return true;
}

bool KVCMStorageBackend::executeRemoteRequest(const RemoteOperationRequestPB& request, FunctionResponsePB& response) {
    const bool success = execute(request, *response.mutable_remote_response());
    return request.op() == REMOTE_OPERATION_READ || request.op() == REMOTE_OPERATION_WRITE || success;
}

}  // namespace rtp_llm
