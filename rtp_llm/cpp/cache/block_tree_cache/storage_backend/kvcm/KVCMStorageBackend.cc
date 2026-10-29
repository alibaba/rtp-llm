#include <cctype>
#include <exception>
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/KVCMStorageBackend.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <mutex>
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
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DecodePublicationAccounting.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DtopOriginMetadata.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/WorkerPayloadProof.h"
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

constexpr std::array<const char*, 5> kQuarantineReasons{
    "sdk_timeout", "rpc_timeout", "rpc_error", "invalid_response", "sdk_error"};

class BufferSafeTransferFailure: public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

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
        TIMEOUT,
        COMPLETION_UNKNOWN,
    };

    // No back-reference to Impl/StorageBackend. A quarantined task can retain
    // the registered SDK and its unresolved remote session after owner shutdown
    // was requested. Pool ownership lives in the generic task's pinned pages.
    struct CompletionResources {
        struct Operation {
            std::vector<std::string> tags;
            std::vector<int32_t> blocks;
            kv_cache_manager::UriStrVec uris;
            kv_cache_manager::BlockBuffers buffers;
            std::vector<DeviceBlockPoolPtr> pools;
            std::shared_ptr<kvcm::ClientWrapper> client;
            std::shared_ptr<Operation> next;
        };
        struct Session {
            std::string id;
            std::shared_ptr<Session> next;
        };
        explicit CompletionResources(std::shared_ptr<kvcm::ClientWrapper> value): client(std::move(value)) {}
        void holdSession(const std::shared_ptr<Session>& session, std::string id) {
            // Preserve identity without allocating after StartWrite. This does
            // not renew the unchanged Manager lease or prove destination drain.
            session->id = std::move(id);
            std::lock_guard<std::mutex> lock(mutex);
            session->next = std::move(sessions);
            sessions = session;
        }
        void releaseSession(const std::shared_ptr<Session>& session) noexcept {
            std::lock_guard<std::mutex> lock(mutex);
            auto* link = &sessions;
            while (*link && *link != session) {
                link = &(*link)->next;
            }
            if (*link) {
                *link = std::move(session->next);
            }
        }
        void holdOperation(const std::shared_ptr<Operation>& operation) noexcept {
            std::lock_guard<std::mutex> lock(mutex);
            operation->next = std::move(operations);
            operations = operation;
        }
        void completeOperation(const std::shared_ptr<Operation>& operation, bool unknown) noexcept {
            std::lock_guard<std::mutex> lock(mutex);
            if (unknown) {
                ++unresolved;
                return;
            }
            auto* link = &operations;
            while (*link && *link != operation) link = &(*link)->next;
            if (*link) *link = std::move(operation->next);
        }
        std::shared_ptr<kvcm::ClientWrapper> client;
        std::mutex mutex;
        std::shared_ptr<Session> sessions;
        std::shared_ptr<Operation> operations;
        std::atomic<size_t> unresolved{0};
    };

    Impl(const CacheConfig&                            cache_config,
         const KVCacheConfig&                          kv_cache_config,
         const RuntimeConfig&                          runtime_config,
         const ParallelismConfig&                      parallelism_config,
         const SpeculativeExecutionConfig&             sp_config,
         std::shared_ptr<BroadcastManager>             broadcast_manager,
         bool                                          gdr_enabled,
         std::shared_ptr<kmonitor::MetricsReporter>    metrics_reporter,
         std::shared_ptr<kvcm::ClientWrapper>          client_wrapper):
        cache_config_(cache_config),
        kv_cache_config_(kv_cache_config),
        runtime_config_(runtime_config),
        parallelism_config_(parallelism_config),
        sp_config_(sp_config),
        broadcast_manager_(std::move(broadcast_manager)),
        gdr_enabled_(gdr_enabled),
        metrics_reporter_(std::move(metrics_reporter)),
        client_wrapper_(std::move(client_wrapper)),
        sdk_check_enabled_(autil::EnvUtil::getEnv("KVCM_SDK_CHECK", autil::EnvUtil::getEnv("RECO_SDK_CHECK", false))) {}

    bool init(const CacheTopology&                                                topology,
              StorageBackend::BufferResolver                                      buffer_resolver,
              const std::function<const DeviceBlockPoolPtr&(const std::string&)>& pool_resolver) {
        RTP_LLM_LOG_INFO("start init BlockTree KVCM storage backend");
        if (!configureDecodePublication(topology, buffer_resolver, pool_resolver)) {
            return false;
        }
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
            group_block_size_bytes.emplace(group.tag, publicationWireBytes(group.tag));
        }
        if (decode_publication_enabled_ && parallelism_config_.role_type == RoleType::PREFILL
            && parallelism_config_.tp_rank == 0) {
            for (const auto& group : groups) {
                if (!group.policy.enable_prefix_reuse)
                    continue;
                const auto& pool = pool_resolver(group.tag);
                // Enabling is idempotent, including shared packed pools. Avoid
                // a separate allocating dedup table for optional provenance.
                if (pool && !pool->enableDecodeOriginTracking())
                    RTP_LLM_LOG_WARNING("KVCM physical source tracking unavailable; cache data remains usable");
            }
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
                client_config_map = genClientConfig();
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
        try {
            if (const auto reporter = remoteCachePayloadReporter(
                    decode_publication_enabled_, parallelism_config_.tp_rank, metrics_reporter_)) {
                kmonitor::MetricsTags read_tags("mode", "read");
                read_tags.AddTag("tp_rank", std::to_string(parallelism_config_.tp_rank));
                read_tags.AddTag("dp_rank", std::to_string(parallelism_config_.dp_rank));
                kmonitor::MetricsTags write_tags("mode", "write");
                write_tags.AddTag("tp_rank", std::to_string(parallelism_config_.tp_rank));
                write_tags.AddTag("dp_rank", std::to_string(parallelism_config_.dp_rank));
                client_wrapper_->setTransferObserver(
                    [reporter, read_tags, write_tags](bool read, size_t bytes, bool unknown) {
                        RtpLLMRemoteCachePayloadMetricsCollector collector;
                        collector.completed_payload_bytes = static_cast<int64_t>(bytes);
                        collector.completion_unknown      = unknown;
                        reporter->report<RtpLLMRemoteCachePayloadMetrics, RtpLLMRemoteCachePayloadMetricsCollector>(
                            read ? &read_tags : &write_tags, &collector);
                    });
            }
        } catch (const std::exception& error) {
            // Optional telemetry allocation must not invalidate registered GPU
            // pools or change inference/SDK outcomes.
            RTP_LLM_LOG_WARNING("KVCM payload observer unavailable: %s", error.what());
        }
        if (decode_publication_enabled_)
            completion_resources_ = std::make_shared<CompletionResources>(client_wrapper_);
        if (completion_accounting_)
            completion_accounting_->startSampling();
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

    StorageMatchResult match(const StorageRequest& request) {
        if (decode_publication_enabled_ && (is_decode_role_ || completion_stopped_.load())) {
            return {request.local_matched_blocks_num, nullptr};
        }
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
        const char*                            lookup_outcome = "failed";
        const auto                            start = Clock::now();
        block_tree_cache_detail::ScopeRollback report([&] {
            collector.remote_match_time_us = elapsedTimeUs(start);
            reportRemoteMetrics<RtpLLMRemoteCacheMatchMetrics>(collector);
            if (decode_publication_enabled_) {
                RtpLLMRemoteCacheOutcomeMetricsCollector result;
                result.stage   = "lookup";
                result.outcome = lookup_outcome;
                reportRemoteMetrics<RtpLLMRemoteCacheOutcomeMetrics>(result);
            }
        });
        const auto trace_id       = nextTraceId("match", match_trace_sequence_);
        auto query_type = static_cast<kv_cache_manager::QueryType>(default_query_type_);
        if (has_swa_) {
            // Fetch the FULL prefix as well as SWA locations, then apply each
            // group's model-defined reuse window locally.
            query_type = kv_cache_manager::QueryType::QT_BATCH_GET;
        }
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
        if (decode_publication_enabled_) {
            const bool found_any =
                std::any_of(locations.begin(), locations.end(), [](const auto& row) { return !row.empty(); });
            lookup_outcome = found_any ? "found_unusable" : "not_found";
        }
        if (positional) {
            auto meta = std::make_shared<KVCMMatchMeta>();
            meta->locations = reusableLocations(request, std::move(locations));
            meta->filtered = true;
            collector.remote_match_fail_qps        = false;
            collector.remote_match_reuse_block_num = static_cast<int64_t>(meta->locations.size());
            if (!meta->locations.empty())
                lookup_outcome = "usable_found";
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
        if (!locations_view.empty())
            lookup_outcome = "usable_found";
        if (!locations_view.empty()) {
            collector.remote_valid_hit_ratio =
                static_cast<double>(request.local_matched_blocks_num + locations_view.size()) / keys.size();
        }
        return {request.local_matched_blocks_num + locations_view.size(), std::move(meta)};
    }

    void read(const StorageRequest& request, const std::shared_ptr<StorageBackendMatchMeta>& match_meta) {
        if (!decode_publication_enabled_) {
            readTransaction(request, match_meta, nullptr);
            return;
        }
        if (completion_stopped_.load()) {
            std::rethrow_exception(buffer_safe_failure_);
        }
        bool source_may_be_touched = false;
        const auto safe_failure = buffer_safe_failure_;
        const auto unknown = completion_unknown_failure_;
        try {
            readTransaction(request, match_meta, &source_may_be_touched);
        } catch (const StorageOperationBufferSafeFailure&) {
            throw;
        } catch (const StorageOperationCompletionUnknown&) {
            if (source_may_be_touched) {
                throw;
            }
            std::rethrow_exception(safe_failure);
        } catch (...) {
            std::rethrow_exception(source_may_be_touched ? unknown : safe_failure);
        }
    }

    void readTransaction(const StorageRequest& request,
                         const std::shared_ptr<StorageBackendMatchMeta>& match_meta,
                         bool* source_may_be_touched) {
        RTP_LLM_CHECK_WITH_INFO(parallelism_config_.tp_rank == 0,
                                "KVCM metadata read must run on tp rank 0, got %ld",
                                parallelism_config_.tp_rank);
        RtpLLMRemoteCacheReadMetricsCollector collector;
        const char*                            read_outcome = "failed";
        const auto                           start = Clock::now();
        block_tree_cache_detail::ScopeRollback report([&] {
            collector.remote_read_task_cost_time_us = elapsedTimeUs(start);
            reportRemoteMetrics<RtpLLMRemoteCacheReadMetrics>(collector);
            if (decode_publication_enabled_) {
                RtpLLMRemoteCacheOutcomeMetricsCollector result;
                result.stage   = "read";
                result.outcome = read_outcome;
                reportRemoteMetrics<RtpLLMRemoteCacheOutcomeMetrics>(result);
            }
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
        struct OriginBinding {
            size_t                        key;
            DeviceBlockPoolPtr            pool;
            BlockIdxType                  block;
            uint64_t                      incarnation;
            std::string                   tag;
            PhysicalDecodeOrigins::Origin origin;
        };
        std::vector<OriginBinding> origin_bindings;
        const bool track_origins    = decode_publication_enabled_ && parallelism_config_.role_type == RoleType::PREFILL;
        bool       collect_bindings = track_origins;
        size_t     validated_source_keys = 0;
        if (collect_bindings) {
            try {
                size_t count = 0;
                for (const auto& row : locations_view)
                    count += row.size();
                origin_bindings.reserve(count);
            } catch (const std::bad_alloc&) {
                collect_bindings = false;
            }
        }
        for (size_t location_idx = 0; location_idx < locations_view.size(); ++location_idx) {
            const size_t key_idx = request.local_matched_blocks_num + location_idx;
            PhysicalDecodeOrigins::Origin origin;
            if (collect_bindings) {
                try {
                    auto source = kvcm::readDtopOriginMetadata(
                        locations_view[location_idx], key_idx, topology_->groups().front().cacheKeyTokenStride());
                    if (source) {
                        origin = std::make_shared<const DecodeCacheOrigin>(std::move(*source));
                        ++validated_source_keys;
                    }
                } catch (const std::bad_alloc&) {
                    // Unknown provenance never rejects an otherwise valid load.
                }
            }
            for (const auto& location_spec : locations_view[location_idx]) {
                const auto        info = spec_info.find(location_spec.spec_name);
                const std::string spec_name(location_spec.spec_name);
                RTP_LLM_CHECK_WITH_INFO(info != spec_info.end(), "KVCM read has unknown spec [%s]", spec_name.c_str());
                const StorageBlockHandle* handle = findHandle(request.handles[key_idx], info->second.tag);
                RTP_LLM_CHECK_WITH_INFO(handle != nullptr,
                                        "KVCM read has no destination handle for key=%zu tag=%s",
                                        key_idx,
                                        info->second.tag.c_str());
                auto* remote = requests.at(static_cast<size_t>(info->second.tp_rank)).mutable_remote_request();
                remote->add_group_tags(info->second.tag);
                remote->add_block_ids(handle->block);
                remote->add_uris(std::string(location_spec.uri));
                if (track_origins && info->second.tp_rank == parallelism_config_.tp_rank) {
                    const auto& pool        = pools_by_tag_.at(info->second.tag);
                    const auto  incarnation = pool->decodeOriginIncarnation(handle->block);
                    // A destination overwrite must not retain its previous
                    // source, including a legacy or failed remote read. This
                    // does not invalidate siblings sharing the physical page.
                    pool->bindDecodeOrigin(handle->block, incarnation, info->second.tag, nullptr);
                    if (origin && incarnation != 0) {
                        try {
                            origin_bindings.push_back(
                                {key_idx, pool, handle->block, incarnation, info->second.tag, origin});
                        } catch (const std::bad_alloc&) {
                            origin_bindings.clear();
                            collect_bindings = false;
                            origin.reset();
                        }
                    }
                }
            }
        }
        if (source_may_be_touched) *source_may_be_touched = true;
        (void)dispatchRequests(requests, kv_cache_config_.kvcm_get_broadcast_timeout, request.request_id,
                               source_may_be_touched);
        if (source_may_be_touched) *source_may_be_touched = false;
        // StorageBackend's task still owns the destination pins here. Publish
        // provenance only after every required rank has completed its actual
        // GPU read. A stale bind can never clear a newer allocation's source.
        size_t                              bound_regions = 0;
        std::shared_ptr<DecodeReuseSources> restored_sources;
        if (track_origins && collect_bindings) {
            try {
                restored_sources = std::make_shared<DecodeReuseSources>();
                restored_sources->reserve(remote_blocks);
            } catch (const std::bad_alloc&) {
                restored_sources.reset();
            }
        }
        for (size_t begin = 0; begin < origin_bindings.size();) {
            size_t end = begin + 1;
            while (end < origin_bindings.size() && origin_bindings[end].key == origin_bindings[begin].key)
                ++end;
            bool bound = true;
            for (size_t i = begin; i < end; ++i) {
                const auto& value = origin_bindings[i];
                bound = value.pool->bindDecodeOrigin(value.block, value.incarnation, value.tag, value.origin) && bound;
            }
            if (!bound) {
                for (size_t i = begin; i < end; ++i) {
                    const auto& value = origin_bindings[i];
                    value.pool->bindDecodeOrigin(value.block, value.incarnation, value.tag, nullptr);
                }
            } else {
                bound_regions += end - begin;
                if (restored_sources)
                    restored_sources->push_back({origin_bindings[begin].origin, Tier::REMOTE});
            }
            begin = end;
        }
        meta->decode_sources =
            restored_sources && !restored_sources->empty() ? std::move(restored_sources) : DecodeReuseSourcesPtr{};
        if (meta->decode_sources) {
            try {
                const auto unit = topology_->groups().front().cacheKeyTokenStride();
                const auto adopted_end =
                    unit != 0
                            && request.handles.size()
                                   <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / unit ?
                        static_cast<int64_t>(request.handles.size() * unit) :
                        int64_t{0};
                const auto counts = countDecodeReuse(*meta->decode_sources, adopted_end);
                if (counts.total() > 0) {
                    RtpLLMDecodeCacheTokenMetricsCollector restored;
                    restored.stage          = "restored";
                    restored.logical_tokens = counts.total();
                    reportRemoteMetrics<RtpLLMDecodeCacheTokenMetrics>(restored);
                }
            } catch (const std::bad_alloc&) {
                // Attribution does not change the completed payload result.
            }
        }
        if (track_origins) {
            RtpLLMRemoteCacheOutcomeMetricsCollector source_result;
            source_result.stage = "read_source";
            source_result.outcome =
                collect_bindings && validated_source_keys == remote_blocks ? "validated" : "unknown_or_invalid";
            reportRemoteMetrics<RtpLLMRemoteCacheOutcomeMetrics>(source_result);
            RTP_LLM_LOG_INFO("DTOP_GPU read_source_bound_regions=%zu source_counts=awaiting_allocator_adoption",
                             bound_regions);
        }
        // A canonical CP key already covers every shard. Do not multiply this
        // logical token count by the number of TP ranks or payload groups.
        collector.remote_read_token_num =
            static_cast<int64_t>(remote_blocks * topology_->groups().front().cacheKeyTokenStride());
        collector.remote_read_fail_qps = false;
        read_outcome                   = "read_complete";
    }

    void write(const StorageRequest& request) {
        if (decode_publication_enabled_ && is_decode_role_ && !request.decode_completion) {
            return;
        }
        if (!decode_publication_enabled_) {
            writeTransaction(request, nullptr);
            return;
        }
        if (completion_stopped_.load()) {
            std::rethrow_exception(buffer_safe_failure_);
        }
        bool source_may_be_touched = false;
        // Error payloads are allocated before the first SDK operation. Metadata
        // failures before submission/after complete payloads are buffer-safe.
        const auto safe_failure = buffer_safe_failure_;
        const auto unknown = completion_unknown_failure_;
        try {
            writeTransaction(request, &source_may_be_touched);
        } catch (const StorageOperationBufferSafeFailure&) {
            throw;
        } catch (const StorageOperationCompletionUnknown&) {
            if (source_may_be_touched) {
                throw;
            }
            std::rethrow_exception(safe_failure);
        } catch (...) {
            std::rethrow_exception(source_may_be_touched ? unknown : safe_failure);
        }
    }

    void writeTransaction(const StorageRequest& request, bool* source_may_be_touched) {
        // Preserve ordinary Prefill publication. D publishes through the EOS
        // transaction only; no benchmark flag may disable P's normal cache path.
        if (decode_publication_enabled_ && is_decode_role_ && !request.decode_completion)
            return;
        RTP_LLM_CHECK_WITH_INFO(parallelism_config_.tp_rank == 0,
                                "KVCM metadata write must run on tp rank 0, got %ld",
                                parallelism_config_.tp_rank);
        RTP_LLM_CHECK(request.keys != nullptr && request.keys->size() == request.handles.size());
        const size_t valid_keys_size = request.keys->size();
        if (valid_keys_size == 0) {
            return;
        }
        RtpLLMRemoteCacheWriteMetricsCollector collector;
        const auto                            start = Clock::now();
        block_tree_cache_detail::ScopeRollback report([&] {
            collector.remote_write_task_cost_time_us = elapsedTimeUs(start);
            reportRemoteMetrics<RtpLLMRemoteCacheWriteMetrics>(collector);
        });
        CacheKeysType            keys(request.keys->begin(), request.keys->begin() + valid_keys_size);
        std::vector<std::string> location_spec_group_names;
        RTP_LLM_CHECK_WITH_INFO(group_policy_->getNeedWriteGroups(request, valid_keys_size, location_spec_group_names),
                                "KVCM write group selection failed");
        const auto        write_started_us  = autil::TimeUtility::currentTimeInMicroSeconds();
        size_t            transferred_bytes = 0;
        const std::string trace_id     = nextTraceId("write", write_trace_sequence_);
        auto session = completion_resources_ ? std::make_shared<CompletionResources::Session>() : nullptr;
        auto [success, write_location] = [&] {
            const auto stage_start = Clock::now();
            block_tree_cache_detail::ScopeRollback timer([&] {
                collector.remote_get_write_location_time_us = elapsedTimeUs(stage_start);
            });
            return client_wrapper_->getWriteLocation(
                "", trace_id, keys, /*tokens=*/{}, location_spec_group_names, /*write_timeout_seconds=*/600,
                kv_cache_config_.kvcm_min_replica_count);
        }();
        RTP_LLM_CHECK_WITH_INFO(success, "KVCM StartWrite failed");
        if (session) completion_resources_->holdSession(session, std::move(write_location.write_session_id));
        const std::string& write_session_id = session ? session->id : write_location.write_session_id;
        static const kv_cache_manager::Locations empty_locations;
        bool                                     finish_attempted = false;
        bool                                     transfer_attempted = false;
        bool                                     retain_remote_session = false;
        block_tree_cache_detail::ScopeRollback release_known_session([&] {
            if (session && !retain_remote_session && (!source_may_be_touched || !*source_may_be_touched))
                completion_resources_->releaseSession(session);
        });
        try {
            std::vector<FunctionRequestPB> requests(
                static_cast<size_t>(request.decode_completion ? target_prefill_tp_ : parallelism_config_.tp_size));
            ActualUriGather                actual_uri_gather(requests.size());
            initializeRequests(requests, REMOTE_OPERATION_WRITE, trace_id);
            const auto key_indices = unmaskedKeyIndices(write_location.block_mask, valid_keys_size);
            RTP_LLM_CHECK_WITH_INFO(key_indices.size() == write_location.locations.size(),
                                    "KVCM write mask/location mismatch: keys=%zu locations=%zu",
                                    key_indices.size(),
                                    write_location.locations.size());
            if (write_location.locations.empty()) {
                if (!write_session_id.empty()) {
                    // The server can create an empty, short-lived session when
                    // every key already has enough replicas. Close it normally.
                    finish_attempted = true;
                    const auto stage_start = Clock::now();
                    block_tree_cache_detail::ScopeRollback timer([&] {
                        collector.remote_finish_write_time_us += elapsedTimeUs(stage_start);
                    });
                    try {
                        if (!client_wrapper_->finishWrite(
                            "", nextTraceId("finish_write", finish_write_trace_sequence_),
                            write_session_id, kv_cache_manager::BlockMaskOffset{0}, empty_locations)) {
                            RTP_LLM_LOG_WARNING("KVCM failed to close empty write session [%s]",
                                                write_session_id.c_str());
                        }
                    } catch (...) {
                        RTP_LLM_LOG_WARNING("KVCM failed to close empty write session [%s]",
                                            write_session_id.c_str());
                    }
                }
                // Keep the legacy task QPS, but report no payload blocks or
                // transfer time when all keys already have enough replicas.
                collector.remote_write_fail_qps = false;
                if (const auto lease = std::dynamic_pointer_cast<kvcm::DecodePublicationAccounting::Lease>(
                        request.completion_admission))
                    lease->recordOutcome("already_present");
                return;
            }
            const auto& spec_info = group_policy_->spec_info_map();
            for (size_t location_idx = 0; location_idx < write_location.locations.size(); ++location_idx) {
                const size_t key_idx = key_indices[location_idx];
                for (auto& location_spec : write_location.locations[location_idx]) {
                    const auto info = spec_info.find(location_spec.spec_name);
                    RTP_LLM_CHECK_WITH_INFO(
                        info != spec_info.end(), "KVCM write has unknown spec [%s]", location_spec.spec_name.c_str());
                    const StorageBlockHandle* handle = findHandle(request.handles[key_idx], info->second.tag);
                    RTP_LLM_CHECK_WITH_INFO(handle != nullptr,
                                            "KVCM write has no source handle for key=%zu tag=%s",
                                            key_idx,
                                            info->second.tag.c_str());
                    transferred_bytes += group_policy_->groups().at(info->second.group_id).block_size_bytes;
                    const size_t rank   = static_cast<size_t>(info->second.tp_rank);
                    auto*        remote = requests.at(rank).mutable_remote_request();
                    remote->add_group_tags(info->second.tag);
                    remote->add_block_ids(request.decode_completion && !handle->peer_blocks.empty() ?
                                              handle->peer_blocks.at(rank) :
                                              handle->block);
                    remote->add_uris(location_spec.uri);
                    actual_uri_gather[rank].push_back(&location_spec);
                }
            }

            const auto responses = [&] {
                const auto                             stage_start = Clock::now();
                block_tree_cache_detail::ScopeRollback timer(
                    [&] { collector.remote_write_broadcast_time_us = elapsedTimeUs(stage_start); });
                if (request.decode_completion) {
                    RTP_LLM_CHECK(source_may_be_touched != nullptr);
                    return transferCompletedDecode(requests, transfer_attempted, *source_may_be_touched);
                }
                transfer_attempted = true;
                if (source_may_be_touched) *source_may_be_touched = true;
                auto responses = dispatchRequests(requests, kv_cache_config_.kvcm_put_broadcast_timeout,
                                                   request.request_id, source_may_be_touched);
                if (source_may_be_touched) *source_may_be_touched = false;
                return responses;
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
            std::optional<kvcm::DtopOriginMetadata> source_metadata;
            if (request.decode_completion && request.prefill_computed_tokens) {
                try {
                    source_metadata = kvcm::makeDtopOriginMetadata(write_location.locations,
                                                                   write_session_id,
                                                                   key_indices,
                                                                   static_cast<size_t>(cache_config_.seq_size_per_block)
                                                                       * target_physical_shards_,
                                                                   request.published_tokens,
                                                                   request.computed_tokens,
                                                                   *request.prefill_computed_tokens);
                } catch (const std::bad_alloc&) {
                    // Source attribution is optional metadata. A completed GPU
                    // payload must not fail because its source could not be stored.
                }
                if (!source_metadata)
                    RTP_LLM_LOG_WARNING("DTOP source metadata unavailable; retaining ordinary payload publication");
            }
            const auto& actual_locations = source_metadata ? source_metadata->locations :
                                           has_actual_uri  ? write_location.locations :
                                                             empty_locations;
            finish_attempted             = true;
            {
                const auto                             stage_start = Clock::now();
                block_tree_cache_detail::ScopeRollback timer(
                    [&] { collector.remote_finish_write_time_us += elapsedTimeUs(stage_start); });
                RTP_LLM_CHECK_WITH_INFO(
                    client_wrapper_->finishWrite("",
                                                 nextTraceId("finish_write", finish_write_trace_sequence_),
                                                 write_session_id,
                                                 write_location.locations.size(),
                                                 actual_locations),
                    "KVCM FinishWrite failed");
            }
            collector.remote_write_cache_block_num = static_cast<int64_t>(write_location.locations.size());
            collector.remote_write_fail_qps        = false;
            if (const auto lease =
                    std::dynamic_pointer_cast<kvcm::DecodePublicationAccounting::Lease>(request.completion_admission))
                lease->recordOutcome("committed", /*acknowledged=*/true);
            if (request.decode_completion) {
                // An older Manager ignores actual_locations. An ACK proves the
                // data transaction, not persistence of this source assertion;
                // do not turn origin_requested into confirmed D-token metrics.
                RTP_LLM_LOG_INFO("DTOP_GPU source origin_requested=%d origin_persistence=unconfirmed",
                                 source_metadata.has_value());
                RTP_LLM_LOG_INFO(
                    "DTOP_GPU committed computed_end=%zu canonical_keys=%zu new_locations=%zu bytes=%zu dp_rank=%ld",
                    request.published_tokens,
                    request.keys->size(),
                    write_location.locations.size(),
                    transferred_bytes,
                    parallelism_config_.dp_rank);
                RTP_LLM_LOG_INFO("DTOP_GPU commit actual_computed_end=%zu published=%zu tail=%zu CPU_snapshot_bytes=0",
                                 request.computed_tokens,
                                 request.published_tokens,
                                 request.computed_tokens - request.published_tokens);
                RTP_LLM_LOG_INFO("DTOP_GPU write_duration_us=%ld computed_end=%zu bytes=%zu",
                                 autil::TimeUtility::currentTimeInMicroSeconds() - write_started_us,
                                 request.published_tokens,
                                 transferred_bytes);
            }
        } catch (...) {
            if (decode_publication_enabled_ && finish_attempted && collector.remote_write_fail_qps) {
                retain_remote_session = true;
                // All payload calls have completed before FinishWrite. A lost
                // metadata acknowledgement leaves publication uncertain, not
                // the caller GPU buffer's DMA lifetime. Do not quarantine safe
                // pages or retry an abort against a possibly committed session.
                if (const auto lease = std::dynamic_pointer_cast<kvcm::DecodePublicationAccounting::Lease>(
                        request.completion_admission))
                    lease->recordOutcome("completion_unknown");
            }
            if (decode_publication_enabled_ && transfer_attempted && !finish_attempted) {
                retain_remote_session = true;
                // A BUFFER_SAFE failure permits releasing the caller's GPU
                // pages, but does not prove detached Consumer writes stopped
                // touching the remote allocation. Do not explicitly recycle
                // that allocation without an independent destination fence.
                // The Manager lease/expiry contract remains separate.
                RTP_LLM_LOG_WARNING(
                    "KVCM remote destination completion unproven; retaining pending session [%s] without abort",
                    write_session_id.c_str());
                RtpLLMRemoteCacheOutcomeMetricsCollector allocation_result;
                allocation_result.stage   = "write_allocation";
                allocation_result.outcome = "retained_unproven";
                reportRemoteMetrics<RtpLLMRemoteCacheOutcomeMetrics>(allocation_result);
            } else if (!finish_attempted) {
                const auto                             stage_start = Clock::now();
                block_tree_cache_detail::ScopeRollback timer(
                    [&] { collector.remote_finish_write_time_us += elapsedTimeUs(stage_start); });
                try {
                    if (!client_wrapper_->finishWrite("",
                                                      nextTraceId("abort_write", finish_write_trace_sequence_),
                                                      write_session_id,
                                                      /*block_mask=*/kv_cache_manager::BlockMaskOffset{0},
                                                      empty_locations)) {
                        RTP_LLM_LOG_WARNING("KVCM failed to abort write session [%s]",
                                            write_session_id.c_str());
                    }
                } catch (...) {
                    RTP_LLM_LOG_WARNING("KVCM abort write session threw, session=[%s]",
                                        write_session_id.c_str());
                }
            }
            throw;
        }
    }

    bool execute(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response) {
        if (request.op() != REMOTE_OPERATION_READ && request.op() != REMOTE_OPERATION_WRITE) {
            return executeMetadata(request, response);
        }
        if (decode_publication_enabled_) {
            return executeProtectedPayload(request, response);
        }
        const std::vector<std::string>    tags(request.group_tags().begin(), request.group_tags().end());
        const std::vector<int32_t>        blocks(request.block_ids().begin(), request.block_ids().end());
        const kv_cache_manager::UriStrVec uris(request.uris().begin(), request.uris().end());
        if (tags.size() != blocks.size() || blocks.size() != uris.size()) {
            RTP_LLM_LOG_WARNING("KVCM transfer tag/block/URI count mismatch");
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
            return false;
        }
        if (tags.empty()) {
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_SUCCESS);
            return true;
        }
        // Validate the complete route before resolving any block into an IOV.
        // Requests arrive over RPC, so an unknown tag is a request failure and
        // must not escape as an assertion or allow a valid prefix to start I/O.
        for (const auto& tag : tags) {
            if (pools_by_tag_.find(tag) == pools_by_tag_.end()) {
                RTP_LLM_LOG_ERROR("KVCM transfer contains unregistered tag [%s]", tag.c_str());
                response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
                return false;
            }
        }
        setCudaDevice();
        kv_cache_manager::BlockBuffers buffers;
        if (!group_policy_->genBlockBuffers(tags, blocks, buffers)) {
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
            return false;
        }
        const auto result = executeTagTransfers(request.op(), tags, blocks, uris, buffers, response);
        response.set_transfer_status(result == TransferResult::SUCCESS ? REMOTE_TRANSFER_STATUS_SUCCESS :
                                     result == TransferResult::TIMEOUT ? REMOTE_TRANSFER_STATUS_TIMEOUT :
                                                                         REMOTE_TRANSFER_STATUS_FAILED);
        return result == TransferResult::SUCCESS;
    }

    bool executeProtectedPayload(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response) {
        // Preflight rejection cannot have submitted a payload. The SDK scope
        // below replaces this proof before any potentially asynchronous I/O.
        response.set_payload_buffer_safe(true);
        if (completion_stopped_.load() || request.group_tags_size() != request.block_ids_size()
            || request.group_tags_size() != request.uris_size()) {
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
            return false;
        }
        if (request.group_tags_size() == 0) {
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_SUCCESS);
            return true;
        }
        auto operation = std::make_shared<CompletionResources::Operation>();
        operation->tags.assign(request.group_tags().begin(), request.group_tags().end());
        operation->blocks.assign(request.block_ids().begin(), request.block_ids().end());
        operation->uris.assign(request.uris().begin(), request.uris().end());
        if (!std::all_of(operation->uris.begin(), operation->uris.end(), [](const auto& uri) {
                return uri.compare(0, 7, "pace://") == 0;
            })) {
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
            return false;
        }
        operation->client = client_wrapper_;
        for (const auto& tag : operation->tags) {
            if (!pools_by_tag_.count(tag)) {
                response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
                return false;
            }
        }
        for (const auto& pool : pools_by_tag_) operation->pools.push_back(pool.second);
        setCudaDevice();
        if (!group_policy_->genBlockBuffers(operation->tags, operation->blocks, operation->buffers)) {
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
            return false;
        }
        // The controller's prepared task pins the globally shared block IDs.
        // This independent local record retains the worker's actual backing,
        // SDK registrations and borrowed IOV/URI descriptors across UNKNOWN.
        bool unknown = false, attempted = false;
        completion_resources_->holdOperation(operation);
        block_tree_cache_detail::ScopeRollback retain_worker_unknown([&] {
            response.set_payload_buffer_safe(!unknown);
            completion_resources_->completeOperation(operation, unknown);
            if (unknown) {
                completion_stopped_.store(true);
                if (completion_accounting_) completion_accounting_->stopAdmission();
            }
        });
        response.set_payload_buffer_safe(false);
        const auto result = executeTagTransfers(request.op(), operation->tags, operation->blocks,
            operation->uris, operation->buffers, response, &unknown, &attempted);
        response.set_transfer_status(result == TransferResult::SUCCESS ? REMOTE_TRANSFER_STATUS_SUCCESS :
                                     result == TransferResult::TIMEOUT ? REMOTE_TRANSFER_STATUS_TIMEOUT :
                                                                         REMOTE_TRANSFER_STATUS_FAILED);
        return result == TransferResult::SUCCESS;
    }

    void shutdown() noexcept {
        completion_stopped_.store(true);
        if (completion_accounting_) completion_accounting_->stopAdmission();
        if (client_wrapper_) {
            client_wrapper_->shutdown();
        }
        if (completion_accounting_) {
            completion_accounting_->stopSampling();
            completion_accounting_->reportSnapshot();
        }
        // Upstream keeps uncertain source pages FROZEN until pool destruction.
    }

    bool protectsUnknownCompletion() const noexcept {
        return decode_publication_enabled_;
    }
    std::shared_ptr<void> completionResourceOwner() const {
        return completion_resources_;
    }
    bool hasUnresolvedExternalIo() const noexcept {
        return completion_resources_ && completion_resources_->unresolved.load() > 0;
    }


#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DtopCompletionBridge.inc"

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
            case StorageQuarantineReason::SDK_ERROR:
                collector.reason = "sdk_error";
                break;
        }
        reportRemoteMetrics<RtpLLMRemoteCacheQuarantineMetrics>(collector);
    }

    void reportQuarantineState(uint64_t                          generation,
                               size_t                            task_count,
                               size_t                            block_count,
                               const StorageQuarantineBreakdown& breakdown) noexcept {
        reportQuarantineMetrics(generation, task_count, block_count, breakdown);
        if (completion_accounting_)
            completion_accounting_->reportSnapshot();
        if (decode_publication_enabled_) {
            if (task_count > 0) {
                completion_stopped_.store(true);
                if (completion_accounting_) completion_accounting_->stopAdmission();
            }
            return;
        }
        if (task_count > 0 && parallelism_config_.tp_rank == 0 && quarantine_abort_ratio_ > 0.0
            && total_device_block_count_ > 0) {
            // Quarantine metrics count per-task pins; capacity counts each
            // physical block once, using the same FROZEN state as BTC.
            size_t frozen_block_count = 0;
            for (const auto& [tag, pool] : pools_by_tag_) {
                frozen_block_count += pool->referencedBlocksNum(BlockTreeRefType::FROZEN);
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
        return type == 1 || type == 2 || type == 3 || type == 4 || type == 5 || type == 9;
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
                    const auto name = kvcm::genLocationSpecName(rank, group.group_name, runtime_config_.model_name);
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
            group_policy_->buildLocationSpecGroups(
                decode_publication_enabled_ ? target_prefill_tp_ : static_cast<int>(parallelism_config_.tp_size),
                *groups,
                runtime_config_.model_name),
            "failed to build KVCM location spec groups");
        for (const auto& [group_id, group] : group_policy_->groups()) {
            for (int rank = 0; rank < (decode_publication_enabled_ ? target_prefill_tp_ : parallelism_config_.tp_size);
                 ++rank) {
                const std::string spec_name =
                    kvcm::genLocationSpecName(rank, group.group_name, runtime_config_.model_name);
                infos->emplace(spec_name, group.block_size_bytes);
            }
        }
        return {std::move(infos), std::move(groups)};
    }

    kvcm::ClientWrapper::ConfigMap genClientConfig() {
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
        auto [location_infos, location_groups] = genLocationSpecs();

        const std::string model_name = runtime_config_.model_name;
        const std::string dtype      = getDataTypeStr(cache_config_.dtype);
        std::string       extra      = kv_cache_config_.kvcm_model_extra_info;
        extra += '/' + autil::EnvUtil::getEnv("BIZ_NAME", std::string("")) + '/'
                 + std::to_string(hashString(autil::EnvUtil::getEnv("CHECKPOINT_PATH", std::string(""))));
        if (decode_publication_enabled_) {
            extra += "/decode-publish-v2/cp=" + std::to_string(target_prefill_cp_)
                     + "/shards=" + std::to_string(target_physical_shards_)
                     + "/base=" + std::to_string(cache_config_.seq_size_per_block);
        }
        std::string draft_info;
        if (!cache_config_.mtp_sub_configs.empty()) {
            draft_info = '{' + sp_config_.to_string() + '}';
        }
        const auto wire_tp = decode_publication_enabled_ ? target_prefill_tp_ : parallelism_config_.tp_size;
        const auto wire_dp = decode_publication_enabled_ ? 1 : parallelism_config_.dp_size;
        // Match the baseline P writer identity exactly. Canonical CP keys are
        // selected by CPSlotMapper; the KVCM base block comes from CacheConfig.
        const auto        wire_block = cache_config_.seq_size_per_block;
        std::stringstream identity;
        identity << "instance_group: " << kv_cache_config_.kvcm_instance_group << ";block_size:" << wire_block
                 << ";model_name:" << model_name << ";dtype_str:" << dtype << ";use_mla:" << cache_config_.use_mla
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

        if (decode_publication_enabled_) {
            RTP_LLM_LOG_INFO("DTOP_GPU identity instance=%s tp=%ld dp=%ld base_tokens=%zu",
                             instance_id.c_str(),
                             static_cast<long>(wire_tp),
                             static_cast<long>(wire_dp),
                             wire_block);
            for (const auto& group : cache_config_.topology().groups()) {
                std::ostringstream layers;
                for (const int layer : cache_config_.layerIdsForGroup(group.tag)) {
                    layers << layer << ',';
                }
                RTP_LLM_LOG_INFO("DTOP_GPU topology tag=%s reusable=%d layers=%s physical_bytes=%zu",
                                 group.tag.c_str(),
                                 static_cast<int>(group.policy.enable_prefix_reuse),
                                 layers.str().c_str(),
                                 cache_config_.blockSizeBytesForGroup(group.tag));
            }
        }
        auto config = std::make_shared<kvcm::KVCMConfig>(kv_cache_config_.kvcm_enable_vipserver,
                                                         kv_cache_config_.kvcm_vipserver_domain,
                                                         static_cast<int32_t>(wire_block),
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
                                                                               static_cast<int32_t>(wire_tp),
                                                                               static_cast<int32_t>(wire_dp),
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
                const size_t pool_block_count = pool->totalBlocksNum();
                if (pool_block_count > std::numeric_limits<size_t>::max() - total_device_block_count) {
                    RTP_LLM_LOG_ERROR("KVCM total device block capacity overflow");
                    return {};
                }
                total_device_block_count += pool_block_count;
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
                                          runtime_config_.model_name)};
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
                if (decode_publication_enabled_) {
                    RTP_LLM_LOG_ERROR(
                        "Requested GDR cannot register group %d tag=%s as dedicated GPU memory; refusing CPU fallback",
                        group_id,
                        tag.c_str());
                    return {};
                }
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
            const int peers = decode_publication_enabled_ && is_decode_role_ ? target_prefill_tp_ : 1;
            for (int peer = 0; peer < peers; ++peer) {
                auto wire_registration               = registration;
                wire_registration.location_spec_name = kvcm::genLocationSpecName(
                    decode_publication_enabled_ && is_decode_role_ ? peer :
                                                                     static_cast<int>(parallelism_config_.tp_rank),
                    groups.at(group_id).group_name,
                    runtime_config_.model_name);
                registration_tags.push_back(publicationPeerTag(tag, peer));
                registrations.push_back(std::move(wire_registration));
            }
        }
        pools_by_tag_      = std::move(pools_by_tag);
        registration_tags_ = std::move(registration_tags);
        total_device_block_count_ = total_device_block_count;
        return registrations;
    }

    TransferResult executeClientTransfer(RemoteOpType                       operation,
                                         const std::string&                 client_tag,
                                         const std::vector<int32_t>&        blocks,
                                         const kv_cache_manager::UriStrVec& uris,
                                         kv_cache_manager::BlockBuffers&    buffers,
                                         kv_cache_manager::UriStrVec&       actual_uris,
                                         bool* source_may_be_touched = nullptr,
                                         bool* transfer_attempted = nullptr) {
        const auto                           trace_info = makeTransferTraceInfo(blocks);
        RtpLLMRemoteCacheSDKMetricsCollector collector;
        collector.remote_sdk_block_num               = static_cast<int64_t>(blocks.size());
        const auto                             start = Clock::now();
        block_tree_cache_detail::ScopeRollback report([&] {
            collector.remote_sdk_cost_time_us = elapsedTimeUs(start);
            reportRemoteMetrics<RtpLLMRemoteCacheSDKMetrics>(collector,
                                                             operation == REMOTE_OPERATION_READ ? "read" : "write");
        });
        auto completion = kvcm::ClientWrapper::TransferCompletion::NOT_SUBMITTED;
        block_tree_cache_detail::ScopeRollback reconcile_completion([&] {
            if (source_may_be_touched) {
                *source_may_be_touched = completion == kvcm::ClientWrapper::TransferCompletion::UNKNOWN;
            }
            if (transfer_attempted && completion != kvcm::ClientWrapper::TransferCompletion::NOT_SUBMITTED) {
                *transfer_attempted = true;
            }
        });
        if (operation == REMOTE_OPERATION_READ) {
            if (source_may_be_touched) *source_may_be_touched = true;
            const auto ec = client_wrapper_->loadKvCachesForTagWithStatus(
                client_tag, uris, buffers, trace_info, source_may_be_touched ? &completion : nullptr);
            if (ec == kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT) {
                return TransferResult::TIMEOUT;
            }
            if (ec != kv_cache_manager::ClientErrorCode::ER_OK) {
                return TransferResult::FAILED;
            }
        } else {
            if (source_may_be_touched) *source_may_be_touched = true;
            auto [ec, result] = client_wrapper_->saveKvCachesForTagWithStatus(
                client_tag, uris, buffers, trace_info, source_may_be_touched ? &completion : nullptr);
            if (ec == kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT) {
                return TransferResult::TIMEOUT;
            }
            if (ec != kv_cache_manager::ClientErrorCode::ER_OK || (!result.empty() && result.size() != uris.size())) {
                return TransferResult::FAILED;
            }
            actual_uris = std::move(result);
        }
        collector.remote_sdk_fail_qps = false;
        return TransferResult::SUCCESS;
    }

    TransferResult executeTagTransfers(RemoteOpType                       operation,
                                       const std::vector<std::string>&    tags,
                                       const std::vector<int32_t>&        blocks,
                                       const kv_cache_manager::UriStrVec& uris,
                                       kv_cache_manager::BlockBuffers&    buffers,
                                       RemoteOperationResponsePB&         response,
                                       bool* source_may_be_touched = nullptr,
                                       bool* transfer_attempted = nullptr) {
        if (operation != REMOTE_OPERATION_READ && operation != REMOTE_OPERATION_WRITE) {
            RTP_LLM_LOG_WARNING("KVCM transfer has invalid operation [%d]", operation);
            return TransferResult::FAILED;
        }
        kv_cache_manager::UriStrVec actual_uris;
        if (std::all_of(uris.begin(), uris.end(), [](const auto& uri) { return uri.compare(0, 7, "pace://") == 0; })) {
            // The first client owns all PACE GPU registrations. Forward the whole
            // request unchanged; the SDK selects normal or gather/scatter mode.
            const auto result =
                executeClientTransfer(operation, registration_tags_.front(), blocks, uris, buffers, actual_uris,
                                      source_may_be_touched, transfer_attempted);
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
                    executeClientTransfer(operation, tag, tag_blocks, tag_uris, tag_buffers, tag_actual_uris);
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
            std::array<StorageQuarantineCounts, kQuarantineReasons.size()> reason_totals{};
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

    std::vector<FunctionResponsePB> dispatchRequests(const std::vector<FunctionRequestPB>& requests,
                                                     int                                   timeout_ms,
                                                     std::optional<int64_t>                request_id,
                                                     bool* source_may_be_touched = nullptr) {
        const auto&                     remote    = requests.front().remote_request();
        const std::string               direction = remote.op() == REMOTE_OPERATION_READ ? "REMOTE2D" : "D2REMOTE";
        std::vector<FunctionResponsePB> responses;
        if (!broadcast_manager_) {
            RTP_LLM_CHECK_WITH_INFO(
                requests.size() == 1, "KVCM local transfer requires exactly one request, got %zu", requests.size());
            FunctionResponsePB response;
            execute(requests.front().remote_request(), *response.mutable_remote_response());
            responses.push_back(std::move(response));
        } else {
            auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                               const std::shared_ptr<grpc::ClientContext>& context,
                               const FunctionRequestPB&                    request,
                               grpc::CompletionQueue*                      queue) {
                return stub->AsyncExecuteFunction(context.get(), request, queue);
            };
            auto result = broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
                requests, timeout_ms, rpc_call, request_id, direction);
            if (result == nullptr) {
                throw StorageOperationCompletionUnknown("KVCM broadcast dispatch failed",
                                                        StorageQuarantineReason::RPC_ERROR);
            }
            try {
                result->waitDone();
            } catch (const std::exception& exception) {
                if (result->deadlineExceeded()) {
                    throw StorageOperationTimeout(std::string("KVCM broadcast timed out: ") + exception.what(),
                                                  StorageQuarantineReason::RPC_TIMEOUT);
                }
                throw StorageOperationCompletionUnknown(std::string("KVCM broadcast completion is unknown: ")
                                                            + exception.what(),
                                                        StorageQuarantineReason::RPC_ERROR);
            } catch (...) {
                throw StorageOperationCompletionUnknown("KVCM broadcast completion is unknown: "
                                                            + result->failureSummary(),
                                                        StorageQuarantineReason::RPC_ERROR);
            }
            if (!result->success()) {
                throw StorageOperationCompletionUnknown("KVCM broadcast completion is unknown: "
                                                            + result->failureSummary(),
                                                        StorageQuarantineReason::RPC_ERROR);
            }
            responses = result->responses();
        }

        if (responses.size() != requests.size()) {
            throw StorageOperationCompletionUnknown("KVCM worker response count mismatch",
                                                    StorageQuarantineReason::INVALID_RESPONSE);
        }
        if (decode_publication_enabled_) {
            kvcm::WorkerPayloadProof proof;
            bool any_timeout = false;
            for (const auto& response : responses) {
                if (!response.has_remote_response()) {
                    proof.observe(false, false, false);
                    continue;
                }
                const auto& payload = response.remote_response();
                any_timeout = any_timeout || payload.transfer_status() == REMOTE_TRANSFER_STATUS_TIMEOUT;
                proof.observe(payload.payload_buffer_safe(),
                              payload.transfer_status() == REMOTE_TRANSFER_STATUS_SUCCESS,
                              payload.transfer_status() == REMOTE_TRANSFER_STATUS_FAILED);
            }
            const auto disposition = proof.result();
            if (source_may_be_touched && disposition != kvcm::WorkerPayloadProof::Result::UNKNOWN) {
                *source_may_be_touched = false;
            }
            if (const auto failure = proof.failure(any_timeout, buffer_safe_failure_, completion_unknown_failure_,
                                                    completion_timeout_failure_)) {
                std::rethrow_exception(failure);
            }
            return responses;
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
    makeTransferTraceInfo(const std::vector<int32_t>& block_ids) const {
        if (!sdk_check_enabled_) {
            return nullptr;
        }
        auto trace_info        = std::make_shared<kv_cache_manager::TransferTraceInfo>();
        trace_info->need_print = true;
        trace_info->block_ids.reserve(block_ids.size());
        for (const auto block_id : block_ids) {
            trace_info->block_ids.push_back(std::to_string(block_id));
        }
        return trace_info;
    }

private:
    int                            target_prefill_cp_{1}, target_prefill_tp_{1}, target_physical_shards_{1};
    bool                           decode_publication_enabled_{false}, pace_backend_{false}, is_decode_role_{false};
    std::atomic<bool> completion_stopped_{false};
    std::shared_ptr<CompletionResources> completion_resources_;
    std::exception_ptr buffer_safe_failure_, completion_unknown_failure_, completion_timeout_failure_;
    StorageBackend::BufferResolver file_resolver_;
    std::shared_ptr<kvcm::DecodePublicationAccounting>   completion_accounting_;
    DecodePublicationLimits                              completion_limits_;
    CacheConfig                          cache_config_;
    KVCacheConfig                        kv_cache_config_;
    RuntimeConfig                        runtime_config_;
    ParallelismConfig                    parallelism_config_;
    SpeculativeExecutionConfig           sp_config_;
    std::shared_ptr<BroadcastManager>                    broadcast_manager_;
    const bool                                           gdr_enabled_;
    std::unique_ptr<kvcm::GroupPolicy>                   group_policy_;
    std::shared_ptr<kmonitor::MetricsReporter>           metrics_reporter_;
    std::shared_ptr<kvcm::ClientWrapper>                 client_wrapper_;
    std::vector<std::string>                             registration_tags_;
    std::unordered_map<std::string, DeviceBlockPoolPtr> pools_by_tag_;
    size_t                                               total_device_block_count_{0};
    double                                               quarantine_abort_ratio_{0.0};
    mutable std::mutex                                   quarantine_metrics_mutex_;
    mutable bool                                         quarantine_metrics_reported_{false};
    mutable uint64_t                                     last_quarantine_metrics_generation_{0};
    // Preserve KVCM's operation-local, one-based request
    // order. Abort and finish share a sequence because both call FinishWrite.
    std::atomic<uint64_t> match_trace_sequence_{1};
    std::atomic<uint64_t> read_trace_sequence_{1};
    std::atomic<uint64_t> write_trace_sequence_{1};
    std::atomic<uint64_t> finish_write_trace_sequence_{1};
    const bool            sdk_check_enabled_;
    const CacheTopology*  topology_ = nullptr;
    bool                  has_swa_ = false;
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
                   parallelism_config.world_rank),
    impl_(std::make_unique<Impl>(cache_config,
                                 kv_cache_config,
                                 runtime_config,
                                 parallelism_config,
                                 sp_config,
                                 std::move(broadcast_manager),
                                 gdr_enabled,
                                 std::move(metrics_reporter),
                                 std::move(client_wrapper))) {}

KVCMStorageBackend::~KVCMStorageBackend() {
    shutdown();
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

bool KVCMStorageBackend::protectUnknownCompletion() const noexcept {
    return impl_->protectsUnknownCompletion();
}

std::shared_ptr<void> KVCMStorageBackend::completionResourceOwner() const {
    return impl_->completionResourceOwner();
}

bool KVCMStorageBackend::hasUnresolvedExternalIo() const noexcept {
    return impl_->hasUnresolvedExternalIo();
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
    if (!impl_->protectsUnknownCompletion()) {
        return impl_->execute(request, response);
    }
    auto guard = trackExternalIo();
    if (!guard) {
        response.set_payload_buffer_safe(true);
        response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
        return false;
    }
    try {
        return impl_->execute(request, response);
    } catch (...) {
        // The local operation scope has already retained any uncertain SDK
        // access before this guard can let shutdown observe an idle worker.
        response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
        return false;
    }
}

}  // namespace rtp_llm

namespace rtp_llm {
bool KVCMStorageBackend::captureCompletedDecode(const BatchKVCacheResource& resource, size_t end, bool finished) {
    // A progress callback is intentionally a no-op: no GPU query, pin or copy.
    if (!finished)
        return true;
    try {
        auto request = impl_->planCompletedDecode(resource, end);
        if (request.empty())
            return true;
        auto       task     = prepareWrite(std::move(request));
        const bool admitted = write(std::move(task));
        if (admitted)
            resource.decodeCompletionSubmitted() = true;
        RTP_LLM_LOG_INFO("DTOP_GPU EOS_queued=%d computed_end=%zu", static_cast<int>(admitted), end);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("DTOP_GPU EOS admission failed; no publication claimed: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("DTOP_GPU EOS admission failed; unknown error");
    }
    return true;
}
}  // namespace rtp_llm
