#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/KVCMStorageBackend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <unordered_map>
#include <variant>

#include "autil/EnvUtil.h"
#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientWrapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/GroupPolicy.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

namespace rtp_llm {
namespace {

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

}  // namespace

class KVCMStorageBackend::Impl {
public:
    using ActualUriGather = std::vector<std::vector<kv_cache_manager::LocationSpecUnit*>>;

    Impl(const CacheConfig&                   cache_config,
         const KVCacheConfig&                 kv_cache_config,
         const RuntimeConfig&                 runtime_config,
         const ParallelismConfig&             parallelism_config,
         const SpeculativeExecutionConfig&    sp_config,
         std::shared_ptr<BroadcastManager>    broadcast_manager,
         std::shared_ptr<kvcm::ClientWrapper> client_wrapper):
        cache_config_(cache_config),
        kv_cache_config_(kv_cache_config),
        runtime_config_(runtime_config),
        parallelism_config_(parallelism_config),
        sp_config_(sp_config),
        broadcast_manager_(std::move(broadcast_manager)),
        client_wrapper_(std::move(client_wrapper)),
        sdk_check_enabled_(autil::EnvUtil::getEnv("KVCM_SDK_CHECK", autil::EnvUtil::getEnv("RECO_SDK_CHECK", false))) {}

    bool init(const CacheTopology&                                                topology,
              StorageBackend::BufferResolver                                      buffer_resolver,
              const std::function<const DeviceBlockPoolPtr&(const std::string&)>& pool_resolver) {
        RTP_LLM_LOG_INFO("start init BlockTree KVCM storage backend");
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
            group_policy_ = std::make_unique<kvcm::FullLinearLayerGroupPolicy>(topology,
                                                                               buffer_resolver,
                                                                               full_group_tags,
                                                                               other_group_tags,
                                                                               std::max(1, cache_config_.linear_step),
                                                                               std::move(group_block_size_bytes));
        }
        if (!group_policy_->init()) {
            RTP_LLM_LOG_ERROR("BlockTree KVCM group policy init failed");
            return false;
        }
        kvcm::ClientWrapper::ConfigMap client_config_map;
        try {
            if (!kv_cache_config_.kvcm_client_config.empty()) {
                // Custom client JSON owns the serialized location specs, while
                // the runtime policy still needs the same deterministic spec
                // name-to-group/rank mapping for payload routing.
                (void)genLocationSpecs();
                autil::legacy::FromJsonString(client_config_map, kv_cache_config_.kvcm_client_config);
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
        default_query_type_ = config->default_query_type();
        const int query_type = resolveQueryType(kv_cache_config_.kvcm_query_type);
        if (default_query_type_ < 1 || default_query_type_ > 4 || query_type < 1 || query_type > 4
            || kv_cache_config_.kvcm_min_replica_count < 0
            || (query_type == 3 && kv_cache_config_.kvcm_sw_size <= 0)
            || (query_type == 4 && (other_group_tags.empty() || has_swa_))
            || !config->sdk_wrapper_config() || !config->sdk_wrapper_config()->drain_on_timeout()
            || (kv_cache_config_.kvcm_read_backend_type != 0
                && (!isPayloadBackend(kv_cache_config_.kvcm_read_backend_type)
                    || kv_cache_config_.kvcm_query_type > 1))) {
            RTP_LLM_LOG_ERROR("invalid KVCM query/replica/backend config or drain_on_timeout is disabled");
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
        const auto tp_rank = parallelism_config_.tp_rank;
        RTP_LLM_LOG_INFO("BlockTree KVCM storage backend initialized, tp_rank=%ld tp_size=%ld policy={%s}",
                         tp_rank,
                         parallelism_config_.tp_size,
                         group_policy_->debugString().c_str());
        return true;
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
        const auto trace_id       = nextTraceId("match", match_trace_sequence_);
        const auto query_type = static_cast<kv_cache_manager::QueryType>(
            resolveQueryType(kv_cache_config_.kvcm_query_type));
        bool success = false;
        kv_cache_manager::Locations locations;
        bool positional = query_type == kv_cache_manager::QueryType::QT_BATCH_GET
                          || query_type == kv_cache_manager::QueryType::QT_REVERSE_ROLL_SW_MATCH;
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
            kv_cache_manager::ForwardContext context;
            context.sw_size = kv_cache_config_.kvcm_sw_size;
            auto result = client_wrapper_->match(
                "", trace_id, query_type, keys, request.local_matched_blocks_num, context);
            success = result.first;
            locations = std::move(result.second);
        }
        if (!success) {
            return {request.local_matched_blocks_num, nullptr};
        }
        if (positional || has_swa_) {
            auto meta = std::make_shared<KVCMMatchMeta>();
            meta->locations = reusableLocations(request, std::move(locations), positional);
            meta->filtered = true;
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
        return {request.local_matched_blocks_num + locations_view.size(), std::move(meta)};
    }

    void read(const StorageRequest& request, const std::shared_ptr<StorageBackendMatchMeta>& match_meta) {
        RTP_LLM_CHECK_WITH_INFO(parallelism_config_.tp_rank == 0,
                                "KVCM metadata read must run on tp rank 0, got %ld",
                                parallelism_config_.tp_rank);
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
                const StorageBlockHandle* handle = findHandle(request.handles[key_idx], info->second.tag);
                RTP_LLM_CHECK_WITH_INFO(handle != nullptr,
                                        "KVCM read has no destination handle for key=%zu tag=%s",
                                        key_idx,
                                        info->second.tag.c_str());
                auto* remote = requests.at(static_cast<size_t>(info->second.tp_rank)).mutable_remote_request();
                remote->add_group_tags(info->second.tag);
                remote->add_block_ids(handle->block);
                remote->add_uris(std::string(location_spec.uri));
            }
        }
        (void)dispatchRequests(requests, kv_cache_config_.kvcm_get_broadcast_timeout);
    }

    void write(const StorageRequest& request) {
        RTP_LLM_CHECK_WITH_INFO(parallelism_config_.tp_rank == 0,
                                "KVCM metadata write must run on tp rank 0, got %ld",
                                parallelism_config_.tp_rank);
        RTP_LLM_CHECK(request.keys != nullptr && request.keys->size() == request.handles.size());
        const size_t valid_keys_size = request.keys->size();
        if (valid_keys_size == 0) {
            return;
        }
        CacheKeysType            keys(request.keys->begin(), request.keys->begin() + valid_keys_size);
        std::vector<std::string> location_spec_group_names;
        RTP_LLM_CHECK_WITH_INFO(group_policy_->getNeedWriteGroups(request, valid_keys_size, location_spec_group_names),
                                "KVCM write group selection failed");
        const std::string trace_id     = nextTraceId("write", write_trace_sequence_);
        auto [success, write_location] = client_wrapper_->getWriteLocation(
            "", trace_id, keys, /*tokens=*/{}, location_spec_group_names, /*write_timeout_seconds=*/600,
            kv_cache_config_.kvcm_min_replica_count);
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
                    finish_attempted = true;
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
                    const size_t rank   = static_cast<size_t>(info->second.tp_rank);
                    auto*        remote = requests.at(rank).mutable_remote_request();
                    remote->add_group_tags(info->second.tag);
                    remote->add_block_ids(handle->block);
                    remote->add_uris(location_spec.uri);
                    actual_uri_gather[rank].push_back(&location_spec);
                }
            }

            const auto responses      = dispatchRequests(requests, kv_cache_config_.kvcm_put_broadcast_timeout);
            bool       has_actual_uri = false;
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
            finish_attempted             = true;
            RTP_LLM_CHECK_WITH_INFO(
                client_wrapper_->finishWrite("",
                                             nextTraceId("finish_write", finish_write_trace_sequence_),
                                             write_location.write_session_id,
                                             write_location.locations.size(),
                                             actual_locations),
                "KVCM FinishWrite failed");
        } catch (...) {
            if (!finish_attempted) {
                try {
                    if (!client_wrapper_->finishWrite("",
                                                      nextTraceId("abort_write", finish_write_trace_sequence_),
                                                      write_location.write_session_id,
                                                      /*block_mask=*/kv_cache_manager::BlockMaskOffset{0},
                                                      empty_locations)) {
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

    bool ownsAllocator() const {
        return parallelism_config_.tp_rank == 0;
    }

    bool execute(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response) {
        if (request.op() != REMOTE_OPERATION_READ && request.op() != REMOTE_OPERATION_WRITE) {
            return executeMetadata(request, response);
        }
        const std::vector<std::string>    tags(request.group_tags().begin(), request.group_tags().end());
        const std::vector<int32_t>        blocks(request.block_ids().begin(), request.block_ids().end());
        const kv_cache_manager::UriStrVec uris(request.uris().begin(), request.uris().end());
        if (tags.size() != blocks.size() || blocks.size() != uris.size()) {
            RTP_LLM_LOG_WARNING("KVCM transfer tag/block/URI count mismatch");
            return false;
        }
        setCudaDevice();
        kv_cache_manager::BlockBuffers buffers;
        if (!group_policy_->genBlockBuffers(tags, blocks, buffers)) {
            return false;
        }
        return executeTagTransfers(request.op(), tags, blocks, uris, buffers, response);
    }

    void shutdown() noexcept {
        if (client_wrapper_) {
            client_wrapper_->shutdown();
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
                                                   kv_cache_manager::Locations locations,
                                                   bool positional) const {
        const size_t local = request.local_matched_blocks_num;
        const size_t key_count = request.keys->size();
        if (positional) {
            RTP_LLM_CHECK_WITH_INFO(locations.size() == key_count, "KVCM positional query shape mismatch");
        } else {
            RTP_LLM_CHECK_WITH_INFO(locations.size() <= key_count - local, "KVCM prefix query shape mismatch");
            locations.insert(locations.begin(), local, kv_cache_manager::Location{});
        }
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
                    const auto name = kvcm::genLocationSpecName(rank, group.group_name);
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
        if (type < 1 || type > 4 || query.detail_level() < 0 || query.p2p_host_count() < 0) {
            return false;
        }
        if (type == 4 && has_swa_
            && (request.op() == REMOTE_OPERATION_MATCH_LOCATION
                || request.op() == REMOTE_OPERATION_MATCH_LOCATION_LEN
                || request.op() == REMOTE_OPERATION_GET_HOST_CACHE_STATE)) {
            RTP_LLM_LOG_WARNING("KVCM Mamba queries require a FULL+LINEAR layout without SWA groups");
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
            case REMOTE_OPERATION_MATCH_LOCATION_LEN: {
                auto [success, length] = client_wrapper_->matchLocationLen(
                    "", request.trace_id(), query_type, keys, tokens, query.sw_size());
                if (success) {
                    response.set_matched_blocks(length);
                }
                return success;
            }
            case REMOTE_OPERATION_MATCH_META: {
                auto [success, metas] = client_wrapper_->matchMeta(
                    "", request.trace_id(), keys, tokens, mask, query.detail_level());
                if (success) {
                    for (const auto& location : metas.locations) {
                        appendLocation(location, response.add_locations());
                    }
                    for (const auto& meta : metas.metas) {
                        response.add_metas(meta);
                    }
                }
                return success;
            }
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
                if (type != 2 && type != 4) {
                    RTP_LLM_LOG_WARNING("KVCM host-state queries require prefix or Mamba mode, got %d", type);
                    return false;
                }
                const std::vector<std::string> medium(query.medium().begin(), query.medium().end());
                auto [success, hosts] = client_wrapper_->getHostCacheState(
                    "", request.trace_id(), query_type, keys, medium, query.p2p_host_count());
                if (success) {
                    for (const auto& host : hosts) {
                        auto* output = response.add_hosts();
                        output->set_host_ip_port(host.host_ip_port);
                        output->set_local(host.local);
                        output->set_p2p_1_fetch(host.p2p_1_fetch);
                        output->set_p2p_1_total_match(host.p2p_1_total_match);
                    }
                }
                return success;
            }
            case REMOTE_OPERATION_MATCH_LOCATION: {
                auto [success, locations] = client_wrapper_->queryLocations(
                    "", request.trace_id(), query_type, keys, tokens, mask, query.sw_size(), names);
                if (success) {
                    for (const auto& location : locations) {
                        appendLocation(location, response.add_locations());
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
            group_policy_->buildLocationSpecGroups(static_cast<int>(parallelism_config_.tp_size), *groups),
            "failed to build KVCM location spec groups");
        for (const auto& [group_id, group] : group_policy_->groups()) {
            for (int rank = 0; rank < parallelism_config_.tp_size; ++rank) {
                const std::string spec_name = kvcm::genLocationSpecName(rank, group.group_name);
                infos->emplace(spec_name, cache_config_.blockSizeBytesForGroup(group.tag));
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
                 << ";default_query_type:" << kv_cache_config_.kvcm_default_query_type
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
        config->set_default_query_type(kv_cache_config_.kvcm_default_query_type);
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
        std::vector<kvcm::ClientWrapper::PoolRegistration> registrations;
        for (int32_t group_id : ordered_groups) {
            const auto& tag  = groups.at(group_id).tag;
            const auto& pool = pool_resolver(tag);
            if (!pool->getBaseAddress() || pool->getTotalSizeBytes() == 0) {
                RTP_LLM_LOG_ERROR("KVCM group %d has no valid registration span", group_id);
                return {};
            }
            registration_tags_.push_back(tag);
            registrations.push_back({{pool->getBaseAddress(), pool->getTotalSizeBytes()},
                                     kvcm::genLocationSpecName(static_cast<int>(parallelism_config_.tp_rank),
                                                               groups.at(group_id).group_name)});
        }
        return registrations;
    }

    bool executeTagTransfers(RemoteOpType                       operation,
                             const std::vector<std::string>&    tags,
                             const std::vector<int32_t>&        blocks,
                             const kv_cache_manager::UriStrVec& uris,
                             kv_cache_manager::BlockBuffers&    buffers,
                             RemoteOperationResponsePB&         response) {
        if (operation != REMOTE_OPERATION_READ && operation != REMOTE_OPERATION_WRITE) {
            RTP_LLM_LOG_WARNING("KVCM transfer has invalid operation [%d]", operation);
            return false;
        }
        std::unordered_map<std::string, std::vector<size_t>> indices_by_tag;
        for (size_t index = 0; index < tags.size(); ++index) {
            indices_by_tag[tags[index]].push_back(index);
        }
        auto actual_uris = uris;
        for (const auto& tag : registration_tags_) {
            const auto found = indices_by_tag.find(tag);
            if (found == indices_by_tag.end()) {
                continue;
            }
            const auto&                    indices = found->second;
            kv_cache_manager::UriStrVec    batch_uris;
            kv_cache_manager::BlockBuffers batch_buffers;
            std::vector<int32_t>           batch_blocks;
            for (size_t index : indices) {
                batch_uris.push_back(uris[index]);
                batch_buffers.push_back(std::move(buffers[index]));
                batch_blocks.push_back(blocks[index]);
            }
            const auto trace_info = makeTransferTraceInfo(batch_blocks);
            if (operation == REMOTE_OPERATION_READ) {
                if (!client_wrapper_->loadKvCachesForTag(tag, batch_uris, batch_buffers, trace_info)) {
                    return false;
                }
            } else {
                auto [success, result] =
                    client_wrapper_->saveKvCachesForTag(tag, batch_uris, batch_buffers, trace_info);
                if (!success || (!result.empty() && result.size() != batch_uris.size())) {
                    return false;
                }
                for (size_t index = 0; index < result.size(); ++index) {
                    actual_uris[indices[index]] = std::move(result[index]);
                }
            }
        }
        if (operation == REMOTE_OPERATION_WRITE && actual_uris != uris) {
            for (auto& uri : actual_uris) {
                *response.add_actual_uris() = std::move(uri);
            }
        }
        return true;
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

    std::vector<FunctionResponsePB> dispatchRequests(const std::vector<FunctionRequestPB>& requests, int timeout_ms) {
        if (!broadcast_manager_) {
            RTP_LLM_CHECK_WITH_INFO(
                requests.size() == 1, "KVCM local transfer requires exactly one request, got %zu", requests.size());
            FunctionResponsePB response;
            RTP_LLM_CHECK_WITH_INFO(execute(requests.front().remote_request(), *response.mutable_remote_response()),
                                    "KVCM local transfer failed");
            return {std::move(response)};
        }
        auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                           const std::shared_ptr<grpc::ClientContext>& context,
                           const FunctionRequestPB&                    request,
                           grpc::CompletionQueue*                      queue) {
            return stub->AsyncExecuteFunction(context.get(), request, queue);
        };
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        // A gRPC deadline completes the client while a peer may still use its
        // physical blocks. Keep the controller's pins until peer I/O drains.
        auto result = broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
            requests, timeout_ms, rpc_call, /*enforce_rpc_deadline=*/false);
        RTP_LLM_CHECK_WITH_INFO(result != nullptr, "KVCM broadcast dispatch failed");
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        const bool in_budget = remaining > 0 && result->waitDone(static_cast<int>(remaining));
        if (!in_budget) {
            RTP_LLM_LOG_WARNING("KVCM broadcast exceeded %d ms; waiting for peer I/O to drain", timeout_ms);
            result->waitDone();
        }
        RTP_LLM_CHECK_WITH_INFO(in_budget, "KVCM broadcast timed out after peer I/O drained, timeout_ms=%d", timeout_ms);
        RTP_LLM_CHECK_WITH_INFO(result->success(), "KVCM broadcast transfer failed");
        return result->responses();
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
    CacheConfig                          cache_config_;
    KVCacheConfig                        kv_cache_config_;
    RuntimeConfig                        runtime_config_;
    ParallelismConfig                    parallelism_config_;
    SpeculativeExecutionConfig           sp_config_;
    std::shared_ptr<BroadcastManager>    broadcast_manager_;
    std::unique_ptr<kvcm::GroupPolicy>   group_policy_;
    std::shared_ptr<kvcm::ClientWrapper> client_wrapper_;
    std::vector<std::string>             registration_tags_;
    // Preserve KVCM's operation-local, one-based request
    // order. Abort and finish share a sequence because both call FinishWrite.
    std::atomic<uint64_t> match_trace_sequence_{1};
    std::atomic<uint64_t> read_trace_sequence_{1};
    std::atomic<uint64_t> write_trace_sequence_{1};
    std::atomic<uint64_t> finish_write_trace_sequence_{1};
    const bool            sdk_check_enabled_;
    const CacheTopology*  topology_ = nullptr;
    bool                  has_swa_ = false;
    int32_t               default_query_type_ = 2;
};

KVCMStorageBackend::KVCMStorageBackend(const CacheConfig&                   cache_config,
                                       const KVCacheConfig&                 kv_cache_config,
                                       const RuntimeConfig&                 runtime_config,
                                       const ParallelismConfig&             parallelism_config,
                                       const SpeculativeExecutionConfig&    sp_config,
                                       std::shared_ptr<BroadcastManager>    broadcast_manager,
                                       std::shared_ptr<kvcm::ClientWrapper> client_wrapper):
    StorageBackend(makeStorageBackendExecutor(kv_cache_config.kvcm_asyncwrapper_thread_num,
                                              kv_cache_config.kvcm_asyncwrapper_queue_size)),
    impl_(std::make_unique<Impl>(cache_config,
                                 kv_cache_config,
                                 runtime_config,
                                 parallelism_config,
                                 sp_config,
                                 std::move(broadcast_manager),
                                 std::move(client_wrapper))) {}

KVCMStorageBackend::~KVCMStorageBackend() = default;

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

bool KVCMStorageBackend::execute(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response) {
    try {
        StorageWriteTask pins;
        std::vector<DeviceBlockPoolPtr> follower_pools;
        if ((request.op() == REMOTE_OPERATION_READ || request.op() == REMOTE_OPERATION_WRITE)
            && request.group_tags_size() == request.block_ids_size()
            && request.block_ids_size() == request.uris_size()) {
            StorageRequest transfer;
            transfer.keys = std::make_shared<CacheKeysType>(request.block_ids_size(), 0);
            transfer.handles.resize(request.block_ids_size());
            for (int i = 0; i < request.block_ids_size(); ++i) {
                transfer.handles[i].push_back({request.group_tags(i), request.block_ids(i)});
                if (!impl_->ownsAllocator()) {
                    const auto& pool = devicePool(request.group_tags(i));
                    RTP_LLM_CHECK_WITH_INFO(!isNullBlockIdx(request.block_ids(i))
                                               && pool->validBlock(request.block_ids(i)),
                                            "KVCM follower received an invalid physical block [%d]", request.block_ids(i));
                    follower_pools.push_back(pool);
                }
            }
            // Only the controller owns allocation metadata. Followers retain
            // their backing pools while the controller pins the shared IDs.
            if (impl_->ownsAllocator()) {
                pins = prepareWrite(std::move(transfer));
            }
        }
        return impl_->execute(request, response);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("KVCM remote operation failed: %s", error.what());
        return false;
    }
}

}  // namespace rtp_llm
