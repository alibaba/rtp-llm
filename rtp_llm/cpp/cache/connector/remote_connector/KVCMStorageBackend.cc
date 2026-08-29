#include "rtp_llm/cpp/cache/connector/remote_connector/KVCMStorageBackend.h"

#include <algorithm>
#include <atomic>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "autil/EnvUtil.h"
#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/ClientWrapper.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/GroupPolicy.h"
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

std::string nextTraceId(const char* operation) {
    static std::atomic<uint64_t> sequence{0};
    return std::string("block_tree_") + operation + "_" + std::to_string(sequence.fetch_add(1));
}

struct KVCMMatchMeta final: StorageBackendMatchMeta {
    kv_cache_manager::Locations locations;
};

const StorageBlockHandle* findHandle(const std::vector<StorageBlockHandle>& handles, size_t group_id) {
    const auto it = std::find_if(handles.begin(), handles.end(), [group_id](const StorageBlockHandle& handle) {
        return handle.group_id == group_id && !isNullBlockIdx(handle.block);
    });
    return it == handles.end() ? nullptr : &*it;
}

}  // namespace

class KVCMStorageBackend::Impl {
public:
    using ActualUriGather = std::vector<std::vector<kv_cache_manager::LocationSpecUnit*>>;

    Impl(const CacheConfig&                cache_config,
         const KVCacheConfig&              kv_cache_config,
         const RuntimeConfig&              runtime_config,
         const ParallelismConfig&          parallelism_config,
         const SpeculativeExecutionConfig& sp_config,
         std::shared_ptr<BroadcastManager> broadcast_manager):
        cache_config_(cache_config),
        kv_cache_config_(kv_cache_config),
        runtime_config_(runtime_config),
        parallelism_config_(parallelism_config),
        sp_config_(sp_config),
        broadcast_manager_(std::move(broadcast_manager)) {}

    bool init(const CacheTopology&                   topology,
              StorageBackend::BufferResolver         buffer_resolver,
              const std::vector<DeviceBlockPoolPtr>& device_pools) {
        RTP_LLM_LOG_INFO("start init BlockTree KVCM storage backend");
        if (cache_config_.use_independent_block_pools) {
            RTP_LLM_LOG_ERROR("BlockTree KVCM does not support independent device block pools");
            return false;
        }
        std::vector<int32_t> full_group_ids;
        std::vector<int32_t> other_group_ids;
        for (int32_t group_id = 0; group_id < cache_config_.groupNums(); ++group_id) {
            if (cache_config_.typeForGroup(static_cast<size_t>(group_id)) == CacheGroupType::FULL) {
                full_group_ids.push_back(group_id);
            } else {
                other_group_ids.push_back(group_id);
            }
        }
        if (other_group_ids.empty()) {
            group_policy_ = std::make_unique<remote_connector::FullLayerGroupPolicy>(
                topology, buffer_resolver, full_group_ids, other_group_ids);
        } else {
            group_policy_ = std::make_unique<remote_connector::FullLinearLayerGroupPolicy>(
                topology, buffer_resolver, full_group_ids, other_group_ids, std::max(1, cache_config_.linear_step));
        }
        if (!group_policy_->init()) {
            RTP_LLM_LOG_ERROR("BlockTree KVCM group policy init failed");
            return false;
        }
        remote_connector::ClientWrapper::ConfigMap client_config_map;
        try {
            if (!kv_cache_config_.reco_client_config.empty()) {
                // Custom client JSON owns the serialized location specs, while
                // the runtime policy still needs the same deterministic spec
                // name-to-group/rank mapping for payload routing.
                (void)genLocationSpecs();
                autil::legacy::FromJsonString(client_config_map, kv_cache_config_.reco_client_config);
            } else {
                client_config_map = genClientConfig();
            }
        } catch (const autil::legacy::ExceptionBase& error) {
            RTP_LLM_LOG_ERROR("parse RECO_CLIENT_CONFIG failed: %s", error.what());
            return false;
        } catch (const std::exception& error) {
            RTP_LLM_LOG_ERROR("initialize BlockTree KVCM client config failed: %s", error.what());
            return false;
        }

        const auto registration_pool = chooseRegistrationPool(device_pools);
        if (!registration_pool || !registration_pool->getBaseAddress() || registration_pool->getTotalSizeBytes() == 0) {
            RTP_LLM_LOG_ERROR("BlockTree KVCM has no valid device registration pool");
            return false;
        }
        RTP_LLM_CHECK_WITH_INFO(!group_policy_->groups().empty(), "KVCM requires at least one cache group");
        const auto registration_group = std::min_element(
            group_policy_->groups().begin(), group_policy_->groups().end(), [](const auto& lhs, const auto& rhs) {
                return std::make_pair(!lhs.second.is_full, lhs.second.group_name)
                       < std::make_pair(!rhs.second.is_full, rhs.second.group_name);
            });
        kv_cache_manager::RegistSpan regist_span{registration_pool->getBaseAddress(),
                                                 registration_pool->getTotalSizeBytes()};
        const int64_t                tp_rank = parallelism_config_.tp_rank;
        kv_cache_manager::InitParams client_init_params{
            tp_rank == 0 ? kv_cache_manager::RoleType::HYBRID : kv_cache_manager::RoleType::WORKER,
            &regist_span,
            remote_connector::genLocationSpecName(static_cast<int>(tp_rank), registration_group->second.group_name)};
        client_wrapper_ = std::make_shared<remote_connector::ClientWrapper>();
        if (!client_wrapper_->init(client_config_map, client_init_params)) {
            RTP_LLM_LOG_ERROR("create BlockTree KVCM client failed");
            return false;
        }
        if (tp_rank == 0 && parallelism_config_.tp_size > 1 && !broadcast_manager_) {
            RTP_LLM_LOG_ERROR("BlockTree KVCM rank 0 requires a broadcast manager for tp_size=%ld",
                              parallelism_config_.tp_size);
            return false;
        }
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
        CacheKeysType keys = *request.keys;
        if (!keys.empty()) {
            keys.pop_back();
        }
        if (request.local_matched_blocks_num >= keys.size()) {
            return {request.local_matched_blocks_num, nullptr};
        }
        const auto trace_id       = nextTraceId("match");
        auto [success, locations] = client_wrapper_->match(
            "", trace_id, kv_cache_manager::QueryType::QT_PREFIX_MATCH, keys, request.local_matched_blocks_num, {});
        if (!success) {
            throw std::runtime_error("KVCM prefix match failed");
        }
        remote_connector::LocationsView locations_view;
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
        remote_connector::LocationsView locations_view;
        RTP_LLM_CHECK_WITH_INFO(
            group_policy_->filterNeedLoadLocations(meta->locations, locations_view, /*block_mask=*/0),
            "KVCM read location filtering failed");
        const size_t remote_blocks = request.handles.size() - request.local_matched_blocks_num;
        RTP_LLM_CHECK_WITH_INFO(locations_view.size() == remote_blocks,
                                "KVCM read shape mismatch: locations=%zu remote_blocks=%zu",
                                locations_view.size(),
                                remote_blocks);

        std::vector<FunctionRequestPB> requests(static_cast<size_t>(parallelism_config_.tp_size));
        const auto&                    spec_info = group_policy_->spec_info_map();
        const std::string              trace_id  = nextTraceId("read");
        initializeRequests(requests, REMOTE_OPERATION_READ, trace_id);
        for (size_t location_idx = 0; location_idx < locations_view.size(); ++location_idx) {
            const size_t key_idx = request.local_matched_blocks_num + location_idx;
            for (const auto& location_spec : locations_view[location_idx]) {
                const auto        info = spec_info.find(location_spec.spec_name);
                const std::string spec_name(location_spec.spec_name);
                RTP_LLM_CHECK_WITH_INFO(info != spec_info.end(), "KVCM read has unknown spec [%s]", spec_name.c_str());
                const StorageBlockHandle* handle = findHandle(request.handles[key_idx], info->second.group_id);
                RTP_LLM_CHECK_WITH_INFO(handle != nullptr,
                                        "KVCM read has no destination handle for key=%zu group=%d",
                                        key_idx,
                                        info->second.group_id);
                auto* remote = requests.at(static_cast<size_t>(info->second.tp_rank)).mutable_remote_request();
                remote->add_group_tags(info->second.tag);
                remote->add_block_ids(handle->block);
                remote->add_uris(std::string(location_spec.uri));
            }
        }
        (void)dispatchRequests(requests, kv_cache_config_.reco_get_broadcast_timeout);
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
        const std::string trace_id     = nextTraceId("write");
        auto [success, write_location] = client_wrapper_->getWriteLocation(
            "", trace_id, keys, /*tokens=*/{}, location_spec_group_names, /*write_timeout_seconds=*/600);
        RTP_LLM_CHECK_WITH_INFO(success, "KVCM StartWrite failed");
        if (write_location.locations.empty()) {
            return;
        }

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
            const auto& spec_info = group_policy_->spec_info_map();
            for (size_t location_idx = 0; location_idx < write_location.locations.size(); ++location_idx) {
                const size_t key_idx = key_indices[location_idx];
                for (auto& location_spec : write_location.locations[location_idx]) {
                    const auto info = spec_info.find(location_spec.spec_name);
                    RTP_LLM_CHECK_WITH_INFO(
                        info != spec_info.end(), "KVCM write has unknown spec [%s]", location_spec.spec_name.c_str());
                    const StorageBlockHandle* handle = findHandle(request.handles[key_idx], info->second.group_id);
                    RTP_LLM_CHECK_WITH_INFO(handle != nullptr,
                                            "KVCM write has no source handle for key=%zu group=%d",
                                            key_idx,
                                            info->second.group_id);
                    const size_t rank   = static_cast<size_t>(info->second.tp_rank);
                    auto*        remote = requests.at(rank).mutable_remote_request();
                    remote->add_group_tags(info->second.tag);
                    remote->add_block_ids(handle->block);
                    remote->add_uris(location_spec.uri);
                    actual_uri_gather[rank].push_back(&location_spec);
                }
            }

            const auto responses      = dispatchRequests(requests, kv_cache_config_.reco_put_broadcast_timeout);
            bool       has_actual_uri = false;
            for (size_t rank = 0; rank < responses.size(); ++rank) {
                const auto& actual_uris = responses[rank].remote_response().actual_uris();
                RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(actual_uris.size()) <= actual_uri_gather[rank].size(),
                                        "KVCM write returned too many actual URIs for rank=%zu",
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
            RTP_LLM_CHECK_WITH_INFO(client_wrapper_->finishWrite("",
                                                                 nextTraceId("finish_write"),
                                                                 write_location.write_session_id,
                                                                 write_location.locations.size(),
                                                                 actual_locations),
                                    "KVCM FinishWrite failed");
        } catch (...) {
            if (!finish_attempted) {
                try {
                    if (!client_wrapper_->finishWrite("",
                                                      nextTraceId("abort_write"),
                                                      write_location.write_session_id,
                                                      /*block_mask=*/0,
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

    bool execute(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response) {
        const std::vector<std::string>    tags(request.group_tags().begin(), request.group_tags().end());
        const std::vector<int32_t>        blocks(request.block_ids().begin(), request.block_ids().end());
        const kv_cache_manager::UriStrVec uris(request.uris().begin(), request.uris().end());
        if (tags.size() != blocks.size() || blocks.size() != uris.size()) {
            RTP_LLM_LOG_WARNING("KVCM transfer tag/block/URI count mismatch");
            return false;
        }
        setCudaDevice();
        kv_cache_manager::BlockBuffers buffers;
        if (!group_policy_->genBlockBuffersByTag(tags, blocks, buffers)) {
            return false;
        }
        if (request.op() == REMOTE_OPERATION_READ) {
            return client_wrapper_->loadKvCaches(uris, buffers);
        }
        if (request.op() == REMOTE_OPERATION_WRITE) {
            auto [success, actual_uris] = client_wrapper_->saveKvCaches(uris, buffers);
            if (!success || (!actual_uris.empty() && actual_uris.size() != uris.size())) {
                return false;
            }
            for (auto& uri : actual_uris) {
                *response.add_actual_uris() = std::move(uri);
            }
            return true;
        }
        RTP_LLM_LOG_WARNING("KVCM transfer has invalid operation [%d]", request.op());
        return false;
    }

    void shutdown() noexcept {
        if (client_wrapper_) {
            client_wrapper_->shutdown();
        }
    }

private:
    std::pair<std::shared_ptr<RemoteConnectorConfig::LocationSpecInfoMap>,
              std::shared_ptr<RemoteConnectorConfig::LocationSpecGroups>>
    genLocationSpecs() {
        auto infos  = std::make_shared<RemoteConnectorConfig::LocationSpecInfoMap>();
        auto groups = std::make_shared<RemoteConnectorConfig::LocationSpecGroups>();
        RTP_LLM_CHECK_WITH_INFO(
            group_policy_->buildLocationSpecGroups(static_cast<int>(parallelism_config_.tp_size), *groups),
            "failed to build KVCM location spec groups");
        for (const auto& [group_id, group] : group_policy_->groups()) {
            for (int rank = 0; rank < parallelism_config_.tp_size; ++rank) {
                const std::string spec_name = remote_connector::genLocationSpecName(rank, group.group_name);
                infos->emplace(spec_name, cache_config_.blockSizeBytesForGroup(static_cast<size_t>(group_id)));
            }
        }
        return {std::move(infos), std::move(groups)};
    }

    remote_connector::ClientWrapper::ConfigMap genClientConfig() {
        std::vector<std::string> addresses;
        if (!kv_cache_config_.reco_server_address.empty()) {
            addresses.push_back(kv_cache_config_.reco_server_address);
        }
        auto channel = std::make_shared<MetaChannelConfig>(kv_cache_config_.reco_meta_channel_retry_time,
                                                           kv_cache_config_.reco_meta_channel_connection_timeout,
                                                           kv_cache_config_.reco_meta_channel_call_timeout);
        auto sdk     = std::make_shared<SdkWrapperConfig>(kv_cache_config_.reco_storage_thread_num,
                                                      kv_cache_config_.reco_storage_queue_size,
                                                      kv_cache_config_.reco_put_timeout_ms,
                                                      kv_cache_config_.reco_get_timeout_ms);
        autil::legacy::FromJsonString(sdk->sdk_backend_configs(), kv_cache_config_.reco_model_sdk_config);
        auto [location_infos, location_groups] = genLocationSpecs();

        const std::string model_name = runtime_config_.model_name;
        const std::string dtype      = getDataTypeStr(cache_config_.dtype);
        std::string       extra      = kv_cache_config_.reco_model_extra_info;
        extra += '/' + autil::EnvUtil::getEnv("BIZ_NAME", std::string("")) + '/'
                 + std::to_string(hashString(autil::EnvUtil::getEnv("CHECKPOINT_PATH", std::string(""))));
        std::string draft_info;
        if (!cache_config_.mtp_sub_configs.empty()) {
            draft_info = '{' + sp_config_.to_string() + '}';
        }
        std::stringstream identity;
        identity << "instance_group: " << kv_cache_config_.reco_instance_group
                 << ";block_size:" << cache_config_.seq_size_per_block << ";model_name:" << model_name
                 << ";dtype_str:" << dtype << ";use_mla:" << cache_config_.use_mla
                 << ";fp8_kv_cache:" << kv_cache_config_.fp8_kv_cache << ";tp_size:" << parallelism_config_.tp_size
                 << ";dp_size:" << parallelism_config_.dp_size << ";extra_info:" << extra
                 << ";location_spec_info:" << autil::legacy::ToJsonString(location_infos, true)
                 << ";draft_model_info:" << draft_info;
        std::string instance_id = kv_cache_config_.reco_instance_id_salt;
        if (!instance_id.empty()) {
            instance_id += '_';
        }
        instance_id += std::to_string(hashString(identity.str()));

        auto config =
            std::make_shared<RemoteConnectorConfig>(kv_cache_config_.reco_enable_vipserver,
                                                    kv_cache_config_.reco_vipserver_domain,
                                                    static_cast<int32_t>(cache_config_.seq_size_per_block),
                                                    kv_cache_config_.reco_instance_group,
                                                    instance_id,
                                                    addresses,
                                                    location_infos,
                                                    channel,
                                                    sdk,
                                                    location_groups,
                                                    ModelDeployment(model_name,
                                                                    dtype,
                                                                    cache_config_.use_mla,
                                                                    static_cast<int32_t>(parallelism_config_.tp_size),
                                                                    static_cast<int32_t>(parallelism_config_.dp_size),
                                                                    1,
                                                                    extra,
                                                                    kv_cache_config_.reco_model_user_data));
        return {{"", std::move(config)}};
    }

    DeviceBlockPoolPtr chooseRegistrationPool(const std::vector<DeviceBlockPoolPtr>& pools) const {
        if (pools.empty()) {
            return nullptr;
        }
        const auto* expected_pool = pools.front().get();
        if (expected_pool == nullptr
            || std::any_of(pools.begin(), pools.end(), [expected_pool](const DeviceBlockPoolPtr& pool) {
                   return pool.get() != expected_pool;
               })) {
            RTP_LLM_LOG_ERROR("BlockTree KVCM requires one shared contiguous device pool");
            return nullptr;
        }
        return pools.front();
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
        auto result =
            broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(requests, timeout_ms, rpc_call);
        RTP_LLM_CHECK_WITH_INFO(result != nullptr, "KVCM broadcast dispatch failed");
        result->waitDone();
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

private:
    CacheConfig                                      cache_config_;
    KVCacheConfig                                    kv_cache_config_;
    RuntimeConfig                                    runtime_config_;
    ParallelismConfig                                parallelism_config_;
    SpeculativeExecutionConfig                       sp_config_;
    std::shared_ptr<BroadcastManager>                broadcast_manager_;
    std::unique_ptr<remote_connector::GroupPolicy>   group_policy_;
    std::shared_ptr<remote_connector::ClientWrapper> client_wrapper_;
};

KVCMStorageBackend::KVCMStorageBackend(const CacheConfig&                cache_config,
                                       const KVCacheConfig&              kv_cache_config,
                                       const RuntimeConfig&              runtime_config,
                                       const ParallelismConfig&          parallelism_config,
                                       const SpeculativeExecutionConfig& sp_config,
                                       std::shared_ptr<BroadcastManager> broadcast_manager):
    StorageBackend(makeStorageBackendExecutor(kv_cache_config.reco_asyncwrapper_thread_num,
                                              kv_cache_config.reco_asyncwrapper_queue_size)),
    impl_(std::make_unique<Impl>(
        cache_config, kv_cache_config, runtime_config, parallelism_config, sp_config, std::move(broadcast_manager))) {}

KVCMStorageBackend::~KVCMStorageBackend() = default;

bool KVCMStorageBackend::initImpl() {
    return impl_->init(
        topology(),
        [this](int layer_id, int group_id, int block_id) { return convertIndexToBuffer(layer_id, group_id, block_id); },
        devicePools());
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
    return impl_->execute(request, response);
}

}  // namespace rtp_llm
