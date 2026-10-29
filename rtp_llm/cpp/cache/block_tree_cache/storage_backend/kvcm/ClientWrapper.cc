#include "ClientWrapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/SdkBufferSafety.h"
#include "kvcm_client/HostCacheStateCompat.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/Subscriber.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"
#include <algorithm>
#include <exception>
#include <numeric>
#include <pthread.h>
#include <thread>
#include <random>
#include <chrono>
#include <unistd.h>
#include <tuple>
#include "autil/EnvUtil.h"

namespace rtp_llm {
namespace kvcm {

namespace {

// PACE has a finite Consumer task ring. A block may carry many layer IOVs;
// bounding only bytes or block count does not bound the number of tasks.
struct SdkRange {
    size_t begin, end, bytes, iovs;
};
bool boundedSdkRanges(const kv_cache_manager::UriStrVec&    uris,
                      const kv_cache_manager::BlockBuffers& buffers,
                      std::vector<SdkRange>&                ranges) {
    constexpr size_t max_bytes = 64 * 1024 * 1024, max_blocks = 256, max_iovs = 1024;
    if (uris.size() != buffers.size()) {
        return false;
    }
    for (size_t begin = 0; begin < buffers.size();) {
        size_t end = begin, bytes = 0, iovs = 0;
        while (end < buffers.size()) {
            size_t     next_bytes = 0;
            const auto next_iovs  = buffers[end].iovs.size();
            if (next_iovs > max_iovs) {
                return false;
            }
            for (const auto& iov : buffers[end].iovs) {
                if (iov.size > max_bytes - next_bytes) {
                    return false;
                }
                next_bytes += iov.size;
            }
            if (end > begin
                && (end - begin >= max_blocks || next_bytes > max_bytes - bytes || next_iovs > max_iovs - iovs)) {
                break;
            }
            bytes += next_bytes;
            iovs += next_iovs;
            ++end;
        }
        ranges.push_back({begin, end, bytes, iovs});
        begin = end;
    }
    return true;
}

std::shared_ptr<kv_cache_manager::TransferTraceInfo>
traceForRange(const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace, const SdkRange& range) {
    if (!trace || trace->block_ids.empty())
        return trace;
    auto result       = std::make_shared<kv_cache_manager::TransferTraceInfo>(*trace);
    result->block_ids = {trace->block_ids.begin() + range.begin, trace->block_ids.begin() + range.end};
    return result;
}

class ReinitPolicy {
public:
    ReinitPolicy(): rd_(), gen_(rd_()), jitter_dist_(jitter_min_, jitter_max_) {}

    int sleep_time_ms() {
        int sleep_time_ms = next_sleep_time_ms_ + jitter_dist_(gen_);
        next_sleep_time_ms_ *= multiplier_;
        next_sleep_time_ms_ = std::min(next_sleep_time_ms_, max_delay_ms_);
        return sleep_time_ms;
    }

private:
    static constexpr int            base_delay_ms_      = 20;
    static constexpr int            max_delay_ms_       = 20000;
    static constexpr double         multiplier_         = 1.5;
    static constexpr int            jitter_min_         = 0;
    static constexpr int            jitter_max_         = 2000;
    int                             next_sleep_time_ms_ = base_delay_ms_;
    std::random_device              rd_;
    std::mt19937                    gen_;
    std::uniform_int_distribution<> jitter_dist_;
};

}  // namespace

ClientWrapper::ClientWrapper(std::unique_ptr<ClientFactory> client_factory):
    client_factory_(std::move(client_factory)) {
    RTP_LLM_CHECK_WITH_INFO(client_factory_ != nullptr, "ClientWrapper requires a client factory");
}

ClientWrapper::~ClientWrapper() {
    shutdown();
}

bool ClientWrapper::init(const ConfigMap&                    config_map,
                         const kv_cache_manager::InitParams& init_params,
                         const std::string&                  tag) {
    return initImpl(config_map, init_params, {}, {tag});
}

bool ClientWrapper::initForPools(const ConfigMap&                     config_map,
                                 kv_cache_manager::RoleType           role,
                                 const std::vector<PoolRegistration>& registrations,
                                 const std::vector<std::string>&      tags) {
    if (registrations.empty()) {
        RTP_LLM_LOG_ERROR("KVCM requires at least one pool registration");
        return false;
    }
    for (const auto& registration : registrations) {
        if (!registration.span.base || registration.span.size == 0 || registration.location_spec_name.empty()) {
            RTP_LLM_LOG_ERROR("KVCM pool registration requires a valid span and location spec name");
            return false;
        }
    }
    auto                               first_span = registrations.front().span;
    const kv_cache_manager::InitParams params{role, &first_span, registrations.front().location_spec_name};
    return initImpl(config_map, params, registrations, tags);
}

bool ClientWrapper::initImpl(const ConfigMap&                     config_map,
                             const kv_cache_manager::InitParams&  init_params,
                             const std::vector<PoolRegistration>& registrations,
                             const std::vector<std::string>&      tags) {
    std::lock_guard<std::mutex>             shutdown_lock(shutdown_mutex_);
    std::unordered_map<std::string, size_t> pending_tags;
    if (tags.size() != (registrations.empty() ? 1 : registrations.size())) {
        return false;
    }
    for (size_t i = 0; i < tags.size(); ++i) {
        if (tags[i].empty() || !pending_tags.emplace(tags[i], i).second) {
            return false;
        }
    }
    if (config_map.size() != 1) {
        RTP_LLM_LOG_ERROR("KVCM requires exactly one endpoint config, got %zu", config_map.size());
        return false;
    }
    const auto& [unique_id, config] = *config_map.begin();
    if (!config || !config->meta_channel_config()) {
        RTP_LLM_LOG_ERROR("KVCM config [%s] requires a non-null config and meta_channel_config", unique_id.c_str());
        return false;
    }
    if (config->meta_channel_config()->retry_time() <= 0) {
        RTP_LLM_LOG_ERROR("KVCM config [%s] requires a positive meta_channel_config.retry_time", unique_id.c_str());
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(reinit_worker_mutex_);
        if (stop_requested_) {
            RTP_LLM_LOG_ERROR("ClientWrapper cannot initialize after shutdown");
            return false;
        }
    }
    if (!transfer_clients_.empty() || !config_map_.empty() || !meta_client_map_.empty()) {
        RTP_LLM_LOG_ERROR("ClientWrapper cannot be initialized more than once");
        return false;
    }
    auto initialize = [&]() {
        init_params_        = init_params;
        pool_registrations_ = registrations;
        if (pool_registrations_.empty()) {
            pool_registrations_.push_back(
                {init_params.regist_span ? *init_params.regist_span : kv_cache_manager::RegistSpan{},
                 init_params.self_location_spec_name});
        }
        if (init_params.regist_span != nullptr) {
            init_params_.regist_span = &pool_registrations_.front().span;
        }
        if (init_params_.role_type != kv_cache_manager::RoleType::HYBRID) {
            init_params_.role_type = kv_cache_manager::RoleType::SCHEDULER;
        }
        if (!initMetaClient(unique_id, config)) {
            return false;
        }
        init_params_.storage_configs = meta_client_map_.begin()->second->GetStorageConfig();
        RTP_LLM_LOG_INFO("transfer client storage config [%s]", init_params_.storage_configs.c_str());
        if (init_params_.role_type == kv_cache_manager::RoleType::SCHEDULER) {
            meta_client_map_.clear();
            init_params_.role_type = kv_cache_manager::RoleType::WORKER;
        }
        const auto config_json = autil::legacy::ToJsonString(config_map_.begin()->second);
        kv_cache_manager::ClientMemoryRegistrations memory_registrations;
        for (auto& registration : pool_registrations_) {
            if (!registration.memory_registrations) {
                continue;
            }
            for (const auto& gpu : registration.memory_registrations->gpu) {
                const auto duplicate = std::find_if(memory_registrations.gpu.begin(),
                                                    memory_registrations.gpu.end(),
                                                    [&](const auto& other) {
                                                        return gpu.base == other.base && gpu.size == other.size
                                                               && gpu.device_id == other.device_id;
                                                    });
                if (duplicate == memory_registrations.gpu.end()) {
                    memory_registrations.gpu.push_back(gpu);
                }
            }
            registration.memory_registrations->gpu.clear();
        }
        if (!memory_registrations.gpu.empty()) {
            // PACE seals the process's MR registrations during the first client Init.
            // Retain every GPU span in that client's descriptor before creating any client.
            auto& first = pool_registrations_.front().memory_registrations;
            if (!first) {
                first.emplace();
            }
            first->gpu = std::move(memory_registrations.gpu);
        }
        std::vector<std::unique_ptr<kv_cache_manager::TransferClient>> clients;
        clients.reserve(pool_registrations_.size());
        for (size_t index = 0; index < pool_registrations_.size(); ++index) {
            auto params = init_params_;
            if (index != 0) {
                params.role_type               = kv_cache_manager::RoleType::WORKER;
                params.regist_span             = &pool_registrations_[index].span;
                params.self_location_spec_name = pool_registrations_[index].location_spec_name;
            }
            auto client = pool_registrations_[index].memory_registrations.has_value()
                              ? client_factory_->createTransferClientWithMemory(
                                    config_json, params, *pool_registrations_[index].memory_registrations)
                              : client_factory_->createTransferClient(config_json, params);
            if (!client) {
                RTP_LLM_LOG_ERROR("init KVCM transfer client failed for pool %zu", index);
                return false;
            }
            if (transfer_observer_) {
                if (auto* observable = dynamic_cast<SdkTransferObservable*>(client.get())) {
                    observable->setSdkTransferObserver(transfer_observer_);
                }
            }
            clients.push_back(std::move(client));
        }
        transfer_clients_ = std::move(clients);
        tag_to_index_     = std::move(pending_tags);
        try {
            reinit_worker_ = std::thread([this] { reinitWorkerLoop(); });
        } catch (const std::exception& error) {
            RTP_LLM_LOG_ERROR("start KVCM re-registration worker failed: %s", error.what());
            transfer_clients_.clear();
            tag_to_index_.clear();
            return false;
        }
        return true;
    };
    const auto clear_failed_init = [&] {
        transfer_clients_.clear();
        {
            std::lock_guard<std::mutex> lock(pending_transfer_mutex_);
            pending_transfers_.reset();
        }
        tag_to_index_.clear();
        meta_client_map_.clear();
        subscriber_.reset();
        config_map_.clear();
        init_params_.regist_span = nullptr;
        pool_registrations_.clear();
    };
    try {
        if (initialize()) {
            return true;
        }
    } catch (...) {
        clear_failed_init();
        throw;
    }
    clear_failed_init();
    return false;
}

void ClientWrapper::shutdown() noexcept {
    std::lock_guard<std::mutex> shutdown_lock(shutdown_mutex_);
    {
        std::lock_guard<std::mutex> lock(reinit_worker_mutex_);
        stop_requested_   = true;
        reinit_requested_ = false;
    }
    reinit_worker_cv_.notify_all();
    if (reinit_worker_.joinable()) {
        reinit_worker_.join();
    }
    {
        std::unique_lock<std::shared_mutex> transfer_guard(transfer_mutex_);
        transfer_clients_.clear();
        tag_to_index_.clear();
        transfer_observer_ = {};
    }
    {
        std::unique_lock<std::shared_mutex> metadata_guard(rr_mutex_);
        meta_client_map_.clear();
        subscriber_.reset();
        config_map_.clear();
        init_params_.regist_span = nullptr;
        pool_registrations_.clear();
    }
}

bool ClientWrapper::initMetaClient(const std::string& unique_id, KVCMConfigPtr config) {
    RTP_LLM_LOG_INFO(
        "kvcm unique_id [%s], init config [%s]", unique_id.c_str(), autil::legacy::ToJsonString(config).c_str());
    const bool enable_vipserver = config->enable_vipserver();
    subscriber_                 = client_factory_->createSubscriber(enable_vipserver);
    if (!subscriber_) {
        RTP_LLM_LOG_ERROR("unique_id [%s] create subscriber failed", unique_id.c_str());
        return false;
    }
    if (enable_vipserver) {
        if (!subscriber_->init({config->vipserver_domain()})) {
            RTP_LLM_LOG_ERROR("unique_id [%s] init vipserver subscriber failed.", unique_id.c_str());
            return false;
        }
        if (!subscriber_->getAddresses(address_snapshot_)) {
            RTP_LLM_LOG_ERROR("unique_id [%s] get addresses failed", unique_id.c_str());
            return false;
        }
        config->set_addresses(address_snapshot_);
    } else {
        if (!subscriber_->init(config->addresses())) {
            RTP_LLM_LOG_ERROR("unique_id [%s] init direct subscriber failed.", unique_id.c_str());
            return false;
        }
    }
    if (config->addresses().empty()) {
        RTP_LLM_LOG_ERROR("empty kvcm addresses");
        return false;
    }
    auto real_config_str = autil::legacy::ToJsonString(config);
    RTP_LLM_LOG_INFO("init unique_id[%s], kvcm real config[%s]", unique_id.c_str(), real_config_str.c_str());
    config_map_[unique_id] = config;
    std::unique_ptr<kv_cache_manager::MetaClient> meta_client;
    for (int i = 1; i <= config->meta_channel_config()->retry_time(); ++i) {
        RTP_LLM_LOG_INFO("try meta client, try time[%d]", i);
        meta_client = client_factory_->createMetaClient(real_config_str, init_params_);
        if (meta_client) {
            break;
        }
    }
    if (meta_client == nullptr) {
        RTP_LLM_LOG_ERROR("create meta client failed");
        return false;
    }
    meta_client_map_[unique_id] = std::move(meta_client);
    return true;
}

bool ClientWrapper::reinit(const std::string&       unique_id,
                           ConfigMap::iterator&     config_iter,
                           MetaClientMap::iterator& meta_client_iter) {
    auto config = config_iter->second;
    config->set_addresses(address_snapshot_);
    auto real_config_str = autil::legacy::ToJsonString(*config);
    RTP_LLM_LOG_INFO("reinit unique_id[%s], kvcm real config[%s]", unique_id.c_str(), real_config_str.c_str());
    auto meta_client = client_factory_->createMetaClient(real_config_str, init_params_);
    if (meta_client == nullptr) {
        RTP_LLM_LOG_ERROR("create meta client failed");
        return false;
    }
    meta_client_iter->second = std::move(meta_client);
    return true;
}

bool ClientWrapper::tryReinit(const std::string& unique_id) {
    auto config_iter      = config_map_.find(unique_id);
    auto meta_client_iter = meta_client_map_.find(unique_id);
    if (config_iter == config_map_.end() || meta_client_iter == meta_client_map_.end()) {
        RTP_LLM_LOG_WARNING("not find unique_id [%s]", unique_id.c_str());
        return false;
    }
    if (config_iter->second->enable_vipserver()) {
        std::vector<std::string> addresses;
        if (!subscriber_->getAddresses(addresses)) {
            return false;
        }
        {
            std::shared_lock read_guard(reinit_mutex_);
            if (addresses == address_snapshot_ && addresses == config_iter->second->addresses()) {
                return true;
            }
        }
        {
            std::unique_lock write_guard(reinit_mutex_);
            // double check
            if (addresses == address_snapshot_ && addresses == config_iter->second->addresses()) {
                return true;
            }
            std::string current_address_str = "";
            std::string new_address_str     = "";
            auto        join_address        = [](auto a, const auto& b) { return a + "," + b; };
            if (address_snapshot_.size() > 0) {
                current_address_str = std::accumulate(
                    std::next(address_snapshot_.begin()), address_snapshot_.end(), address_snapshot_[0], join_address);
            }
            if (addresses.size() > 0) {
                new_address_str =
                    std::accumulate(std::next(addresses.begin()), addresses.end(), addresses[0], join_address);
            }
            RTP_LLM_LOG_INFO("ClientWrapper [%s] address changed, start reinit, current [%s], new [%s]",
                             unique_id.c_str(),
                             current_address_str.c_str(),
                             new_address_str.c_str());
            address_snapshot_.swap(addresses);
            if (!reinit(unique_id, config_iter, meta_client_iter)) {
                // clear address_snapshot, reinit next time
                address_snapshot_.clear();
                config_iter->second->set_addresses({});
                RTP_LLM_LOG_ERROR("ClientWrapper [%s] reinit failed", unique_id.c_str());
                return false;
            }
        }
        RTP_LLM_LOG_INFO("ClientWrapper [%s] reinit finish", unique_id.c_str());
    }
    return true;
}

std::shared_ptr<kv_cache_manager::MetaClient> ClientWrapper::getMetaClient(const std::string& unique_id) {
    if (!tryReinit(unique_id)) {
        return nullptr;
    }
    std::shared_lock lock(reinit_mutex_);
    return meta_client_map_.at(unique_id);
}

#define DEFER(...) __VA_ARGS__
#define CHECK_INIT_BASE(unique_id, return_value)                                                                       \
    std::shared_lock read_guard(rr_mutex_, std::try_to_lock);                                                          \
    if (!read_guard.owns_lock()) {                                                                                     \
        RTP_LLM_LOG_WARNING("doing re-registration");                                                                  \
        return return_value;                                                                                           \
    }                                                                                                                  \
    const auto client = getMetaClient(unique_id);                                                                      \
    if (!client) {                                                                                                     \
        return return_value;                                                                                           \
    }
#define CHECK_INIT2(unique_id) CHECK_INIT_BASE(unique_id, DEFER({false, {}}))
#define CHECK_INIT1(unique_id) CHECK_INIT_BASE(unique_id, false)

#define CALL_CLIENT2(function_name, ...)                                                                               \
    auto [ec, result] = client->function_name(__VA_ARGS__);                                                            \
    if (!checkError(ec)) {                                                                                             \
        RTP_LLM_LOG_WARNING(#function_name " fail, ec [%d]", ec);                                                      \
        return {false, {}};                                                                                            \
    }                                                                                                                  \
    return {true, std::move(result)};

std::pair<bool, kv_cache_manager::Locations>
ClientWrapper::match(const std::string&                      unique_id,
                     const std::string&                      trace_id,
                     kv_cache_manager::QueryType             query_type,
                     const std::vector<int64_t>&             keys,
                     const kv_cache_manager::BlockMask&      block_mask,
                     const kv_cache_manager::ForwardContext& forward_context) {
    CHECK_INIT2(unique_id);
    CALL_CLIENT2(MatchLocation, trace_id, query_type, keys, {}, block_mask, forward_context.sw_size, {});
}

std::pair<bool, kv_cache_manager::WriteLocation>
ClientWrapper::getWriteLocation(const std::string&              unique_id,
                                const std::string&              trace_id,
                                const std::vector<int64_t>&     keys,
                                const std::vector<int64_t>&     tokens,
                                const std::vector<std::string>& location_spec_group_names,
                                int64_t                         write_timeout_seconds,
                                int32_t                         min_replica_count) {
    CHECK_INIT2(unique_id);
    CALL_CLIENT2(StartWrite, trace_id, keys, tokens, location_spec_group_names, write_timeout_seconds, min_replica_count);
}

bool ClientWrapper::removeCache(const std::string& unique_id, const std::string& trace_id,
                                const std::vector<int64_t>& keys, const std::vector<int64_t>& tokens,
                                const kv_cache_manager::BlockMask& block_mask) {
    CHECK_INIT1(unique_id);
    return checkError(client->RemoveCache(trace_id, keys, tokens, block_mask));
}

std::pair<bool, kv_cache_manager::BackendLocations>
ClientWrapper::getCacheLocationsByBackend(const std::string& unique_id, const std::string& trace_id,
                                          const std::vector<int64_t>& keys, const std::vector<int64_t>& tokens,
                                          const kv_cache_manager::BlockMask& block_mask,
                                          const std::vector<std::string>& location_spec_names,
                                          kv_cache_manager::StorageType backend_type) {
    CHECK_INIT2(unique_id);
    if (!remoteCacheDecodePublicationEnabled()) {
        // Preserve the upstream SDK response contract when publication is off.
        CALL_CLIENT2(GetCacheLocationsByBackend, trace_id, keys, tokens, block_mask, location_spec_names,
                     backend_type, kv_cache_manager::BackendSelectStrategy::LSS_WEIGHTED_RANDOM);
    }
    auto [ec, locations] =
        client->GetCacheLocationsByBackend(trace_id,
                                           keys,
                                           tokens,
                                           block_mask,
                                           location_spec_names,
                                           backend_type,
                                           kv_cache_manager::BackendSelectStrategy::LSS_WEIGHTED_RANDOM);
    if (!checkError(ec)) {
        RTP_LLM_LOG_WARNING("GetCacheLocationsByBackend failed, ec [%d]", ec);
        return {false, {}};
    }
    // Backend results are positional. Older managers can return locations for
    // masked keys; clear those slots without shifting the remaining key indices.
    if (!keys.empty() && locations.size() != keys.size()) {
        RTP_LLM_LOG_WARNING("KVCM backend response has %zu slots for %zu keys", locations.size(), keys.size());
        return {false, {}};
    }
    if (const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&block_mask)) {
        if (*offset > locations.size()) {
            return {false, {}};
        }
        for (size_t i = 0; i < *offset; ++i) {
            locations[i].clear();
        }
    } else {
        const auto& mask = std::get<kv_cache_manager::BlockMaskVector>(block_mask);
        if (!mask.empty() && mask.size() != locations.size()) {
            return {false, {}};
        }
        for (size_t i = 0; i < mask.size(); ++i) {
            if (mask[i]) {
                locations[i].clear();
            }
        }
    }
    return {true, std::move(locations)};
}

std::pair<bool, kv_cache_manager::HostCacheState>
ClientWrapper::getHostCacheState(const std::string& unique_id, const std::string& trace_id,
                                 kv_cache_manager::QueryType query_type, const std::vector<int64_t>& keys,
                                 const std::vector<std::string>& medium, int32_t p2p_host_count) {
    CHECK_INIT2(unique_id);
    auto [ec, result] = getSdkHostCacheState(*client, trace_id, query_type, keys, medium, p2p_host_count);
    if (!checkError(ec)) {
        RTP_LLM_LOG_WARNING("GetHostCacheState fail, ec [%d]", ec);
        return {false, {}};
    }
    return {true, std::move(result)};
}

bool ClientWrapper::finishWrite(const std::string&                 unique_id,
                                const std::string&                 trace_id,
                                const std::string&                 write_session_id,
                                const kv_cache_manager::BlockMask& block_mask,
                                const kv_cache_manager::Locations& locations) {
    CHECK_INIT1(unique_id);
    auto ec = client->FinishWrite(trace_id, write_session_id, block_mask, locations);
    if (!checkError(ec)) {
        RTP_LLM_LOG_WARNING("FinishWrite fail, ec [%d]", ec);
        return false;
    }
    return true;
}

bool ClientWrapper::checkError(kv_cache_manager::ClientErrorCode ec) {
    if (ec == kv_cache_manager::ClientErrorCode::ER_INVALID_GRPCSTATUS) {
        grpc_error_count_.fetch_add(1, std::memory_order_acquire);
    }
    if (ec == kv_cache_manager::ClientErrorCode::ER_OK) {
        return true;
    } else if (ec == kv_cache_manager::ClientErrorCode::ER_SERVICE_INSTANCE_NOT_EXIST
               || ec == kv_cache_manager::ClientErrorCode::ER_SERVICE_NOT_LEADER
               || grpc_error_count_.load(std::memory_order_acquire) > 2) {
        requestReinit();
        return false;
    }
    return false;
}

void ClientWrapper::requestReinit() {
    {
        std::lock_guard<std::mutex> lock(reinit_worker_mutex_);
        if (stop_requested_) {
            return;
        }
        reinit_requested_ = true;
    }
    reinit_worker_cv_.notify_one();
}

void ClientWrapper::reinitWorkerLoop() noexcept {
    pthread_setname_np(pthread_self(), "kvcm_reinit");
    while (true) {
        {
            std::unique_lock<std::mutex> lock(reinit_worker_mutex_);
            reinit_worker_cv_.wait(lock, [this] { return stop_requested_ || reinit_requested_; });
            if (stop_requested_) {
                return;
            }
            reinit_requested_ = false;
        }
        bool succeeded = false;
        try {
            succeeded = reinitAllMetaClients();
        } catch (const std::exception& error) {
            RTP_LLM_LOG_ERROR("KVCM re-registration worker failed: %s", error.what());
        } catch (...) { RTP_LLM_LOG_ERROR("KVCM re-registration worker failed with an unknown exception"); }
        if (succeeded) {
            grpc_error_count_.store(0, std::memory_order_release);
        }
    }
}

bool ClientWrapper::reinitAllMetaClients() {
    std::unique_lock write_guard(rr_mutex_);
    RTP_LLM_LOG_INFO("reinitAllMetaClients start");
    ReinitPolicy policy;
    while (true) {
        RTP_LLM_INTERVAL_LOG(5, INFO, "doing reinitAllMetaClients...");
        try {
            if (subscriber_->getAddresses(address_snapshot_)) {
                bool all_succeed = true;
                for (auto config_iter = config_map_.begin(); config_iter != config_map_.end(); ++config_iter) {
                    const auto& unique_id = config_iter->first;
                    if (auto meta_client_iter = meta_client_map_.find(unique_id);
                        meta_client_iter != meta_client_map_.end()) {
                        if (!reinit(unique_id, config_iter, meta_client_iter)) {
                            all_succeed = false;
                            break;
                        }
                    } else {
                        all_succeed = false;
                        break;
                    }
                }
                if (all_succeed) {
                    RTP_LLM_LOG_INFO("reinitAllMetaClients finish");
                    return true;
                }
            }
        } catch (const std::exception& error) {
            RTP_LLM_LOG_ERROR("KVCM re-registration attempt failed: %s", error.what());
        } catch (...) { RTP_LLM_LOG_ERROR("KVCM re-registration attempt failed with an unknown exception"); }
        if (!waitForRetry(policy.sleep_time_ms())) {
            RTP_LLM_LOG_INFO("reinitAllMetaClients stopped");
            return false;
        }
    }
}

void ClientWrapper::retainTransfer(const std::shared_ptr<PendingTransfer>& transfer) {
    std::lock_guard<std::mutex> lock(pending_transfer_mutex_);
    transfer->next = std::move(pending_transfers_);
    pending_transfers_ = transfer;
}

void ClientWrapper::releaseTransfer(const std::shared_ptr<PendingTransfer>& transfer) {
    std::lock_guard<std::mutex> lock(pending_transfer_mutex_);
    auto* link = &pending_transfers_;
    while (*link && *link != transfer) link = &(*link)->next;
    if (*link) *link = std::move(transfer->next);
}

void ClientWrapper::setTransferObserver(TransferObserver observer) {
    std::unique_lock lock(transfer_mutex_);
    const bool       had_observer = static_cast<bool>(transfer_observer_);
    transfer_observer_            = remoteCacheDecodePublicationEnabled() ? std::move(observer) : TransferObserver{};
    if (transfer_observer_ || had_observer) {
        for (const auto& client : transfer_clients_) {
            if (auto* observable = dynamic_cast<SdkTransferObservable*>(client.get())) {
                observable->setSdkTransferObserver(transfer_observer_);
            }
        }
    }
}

void ClientWrapper::observeTransfer(bool is_read, size_t completed_bytes, bool completion_unknown) const noexcept {
    if (transfer_observer_) {
        try {
            transfer_observer_(is_read, completed_bytes, completion_unknown);
        } catch (...) {
            RTP_LLM_LOG_ERROR("failed to report confirmed SDK payload");
        }
    }
}

ClientWrapper::TransferCompletion ClientWrapper::sdkResultCompletion(kv_cache_manager::ClientErrorCode ec) {
    return sdkCodeProvesBufferSafety(ec) ? TransferCompletion::BUFFER_SAFE : TransferCompletion::UNKNOWN;
}

bool ClientWrapper::waitForRetry(int sleep_time_ms) {
    std::unique_lock<std::mutex> lock(reinit_worker_mutex_);
    return !reinit_worker_cv_.wait_for(
        lock, std::chrono::milliseconds(sleep_time_ms), [this] { return stop_requested_; });
}

bool ClientWrapper::loadKvCachesForTag(const std::string&                                          tag,
                                       const kv_cache_manager::UriStrVec&                          uri_str_vec,
                                       kv_cache_manager::BlockBuffers&                             block_buffers,
                                       const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace_info) {
    return loadKvCachesForTagWithStatus(tag, uri_str_vec, block_buffers, trace_info)
           == kv_cache_manager::ClientErrorCode::ER_OK;
}

kv_cache_manager::ClientErrorCode
ClientWrapper::loadKvCachesForTagWithStatus(const std::string&                                          tag,
                                            const kv_cache_manager::UriStrVec&                          uri_str_vec,
                                            kv_cache_manager::BlockBuffers&                             block_buffers,
                                            const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace_info,
                                            TransferCompletion*                                         completion) {
    if (completion)
        *completion = TransferCompletion::NOT_STARTED;
    std::shared_lock read_guard(transfer_mutex_);
    const auto       slot = tag_to_index_.find(tag);
    if (slot == tag_to_index_.end()) {
        RTP_LLM_LOG_ERROR("kvcm client not find transfer client");
        return kv_cache_manager::ClientErrorCode::ER_CLIENT_NOT_EXISTS;
    }
    if (!remoteCacheDecodePublicationEnabled()) {
        if (completion)
            *completion = TransferCompletion::UNKNOWN;
        const auto ec = transfer_clients_[slot->second]->LoadKvCaches(uri_str_vec, block_buffers, trace_info);
        if (completion)
            *completion = sdkResultCompletion(ec);
        if (ec != kv_cache_manager::ClientErrorCode::ER_OK) {
            RTP_LLM_LOG_ERROR("kvcm client loadKvCaches fail, ec [%d]", ec);
        }
        return ec;
    }
    const auto io_started = std::chrono::steady_clock::now();
    const bool inner_observation =
        dynamic_cast<SdkTransferObservable*>(transfer_clients_[slot->second].get()) != nullptr;
    auto                  ec = kv_cache_manager::ClientErrorCode::ER_OK;
    std::vector<SdkRange> ranges;
    if ((trace_info && !trace_info->block_ids.empty() && trace_info->block_ids.size() != uri_str_vec.size())
        || !boundedSdkRanges(uri_str_vec, block_buffers, ranges)) {
        RTP_LLM_LOG_ERROR("KVCM bounded SDK rejects malformed or oversized block geometry");
        return kv_cache_manager::ClientErrorCode::ER_INVALID_PARAMS;
    }
    for (const auto& range : ranges) {
        auto retained = std::make_shared<PendingTransfer>();
        retained->uris.assign(uri_str_vec.begin() + range.begin, uri_str_vec.begin() + range.end);
        auto& uris = retained->uris;
        // Shallow pointer descriptors; caller pins own all payloads for the whole operation.
        retained->buffers.assign(block_buffers.begin() + range.begin, block_buffers.begin() + range.end);
        auto& buffers = retained->buffers;
        const auto                     began      = std::chrono::steady_clock::now();
        const auto                     part_trace = traceForRange(trace_info, range);
        retained->trace = part_trace;
        auto local_completion = TransferCompletion::UNKNOWN;
        retainTransfer(retained);
        struct RetainedScope {
            ClientWrapper* owner;
            const std::shared_ptr<PendingTransfer>& transfer;
            TransferCompletion& completion;
            ~RetainedScope() {
                if (completion != TransferCompletion::UNKNOWN) owner->releaseTransfer(transfer);
            }
        } preserve_unknown{this, retained, local_completion};
        try {
            if (completion)
                *completion = TransferCompletion::UNKNOWN;
            ec = transfer_clients_[slot->second]->LoadKvCaches(uris, buffers, part_trace);
        } catch (...) {
            if (!inner_observation)
                observeTransfer(/*is_read=*/true, 0, /*completion_unknown=*/true);
            throw;
        }
        local_completion = sdkResultCompletion(ec);
        if (completion)
            *completion = local_completion;
        if (!inner_observation) {
            observeTransfer(/*is_read=*/true,
                            ec == kv_cache_manager::ClientErrorCode::ER_OK ? range.bytes : 0,
                            ec != kv_cache_manager::ClientErrorCode::ER_OK);
        }
        const auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - began).count();
        RTP_LLM_LOG_INFO("KV_SDK_CHUNK op=read pid=%d tag=%s blocks=%zu iovs=%zu bytes=%zu duration_us=%ld ec=%d",
                         static_cast<int>(getpid()),
                         tag.c_str(),
                         uris.size(),
                         range.iovs,
                         range.bytes,
                         static_cast<long>(us),
                         static_cast<int>(ec));
        if (ec != kv_cache_manager::ClientErrorCode::ER_OK) {
            break;
        }
        std::copy(buffers.begin(), buffers.end(), block_buffers.begin() + range.begin);
    }
    const auto io_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - io_started).count();
    size_t io_bytes = 0;
    for (const auto& block : block_buffers) {
        for (const auto& iov : block.iovs) {
            io_bytes += iov.size;
        }
    }
    RTP_LLM_LOG_INFO("KV_IO op=read pid=%d tag=%s blocks=%zu bytes=%zu duration_us=%ld ec=%d",
                     static_cast<int>(getpid()),
                     tag.c_str(),
                     uri_str_vec.size(),
                     io_bytes,
                     static_cast<long>(io_us),
                     static_cast<int>(ec));
    if (ec != kv_cache_manager::ClientErrorCode::ER_OK) {
        RTP_LLM_LOG_ERROR("kvcm client loadKvCaches fail, ec [%d]", ec);
    }
    return ec;
}

std::pair<bool, kv_cache_manager::UriStrVec>
ClientWrapper::saveKvCachesForTag(const std::string&                                          tag,
                                  const kv_cache_manager::UriStrVec&                          uri_str_vec,
                                  const kv_cache_manager::BlockBuffers&                       block_buffers,
                                  const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace_info) {
    auto [ec, result] = saveKvCachesForTagWithStatus(tag, uri_str_vec, block_buffers, trace_info);
    if (ec != kv_cache_manager::ClientErrorCode::ER_OK) {
        return {false, {}};
    }
    return {true, std::move(result)};
}

std::pair<kv_cache_manager::ClientErrorCode, kv_cache_manager::UriStrVec>
ClientWrapper::saveKvCachesForTagWithStatus(const std::string&                                          tag,
                                            const kv_cache_manager::UriStrVec&                          uri_str_vec,
                                            const kv_cache_manager::BlockBuffers&                       block_buffers,
                                            const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace_info,
                                            TransferCompletion*                                         completion) {
    if (completion)
        *completion = TransferCompletion::NOT_STARTED;
    std::shared_lock read_guard(transfer_mutex_);
    const auto       slot = tag_to_index_.find(tag);
    if (slot == tag_to_index_.end()) {
        RTP_LLM_LOG_ERROR("kvcm client not find transfer client");
        return {kv_cache_manager::ClientErrorCode::ER_CLIENT_NOT_EXISTS, {}};
    }
    if (!remoteCacheDecodePublicationEnabled()) {
        if (completion)
            *completion = TransferCompletion::UNKNOWN;
        auto result = transfer_clients_[slot->second]->SaveKvCaches(uri_str_vec, block_buffers, trace_info);
        if (completion)
            *completion = sdkResultCompletion(result.first);
        if (result.first != kv_cache_manager::ClientErrorCode::ER_OK) {
            RTP_LLM_LOG_ERROR("kvcm client saveKvCaches fail, ec [%d]", result.first);
        }
        return result;
    }
    const auto io_started = std::chrono::steady_clock::now();
    const bool inner_observation =
        dynamic_cast<SdkTransferObservable*>(transfer_clients_[slot->second].get()) != nullptr;
    auto                        ec = kv_cache_manager::ClientErrorCode::ER_OK;
    kv_cache_manager::UriStrVec result;
    std::vector<SdkRange>       ranges;
    if ((trace_info && !trace_info->block_ids.empty() && trace_info->block_ids.size() != uri_str_vec.size())
        || !boundedSdkRanges(uri_str_vec, block_buffers, ranges)) {
        RTP_LLM_LOG_ERROR("KVCM bounded SDK rejects malformed or oversized block geometry");
        return {kv_cache_manager::ClientErrorCode::ER_INVALID_PARAMS, {}};
    }
    for (const auto& range : ranges) {
        auto retained = std::make_shared<PendingTransfer>();
        retained->uris.assign(uri_str_vec.begin() + range.begin, uri_str_vec.begin() + range.end);
        auto& uris = retained->uris;
        // Shallow pointer descriptors; caller pins own all payloads for the whole operation.
        retained->buffers.assign(block_buffers.begin() + range.begin, block_buffers.begin() + range.end);
        auto& buffers = retained->buffers;
        const auto                     began      = std::chrono::steady_clock::now();
        const auto                     part_trace = traceForRange(trace_info, range);
        retained->trace = part_trace;
        auto local_completion = TransferCompletion::UNKNOWN;
        retainTransfer(retained);
        struct RetainedScope {
            ClientWrapper* owner;
            const std::shared_ptr<PendingTransfer>& transfer;
            TransferCompletion& completion;
            ~RetainedScope() {
                if (completion != TransferCompletion::UNKNOWN) owner->releaseTransfer(transfer);
            }
        } preserve_unknown{this, retained, local_completion};
        auto                           saved      = [&] {
            try {
                if (completion)
                    *completion = TransferCompletion::UNKNOWN;
                return transfer_clients_[slot->second]->SaveKvCaches(uris, buffers, part_trace);
            } catch (...) {
                if (!inner_observation)
                    observeTransfer(/*is_read=*/false, 0, /*completion_unknown=*/true);
                throw;
            }
        }();
        ec = saved.first;
        local_completion = sdkResultCompletion(ec);
        if (completion)
            *completion = local_completion;
        if (!inner_observation) {
            observeTransfer(/*is_read=*/false,
                            ec == kv_cache_manager::ClientErrorCode::ER_OK ? range.bytes : 0,
                            ec != kv_cache_manager::ClientErrorCode::ER_OK);
        }
        const auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - began).count();
        RTP_LLM_LOG_INFO("KV_SDK_CHUNK op=write pid=%d tag=%s blocks=%zu iovs=%zu bytes=%zu duration_us=%ld ec=%d",
                         static_cast<int>(getpid()),
                         tag.c_str(),
                         uris.size(),
                         range.iovs,
                         range.bytes,
                         static_cast<long>(us),
                         static_cast<int>(ec));
        if (ec != kv_cache_manager::ClientErrorCode::ER_OK) {
            break;
        }
        if (!saved.second.empty() && saved.second.size() != uris.size()) {
            ec = kv_cache_manager::ClientErrorCode::ER_INVALID_PARAMS;
            break;
        }
        const auto& actual = saved.second.empty() ? uris : saved.second;
        result.insert(result.end(), actual.begin(), actual.end());
    }
    if (ec != kv_cache_manager::ClientErrorCode::ER_OK) {
        result.clear();
    }
    const auto io_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - io_started).count();
    size_t io_bytes = 0;
    for (const auto& block : block_buffers) {
        for (const auto& iov : block.iovs) {
            io_bytes += iov.size;
        }
    }
    RTP_LLM_LOG_INFO("KV_IO op=write pid=%d tag=%s blocks=%zu bytes=%zu duration_us=%ld ec=%d",
                     static_cast<int>(getpid()),
                     tag.c_str(),
                     uri_str_vec.size(),
                     io_bytes,
                     static_cast<long>(io_us),
                     static_cast<int>(ec));
    if (ec != kv_cache_manager::ClientErrorCode::ER_OK) {
        RTP_LLM_LOG_ERROR("kvcm client saveKvCaches fail, ec [%d]", ec);
    }
    return {ec, std::move(result)};
}

}  // namespace kvcm
}  // namespace rtp_llm
