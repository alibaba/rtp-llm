#include "ClientWrapper.h"
#include "kvcm_client/HostCacheStateCompat.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/Subscriber.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <algorithm>
#include <exception>
#include <pthread.h>
#include <thread>
#include <chrono>

namespace rtp_llm {
namespace kvcm {

namespace {

thread_local const ClientWrapper* transfer_deadline_owner = nullptr;
thread_local std::optional<std::chrono::steady_clock::time_point> transfer_deadline;
thread_local ClientWrapper::TransferCompletion* transfer_completion = nullptr;

class TransferDeadlineScope {
public:
    TransferDeadlineScope(const ClientWrapper* owner, std::optional<std::chrono::steady_clock::time_point> deadline,
                          ClientWrapper::TransferCompletion* completion = nullptr):
        previous_owner_(transfer_deadline_owner), previous_deadline_(transfer_deadline),
        previous_completion_(transfer_completion) {
        transfer_deadline_owner = owner;
        transfer_deadline = deadline;
        transfer_completion = completion;
    }
    ~TransferDeadlineScope() {
        transfer_deadline_owner = previous_owner_;
        transfer_deadline = previous_deadline_;
        transfer_completion = previous_completion_;
    }
private:
    const ClientWrapper* previous_owner_;
    std::optional<std::chrono::steady_clock::time_point> previous_deadline_;
    ClientWrapper::TransferCompletion* previous_completion_;
};

}  // namespace

ClientWrapper::ClientWrapper(std::unique_ptr<ClientFactory> client_factory,
                             std::chrono::milliseconds reinit_interval):
    client_factory_(std::move(client_factory)),
    reinit_interval_(std::max(reinit_interval, std::chrono::milliseconds(1))) {
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
            clients.push_back(std::move(client));
        }
        transfer_clients_ = std::move(clients);
        tag_to_index_     = std::move(pending_tags);
        try {
            if (!meta_client_map_.empty()) {
                reinit_worker_ = std::thread([this] { reinitWorkerLoop(); });
            }
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
        reinit_action_ = ReinitAction::NONE;
    }
    reinit_worker_cv_.notify_all();
    if (reinit_worker_.joinable()) {
        reinit_worker_.join();
    }
    {
        std::unique_lock<std::shared_mutex> transfer_guard(transfer_mutex_);
        transfer_clients_.clear();
        tag_to_index_.clear();
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
        address_snapshot_ = config->addresses();
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
        // Record the actual bound address. The SDK does not expose it, and
        // some versions stop at the first unreachable address in a list.
        for (const auto& address : address_snapshot_) {
            auto candidate_config = *config;
            candidate_config.set_addresses({address});
            meta_client = client_factory_->createMetaClient(autil::legacy::ToJsonString(candidate_config), init_params_);
            if (meta_client) {
                bound_address_ = address;
                break;
            }
        }
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

std::shared_ptr<kv_cache_manager::MetaClient> ClientWrapper::getMetaClient(const std::string& unique_id) {
    std::lock_guard<std::mutex> worker_lock(reinit_worker_mutex_);
    // Keep in-flight snapshots alive, but do not send new calls to a known non-leader.
    if (stop_requested_ || reinit_action_ == ReinitAction::FIND_LEADER) {
        return nullptr;
    }
    const auto it = meta_client_map_.find(unique_id);
    return it == meta_client_map_.end() ? nullptr : it->second;
}

#define DEFER(...) __VA_ARGS__
#define CHECK_INIT_BASE(unique_id, return_value)                                                                       \
    std::shared_lock read_guard(rr_mutex_, std::try_to_lock);                                                          \
    if (!read_guard.owns_lock()) {                                                                                     \
        RTP_LLM_LOG_WARNING("KVCM metadata client is shutting down");                                                  \
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
    if (!checkError(ec, client)) {                                                                                     \
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
    return checkError(client->RemoveCache(trace_id, keys, tokens, block_mask), client);
}

std::pair<bool, kv_cache_manager::BackendLocations>
ClientWrapper::getCacheLocationsByBackend(const std::string& unique_id, const std::string& trace_id,
                                          const std::vector<int64_t>& keys, const std::vector<int64_t>& tokens,
                                          const kv_cache_manager::BlockMask& block_mask,
                                          const std::vector<std::string>& location_spec_names,
                                          kv_cache_manager::StorageType backend_type) {
    CHECK_INIT2(unique_id);
    CALL_CLIENT2(GetCacheLocationsByBackend, trace_id, keys, tokens, block_mask, location_spec_names,
                 backend_type, kv_cache_manager::BackendSelectStrategy::LSS_WEIGHTED_RANDOM);
}

std::pair<bool, kv_cache_manager::HostCacheState>
ClientWrapper::getHostCacheState(const std::string& unique_id, const std::string& trace_id,
                                 kv_cache_manager::QueryType query_type, const std::vector<int64_t>& keys,
                                 const std::vector<std::string>& medium, int32_t p2p_host_count) {
    CHECK_INIT2(unique_id);
    auto [ec, result] = getSdkHostCacheState(*client, trace_id, query_type, keys, medium, p2p_host_count);
    if (!checkError(ec, client)) {
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
    if (!checkError(ec, client)) {
        RTP_LLM_LOG_WARNING("FinishWrite fail, ec [%d]", ec);
        return false;
    }
    return true;
}

bool ClientWrapper::abortWrite(const std::string& unique_id, const std::string& trace_id,
                               const std::string& write_session_id) {
    if (write_session_id.empty()) {
        return false;
    }
    // SDK retries do not cover CHECK_INIT rejecting an unsent request.
    constexpr int max_attempts = 3;
    auto attempt_abort = [&]() -> std::optional<kv_cache_manager::ClientErrorCode> {
        CHECK_INIT_BASE(unique_id, std::nullopt);
        const auto ec = client->FinishWrite(trace_id, write_session_id, kv_cache_manager::BlockMaskOffset{0}, {});
        checkError(ec, client);
        return ec;
    };
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        std::optional<kv_cache_manager::ClientErrorCode> ec;
        try {
            ec = attempt_abort();
        } catch (...) {
            RTP_LLM_LOG_WARNING("KVCM abort attempt threw, session=%s", write_session_id.c_str());
        }
        if (ec && *ec == kv_cache_manager::ClientErrorCode::ER_OK) {
            return true;
        }
        RTP_LLM_LOG_WARNING("KVCM abort unconfirmed, session=%s attempt=%d code=%d",
                            write_session_id.c_str(), attempt + 1, ec ? static_cast<int>(*ec) : -1);
        if (ec && *ec != kv_cache_manager::ClientErrorCode::ER_INVALID_GRPCSTATUS
            && *ec != kv_cache_manager::ClientErrorCode::ER_INVALID_STUB
            && *ec != kv_cache_manager::ClientErrorCode::ER_CONNECT_FAIL) {
            return false;
        }
        if (attempt + 1 < max_attempts && !waitForRetry(/*sleep_time_ms=*/20)) {
            return false;
        }
    }
    return false;
}

bool ClientWrapper::checkError(kv_cache_manager::ClientErrorCode ec,
                               const std::shared_ptr<kv_cache_manager::MetaClient>& client) {
    if (ec == kv_cache_manager::ClientErrorCode::ER_OK) {
        return true;
    }
    if (ec == kv_cache_manager::ClientErrorCode::ER_SERVICE_INSTANCE_NOT_EXIST
        || ec == kv_cache_manager::ClientErrorCode::ER_SERVICE_NOT_LEADER) {
        {
            std::lock_guard<std::mutex> lock(reinit_worker_mutex_);
            if (stop_requested_ || meta_client_map_.empty() || meta_client_map_.begin()->second != client) {
                return false;
            }
            if (ec == kv_cache_manager::ER_SERVICE_NOT_LEADER) {
                reinit_action_ = ReinitAction::FIND_LEADER;
            } else if (reinit_action_ == ReinitAction::NONE) {
                reinit_action_ = ReinitAction::REGISTER;
            }
        }
        reinit_worker_cv_.notify_all();
    }
    return false;
}

void ClientWrapper::reinitWorkerLoop() noexcept {
    pthread_setname_np(pthread_self(), "kvcm_reinit");
    bool attempted = false;
    size_t attempt_index = 0;
    std::vector<std::string> attempt_addresses;
    std::string attempt_preferred;
    const auto& [unique_id, config] = *config_map_.begin();
    std::unique_lock<std::mutex> lock(reinit_worker_mutex_);
    while (!stop_requested_) {
        const auto ready = [this, &attempted] {
            return stop_requested_ || (!attempted && reinit_action_ != ReinitAction::NONE);
        };
        // Only VIP refresh and an attempt's cooldown need a timed wait.
        if (attempted || config->enable_vipserver()) {
            reinit_worker_cv_.wait_for(lock, reinit_interval_, ready);
        } else {
            reinit_worker_cv_.wait(lock, ready);
        }
        if (stop_requested_) {
            return;
        }

        auto addresses = address_snapshot_;
        lock.unlock();
        try {
            if (config->enable_vipserver()) {
                std::vector<std::string> discovered;
                if (subscriber_->getAddresses(discovered) && !discovered.empty()) {
                    addresses = std::move(discovered);
                }
            }
        } catch (const std::exception& error) {
            RTP_LLM_LOG_WARNING("KVCM VIP refresh failed: %s", error.what());
        } catch (...) { RTP_LLM_LOG_WARNING("KVCM VIP refresh threw"); }
        lock.lock();
        if (stop_requested_) {
            return;
        }
        const bool bound_present = std::find(addresses.begin(), addresses.end(), bound_address_) != addresses.end();
        const bool removed = !bound_present
                             && std::find(address_snapshot_.begin(), address_snapshot_.end(), bound_address_)
                                    != address_snapshot_.end();
        address_snapshot_ = addresses;
        if (removed && reinit_action_ == ReinitAction::NONE) {
            reinit_action_ = ReinitAction::REGISTER;
        }
        attempted = reinit_action_ != ReinitAction::NONE;
        if (!attempted) {
            continue;
        }

        const auto action = reinit_action_;
        std::string preferred;
        if (!config->enable_vipserver() || bound_present) {
            preferred = bound_address_;
        }
        lock.unlock();
        if (action == ReinitAction::FIND_LEADER) {
            preferred.clear();
            // The last bound node may be gone; configured peers can still report the leader.
            for (size_t i = 0; i <= addresses.size() && preferred.empty(); ++i) {
                const auto& source = i == 0 ? bound_address_ : addresses[i - 1];
                if (i != 0 && source == bound_address_) {
                    continue;
                }
                {
                    std::lock_guard<std::mutex> stop_lock(reinit_worker_mutex_);
                    if (stop_requested_) return;
                }
                try {
                    preferred = client_factory_->getLeaderAddress(
                        source, config->instance_id(), config->meta_channel_config()->call_timeout());
                } catch (const std::exception& error) {
                    RTP_LLM_LOG_WARNING("KVCM leader discovery failed: %s", error.what());
                } catch (...) { RTP_LLM_LOG_WARNING("KVCM leader discovery threw"); }
            }
        }
        if (!preferred.empty()) {
            addresses.erase(std::remove(addresses.begin(), addresses.end(), preferred), addresses.end());
            addresses.insert(addresses.begin(), preferred);
        }
        // Keep fallback order stable across VIP reordering, but try a newly discovered leader first.
        if (preferred != attempt_preferred
            || !std::is_permutation(addresses.begin(), addresses.end(),
                                    attempt_addresses.begin(), attempt_addresses.end())) {
            attempt_index = 0;
            attempt_preferred = preferred;
            attempt_addresses = addresses;
        }
        const auto address = attempt_addresses[attempt_index % attempt_addresses.size()];
        std::shared_ptr<kv_cache_manager::MetaClient> candidate;
        lock.lock();
        if (stop_requested_) {
            return;
        }
        lock.unlock();
        try {
            auto candidate_config = *config;
            candidate_config.set_addresses({address});
            candidate = client_factory_->createMetaClient(autil::legacy::ToJsonString(candidate_config), init_params_);
        } catch (const std::exception& error) {
            RTP_LLM_LOG_ERROR("KVCM re-registration attempt failed: %s", error.what());
        } catch (...) { RTP_LLM_LOG_ERROR("KVCM re-registration attempt failed with an unknown exception"); }
        lock.lock();
        if (candidate && !stop_requested_) {
            candidate.swap(meta_client_map_.at(unique_id));
            bound_address_ = address;
            reinit_action_ = ReinitAction::NONE;
            attempt_index = 0;
            attempt_addresses.clear();
            attempt_preferred.clear();
            RTP_LLM_LOG_INFO("KVCM metadata client recovered, address=%s", address.c_str());
        } else if (!stop_requested_) {
            attempt_index = reinit_action_ == action ? attempt_index + 1 : 0;
            RTP_LLM_INTERVAL_LOG(5, WARN, "KVCM metadata recovery failed, address=%s", address.c_str());
        }
        // Destruction can run SDK cleanup; keep it outside publication/state locks.
        lock.unlock();
        candidate.reset();
        lock.lock();
    }
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
                                            const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace_info) {
    const auto deadline = transfer_deadline_owner == this ? transfer_deadline : std::nullopt;
    std::shared_lock read_guard(transfer_mutex_);
    if (deadline && std::chrono::steady_clock::now() >= *deadline) {
        return kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT;
    }
    const auto       slot = tag_to_index_.find(tag);
    if (slot == tag_to_index_.end()) {
        RTP_LLM_LOG_ERROR("kvcm client not find transfer client");
        return kv_cache_manager::ClientErrorCode::ER_CLIENT_NOT_EXISTS;
    }
    auto* completion = transfer_deadline_owner == this ? transfer_completion : nullptr;
    if (completion) *completion = TransferCompletion::UNKNOWN;
    auto ec = transfer_clients_[slot->second]->LoadKvCaches(uri_str_vec, block_buffers, trace_info);
    if (completion && ec == kv_cache_manager::ClientErrorCode::ER_OK)
        *completion = TransferCompletion::PAYLOAD_COMPLETE;
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
                                            const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace_info) {
    const auto deadline = transfer_deadline_owner == this ? transfer_deadline : std::nullopt;
    std::shared_lock read_guard(transfer_mutex_);
    if (deadline && std::chrono::steady_clock::now() >= *deadline) {
        return {kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT, {}};
    }
    const auto       slot = tag_to_index_.find(tag);
    if (slot == tag_to_index_.end()) {
        RTP_LLM_LOG_ERROR("kvcm client not find transfer client");
        return {kv_cache_manager::ClientErrorCode::ER_CLIENT_NOT_EXISTS, {}};
    }
    auto* completion = transfer_deadline_owner == this ? transfer_completion : nullptr;
    if (completion) *completion = TransferCompletion::UNKNOWN;
    auto [ec, result] = transfer_clients_[slot->second]->SaveKvCaches(uri_str_vec, block_buffers, trace_info);
    if (completion && ec == kv_cache_manager::ClientErrorCode::ER_OK)
        *completion = TransferCompletion::PAYLOAD_COMPLETE;
    if (ec != kv_cache_manager::ClientErrorCode::ER_OK) {
        RTP_LLM_LOG_ERROR("kvcm client saveKvCaches fail, ec [%d]", ec);
    }
    return {ec, std::move(result)};
}



kv_cache_manager::ClientErrorCode ClientWrapper::loadKvCachesForTagWithStatus(
    const std::string& tag, const kv_cache_manager::UriStrVec& uris,
    kv_cache_manager::BlockBuffers& buffers,
    const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace,
    std::optional<std::chrono::steady_clock::time_point> deadline) {
    if (deadline && std::chrono::steady_clock::now() >= *deadline) return kv_cache_manager::ER_SDK_TIMEOUT;
    TransferDeadlineScope scope(this, deadline);
    return loadKvCachesForTagWithStatus(tag, uris, buffers, trace);
}

std::pair<kv_cache_manager::ClientErrorCode, kv_cache_manager::UriStrVec> ClientWrapper::saveKvCachesForTagWithStatus(
    const std::string& tag, const kv_cache_manager::UriStrVec& uris,
    const kv_cache_manager::BlockBuffers& buffers,
    const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace,
    std::optional<std::chrono::steady_clock::time_point> deadline) {
    if (deadline && std::chrono::steady_clock::now() >= *deadline) return {kv_cache_manager::ER_SDK_TIMEOUT, {}};
    TransferDeadlineScope scope(this, deadline);
    return saveKvCachesForTagWithStatus(tag, uris, buffers, trace);
}

kv_cache_manager::ClientErrorCode ClientWrapper::loadKvCachesForTagWithStatus(
    const std::string& tag, const kv_cache_manager::UriStrVec& uris,
    kv_cache_manager::BlockBuffers& buffers,
    const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace,
    TransferCompletion* completion, std::optional<std::chrono::steady_clock::time_point> deadline) {
    if (completion) *completion = TransferCompletion::NOT_SUBMITTED;
    if (deadline && std::chrono::steady_clock::now() >= *deadline) return kv_cache_manager::ER_SDK_TIMEOUT;
    TransferDeadlineScope scope(this, deadline, completion);
    return loadKvCachesForTagWithStatus(tag, uris, buffers, trace);
}

std::pair<kv_cache_manager::ClientErrorCode, kv_cache_manager::UriStrVec> ClientWrapper::saveKvCachesForTagWithStatus(
    const std::string& tag, const kv_cache_manager::UriStrVec& uris,
    const kv_cache_manager::BlockBuffers& buffers,
    const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace,
    TransferCompletion* completion, std::optional<std::chrono::steady_clock::time_point> deadline) {
    if (completion) *completion = TransferCompletion::NOT_SUBMITTED;
    if (deadline && std::chrono::steady_clock::now() >= *deadline) return {kv_cache_manager::ER_SDK_TIMEOUT, {}};
    TransferDeadlineScope scope(this, deadline, completion);
    return saveKvCachesForTagWithStatus(tag, uris, buffers, trace);
}
}  // namespace kvcm
}  // namespace rtp_llm
