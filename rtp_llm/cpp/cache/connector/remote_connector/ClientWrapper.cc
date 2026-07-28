#include "ClientWrapper.h"
#include "VIPServerSubscriber.h"
#include "DirectSubscriber.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <algorithm>
#include <numeric>
#include <thread>
#include <random>
#include <chrono>

namespace rtp_llm {
namespace remote_connector {

namespace {

using TaggedRequestIndices = std::map<std::string, std::vector<size_t>>;

TaggedRequestIndices groupRequestIndicesByTag(const std::vector<std::string>& tags) {
    TaggedRequestIndices result;
    for (size_t i = 0; i < tags.size(); ++i) {
        RTP_LLM_CHECK_WITH_INFO(!tags[i].empty(), "remote cache request has empty tag at index=%zu", i);
        result[tags[i]].push_back(i);
    }
    return result;
}

std::shared_ptr<kv_cache_manager::TransferTraceInfo>
selectTraceInfo(const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace_info,
                const std::vector<size_t>&                                  indices) {
    if (!trace_info) {
        return nullptr;
    }
    auto selected        = std::make_shared<kv_cache_manager::TransferTraceInfo>();
    selected->need_print = trace_info->need_print;
    selected->block_ids.reserve(indices.size());
    for (const auto index : indices) {
        RTP_LLM_CHECK_WITH_INFO(index < trace_info->block_ids.size(),
                                "remote cache trace block index=%zu out of range=%zu",
                                index,
                                trace_info->block_ids.size());
        selected->block_ids.push_back(trace_info->block_ids[index]);
    }
    return selected;
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

std::unique_ptr<Subscriber>    ClientWrapper::subscriber_;
std::unique_ptr<ClientFactory> ClientWrapper::client_factory_ = std::make_unique<ClientFactory>();

ClientWrapper::~ClientWrapper() = default;

bool ClientWrapper::init(const ConfigMap&               config_map,
                         kv_cache_manager::RoleType     role_type,
                         const TransferRegistrationMap& registrations) {
    RTP_LLM_CHECK_WITH_INFO(!config_map.empty(), "no valid remote connector config");
    RTP_LLM_CHECK_WITH_INFO(!registrations.empty(), "no KV cache transfer registrations");
    transfer_clients_.clear();
    const auto& first_registration = registrations.begin()->second;
    RTP_LLM_CHECK_WITH_INFO(!registrations.begin()->first.empty() && first_registration.address != nullptr
                                && first_registration.size_bytes > 0 && !first_registration.location_spec_name.empty(),
                            "invalid KV cache transfer registration for tag=%s",
                            registrations.begin()->first.c_str());
    init_params_ = kv_cache_manager::InitParams{role_type, nullptr, first_registration.location_spec_name};
    // init all meta_client
    if (init_params_.role_type == kv_cache_manager::RoleType::HYBRID) {
        for (auto& [unique_id, config] : config_map) {
            if (!initMetaClient(unique_id, config)) {
                return false;
            }
        }
    } else {
        init_params_.role_type = kv_cache_manager::RoleType::SCHEDULER;
        const auto& item       = *config_map.begin();
        if (!initMetaClient(item.first, item.second)) {
            return false;
        }
    }
    init_params_.storage_configs = meta_client_map_.begin()->second->GetStorageConfig();
    RTP_LLM_LOG_INFO("transfer client storage config [%s]", init_params_.storage_configs.c_str());
    if (init_params_.role_type == kv_cache_manager::RoleType::SCHEDULER) {
        meta_client_map_.clear();
        init_params_.role_type = kv_cache_manager::RoleType::WORKER;
    }
    const auto transfer_role = init_params_.role_type;
    for (const auto& [tag, registration] : registrations) {
        RTP_LLM_CHECK_WITH_INFO(!tag.empty() && registration.address != nullptr && registration.size_bytes > 0
                                    && !registration.location_spec_name.empty(),
                                "invalid KV cache transfer registration for tag=%s",
                                tag.c_str());
        kv_cache_manager::RegistSpan regist_span{registration.address, registration.size_bytes};
        kv_cache_manager::InitParams transfer_init_params{transfer_role, &regist_span, registration.location_spec_name};
        transfer_init_params.storage_configs = init_params_.storage_configs;
        auto transfer_client                 = client_factory_->CreateTransferClient(
            autil::legacy::ToJsonString(config_map_.begin()->second), transfer_init_params);
        if (!transfer_client) {
            RTP_LLM_LOG_ERROR("init transfer client failed for tag=%s", tag.c_str());
            transfer_clients_.clear();
            return false;
        }
        if (!transfer_clients_.emplace(tag, std::move(transfer_client)).second) {
            RTP_LLM_LOG_ERROR("duplicate transfer client tag=%s", tag.c_str());
            transfer_clients_.clear();
            return false;
        }
    }
    return true;
}

bool ClientWrapper::initMetaClient(const std::string& unique_id, RemoteConnectorConfigPtr config) {
    RTP_LLM_LOG_INFO(
        "kvcm unique_id [%s], init config [%s]", unique_id.c_str(), autil::legacy::ToJsonString(config).c_str());
    if (config->enable_vipserver()) {
        if (subscriber_ == nullptr) {
            subscriber_ = std::make_unique<remote_connector::VIPServerSubscriber>();
        }
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
        if (subscriber_ == nullptr) {
            subscriber_ = std::make_unique<remote_connector::DirectSubscriber>();
        }
        subscriber_->init(config->addresses());
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
        meta_client = client_factory_->CreateMetaClient(real_config_str, init_params_);
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
    auto meta_client = client_factory_->CreateMetaClient(real_config_str, init_params_);
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

#define DEFER(...) __VA_ARGS__
#define CHECK_INIT_BASE(unique_i, return_value)                                                                        \
    std::shared_lock read_guard(rr_mutex_, std::try_to_lock);                                                          \
    if (!read_guard.owns_lock()) {                                                                                     \
        RTP_LLM_LOG_WARNING("doing re-registration");                                                                  \
        return return_value;                                                                                           \
    }                                                                                                                  \
    if (!tryReinit(unique_id)) {                                                                                       \
        return return_value;                                                                                           \
    }
#define CHECK_INIT2(unique_id) CHECK_INIT_BASE(unique_id, DEFER({false, {}}))
#define CHECK_INIT1(unique_id) CHECK_INIT_BASE(unique_id, false)

#define CALL_CLIENT2(unique_id, function_name, ...)                                                                    \
    if (const auto& client_iter = meta_client_map_.find(unique_id); client_iter != meta_client_map_.end()) {           \
        auto [ec, result] = client_iter->second->function_name(__VA_ARGS__);                                           \
        if (!checkError(ec)) {                                                                                         \
            RTP_LLM_LOG_WARNING(#function_name " fail, ec [%d]", ec);                                                  \
            return {false, {}};                                                                                        \
        }                                                                                                              \
        return {true, std::move(result)};                                                                              \
    }                                                                                                                  \
    RTP_LLM_LOG_WARNING("not find client [%s]", unique_id.c_str());                                                    \
    return {false, {}};

std::pair<bool, kv_cache_manager::Locations>
ClientWrapper::match(const std::string&                      unique_id,
                     const std::string&                      trace_id,
                     kv_cache_manager::QueryType             query_type,
                     const std::vector<int64_t>&             keys,
                     const kv_cache_manager::BlockMask&      block_mask,
                     const kv_cache_manager::ForwardContext& forward_context) {
    CHECK_INIT2(unique_id);
    CALL_CLIENT2(unique_id, MatchLocation, trace_id, query_type, keys, {}, block_mask, forward_context.sw_size, {});
}

std::pair<bool, kv_cache_manager::WriteLocation>
ClientWrapper::getWriteLocation(const std::string&              unique_id,
                                const std::string&              trace_id,
                                const std::vector<int64_t>&     keys,
                                const std::vector<int64_t>&     tokens,
                                const std::vector<std::string>& location_spec_group_names,
                                int64_t                         write_timeout_seconds) {
    CHECK_INIT2(unique_id);
    CALL_CLIENT2(unique_id, StartWrite, trace_id, keys, tokens, location_spec_group_names, write_timeout_seconds);
}

bool ClientWrapper::finishWrite(const std::string&                 unique_id,
                                const std::string&                 trace_id,
                                const std::string&                 write_session_id,
                                const kv_cache_manager::BlockMask& block_mask,
                                const kv_cache_manager::Locations& locations) {
    CHECK_INIT1(unique_id);
    if (const auto& client_iter = meta_client_map_.find(unique_id); client_iter != meta_client_map_.end()) {
        auto ec = client_iter->second->FinishWrite(trace_id, write_session_id, block_mask, locations);
        if (!checkError(ec)) {
            RTP_LLM_LOG_WARNING("FinishWrite fail, ec [%d]", ec);
            return false;
        }
        return true;
    }
    RTP_LLM_LOG_WARNING("not find client [%s]", unique_id.c_str());
    return false;
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
        std::thread([self = shared_from_this()]() { self->reinitAllMetaClients(); }).detach();
        return false;
    }
    return false;
}

void ClientWrapper::reinitAllMetaClients() {
    auto expected = false;
    if (!rr_other_working_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        RTP_LLM_LOG_INFO("other thread is working");
        return;
    }
    std::unique_lock write_guard(rr_mutex_);
    RTP_LLM_LOG_INFO("reinitAllMetaClients start");
    ReinitPolicy policy;
    while (true) {
        RTP_LLM_INTERVAL_LOG(5, INFO, "doing reinitAllMetaClients...");
        std::this_thread::sleep_for(std::chrono::milliseconds(policy.sleep_time_ms()));
        if (!subscriber_->getAddresses(address_snapshot_)) {
            continue;
        }
        bool all_succeed = false;
        for (auto config_iter = config_map_.begin(); config_iter != config_map_.end(); ++config_iter) {
            const auto& unique_id = config_iter->first;
            if (auto meta_client_iter = meta_client_map_.find(unique_id); meta_client_iter != meta_client_map_.end()) {
                all_succeed = reinit(unique_id, config_iter, meta_client_iter);
                if (!all_succeed) {
                    break;
                }
            } else {
                continue;
            }
        }
        if (all_succeed) {
            break;
        }
    }
    RTP_LLM_LOG_INFO("reinitAllMetaClients finish");
    grpc_error_count_.store(0, std::memory_order_release);
    rr_other_working_.store(false, std::memory_order_release);
}

bool ClientWrapper::loadKvCaches(const std::vector<std::string>&                             tags,
                                 const kv_cache_manager::UriStrVec&                          uri_str_vec,
                                 kv_cache_manager::BlockBuffers&                             block_buffers,
                                 const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace_info) {
    if (tags.size() != uri_str_vec.size() || tags.size() != block_buffers.size()
        || (trace_info && trace_info->block_ids.size() != tags.size())) {
        RTP_LLM_LOG_ERROR("remote cache transfer count mismatch: tags=%zu uris=%zu buffers=%zu trace_blocks=%zu",
                          tags.size(),
                          uri_str_vec.size(),
                          block_buffers.size(),
                          trace_info ? trace_info->block_ids.size() : 0);
        return false;
    }
    for (const auto& tag : tags) {
        if (tag.empty() || transfer_clients_.find(tag) == transfer_clients_.end()) {
            RTP_LLM_LOG_ERROR("kvcm client not find transfer client for tag=%s", tag.c_str());
            return false;
        }
    }
    for (const auto& [tag, indices] : groupRequestIndicesByTag(tags)) {
        const auto client_it = transfer_clients_.find(tag);
        if (client_it == transfer_clients_.end()) {
            RTP_LLM_LOG_ERROR("kvcm client not find transfer client for tag=%s", tag.c_str());
            return false;
        }
        kv_cache_manager::UriStrVec    tagged_uris;
        kv_cache_manager::BlockBuffers tagged_buffers;
        tagged_uris.reserve(indices.size());
        tagged_buffers.reserve(indices.size());
        for (const auto index : indices) {
            tagged_uris.push_back(uri_str_vec[index]);
            tagged_buffers.push_back(block_buffers[index]);
        }
        const auto ec =
            client_it->second->LoadKvCaches(tagged_uris, tagged_buffers, selectTraceInfo(trace_info, indices));
        if (ec != kv_cache_manager::ClientErrorCode::ER_OK) {
            RTP_LLM_LOG_ERROR("kvcm client loadKvCaches fail for tag=%s, ec [%d]", tag.c_str(), ec);
            return false;
        }
    }
    return true;
}

std::pair<bool, kv_cache_manager::UriStrVec>
ClientWrapper::saveKvCaches(const std::vector<std::string>&                             tags,
                            const kv_cache_manager::UriStrVec&                          uri_str_vec,
                            const kv_cache_manager::BlockBuffers&                       block_buffers,
                            const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace_info) {
    if (tags.size() != uri_str_vec.size() || tags.size() != block_buffers.size()
        || (trace_info && trace_info->block_ids.size() != tags.size())) {
        RTP_LLM_LOG_ERROR("remote cache transfer count mismatch: tags=%zu uris=%zu buffers=%zu trace_blocks=%zu",
                          tags.size(),
                          uri_str_vec.size(),
                          block_buffers.size(),
                          trace_info ? trace_info->block_ids.size() : 0);
        return {false, {}};
    }
    for (const auto& tag : tags) {
        if (tag.empty() || transfer_clients_.find(tag) == transfer_clients_.end()) {
            RTP_LLM_LOG_ERROR("kvcm client not find transfer client for tag=%s", tag.c_str());
            return {false, {}};
        }
    }
    auto resolved_uris  = uri_str_vec;
    bool has_actual_uri = false;
    for (const auto& [tag, indices] : groupRequestIndicesByTag(tags)) {
        const auto client_it = transfer_clients_.find(tag);
        if (client_it == transfer_clients_.end()) {
            RTP_LLM_LOG_ERROR("kvcm client not find transfer client for tag=%s", tag.c_str());
            return {false, {}};
        }
        kv_cache_manager::UriStrVec    tagged_uris;
        kv_cache_manager::BlockBuffers tagged_buffers;
        tagged_uris.reserve(indices.size());
        tagged_buffers.reserve(indices.size());
        for (const auto index : indices) {
            tagged_uris.push_back(uri_str_vec[index]);
            tagged_buffers.push_back(block_buffers[index]);
        }
        auto [ec, actual_uris] =
            client_it->second->SaveKvCaches(tagged_uris, tagged_buffers, selectTraceInfo(trace_info, indices));
        if (ec != kv_cache_manager::ClientErrorCode::ER_OK) {
            RTP_LLM_LOG_ERROR("kvcm client saveKvCaches fail for tag=%s, ec [%d]", tag.c_str(), ec);
            return {false, {}};
        }
        if (actual_uris.empty()) {
            continue;
        }
        if (actual_uris.size() != indices.size()) {
            RTP_LLM_LOG_ERROR("kvcm client returned invalid URI count for tag=%s: expected=%zu actual=%zu",
                              tag.c_str(),
                              indices.size(),
                              actual_uris.size());
            return {false, {}};
        }
        for (size_t i = 0; i < indices.size(); ++i) {
            if (resolved_uris[indices[i]] != actual_uris[i]) {
                resolved_uris[indices[i]] = std::move(actual_uris[i]);
                has_actual_uri            = true;
            }
        }
    }
    return {true, has_actual_uri ? std::move(resolved_uris) : kv_cache_manager::UriStrVec{}};
}

}  // namespace remote_connector
}  // namespace rtp_llm
