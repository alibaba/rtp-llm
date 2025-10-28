#include "KVCMClientWrapper.h"
#include "VIPServerSubscriber.h"
#include "KVCMDirectSubscriber.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <algorithm>
#include <numeric>
#include <thread>
#include <random>
#include <chrono>

namespace rtp_llm {

namespace {

class ReRegistrationPolicy {
public:
    ReRegistrationPolicy(): rd_(), gen_(rd_()), jitter_dist_(jitter_min_, jitter_max_) {}

    int sleep_time_ms() {
        int sleep_time_ms = next_sleep_time_ms_ + jitter_dist_(gen_);
        next_sleep_time_ms_ *= multiplier_;
        next_sleep_time_ms_ = std::min(next_sleep_time_ms_, max_delay_ms_);
        return sleep_time_ms;
    }

private:
    static constexpr int            base_delay_ms_      = 100;
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

std::unique_ptr<kv_cache_manager::TransferClient> KVCMClientWrapper::transfer_client_;
std::unique_ptr<KVCMSubscriber>                   KVCMClientWrapper::subscriber_;

KVCMClientWrapper::~KVCMClientWrapper() = default;

bool KVCMClientWrapper::init(const std::map<std::string, std::string>& config_str_map,
                             const kv_cache_manager::InitParams&       init_params) {
    if (config_str_map.empty()) {
        RTP_LLM_LOG_ERROR("no invalid config");
        return false;
    }
    init_params_ = init_params;
    // init all meta_client
    if (init_params_.role_type == kv_cache_manager::RoleType::HYBRID) {
        for (const auto& [unique_id, config_str] : config_str_map) {
            if (!initMetaClient(unique_id, config_str)) {
                return false;
            }
        }
        if (transfer_client_ != nullptr) {
            RTP_LLM_LOG_INFO("transfer client has been inited");
            return true;
        }
    } else {
        if (transfer_client_ != nullptr) {
            RTP_LLM_LOG_INFO("transfer client has been inited");
            return true;
        }
        init_params_.role_type = kv_cache_manager::RoleType::SCHEDULER;
        const auto& item       = *config_str_map.begin();
        if (!initMetaClient(item.first, item.second)) {
            return false;
        }
    }
    // init static transfer client
    init_params_.storage_configs = meta_client_map_.begin()->second->GetStorageConfig();
    RTP_LLM_LOG_INFO("transfer client storage config [%s]", init_params_.storage_configs.c_str());
    if (init_params_.role_type == kv_cache_manager::RoleType::SCHEDULER) {
        meta_client_map_.clear();
        init_params_.role_type = kv_cache_manager::RoleType::WORKER;
    }
    transfer_client_ = kv_cache_manager::TransferClient::Create(
        autil::legacy::ToJsonString(config_map_.begin()->second), init_params_);
    if (!transfer_client_) {
        RTP_LLM_LOG_ERROR("init trasfer client failed");
        return false;
    }
    return true;
}

bool KVCMClientWrapper::initMetaClient(const std::string& unique_id, const std::string& config_str) {
    RTP_LLM_LOG_INFO("kvcm unique_id [%s], init config [%s]", unique_id.c_str(), config_str.c_str());
    auto        config = std::make_shared<KVCMClientWrapperConfig>();
    std::string rtp_llm_clint_config;
    try {
        autil::legacy::FromJsonString(config, config_str);
    } catch (autil::legacy::ExceptionBase& e) {
        RTP_LLM_LOG_ERROR(
            "found exception when parse KVCMClientWrapperConfig.\n %s, exception: [%s]", config_str.c_str(), e.what());
        return false;
    }
    config_map_[unique_id] = config;
    if (config->enable_vipserver()) {
        if (subscriber_ == nullptr) {
            subscriber_ = std::make_unique<VIPServerSubscriber>();
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
            subscriber_ = std::make_unique<KVCMDirectSubscriber>();
        }
        subscriber_->init(config->addresses());
    }
    if (config->addresses().empty()) {
        RTP_LLM_LOG_ERROR("empty kvcm addresses");
        return false;
    }
    auto real_config_str = autil::legacy::ToJsonString(*config);
    RTP_LLM_LOG_INFO("init unique_id[%s], kvcm real config[%s]", unique_id.c_str(), real_config_str.c_str());
    config_map_[unique_id] = config;
    std::unique_ptr<kv_cache_manager::MetaClient> meta_client;
    for (int i = 1; i <= config->meta_channel_config().retry_time(); ++i) {
        RTP_LLM_LOG_INFO("try meta client, try time[%d]", i);
        meta_client = kv_cache_manager::MetaClient::Create(real_config_str, init_params_);
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

bool KVCMClientWrapper::reinit(const std::string&           unique_id,
                               KvcmConfigMap::iterator&     config_iter,
                               KvcmMetaClientMap::iterator& meta_client_iter) {
    auto config          = config_iter->second;
    auto real_config_str = autil::legacy::ToJsonString(*config);
    RTP_LLM_LOG_INFO("reinit unique_id[%s], kvcm real config[%s]", unique_id.c_str(), real_config_str.c_str());
    auto meta_client = kv_cache_manager::MetaClient::Create(real_config_str, init_params_);
    if (meta_client == nullptr) {
        RTP_LLM_LOG_ERROR("create meta client failed");
        return false;
    }
    config->set_addresses(address_snapshot_);
    meta_client_iter->second = std::move(meta_client);
    return true;
}

bool KVCMClientWrapper::tryReinit(const std::string& unique_id) {
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
            if (addresses == address_snapshot_) {
                return true;
            }
        }
        {
            std::unique_lock write_guard(reinit_mutex_);
            // double check
            if (addresses == address_snapshot_) {
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
            RTP_LLM_LOG_INFO("kvcm address changed, start reinit, current [%s], new [%s]",
                             current_address_str.c_str(),
                             new_address_str.c_str());
            address_snapshot_.swap(addresses);
            if (!reinit(unique_id, config_iter, meta_client_iter)) {
                RTP_LLM_LOG_ERROR("KVCMClientWrapper reinit failed");
                return false;
            }
        }
        RTP_LLM_LOG_INFO("kvcm client reinit finish");
    }
    return true;
}

std::pair<bool, kv_cache_manager::LocationsMap>
KVCMClientWrapper::match(const std::string&                      unique_id,
                         const std::string&                      trace_id,
                         kv_cache_manager::QueryType             query_type,
                         const std::vector<int64_t>&             keys,
                         const kv_cache_manager::BlockMask&      block_mask,
                         const kv_cache_manager::ForwardContext& forward_context) {
    std::shared_lock read_guard(rr_mutex_, std::try_to_lock);
    if (!tryReinit(unique_id)) {
        return {false, {}};
    }
    if (const auto& client_iter = meta_client_map_.find(unique_id); client_iter != meta_client_map_.end()) {
        auto [ec, result] =
            client_iter->second->MatchLocation(trace_id, query_type, keys, {}, block_mask, forward_context.sw_size, {});
        if (!checkError(ec)) {
            RTP_LLM_LOG_ERROR("kvcm client match fail, ec [%d]", ec);
            return {false, {}};
        }
        return {true, std::move(result)};
    }
    RTP_LLM_LOG_ERROR("kvcm client not find client [%s]", unique_id.c_str());
    return {false, {}};
}

std::pair<bool, kv_cache_manager::WriteLocation>
KVCMClientWrapper::getWriteLocation(const std::string&                      unique_id,
                                    const std::string&                      trace_id,
                                    const std::vector<int64_t>&             keys,
                                    const std::vector<std::string>&         location_spec_names,
                                    int64_t                                 write_timeout_seconds,
                                    const kv_cache_manager::ForwardContext& forward_context) {
    std::shared_lock read_guard(rr_mutex_, std::try_to_lock);
    if (!tryReinit(unique_id)) {
        return {false, {}};
    }
    if (const auto& client_iter = meta_client_map_.find(unique_id); client_iter != meta_client_map_.end()) {
        auto [ec, result] =
            client_iter->second->StartWrite(trace_id, keys, {}, location_spec_names, write_timeout_seconds);
        if (!checkError(ec)) {
            RTP_LLM_LOG_ERROR("kvcm client getWriteLocation fail, ec [%d]", ec);
            return {false, {}};
        }
        return {true, std::move(result)};
    }
    RTP_LLM_LOG_ERROR("kvcm client not find client [%s]", unique_id.c_str());
    return {false, {}};
}

bool KVCMClientWrapper::finishWrite(const std::string&                    unique_id,
                                    const std::string&                    trace_id,
                                    const std::string&                    write_session_id,
                                    const kv_cache_manager::BlockMask&    block_mask,
                                    const kv_cache_manager::LocationsMap& locations_map) {
    std::shared_lock read_guard(rr_mutex_, std::try_to_lock);
    if (!tryReinit(unique_id)) {
        return false;
    }
    if (const auto& client_iter = meta_client_map_.find(unique_id); client_iter != meta_client_map_.end()) {
        auto ec = client_iter->second->FinishWrite(trace_id, write_session_id, block_mask, locations_map);
        if (!checkError(ec)) {
            RTP_LLM_LOG_ERROR("kvcm client finishWrite fail, ec [%d]", ec);
            return false;
        }
        return true;
    }
    RTP_LLM_LOG_ERROR("kvcm client not find client [%s]", unique_id.c_str());
    return false;
}

bool KVCMClientWrapper::checkError(kv_cache_manager::ClientErrorCode ec) {
    if (ec == kv_cache_manager::ClientErrorCode::ER_OK) {
        return true;
    } else if (ec == kv_cache_manager::ClientErrorCode::ER_SERVICE_INSTANCE_NOT_EXIST) {
        std::thread([self = shared_from_this()]() { self->reRegistration(); }).detach();
        return false;
    }
    return false;
}

void KVCMClientWrapper::reRegistration() {
    auto expected = false;
    if (!rr_other_working_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        RTP_LLM_LOG_INFO("other thread is working");
        return;
    }
    std::unique_lock write_guard(rr_mutex_);
    RTP_LLM_LOG_INFO("re-registration start");
    ReRegistrationPolicy policy;
    while (true) {
        RTP_LLM_INTERVAL_LOG(5, INFO, "doing re-registering...");
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
    RTP_LLM_LOG_INFO("re-registration finish");
    rr_other_working_.store(false, std::memory_order_release);
}

bool KVCMClientWrapper::loadKvCaches(const kv_cache_manager::Locations& locations,
                                     kv_cache_manager::BlockBuffers&    block_buffers) {
    if (transfer_client_ == nullptr) {
        RTP_LLM_LOG_ERROR("kvcm client not find transfer client");
        return false;
    }
    auto ec = transfer_client_->LoadKvCaches(locations, block_buffers);
    if (ec != kv_cache_manager::ClientErrorCode::ER_OK) {
        RTP_LLM_LOG_ERROR("kvcm client loadKvCaches fail, ec [%d]", ec);
        return false;
    }
    return true;
}

std::pair<bool, kv_cache_manager::Locations>
KVCMClientWrapper::saveKvCaches(const kv_cache_manager::Locations&    locations,
                                const kv_cache_manager::BlockBuffers& block_buffers) {
    if (transfer_client_ == nullptr) {
        RTP_LLM_LOG_ERROR("kvcm client not find transfer client");
        return {false, {}};
    }
    auto [ec, result] = transfer_client_->SaveKvCaches(locations, block_buffers);
    if (ec != kv_cache_manager::ClientErrorCode::ER_OK) {
        RTP_LLM_LOG_ERROR("kvcm client saveKvCaches fail, ec [%d]", ec);
        return {false, {}};
    }
    return {true, std::move(result)};
}

}  // namespace rtp_llm
