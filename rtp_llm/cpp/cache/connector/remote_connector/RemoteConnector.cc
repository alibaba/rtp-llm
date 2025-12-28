#include "rtp_llm/cpp/cache/connector/remote_connector/RemoteConnector.h"

#include <atomic>
#include <algorithm>
#include <sstream>
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"

namespace rtp_llm {
namespace {

struct MatchMetricsHelper {
    MatchMetricsHelper(const std::string& trace_id, const kmonitor::MetricsReporterPtr& metrics_reporter):
        trace_id(trace_id), begin_us(rtp_llm::currentTimeUs()), metrics_reporter(metrics_reporter) {}
    ~MatchMetricsHelper() {
        collector.remote_match_time_us = rtp_llm::currentTimeUs() - begin_us;
        if (metrics_reporter) {
            metrics_reporter->report<RtpLLMRemoteCacheMatchMetrics, RtpLLMRemoteCacheMatchMetricsCollector>(nullptr,
                                                                                                            &collector);
        } else {
            RTP_LLM_LOG_INFO("MatchMetrics : {trace_id[%s], fail[%d], reuse_block_num[%ld], match_time_us[%ld]}",
                             trace_id.c_str(),
                             static_cast<int>(collector.remote_match_fail_qps),
                             collector.remote_match_reuse_block_num,
                             collector.remote_match_time_us);
        }
    }
    std::string                                     trace_id;
    int64_t                                         begin_us;
    rtp_llm::RtpLLMRemoteCacheMatchMetricsCollector collector;
    const kmonitor::MetricsReporterPtr              metrics_reporter;
};

struct ReadMetricsHelper {
    ReadMetricsHelper(const std::string& trace_id, const kmonitor::MetricsReporterPtr& metrics_reporter):
        trace_id(trace_id), begin_us(rtp_llm::currentTimeUs()), metrics_reporter(metrics_reporter) {}
    ~ReadMetricsHelper() {
        collector.remote_read_task_cost_time_us = rtp_llm::currentTimeUs() - begin_us;
        if (metrics_reporter) {
            metrics_reporter->report<RtpLLMRemoteCacheReadMetrics, RtpLLMRemoteCacheReadMetricsCollector>(nullptr,
                                                                                                          &collector);
        } else {
            RTP_LLM_LOG_INFO("ReadMetrics : {trace_id[%s], fail[%d], read_task_cost_time_us[%ld]}",
                             trace_id.c_str(),
                             static_cast<int>(collector.remote_read_fail_qps),
                             collector.remote_read_task_cost_time_us);
        }
    }

    std::string                                    trace_id;
    int64_t                                        begin_us;
    rtp_llm::RtpLLMRemoteCacheReadMetricsCollector collector;
    const kmonitor::MetricsReporterPtr             metrics_reporter;
};

struct WriteMetricsHelper {
    WriteMetricsHelper(const std::string& trace_id, const kmonitor::MetricsReporterPtr& metrics_reporter):
        trace_id(trace_id), begin_us(rtp_llm::currentTimeUs()), metrics_reporter(metrics_reporter) {}
    ~WriteMetricsHelper() {
        collector.remote_write_task_cost_time_us = rtp_llm::currentTimeUs() - begin_us;
        if (metrics_reporter) {
            metrics_reporter->report<RtpLLMRemoteCacheWriteMetrics, RtpLLMRemoteCacheWriteMetricsCollector>(nullptr,
                                                                                                            &collector);
        } else {
            RTP_LLM_LOG_INFO(
                "WriteMetrics : {trace_id[%s], fail[%d], write_cache_block_num[%ld], write_task_cost_time_us[%ld], get_write_location_time_us[%ld], broadcast_time_us[%ld], finish_write_time_us[%ld]}",
                trace_id.c_str(),
                static_cast<int>(collector.remote_write_fail_qps),
                collector.remote_write_cache_block_num,
                collector.remote_write_task_cost_time_us,
                collector.remote_get_write_location_time_us,
                collector.remote_write_broadcast_time_us,
                collector.remote_finish_write_time_us);
        }
    }

    std::string                                     trace_id;
    int64_t                                         begin_us;
    rtp_llm::RtpLLMRemoteCacheWriteMetricsCollector collector;
    const kmonitor::MetricsReporterPtr              metrics_reporter;
};

struct SdkMetricsHelper {
    SdkMetricsHelper(const std::string& trace_id, bool is_read, const kmonitor::MetricsReporterPtr& metrics_reporter):
        trace_id(trace_id), is_read(is_read), begin_us(rtp_llm::currentTimeUs()), metrics_reporter(metrics_reporter) {}
    ~SdkMetricsHelper() {
        collector.remote_sdk_cost_time_us = rtp_llm::currentTimeUs() - begin_us;
        static kmonitor::MetricsTags read_tag("mode", "read");
        static kmonitor::MetricsTags write_tag("mode", "write");

        if (metrics_reporter) {
            metrics_reporter->report<RtpLLMRemoteCacheSDKMetrics, RtpLLMRemoteCacheSDKMetricsCollector>(
                is_read ? &read_tag : &write_tag, &collector);
        } else {
            RTP_LLM_LOG_DEBUG(
                "trace_id[%s], SDKMetrics : {mode[%s], fail[%d], sdk_block_num[%ld], sdk_cost_time_us[%ld]}",
                trace_id.c_str(),
                is_read ? "read" : "write",
                static_cast<int>(collector.remote_sdk_fail_qps),
                collector.remote_sdk_block_num,
                collector.remote_sdk_cost_time_us);
        }
    }

    std::string                                   trace_id;
    bool                                          is_read;
    int64_t                                       begin_us;
    rtp_llm::RtpLLMRemoteCacheSDKMetricsCollector collector;
    const kmonitor::MetricsReporterPtr            metrics_reporter;
};

inline std::size_t hashString(const std::string& str) {
    std::hash<std::string> hasher;
    return hasher(str);
}

inline std::string genLocationSpecName(int tp_rank, const std::string& group_name) {
    static std::string location_spec_name("tp");
    return location_spec_name + std::to_string(tp_rank) + "_" + group_name;
}

}  // namespace

bool RemoteConnectorState::doneImpl() const {
    auto cur_state = state();
    return cur_state >= State::RCS_SUCCESS || cur_state == State::RCS_ERROR;
}

bool RemoteConnectorState::successImpl() const {
    auto state = state_.load(std::memory_order_acquire);
    if (state == State::RCS_SUCCESS) {
        return true;
    }
    if (state > State::RCS_SUCCESS) {
        RTP_LLM_LOG_WARNING("state error [%d], ignore it", static_cast<int>(state));
        return true;
    }
    return false;
}

bool RemoteConnectorAsyncContext::done() const {
    return doneImpl();
}

bool RemoteConnectorAsyncContext::success() const {
    return successImpl();
}

bool RemoteAsyncMatchContext::done() const {
    return doneImpl();
}

bool RemoteAsyncMatchContext::success() const {
    return successImpl();
}

RemoteConnector::RemoteConnector(const CacheConfig&                        cache_config,
                                 const KVCacheConfig&                      kv_cache_config,
                                 const RuntimeConfig&                      runtime_config,
                                 const ParallelismConfig&                  parallelism_config,
                                 DeviceBase*                               device,
                                 void*                                     register_buffer_addr,
                                 size_t                                    register_buffer_size,
                                 std::shared_ptr<KVCacheAllocator>         allocator,
                                 RemoteConnectorGroupMode                  group_mode,
                                 const std::vector<int32_t>&               full_group_ids,
                                 const std::vector<int32_t>&               other_group_ids,
                                 const kmonitor::MetricsReporterPtr        metrics_reporter,
                                 uint32_t                                  linear_attention_write_interval,
                                 size_t                                    sink_size,
                                 size_t                                    sw_size,
                                 const std::map<std::string, std::string>& lora_info_map):
    metrics_reporter_(metrics_reporter) {
    RemoteConnector::InitParams init_params{cache_config,
                                            kv_cache_config,
                                            runtime_config,
                                            parallelism_config,
                                            device,
                                            register_buffer_addr,
                                            register_buffer_size,
                                            lora_info_map};
    init_params_ = std::make_shared<RemoteConnector::InitParams>(std::move(init_params));
    switch (group_mode) {
        case RemoteConnectorGroupMode::RCGM_LAYER_DEFAULT: {
            group_policy_ =
                std::make_unique<remote_connector::DefaultLayerGroupPolicy>(allocator, full_group_ids, other_group_ids);
            break;
        }
        case RemoteConnectorGroupMode::RCGM_ONLY_FULL_LAYER: {
            group_policy_ =
                std::make_unique<remote_connector::FullLayerGroupPolicy>(allocator, full_group_ids, other_group_ids);
            break;
        }
        case RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER: {
            group_policy_ = std::make_unique<remote_connector::FullLinearLayerGroupPolicy>(
                allocator, full_group_ids, other_group_ids, linear_attention_write_interval);
            break;
        }
        case RemoteConnectorGroupMode::RCGM_FULL_SW_LAYER: {
            group_policy_ = std::make_unique<remote_connector::FullSWLayerGroupPolicy>(
                allocator, full_group_ids, other_group_ids, sink_size, sw_size);
            break;
        }
    }
}

RemoteConnector::~RemoteConnector() {
    if (thread_pool_) {
        thread_pool_->stop();
        thread_pool_->waitFinish();
        thread_pool_.reset();
    }
    broadcaster_.reset();
}

std::pair<std::shared_ptr<RemoteConnectorConfig::LocationSpecInfoMap>,
          std::shared_ptr<RemoteConnectorConfig::LocationSpecGroups>>
RemoteConnector::genLocationSpecInfoMapAndGroups(int64_t tp_size) {
    auto   location_spec_groups_ptr = std::make_shared<RemoteConnectorConfig::LocationSpecGroups>();
    size_t group_size               = group_policy_->groups().size();
    assert(group_size > 0);
    auto location_spec_info_map_ptr = std::make_shared<RemoteConnectorConfig::LocationSpecInfoMap>();
    // TODO : support different byte_size_per_block (transfer client not support now)
    // TODO : is this correct ?
    // TODO(zhoushipei.zsp) is this correct ? block_size --> block_size_bytes
    size_t                   byte_size_per_block = init_params_->cache_config.block_size_bytes;
    std::vector<std::string> all_group_names;
    std::vector<uint64_t>    all_group_name_bithashs;
    all_group_names.reserve(group_size);
    for (const auto& entry : group_policy_->groups()) {
        const auto& group = entry.second;
        all_group_names.push_back(group.group_name);
        all_group_name_bithashs.push_back(group.group_name_bithash);
        auto [iter, success] = location_spec_groups_ptr->insert({group.group_name, {}});
        assert(success);
        group_policy_->addLocationSpecGroup(group.group_name_bithash, group.group_name);
        for (int r = 0; r < tp_size; ++r) {
            std::string location_spec_name = genLocationSpecName(r, group.group_name);
            location_spec_info_map_ptr->emplace(location_spec_name, byte_size_per_block);
            iter->second.push_back(location_spec_name);
            group_policy_->addSpecInfo(location_spec_name, entry.first, r);
        }
    }
    for (int sub_group = 2; sub_group <= group_size; ++sub_group) {
        std::string bitmask(sub_group, 1);
        bitmask.resize(group_size, 0);
        do {
            std::stringstream        ss_group_name;
            std::vector<std::string> spec_names;
            uint64_t                 groups_name_bithash = 0;
            for (int i = 0; i < group_size; ++i) {
                if (static_cast<bool>(bitmask[i])) {
                    ss_group_name << all_group_names[i];
                    groups_name_bithash |= all_group_name_bithashs[i];
                    const auto& sub_group_names = location_spec_groups_ptr->at(all_group_names[i]);
                    spec_names.insert(spec_names.end(), sub_group_names.begin(), sub_group_names.end());
                }
            }
            std::string groups_name = ss_group_name.str();
            group_policy_->addLocationSpecGroup(groups_name_bithash, groups_name);
            location_spec_groups_ptr->insert({groups_name, std::move(spec_names)});
        } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
    }
    /*
        location_spec_groups has all combinations of groups, which means:
        location_spec_groups.size() == Sum(Comb(group_size, i)), i = [1, 2, ..., group_size]

        example:
        tp_size = 2, one full attn group F0, two linear attn groups L1, L2;
        location_spec_groups.size() == Comb(3, 1) +  Comb(3, 2) +  Comb(3, 3) == 7
        location_spec_groups ==
        {
            "F0" : ["tp0_F0", "tp1_F0"],
            "L1" : ["tp0_L1", "tp1_L1"],
            "L2" : ["tp0_L2", "tp1_L2"],
            "F0L1" : ["tp0_F0", "tp1_F0", "tp0_L1", "tp1_L1"],
            "F0L2" : ["tp0_F0", "tp1_F0", "tp0_L2", "tp1_L2"],
            "L1L2" : ["tp0_L1", "tp1_L1", "tp0_L2", "tp1_L2"],
            "F0L1L2" : ["tp0_F0", "tp1_F0", "tp0_L1", "tp1_L1", "tp0_L2", "tp1_L2"]
        }
    */
    return {location_spec_info_map_ptr, location_spec_groups_ptr};
}

remote_connector::ClientWrapper::ConfigMap RemoteConnector::genClientConfig() {
    bool enable_vipserver = autil::EnvUtil::getEnv("RECO_ENABLE_VIPSERVER", false);
    auto vipserver_domain = autil::EnvUtil::getEnv("RECO_VIPSERVER_DOMAIN", std::string(""));

    std::vector<std::string> addresses;
    if (auto address = autil::EnvUtil::getEnv("RECO_SERVER_ADDRESS", std::string("")); !address.empty()) {
        addresses.push_back(std::move(address));
    }
    auto instance_group      = autil::EnvUtil::getEnv("RECO_INSTANCE_GROUP", std::string("default"));
    auto meta_channel_config = std::make_shared<MetaChannelConfig>(
        autil::EnvUtil::getEnv("RECO_META_CHANNEL_RETRY_TIME", (uint32_t)3),
        autil::EnvUtil::getEnv("RECO_META_CHANNEL_CONNECTION_TIMEOUT", (uint32_t)6000),
        autil::EnvUtil::getEnv("RECO_META_CHANNEL_CALL_TIMEOUT", (uint32_t)100));

    auto sdk_wrapper_config =
        std::make_shared<SdkWrapperConfig>(autil::EnvUtil::getEnv("RECO_STORAGE_THREAD_NUM", 4),
                                           autil::EnvUtil::getEnv("RECO_STORAGE_QUEUE_SIZE", 2000),
                                           autil::EnvUtil::getEnv("RECO_PUT_TIMEOUT_MS", 2000),
                                           autil::EnvUtil::getEnv("RECO_GET_TIMEOUT_MS", 2000));
    auto sdk_backend_configs_str =
        autil::EnvUtil::getEnv("RECO_MODEL_SDK_CONFIG", std::string(R"([{"type":"local","sdk_log_level":"DEBUG"}])"));
    autil::legacy::FromJsonString(sdk_wrapper_config->sdk_backend_configs(), sdk_backend_configs_str);

    uint32_t block_size = init_params_->cache_config.seq_size_per_block;

    // ModelDeployment
    const auto& model_name = init_params_->runtime_config.model_name;
    const auto& dtype_str =
        getDataTypeStr(init_params_->cache_config.cache_specs[0]->dtype);  // TODO(zhoushipei.zsp) is this right?
    bool    use_mla      = init_params_->cache_config.use_mla;
    int64_t tp_size      = init_params_->parallelism_config.tp_size;
    int64_t dp_size      = init_params_->parallelism_config.dp_size;
    int     fp8_kv_cache = init_params_->kv_cache_config.fp8_kv_cache;
    auto    user_data    = autil::EnvUtil::getEnv("RECO_MODEL_USER_DATA", std::string(""));
    auto    extra_info   = autil::EnvUtil::getEnv("RECO_MODEL_EXTRA_INFO", std::string(""));
    if (extra_info.empty()) {
        // legacy info
        auto biz_name  = autil::EnvUtil::getEnv("BIZ_NAME", std::string(""));
        auto ckpt_path = autil::EnvUtil::getEnv("CHECKPOINT_PATH", std::string(""));
        extra_info += biz_name + '/' + std::to_string(hashString(ckpt_path));
    }

    auto [location_spec_info_map, location_spec_groups] = genLocationSpecInfoMapAndGroups(tp_size);

    if (init_params_->lora_info_map.empty()) {
        init_params_->lora_info_map[""] = "";  // default : no lora
    }

    auto instance_id_salt = autil::EnvUtil::getEnv("RECO_INSTANCE_ID_SALT", std::string(""));

    remote_connector::ClientWrapper::ConfigMap result;
    for (const auto& [lora_adapter_name, lora_path] : init_params_->lora_info_map) {
        std::string lora_info_str;
        if (!lora_adapter_name.empty()) {
            lora_info_str = lora_adapter_name + '_' + std::to_string(hashString(lora_path));
        }
        std::stringstream instance_id_hash_ss;
        instance_id_hash_ss << "instance_group: " << instance_group << "block_size:" << block_size
                            << ";model_name:" << model_name << ";dtype_str:" << dtype_str << ";use_mla:" << use_mla
                            << ";fp8_kv_cache:" << fp8_kv_cache << ";tp_size:" << tp_size << ";dp_size:" << dp_size
                            << ";extra_info:" << extra_info << ";lora_info:" << lora_info_str
                            << ";location_spec_info:" << autil::legacy::ToJsonString(location_spec_info_map, true);
        std::string instace_id_hash_str = instance_id_hash_ss.str();
        std::string instance_id_hash    = std::to_string(hashString(instace_id_hash_str));
        std::string instance_id(instance_id_salt);
        if (instance_id.empty()) {
            instance_id = std::move(instance_id_hash);
        } else {
            instance_id += "_" + instance_id_hash;
        }

        RTP_LLM_LOG_INFO("lora_adapter_name[%s], instance_id[%s], instace_id_hash_str[%s]",
                         lora_adapter_name.c_str(),
                         instance_id.c_str(),
                         instace_id_hash_str.c_str());
        auto config = std::make_shared<RemoteConnectorConfig>(
            enable_vipserver,
            vipserver_domain,
            block_size,
            instance_group,
            instance_id,
            addresses,
            location_spec_info_map,
            meta_channel_config,
            sdk_wrapper_config,
            location_spec_groups,
            ModelDeployment(model_name, dtype_str, use_mla, tp_size, dp_size, lora_info_str, 1, extra_info, user_data));
        result[lora_adapter_name] = config;
    }
    return result;
}

bool RemoteConnector::init() {
    RTP_LLM_LOG_INFO("start init remote connector");
    if (!group_policy_->init()) {
        RTP_LLM_LOG_ERROR("init group policy failed");
        return false;
    }

    auto client_config_map_str = autil::EnvUtil::getEnv("RECO_CLIENT_CONFIG", std::string(""));
    remote_connector::ClientWrapper::ConfigMap client_config_map;
    try {
        if (!client_config_map_str.empty()) {
            autil::legacy::FromJsonString(client_config_map, client_config_map_str);
        } else {
            client_config_map = genClientConfig();
        }
    } catch (const autil::legacy::ExceptionBase& e) {
        RTP_LLM_LOG_ERROR(
            "parse RECO_CLIENT_CONFIG [%s] fail.\n %s, exception: [%s]", client_config_map_str.c_str(), e.what());
        return false;
    }
    auto tp_rank    = init_params_->device->getDeviceProperties().tp_rank;
    client_wrapper_ = std::make_shared<remote_connector::ClientWrapper>();
    kv_cache_manager::RegistSpan regist_span{init_params_->register_buffer_addr, init_params_->register_buffer_size};
    constexpr int32_t            full_group_idx = 0;  // TODO : transfer client not support diffent spec size
    kv_cache_manager::InitParams client_init_params{
        tp_rank == 0 ? kv_cache_manager::RoleType::HYBRID : kv_cache_manager::RoleType::WORKER,
        &regist_span,
        genLocationSpecName(tp_rank, group_policy_->groups().at(full_group_idx).group_name)};
    if (!client_wrapper_->init(client_config_map, client_init_params)) {
        RTP_LLM_LOG_ERROR("create remote kv cache client failed");
        return false;
    }
    size_t thread_num = autil::EnvUtil::getEnv("RECO_ASYNCWRAPPER_THREAD_NUM", 16);
    size_t queue_size = autil::EnvUtil::getEnv("RECO_ASYNCWRAPPER_QUEUE_SIZE", 1000);
    thread_pool_      = std::make_unique<autil::LockFreeThreadPool>(thread_num, queue_size, nullptr, "RECOThreadPool");
    if (!thread_pool_->start()) {
        RTP_LLM_LOG_ERROR(
            "init failed, start thread pool failed, thread num: %zu, queue size: %zu", thread_num, queue_size);
        return false;
    }
    get_broadcast_timeout_ = autil::EnvUtil::getEnv("RECO_GET_BROADCAST_TIMEOUT", get_broadcast_timeout_);
    put_broadcast_timeout_ = autil::EnvUtil::getEnv("RECO_PUT_BROADCAST_TIMEOUT", put_broadcast_timeout_);
    broadcaster_           = std::make_shared<TpBroadcastManager>(init_params_->runtime_config.worker_grpc_addrs);
    if (!broadcaster_->init()) {
        RTP_LLM_LOG_ERROR("failed to init broadcast manager");
        return false;
    }
    RTP_LLM_LOG_INFO("init remote connector success");
    printInfo();
    return true;
}

void RemoteConnector::printInfo() const {
    std::stringstream debug_ss;
    debug_ss << '\n';
    debug_ss << "get_broadcast_timeout : " << get_broadcast_timeout_ << '\n';
    debug_ss << "put_broadcast_timeout : " << put_broadcast_timeout_ << '\n';
    debug_ss << "group_policy : {" << '\n' << group_policy_->debugString() << "}\n";

    RTP_LLM_LOG_INFO("%s", debug_ss.str().c_str());
}

std::shared_ptr<KVCacheConnector::AsyncMatchContext>
RemoteConnector::asyncMatch(const std::shared_ptr<KVCacheResourceV1>& resource, const std::shared_ptr<Meta>& meta) {
    if (const auto& remote_connector_meta = std::dynamic_pointer_cast<RemoteConnectorMeta>(meta);
        remote_connector_meta != nullptr) {
        auto async_match_context = std::make_shared<RemoteAsyncMatchContext>(resource->reuseBlocksNum());
        auto ec                  = thread_pool_->pushTask(
            [this, resource, remote_connector_meta, async_match_context]() {
                async_match_context->setState(RemoteConnectorState::State::RCS_START);
                this->asyncMatchTask(resource, remote_connector_meta, async_match_context);
            },
            false);
        if (ec != autil::ThreadPoolBase::ERROR_TYPE::ERROR_NONE) {
            async_match_context->setState(RemoteConnectorState::State::RCS_THREADPOOL_FULL);
            RTP_LLM_LOG_WARNING("asyncRead push task failed, ec [%d]", ec);
            return nullptr;
        }
        return async_match_context;
    }
    RTP_LLM_LOG_WARNING("cast meta to RemoteConnectorMeta failed");
    return nullptr;
}

std::shared_ptr<AsyncContext> RemoteConnector::asyncRead(const std::shared_ptr<KVCacheResourceV1>& resource,
                                                         const std::shared_ptr<Meta>&              meta,
                                                         const std::shared_ptr<AsyncMatchContext>& match_context) {
    if (const auto& remote_match_context = std::dynamic_pointer_cast<RemoteAsyncMatchContext>(match_context);
        remote_match_context != nullptr) {
        auto async_context = std::make_shared<RemoteConnectorAsyncContext>();
        auto ec            = thread_pool_->pushTask(
            [this, resource, meta, async_context, remote_match_context]() {
                async_context->setState(RemoteConnectorAsyncContext::State::RCS_START);
                this->asyncReadTask(resource, meta, async_context, remote_match_context);
            },
            false);
        if (ec != autil::ThreadPoolBase::ERROR_TYPE::ERROR_NONE) {
            async_context->setState(RemoteConnectorAsyncContext::State::RCS_THREADPOOL_FULL);
            RTP_LLM_LOG_WARNING("asyncRead push task failed, ec [%d]", ec);
            return nullptr;
        }
        return async_context;
    }
    RTP_LLM_LOG_WARNING("cast meta to RemoteConnectorMeta failed");
    return nullptr;
}

std::shared_ptr<AsyncContext> RemoteConnector::asyncWrite(const std::shared_ptr<KVCacheResourceV1>&      resource,
                                                          const std::shared_ptr<KVCacheConnector::Meta>& meta) {
    if (const auto& remote_connector_meta = std::dynamic_pointer_cast<RemoteConnectorMeta>(meta);
        remote_connector_meta != nullptr) {
        auto async_context = std::make_shared<RemoteConnectorAsyncContext>();
        auto ec            = thread_pool_->pushTask(
            [this, resource, remote_connector_meta, async_context]() {
                async_context->setState(RemoteConnectorAsyncContext::State::RCS_START);
                this->asyncWriteTask(resource, remote_connector_meta, async_context);
            },
            false);
        if (ec != autil::ThreadPoolBase::ERROR_TYPE::ERROR_NONE) {
            RTP_LLM_LOG_WARNING("asyncWrite push task failed, ec [%d]", ec);
            return nullptr;
        }
        return async_context;
    }
    RTP_LLM_LOG_WARNING("cast meta to RemoteConnectorMeta failed");
    return nullptr;
}

std::shared_ptr<AsyncContext> RemoteConnector::asyncWriteByLayer(int                                       layer_id,
                                                                 const std::shared_ptr<KVCacheResourceV1>& resource,
                                                                 const std::shared_ptr<KVCacheConnector::Meta>& meta) {
    throw std::runtime_error("Not Implement");
    return nullptr;
}

bool RemoteConnector::copyCache(const RemoteBroadcastTpRequestPB& request, RemoteBroadcastTpResponsePB& response) {
    const auto&                 trace_id = request.trace_id();
    std::vector<int32_t>        group_ids(request.group_ids().begin(), request.group_ids().end());
    std::vector<int32_t>        block_ids(request.block_ids().begin(), request.block_ids().end());
    kv_cache_manager::UriStrVec uris(request.uris().begin(), request.uris().end());
    switch (request.op()) {
        case ::RemoteBroadcastTpOp::REMOTE_BROADCAST_READ: {
            if (!Read(trace_id, group_ids, block_ids, uris)) {
                RTP_LLM_LOG_WARNING("broadcastTp Read failed");
                return false;
            }
            break;
        }
        case ::RemoteBroadcastTpOp::REMOTE_BROADCAST_WRITE: {
            kv_cache_manager::UriStrVec out_uris;
            if (!Write(trace_id, group_ids, block_ids, uris, out_uris)) {
                RTP_LLM_LOG_WARNING("broadcastTp Write failed");
                return false;
            }
            if (!out_uris.empty()) {
                auto mutable_actual_uris = response.mutable_actual_uris();
                mutable_actual_uris->Reserve(out_uris.size());
                for (const auto& uri_str : out_uris) {
                    auto actual_uri = mutable_actual_uris->Add();
                    *actual_uri     = uri_str;
                }
            }
            break;
        }
        default: {
            RTP_LLM_LOG_WARNING("invalid operation [%d]", request.op());
            return false;
        }
    }
    return true;
}

#define RETURN_IF(condition, method)                                                                                   \
    do {                                                                                                               \
        if (condition) {                                                                                               \
            async_context->setState(RemoteConnectorAsyncContext::State::RCS_SUCCESS);                                  \
            helper.collector.remote_##method##_fail_qps = false;                                                       \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define CHECK_AND_LOG(condition, state, format, args...)                                                               \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            RTP_LLM_LOG_WARNING(format, ##args);                                                                       \
            async_context->setState(RemoteConnectorAsyncContext::State::state);                                        \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define CHECK_AND_LOG_WITH_DEFER(condition, defer_lambda, format, args...)                                             \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            RTP_LLM_LOG_WARNING(format, ##args);                                                                       \
            defer_lambda();                                                                                            \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

void RemoteConnector::asyncMatchTask(const std::shared_ptr<KVCacheResourceV1>&       resource,
                                     const std::shared_ptr<RemoteConnectorMeta>&     meta,
                                     const std::shared_ptr<RemoteAsyncMatchContext>& async_context) {
    RTP_LLM_LOG_DEBUG("asyncMatchTask, [%d] [%zu]", resource->reuseBlocksNum(), resource->cacheKeys().size());
    std::string match_trace_id = "match_" + meta->trace_id;
    // only gpu reuse blocks num now
    kv_cache_manager::BlockMaskOffset block_mask = resource->reuseBlocksNum();
    const std::vector<int64_t>&       keys       = resource->cacheKeys();
    MatchMetricsHelper                helper(match_trace_id, metrics_reporter_);
    RETURN_IF(block_mask >= keys.size(), match);
    const std::string&          unique_id  = meta->unique_id;
    kv_cache_manager::QueryType query_type = kv_cache_manager::QueryType::QT_PREFIX_MATCH;
    async_context->setState(RemoteConnectorAsyncContext::State::RCS_READ_MATCH);
    auto match_result = client_wrapper_->match(unique_id, match_trace_id, query_type, keys, block_mask, {});
    CHECK_AND_LOG(match_result.first, RCS_READ_MATCH_ERROR, "asyncGet match failed, [%s]", match_trace_id.c_str());
    async_context->set_trace_id(match_trace_id);
    async_context->set_locations(std::move(match_result.second));
    async_context->set_matched_block_count(async_context->locations_ptr()->size()
                                           + async_context->prev_reuse_blocks_num());  // gpu + remote
    async_context->setState(RemoteConnectorAsyncContext::State::RCS_SUCCESS);
}

void RemoteConnector::asyncReadTask(const std::shared_ptr<KVCacheResourceV1>&           resource,
                                    const std::shared_ptr<Meta>&                        meta,
                                    const std::shared_ptr<RemoteConnectorAsyncContext>& async_context,
                                    const std::shared_ptr<RemoteAsyncMatchContext>&     match_context) {
    RTP_LLM_LOG_DEBUG("asyncReadTask, [%d] [%zu]", resource->reuseBlocksNum(), resource->cacheKeys().size());
    const std::string& match_trace_id    = match_context->trace_id();
    auto [start_block_index, reuse_size] = meta->blockRange();
    RTP_LLM_LOG_DEBUG("start_block_index:[%d], reuse_size:[%d]", start_block_index, reuse_size);
    assert(start_block_index >= match_context->prev_reuse_blocks_num());
    auto start_location_index =
        start_block_index - match_context->prev_reuse_blocks_num();  // prev_reuse_blocks_num only include gpu now
    const auto&       locations_ptr = match_context->locations_ptr();
    ReadMetricsHelper helper(match_trace_id, metrics_reporter_);
    RETURN_IF(start_location_index >= locations_ptr->size(), read);

    std::vector<BroadcastTpRequestPB> requests;
    size_t                            new_reuse_block_num = 0;
    CHECK_AND_LOG(genReadRequest(broadcaster_->workerNum(),
                                 *locations_ptr,
                                 start_block_index,
                                 start_location_index,
                                 match_trace_id,
                                 resource,
                                 requests,
                                 new_reuse_block_num),
                  RCS_ERROR,
                  "remote_connector_get genRequest failed");
    RETURN_IF(new_reuse_block_num == 0, read);
    async_context->setState(RemoteConnectorAsyncContext::State::RCS_READ_BROADCAST);
    auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& context,
                       const BroadcastTpRequestPB&                 request,
                       grpc::CompletionQueue*                      completion_queue) {
        return stub->AsyncBroadcastTp(context.get(), request, completion_queue);
    };
    auto broadcast_result = broadcaster_->broadcast<BroadcastTpRequestPB, BroadcastTpResponsePB>(
        requests, get_broadcast_timeout_, rpc_call);
    broadcast_result->waitDone();
    CHECK_AND_LOG(
        broadcast_result->success(), RCS_ERROR, "Read failed for grpc status, trace_id [%s]", match_trace_id.c_str());
    // TODO : maybe not all locations are loaded successfuly
    helper.collector.remote_read_fail_qps = false;
    async_context->setState(RemoteConnectorAsyncContext::State::RCS_SUCCESS);
    resource->setRemoteReuseBlocksNum(new_reuse_block_num);
    resource->setReuseBlocksNum(start_block_index + new_reuse_block_num);  // TODO(zhoushipei.zsp) reuse_len to be fixed
}

void RemoteConnector::asyncWriteTask(const std::shared_ptr<KVCacheResourceV1>&           resource,
                                     const std::shared_ptr<RemoteConnectorMeta>&         meta,
                                     const std::shared_ptr<RemoteConnectorAsyncContext>& async_context) {
    RTP_LLM_LOG_DEBUG("asyncWriteTask, [%d] [%zu]", resource->reuseBlocksNum(), resource->cacheKeys().size());
    WriteMetricsHelper          helper(meta->trace_id, metrics_reporter_);
    const std::string&          unique_id            = meta->unique_id;
    std::string                 start_write_trace_id = "start_write_" + meta->trace_id;
    const std::vector<int64_t>& keys                 = resource->cacheKeys();
    const std::vector<int64_t>& tokens               = meta->tokens;
    std::vector<std::string>    location_spec_group_names;
    RETURN_IF(keys.empty(), write);
    CHECK_AND_LOG(group_policy_->getNeedWriteGroups(resource, location_spec_group_names),
                  RCS_ERROR,
                  "trace_id [%s] filter need write groups failed",
                  start_write_trace_id.c_str());
    // 1. for meta_client : get cache location from remote service
    async_context->setState(RemoteConnectorAsyncContext::State::RCS_WRITE_START);
    auto [start_result, write_location] = client_wrapper_->getWriteLocation(
        unique_id, start_write_trace_id, keys, tokens, location_spec_group_names, 600);
    helper.collector.remote_get_write_location_time_us = currentTimeUs() - helper.begin_us;
    CHECK_AND_LOG(start_result, RCS_ERROR, "asyncPut getWriteLocation failed, [%s]", start_write_trace_id.c_str());
    RETURN_IF(write_location.locations.empty(), write);
    // TODO : support finish partially
    const std::string&                 write_session_id = write_location.write_session_id;
    static kv_cache_manager::Locations empty_locations;
    kv_cache_manager::Locations*       actual_locations  = &empty_locations;
    size_t                             succeed_block_num = 0;
    auto                               finish_write_task = [&, this](bool success = false) {
        int64_t     finish_write_begin_us = currentTimeUs();
        std::string finish_write_trace_id = "finish_write_" + meta->trace_id;
        async_context->setState(RemoteConnectorAsyncContext::State::RCS_WRITE_FINISH);
        bool finish_result = this->client_wrapper_->finishWrite(
            unique_id, finish_write_trace_id, write_session_id, succeed_block_num, *actual_locations);
        helper.collector.remote_finish_write_time_us = currentTimeUs() - finish_write_begin_us;
        CHECK_AND_LOG(finish_result, RCS_ERROR, "asyncPut finishWrite failed, [%s]", finish_write_trace_id.c_str());
        if (success) {
            async_context->setState(RemoteConnectorAsyncContext::State::RCS_SUCCESS);
            helper.collector.remote_write_fail_qps = false;
        } else {
            async_context->setState(RemoteConnectorAsyncContext::State::RCS_ERROR);
        }
    };
    // 2. sync call all rank write
    std::vector<BroadcastTpRequestPB> requests;
    ActualUriGather                   actual_uri_gather;
    CHECK_AND_LOG_WITH_DEFER(genWriteRequest(broadcaster_->workerNum(),
                                             write_location.locations,
                                             write_location.block_mask,
                                             start_write_trace_id,
                                             resource,
                                             requests,
                                             actual_uri_gather),
                             finish_write_task,
                             "remote_connector_put genRequest failed");
    helper.collector.remote_write_cache_block_num = requests[0].remote_request().block_ids_size();

    async_context->setState(RemoteConnectorAsyncContext::State::RCS_WRITE_BROADCAST);
    auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& context,
                       const BroadcastTpRequestPB&                 request,
                       grpc::CompletionQueue*                      completion_queue) {
        return stub->AsyncBroadcastTp(context.get(), request, completion_queue);
    };
    int64_t broadcast_begin_us = currentTimeUs();
    auto    broadcast_result   = broadcaster_->broadcast<BroadcastTpRequestPB, BroadcastTpResponsePB>(
        requests, put_broadcast_timeout_, rpc_call);
    broadcast_result->waitDone();
    helper.collector.remote_write_broadcast_time_us = currentTimeUs() - broadcast_begin_us;
    CHECK_AND_LOG_WITH_DEFER(broadcast_result->success(),
                             finish_write_task,
                             "Write failed for grpc status, trace_id [%s]",
                             meta->trace_id.c_str());
    auto responses = broadcast_result->responses();
    // 3. for meta_client : finish write
    // TODO : get real succeed_block_num
    succeed_block_num         = write_location.locations.size();
    bool actual_uri_not_empty = false;
    for (int i = 0; i < broadcaster_->workerNum(); i++) {
        const auto& proto_actual_uris = responses[i].remote_response().actual_uris();
        for (int j = 0; j < proto_actual_uris.size(); j++) {
            if (!proto_actual_uris[j].empty()) {
                actual_uri_not_empty         = true;
                actual_uri_gather[i][j]->uri = proto_actual_uris[j];
            }
        }
    }
    if (actual_uri_not_empty) {
        actual_locations = &write_location.locations;
    }
    finish_write_task(true);
}

#undef RETURN_IF
#undef CHECK_AND_LOG
#undef CHECK_AND_LOG_WITH_DEFER

bool RemoteConnector::genReadRequest(size_t                                    tp_size,
                                     const kv_cache_manager::Locations&        locations,
                                     size_t                                    block_idx,
                                     const kv_cache_manager::BlockMaskOffset&  block_mask,
                                     const std::string&                        trace_id,
                                     const std::shared_ptr<KVCacheResourceV1>& resource,
                                     std::vector<BroadcastTpRequestPB>&        requests,
                                     size_t&                                   new_reuse_block_num) const {
    requests.resize(tp_size, {});
    for (size_t i = 0; i < tp_size; ++i) {
        requests[i].mutable_remote_request()->set_op(::RemoteBroadcastTpOp::REMOTE_BROADCAST_READ);
        requests[i].mutable_remote_request()->set_trace_id(trace_id);
    }
    remote_connector::LocationsView locations_view;
    if (!group_policy_->filterNeedLoadLocations(locations, locations_view, block_mask)) {
        RTP_LLM_LOG_WARNING("trace_id [%s], filterNeedLoadLocations failed");
        return false;
    }
    new_reuse_block_num           = locations_view.size() - block_mask;  // only contains remote
    const auto& spec_name_to_info = group_policy_->spec_info_map();
    for (const auto& location_view : locations_view) {
        if (location_view.empty()) {
            continue;
        }
        for (const auto& location_spec : location_view) {
            const auto iter = spec_name_to_info.find(location_spec.spec_name);
            if (iter == spec_name_to_info.end()) {
                RTP_LLM_LOG_WARNING("trace_id [%s], genReadRequest not find spec_name [%s]",
                                    trace_id.c_str(),
                                    location_spec.spec_name.data());
                return false;
            }
            const auto& spec_info      = iter->second;
            auto        remote_request = requests[spec_info.tp_rank].mutable_remote_request();
            remote_request->add_group_ids(spec_info.group_id);
            const auto& block_indices = resource->groupBlocks().at(spec_info.group_id)->blocks();
            if (block_indices.size() <= block_idx) {
                RTP_LLM_LOG_ERROR("trace_id [%s], group_id [%d] bad block_indices size[%lu], block_idx [%zu]",
                                  trace_id.c_str(),
                                  spec_info.group_id,
                                  block_indices.size(),
                                  block_idx);
                return false;
            }
            auto block_id = block_indices.at(block_idx);
            remote_request->add_block_ids(block_id);
            remote_request->add_uris(location_spec.uri.data());
        }
        block_idx++;
    }
    return true;
}

bool RemoteConnector::genWriteRequest(size_t                                    tp_size,
                                      const kv_cache_manager::Locations&        locations,
                                      const kv_cache_manager::BlockMask&        block_mask,
                                      const std::string&                        trace_id,
                                      const std::shared_ptr<KVCacheResourceV1>& resource,
                                      std::vector<BroadcastTpRequestPB>&        requests,
                                      ActualUriGather&                          actual_uri_gather) const {
    requests.resize(tp_size, {});
    actual_uri_gather.resize(tp_size, {});
    for (size_t i = 0; i < tp_size; ++i) {
        requests[i].mutable_remote_request()->set_op(::RemoteBroadcastTpOp::REMOTE_BROADCAST_WRITE);
        requests[i].mutable_remote_request()->set_trace_id(trace_id);
        actual_uri_gather[i].reserve(locations.size() * 1.2);
    }
    int64_t cache_key_idx = 0;
    if (std::holds_alternative<kv_cache_manager::BlockMaskOffset>(block_mask)) {
        cache_key_idx = std::get<kv_cache_manager::BlockMaskOffset>(block_mask);
        cache_key_idx--;
    }
    const auto& spec_name_to_info = group_policy_->spec_info_map();
    for (const auto& location : locations) {
        std::visit(
            [&cache_key_idx](const auto& block_mask) {
                using T = std::decay_t<decltype(block_mask)>;
                if constexpr (std::is_same_v<kv_cache_manager::BlockMaskOffset, T>) {
                    cache_key_idx++;
                } else if constexpr (std::is_same_v<kv_cache_manager::BlockMaskVector, T>) {
                    for (auto i = cache_key_idx; i < block_mask.size(); i++) {
                        if (block_mask[i]) {
                            cache_key_idx++;
                        }
                        break;
                    }
                }
            },
            block_mask);
        for (const auto& location_spec : location) {
            const auto iter = spec_name_to_info.find(location_spec.spec_name);
            if (iter == spec_name_to_info.end()) {
                RTP_LLM_LOG_WARNING("trace_id [%s], genWriteRequest not find spec_name [%s]",
                                    trace_id.c_str(),
                                    location_spec.spec_name.c_str());
                return false;
            }
            const auto& spec_info      = iter->second;
            auto        remote_request = requests[spec_info.tp_rank].mutable_remote_request();
            remote_request->add_group_ids(spec_info.group_id);
            const auto& block_indices = resource->groupBlocks().at(spec_info.group_id)->blocks();
            if (block_indices.size() <= cache_key_idx) {
                RTP_LLM_LOG_ERROR("trace_id [%s], group_id [%d] bad block_indices size[%lu]",
                                  trace_id.c_str(),
                                  spec_info.group_id,
                                  block_indices.size());
                return false;
            }
            auto block_id = block_indices.at(cache_key_idx);
            remote_request->add_block_ids(block_id);
            remote_request->add_uris(location_spec.uri);
            actual_uri_gather[spec_info.tp_rank].push_back(
                const_cast<kv_cache_manager::LocationSpecUnit*>(&location_spec));
        }
    }
    return true;
}

bool RemoteConnector::Read(const std::string&                 trace_id,
                           const std::vector<int32_t>&        group_ids,
                           const std::vector<int32_t>&        block_ids,
                           const kv_cache_manager::UriStrVec& uri_str_vec) {
    // for transfer client
    // TODO : support only part of the blocks loading successfully
    SdkMetricsHelper helper(trace_id, true, metrics_reporter_);
    helper.collector.remote_sdk_block_num = block_ids.size();
    kv_cache_manager::BlockBuffers block_buffers;
    if (!group_policy_->genBlockBuffers(group_ids, block_ids, block_buffers)) {
        return false;
    }
    if (!client_wrapper_->loadKvCaches(uri_str_vec, block_buffers)) {
        return false;
    }
    helper.collector.remote_sdk_fail_qps = false;
    return true;
}

bool RemoteConnector::Write(const std::string&                 trace_id,
                            const std::vector<int32_t>&        group_ids,
                            const std::vector<int32_t>&        block_ids,
                            const kv_cache_manager::UriStrVec& uri_str_vec,
                            kv_cache_manager::UriStrVec&       out_uri_str_vec) {
    // for transfer client
    // TODO : support finish partially
    SdkMetricsHelper helper(trace_id, false, metrics_reporter_);
    helper.collector.remote_sdk_block_num = block_ids.size();
    kv_cache_manager::BlockBuffers block_buffers;
    if (!group_policy_->genBlockBuffers(group_ids, block_ids, block_buffers)) {
        return false;
    }
    auto result = client_wrapper_->saveKvCaches(uri_str_vec, block_buffers);
    if (!result.first) {
        return false;
    }
    if (!result.second.empty()) {
        if (uri_str_vec.size() != result.second.size()) {
            RTP_LLM_LOG_WARNING("some internal error happens in saveKvCaches, expectd [%lu], actual [%lu]",
                                uri_str_vec.size(),
                                result.second.size());
            return false;
        }

        if (uri_str_vec != result.second) {
            out_uri_str_vec = std::move(result.second);
        }
    }
    helper.collector.remote_sdk_fail_qps = false;
    return true;
}

}  // namespace rtp_llm
