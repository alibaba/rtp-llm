#include "ProxyDistKvCache.h"
#include "ProxyKvCachePlanner.h"
#include "rtp_llm/cpp/cache/DistKvCacheMetrics.h"
#include "rtp_llm/cpp/cache/CacheManager.h"

using namespace rtp_llm::threefs;

namespace rtp_llm {

namespace {
inline std::size_t hashString(const std::string& str) {
    std::hash<std::string> hasher;
    return hasher(str);
}

inline std::string genLocationSpecName(int tp_rank) {
    static std::string location_spec_name("tp_rank_");
    return location_spec_name + std::to_string(tp_rank);
}

inline kv_cache_manager::BlockBuffers genKvcmBlockBuffers(const std::vector<DistStorage::Item>& layout_items) {
    kv_cache_manager::BlockBuffers block_buffers;
    block_buffers.reserve(layout_items.size());
    for (const auto& item : layout_items) {
        kv_cache_manager::BlockBuffer block_buffer;
        block_buffer.iovs.reserve(item.iovs.size());
        for (const auto& item_iov : item.iovs) {
            block_buffer.iovs.push_back(
                {kv_cache_manager::MemoryType::GPU, item_iov.data.get(), item_iov.len, item_iov.ignore});
        }
        block_buffers.push_back(std::move(block_buffer));
    }
    return block_buffers;
}

}  // namespace

ProxyDistKvCache::ProxyDistKvCache(CacheManager*                       cache_manager,
                                   const GptInitParameter&             gpt_init_params,
                                   const kmonitor::MetricsReporterPtr& metrics_reporter):
    DistKvCache(cache_manager, gpt_init_params, metrics_reporter) {}

bool ProxyDistKvCache::init(const DistKvCacheInitParams& init_params) {
    planner_ = std::make_unique<ProxyKvCachePlanner>(cache_manager_, gpt_init_params_, metrics_reporter_);
    if (!initKVCMClientWrapper()) {
        RTP_LLM_LOG_WARNING("init failed, init remote kv cache client failed");
        return false;
    }
    // TODO : how to design wait_match_thread_pool_ / io_thread_pool_

    wait_match_thread_pool_ =
        std::make_unique<autil::LockFreeThreadPool>(thread_num_, queue_size_, nullptr, "WaitMatchThreadPool");
    if (!wait_match_thread_pool_->start()) {
        RTP_LLM_LOG_WARNING("init failed, start wait match thread pool failed, thread num: %zu, queue size: %zu",
                            thread_num_,
                            queue_size_);
        return false;
    }

    rpc_pool_ = std::make_shared<rtp_llm::RPCPool>();

    RTP_LLM_LOG_INFO("proxy dist kv cache init success");
    return true;
}

std::map<std::string, std::string> ProxyDistKvCache::genKVCMClientConfig() const {
    static std::string config_format = R"(
{
"enable_vipserver" : %s,
"vipserver_domain" : "%s",
"instance_group": "%s",
"instance_id": "%s",
"address": ["%s"],
"meta_channel_config": {
    "retry_time":%u,
    "connection_timeout":%u,
    "call_timeout":%u
},
"block_size": %d,
"location_spec_infos": %s,
"sdk_config": {
    "thread_num":%d,
    "queue_size":%d,
    "sdk_backend_configs":%s,
    "timeout_config": {
        "put_timeout_ms":%d,
        "get_timeout_ms":%d
}},
"model_deployment": {
    "model_name": "%s",
    "dtype": "%s",
    "use_mla": %s,
    "tp_size": %ld,
    "dp_size": %ld,
    "lora_name": "%s",
    "pp_size": 1,
    "extra": "%s",
    "user_data": "%s"
}
})";

    bool enable_vip_server = autil::EnvUtil::getEnv("KVCM_ENABLE_VIPSERVER", false);
    auto vipserver_domain  = autil::EnvUtil::getEnv("KVCM_VIPSERVER_DOMAIN", std::string(""));
    auto server_address    = autil::EnvUtil::getEnv("KVCM_SERVER_ADDRESS", std::string(""));
    auto instance_group    = autil::EnvUtil::getEnv("KVCM_INSTANCE_GROUP", std::string("default"));
    auto instance_id       = autil::EnvUtil::getEnv("KVCM_INSTANCE_ID_SALT", std::string(""));
    auto extra_info        = autil::EnvUtil::getEnv("KVCM_MODEL_EXTRA_INFO", std::string(""));

    uint32_t channel_retry_time         = autil::EnvUtil::getEnv("KVCM_META_CHANNEL_RETRY_TIME", 3);
    uint32_t channel_connection_timeout = autil::EnvUtil::getEnv("KVCM_META_CHANNEL_CONNECTION_TIMEOUT", 1000);
    uint32_t channel_call_timeout       = autil::EnvUtil::getEnv("KVCM_META_CHANNEL_CALL_TIMEOUT", 100);
    if (extra_info.empty()) {
        // legacy info
        auto biz_name  = autil::EnvUtil::getEnv("BIZ_NAME", std::string(""));
        auto ckpt_path = autil::EnvUtil::getEnv("CHECKPOINT_PATH", std::string(""));
        extra_info += biz_name + '/' + std::to_string(hashString(ckpt_path));
    }
    auto user_data          = autil::EnvUtil::getEnv("KVCM_MODEL_USER_DATA", std::string(""));
    int  storage_thread_num = autil::EnvUtil::getEnv("KVCM_STORAGE_THREAD_NUM", 4);
    int  storage_queue_size = autil::EnvUtil::getEnv("KVCM_STORAGE_QUEUE_SIZE", 2000);
    int  put_timeout_ms     = autil::EnvUtil::getEnv("KVCM_PUT_TIMEOUT_MS", 2000);
    int  get_timeout_ms     = autil::EnvUtil::getEnv("KVCM_GET_TIMEOUT_MS", 2000);
    auto sdk_backend_configs =
        autil::EnvUtil::getEnv("KVCM_MODEL_SDK_CONFIG", std::string(R"([{"type":"local","sdk_log_level":"DEBUG"}])"));

    const auto& cache_config        = cache_manager_->cacheConfig();
    uint32_t    block_size          = cache_config.seq_size_per_block;
    size_t      byte_size_per_block = cache_config.block_size;
    const auto& model_parameter     = cache_manager_->gptInitParameter();
    const auto& model_name          = model_parameter.model_name_;
    const auto& dtype_str           = model_parameter.data_type_str_;
    bool        use_mla             = model_parameter.use_mla_;
    int64_t     tp_size             = model_parameter.tp_size_;
    int64_t     dp_size             = model_parameter.dp_size_;
    std::string self_location_spec_name;
    auto        location_spec_info_map = KVCMClientWrapperConfig::LocationSpecInfoMap{};
    for (size_t i = 0; i < cache_manager_->device()->getDeviceProperties().tp_size; ++i) {
        location_spec_info_map.emplace(genLocationSpecName(i), byte_size_per_block);
    }
    auto location_spec_infos_str = autil::legacy::ToJsonString(location_spec_info_map, true);

    auto lora_info_map = cache_manager_->lora_info_map();
    if (lora_info_map.empty()) {
        lora_info_map[""] = "";  // default : no lora
    }
    std::map<std::string, std::string> result;
    for (const auto& [lora_adapter_name, lora_path] : lora_info_map) {
        std::array<char, 40960> buffer;
        std::string             lora_info_str;
        if (!lora_adapter_name.empty()) {
            lora_info_str = lora_adapter_name + '_' + std::to_string(hashString(lora_path));
        }
        std::stringstream instance_id_hash_ss;
        instance_id_hash_ss << cache_config.debugString() << ";model_name:" << model_name << ";use_mla:" << use_mla
                            << ";tp_size:" << tp_size << ";dp_size:" << dp_size << ";extra_info:" << extra_info
                            << ";lora_info:" << lora_info_str << ";location_spec_info:" << location_spec_infos_str;
        std::string instance_id_hash = std::to_string(hashString(instance_id_hash_ss.str()));
        if (instance_id.empty()) {
            instance_id = std::move(instance_id_hash);
        } else {
            instance_id += "_" + instance_id_hash;
        }

        RTP_LLM_LOG_INFO("lora_adapter_name[%s], instance_id[%s]", lora_adapter_name.c_str(), instance_id.c_str());
        int n = std::snprintf(buffer.data(),
                              buffer.size(),
                              config_format.c_str(),
                              enable_vip_server ? "true" : "false",
                              vipserver_domain.c_str(),
                              instance_group.c_str(),
                              instance_id.c_str(),
                              server_address.c_str(),
                              channel_retry_time,
                              channel_connection_timeout,
                              channel_call_timeout,
                              block_size,
                              location_spec_infos_str.c_str(),
                              storage_thread_num,
                              storage_queue_size,
                              sdk_backend_configs.c_str(),
                              put_timeout_ms,
                              get_timeout_ms,
                              model_name.c_str(),
                              dtype_str.c_str(),
                              use_mla ? "true" : "false",
                              tp_size,
                              dp_size,
                              lora_info_str.c_str(),
                              extra_info.c_str(),
                              user_data.c_str());

        result[lora_info_str] = std::string(buffer.data(), n);
    }
    return result;
}

bool ProxyDistKvCache::initKVCMClientWrapper() {
    auto client_config_map_str = autil::EnvUtil::getEnv("KVCM_CLIENT_CONFIG", std::string(""));
    std::map<std::string, std::string> client_config_map;
    if (!client_config_map_str.empty()) {
        try {
            autil::legacy::FromJsonString(client_config_map, client_config_map_str);
        } catch (autil::legacy::ExceptionBase& e) {
            RTP_LLM_LOG_ERROR(
                "parse KVCM_CLIENT_CONFIG [%s] fail.\n %s, exception: [%s]", client_config_map_str.c_str(), e.what());
            return false;
        }
    } else {
        client_config_map = genKVCMClientConfig();
    }
    auto tp_rank         = cache_manager_->device()->getDeviceProperties().tp_rank;
    kvcm_client_wrapper_ = std::make_shared<KVCMClientWrapper>();
    kv_cache_manager::RegistSpan regist_span{cache_manager_->kvCacheAllocator()->getCacheBasePtr(),
                                             cache_manager_->kvCacheAllocator()->getCacheBufferSize()};
    kv_cache_manager::InitParams client_init_params{tp_rank == 0 ? kv_cache_manager::RoleType::HYBRID :
                                                                   kv_cache_manager::RoleType::WORKER,
                                                    &regist_span,
                                                    genLocationSpecName(tp_rank)};
    if (!kvcm_client_wrapper_->init(client_config_map, client_init_params)) {
        RTP_LLM_LOG_ERROR("create remote kv cache client failed");
        return false;
    }
    RTP_LLM_LOG_INFO("create remote kv cache client success");
    return true;
}

bool ProxyDistKvCache::get(const std::vector<int64_t>&        cache_keys,
                           const std::vector<int32_t>&        block_indices,
                           const kv_cache_manager::Locations& locations,
                           const kv_cache_manager::BlockMask& block_mask,
                           int64_t                            request_id,
                           std::map<std::string, std::string> extra_metas) const {
    auto layout_items = planner_->layout(cache_keys, block_indices, block_mask, extra_metas);
    if (layout_items.empty()) {
        RTP_LLM_LOG_WARNING("dist kv cache get cache, layout iovs is empty");
        return false;
    }

    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markGetCacheBeginUs(metrics);

    auto block_buffers = genKvcmBlockBuffers(layout_items);
    RTP_LLM_LOG_INFO("kvcm_get: request_id[%ld],key_size=%zu,location_size=%zu,first=%s,buffer_size=%zu",
                     request_id,
                     cache_keys.size(),
                     locations.size(),
                     locations[0].c_str(),
                     block_buffers.size());

    if (!kvcm_client_wrapper_->loadKvCaches(locations, block_buffers)) {
        DistKvCacheMetrics::markGetCacheDoneUs(metrics);
        return false;
    }

    DistKvCacheMetrics::markGetCacheDoneUs(metrics);

    if (!planner_->verify(layout_items, cache_keys, block_indices, extra_metas)) {
        RTP_LLM_LOG_WARNING("dist kv cache get cache, verify cache failed, request key: %s", request_id);
        return false;
    }

    return true;
}

bool ProxyDistKvCache::putAllRankImpl(const std::vector<int64_t>&               cache_keys,
                                      const std::vector<int32_t>&               block_indices,
                                      size_t                                    ignore_block_num,
                                      int64_t                                   request_id,
                                      const std::map<std::string, std::string>& extra_metas) const {
    std::string trace_id = "put_start_" + std::to_string(request_id);
    // TODO : remove timeout ?
    auto [success, write_location] =
        kvcm_client_wrapper_->getWriteLocation(genUniqueId(extra_metas), trace_id, cache_keys, {}, 1800, {});

    if (!success) {
        RTP_LLM_LOG_WARNING("kvcm getWriteLocation failed, [%s]", trace_id.c_str());
        return false;
    }

    if (write_location.locations_map.empty()) {
        RTP_LLM_LOG_WARNING("kvcm getWriteLocation empty, [%s]", trace_id.c_str());
        return true;
    }

    bool result = syncCallAllRank(cache_keys,
                                  block_indices,
                                  write_location.locations_map,
                                  write_location.block_mask,
                                  request_id,
                                  extra_metas,
                                  OpType::OP_PUT);

    // TODO : what should we do if syncCallAllRank failed?
    if (!result) {
        RTP_LLM_LOG_WARNING("kvcm request_id [%ld], syncCallAllRank failed", request_id);
    }

    // TODO : become async finishWrite?
    // TODO : succeed_block useless now
    bool   all_block_succeed = result;
    size_t succeed_block     = all_block_succeed ? write_location.locations_map.begin()->second.size() : 0;
    trace_id                 = "put_finish_" + std::to_string(request_id);
    result                   = kvcm_client_wrapper_->finishWrite(
        genUniqueId(extra_metas), trace_id, write_location.write_session_id, succeed_block, {});
    if (!result) {
        RTP_LLM_LOG_WARNING("kvcm finishWrite failed, [%s]", trace_id.c_str());
    }
    return result;
}

bool ProxyDistKvCache::put(const std::vector<int64_t>&        cache_keys,
                           const std::vector<int32_t>&        block_indices,
                           const kv_cache_manager::Locations& locations,
                           const kv_cache_manager::BlockMask& block_mask,
                           int64_t                            request_id,
                           std::map<std::string, std::string> extra_metas) const {
    auto layout_items = planner_->layout(cache_keys, block_indices, block_mask, extra_metas);
    if (layout_items.empty()) {
        RTP_LLM_LOG_WARNING("dist kv cache put cache, layout iovs is empty");
        return false;
    }

    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markPutCacheBeginUs(metrics);

    auto block_buffers = genKvcmBlockBuffers(layout_items);
    RTP_LLM_LOG_INFO("kvcm_put: request_id[%ld],key_size=%zu,location_size=%zu,buffer_size=%zu,first_location=%s\n",
                     request_id,
                     cache_keys.size(),
                     locations.size(),
                     block_buffers.size(),
                     locations[0].c_str());

    // TODO : saveKvCaches will return real location, should be returned to rank0 and kv_cache_manager
    if (!kvcm_client_wrapper_->saveKvCaches(locations, block_buffers).first) {
        RTP_LLM_LOG_WARNING("SaveKvCaches failed");
        DistKvCacheMetrics::markPutCacheDoneUs(metrics);
        return false;
    }

    DistKvCacheMetrics::markPutCacheDoneUs(metrics);
    return true;
}

int32_t ProxyDistKvCache::matchAllRankImpl(const std::vector<int64_t>&               cache_keys,
                                           LocationsMapPtr                           locations_map_ptr,
                                           size_t                                    ignore_block_num,
                                           int64_t                                   request_id,
                                           const std::map<std::string, std::string>& extra_metas,
                                           const std::shared_ptr<std::atomic<bool>>& stop) const {
    if ((stop && stop->load()) || cache_keys.empty()) {
        return 0;
    }

    RTP_LLM_LOG_DEBUG("cache_keys size: %zu", cache_keys.size());
    RTP_LLM_LOG_DEBUG("ignore_block_num: %zu", ignore_block_num);

    std::string trace_id   = "match_" + std::to_string(request_id);
    auto        query_type = kv_cache_manager::QueryType::QT_PREFIX_MATCH;
    auto [success, locations_map] =
        kvcm_client_wrapper_->match(genUniqueId(extra_metas), trace_id, query_type, cache_keys, ignore_block_num, {});
    if (!success) {
        RTP_LLM_LOG_ERROR("remote match failed");
        return 0;
    }
    *locations_map_ptr = std::move(locations_map);
    return ignore_block_num + (locations_map_ptr->empty() ? 0 : locations_map_ptr->begin()->second.size());
}

std::string ProxyDistKvCache::genUniqueId(const std::map<std::string, std::string>& extra_metas) const {
    const auto& adpater_iter = extra_metas.find("LORA_ADAPTER_NAME");
    if (adpater_iter == extra_metas.end()) {
        return "";
    }
    const auto& lora_adapter_name = adpater_iter->second;
    if (lora_adapter_name.empty()) {
        return "";
    }
    const auto& lora_info_map = cache_manager_->lora_info_map();
    if (const auto& iter = lora_info_map.find(lora_adapter_name); iter != lora_info_map.end()) {
        return lora_adapter_name + '_' + std::to_string(hashString(iter->second));
    }
    return "";
}

bool ProxyDistKvCache::fillDistKvCacheRequestPB(DistKvCacheRequestPB&                     request,
                                                const std::vector<int64_t>&               cache_keys,
                                                const std::vector<int32_t>&               block_indices,
                                                const kv_cache_manager::LocationsMap&     locations_map,
                                                const kv_cache_manager::BlockMask&        block_mask,
                                                int64_t                                   request_id,
                                                const std::map<std::string, std::string>& extra_metas,
                                                DistKvCache::OpType                       op_type,
                                                int                                       rank) const {
    DistKvCache::fillDistKvCacheRequestPB(
        request, cache_keys, block_indices, locations_map, block_mask, request_id, extra_metas, op_type, rank);
    request.set_ignore_block_num(0);
    auto       location_spec_name = genLocationSpecName(rank);
    const auto locations_iter     = locations_map.find(location_spec_name);
    if (locations_iter == locations_map.end()) {
        RTP_LLM_LOG_WARNING("not exist location spce name [%s]", location_spec_name.c_str());
        return false;
    }
    const auto& locations = locations_iter->second;
    for (const auto& location : locations) {
        request.add_kvcm_locations(location);
    }
    return true;
}

}  // namespace rtp_llm