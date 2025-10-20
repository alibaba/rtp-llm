#include "rtp_llm/cpp/cache/DistKvCache.h"

#include <future>
#include <thread>
#include <algorithm>

#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/cache/DistKvCacheMetrics.h"
#include "rtp_llm/cpp/cache/RemoteKvCachePlanner.h"

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

inline void kvcmBlockMaskToPB(const kv_cache_manager::BlockMask& block_mask, KVCMBlockMaskPB* proto) {
    proto->clear_info();
    if (std::holds_alternative<kv_cache_manager::BlockMaskVector>(block_mask)) {
        const auto& mask_vector = std::get<kv_cache_manager::BlockMaskVector>(block_mask);
        auto        bool_masks  = proto->mutable_bool_masks();
        bool_masks->clear_values();
        for (const auto& value : mask_vector) {
            bool_masks->add_values(value);
        }
    } else if (std::holds_alternative<kv_cache_manager::BlockMaskOffset>(block_mask)) {
        const auto& mask_offset = std::get<kv_cache_manager::BlockMaskOffset>(block_mask);
        proto->set_offset(static_cast<int64_t>(mask_offset));
    }
}

}  // namespace

DistKvCache::DistKvCache(CacheManager*                       cache_manager,
                         const GptInitParameter&             gpt_init_params,
                         const kmonitor::MetricsReporterPtr& metrics_reporter):
    cache_manager_(cache_manager), gpt_init_params_(gpt_init_params), metrics_reporter_(metrics_reporter) {}

DistKvCache::~DistKvCache() {
    RTP_LLM_LOG_INFO("DistKvCache destructor");
    if (io_thread_pool_) {
        io_thread_pool_->stop();
        io_thread_pool_->waitFinish();
        io_thread_pool_.reset();
    }
    if (wait_match_thread_pool_) {
        wait_match_thread_pool_->stop();
        wait_match_thread_pool_->waitFinish();
        wait_match_thread_pool_.reset();
    }
    rpc_pool_.reset();
    storage_.reset();
    planner_.reset();
    metrics_reporter_.reset();
    cache_manager_ = nullptr;
}

bool DistKvCache::init(const DistKvCacheInitParams& init_params, bool is_legacy) {
    RTP_LLM_LOG_INFO("dist kvcache init params: [%s], legacy [%d]", init_params.toString().c_str(), is_legacy);
    is_legacy_                       = is_legacy;
    init_params_                     = init_params;
    auto& storage_params             = init_params_.storage_manager_params;
    storage_params.lookup_timeout_ms = init_params.match_timeout_ms;
    storage_params.get_timeout_ms    = init_params.rpc_get_cache_timeout_ms;
    storage_params.put_timeout_ms    = init_params.rpc_put_cache_timeout_ms;

    if (is_legacy_) {
        const auto& init_params_3fs = storage_params.init_params_3fs.value();
        planner_                    = std::make_unique<DefaultDistKvCachePlanner>(
            cache_manager_, gpt_init_params_, init_params_3fs, metrics_reporter_);
        storage_ = std::make_unique<DistStorageManager>(metrics_reporter_);
        if (!storage_->init(storage_params)) {
            RTP_LLM_LOG_WARNING("init failed, init storage failed");
            return false;
        }

        if (!initDefaultMetas()) {
            RTP_LLM_LOG_WARNING("init failed, init default metas failed");
            return false;
        }
    } else {
        planner_ = std::make_unique<RemoteKvCachePlanner>(cache_manager_, gpt_init_params_, metrics_reporter_);
        if (!initRemoteKvCacheClient()) {
            RTP_LLM_LOG_WARNING("init failed, init remote kv cache client failed");
            return false;
        }
    }

    // TODO KVCM client需要简化pool的使用
    wait_match_thread_pool_ =
        std::make_unique<autil::LockFreeThreadPool>(thread_num_, queue_size_, nullptr, "WaitMatchThreadPool");
    if (!wait_match_thread_pool_->start()) {
        RTP_LLM_LOG_WARNING("init failed, start wait match thread pool failed, thread num: %zu, queue size: %zu",
                            thread_num_,
                            queue_size_);
        return false;
    }

    // parallel IO pool for 3FS get/put of multiple items
    io_thread_pool_ =
        std::make_unique<autil::LockFreeThreadPool>(io_thread_num_, io_queue_size_, nullptr, "DistKvCacheIOThreadPool");
    if (!io_thread_pool_->start()) {
        RTP_LLM_LOG_WARNING("init failed, start io thread pool failed, thread num: %zu, queue size: %zu",
                            io_thread_num_,
                            io_queue_size_);
        return false;
    }

    rpc_pool_ = std::make_shared<rtp_llm::RPCPool>();
    RTP_LLM_LOG_INFO("dist kv cache init success");
    return true;
}

bool DistKvCache::initDefaultMetas() {
    const auto& cache_config             = cache_manager_->cacheConfig();
    default_metas_["SEQ_SIZE_PER_BLOCK"] = std::to_string(cache_config.seq_size_per_block);
    default_metas_["DTYPE"]              = std::to_string(static_cast<int>(cache_config.dtype));
    default_metas_["USE_MLA"]            = std::to_string(static_cast<int>(cache_config.use_mla));
    default_metas_["TP_SIZE"]            = std::to_string(gpt_init_params_.tp_size_);
    default_metas_["TP_RANK"]            = std::to_string(gpt_init_params_.tp_rank_);
    default_metas_["LAYOUT_VERSION"]     = "v2";

    auto biz_name = autil::EnvUtil::getEnv("BIZ_NAME", std::string(""));
    if (biz_name.empty()) {
        RTP_LLM_LOG_WARNING("init metas failed, biz name is empty");
        return false;
    }
    default_metas_["BIZ_NAME"] = biz_name;

    const auto ckpt_path = autil::EnvUtil::getEnv("CHECKPOINT_PATH", std::string(""));
    if (ckpt_path.empty()) {
        RTP_LLM_LOG_WARNING("init metas failed, ckpt path is empty");
        return false;
    }
    default_metas_["CKPT_PATH"]      = std::to_string(hashString(ckpt_path));
    default_metas_["LORA_CKPT_PATH"] = "no_lora";

    std::ostringstream oss;
    for (const auto& [key, value] : default_metas_) {
        oss << key << ":" << value << ", ";
    }
    RTP_LLM_LOG_INFO("dist kv cache default metas: [%s]", oss.str().c_str());

    return true;
}

std::map<std::string, std::string> DistKvCache::genKVCMClientConfig() const {
    static std::string config_format = R"(
{
"enable_vipserver" : %s,
"vipserver_domain" : "%s",
"instance_group": "%s",
"instance_id": "%lu",
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
    auto        lora_info_map       = cache_manager_->lora_info_map();
    std::string self_location_spec_name;
    auto        location_spec_info_map = KVCMClientWrapperConfig::LocationSpecInfoMap{};
    for (size_t i = 0; i < cache_manager_->device()->getDeviceProperties().tp_size; ++i) {
        location_spec_info_map.emplace(genLocationSpecName(i), byte_size_per_block);
    }
    auto location_spec_infos_str = autil::legacy::ToJsonString(location_spec_info_map, true);
    lora_info_map[""]            = "";  // default : no lora
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
        std::string instance_id_hash_str = instance_id_hash_ss.str();
        auto        instance_id          = hashString(instance_id_hash_str);
        RTP_LLM_LOG_INFO("lora_adapter_name[%s], instance_id_hash_str[%s], instance_id[%lu]",
                         lora_adapter_name.c_str(),
                         instance_id_hash_str.c_str(),
                         instance_id);
        int n = std::snprintf(buffer.data(),
                              buffer.size(),
                              config_format.c_str(),
                              enable_vip_server ? "true" : "false",
                              vipserver_domain.c_str(),
                              instance_group.c_str(),
                              instance_id,
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

bool DistKvCache::initRemoteKvCacheClient() {
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
    kvcm_client_wrapper_ = std::make_unique<KVCMClientWrapper>();
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

int32_t DistKvCache::matchForAllRank(const std::vector<int64_t>&        cache_keys,
                                     size_t                             ignore_block_num,
                                     int64_t                            request_id,
                                     std::map<std::string, std::string> extra_metas,
                                     kv_cache_manager::LocationsMap&    locations_map) {
    if (cache_keys.empty() || !planner_ || (is_legacy_ && !storage_) || (!is_legacy_ && !kvcm_client_wrapper_)) {
        RTP_LLM_LOG_WARNING("invalid state, ignore match request, %p, %p", planner_.get(), kvcm_client_wrapper_.get());
        return 0;
    }

    if (wait_match_thread_pool_->isFull()) {
        RTP_LLM_LOG_WARNING("match for all rank failed, wait match thread pool is full, something maybe wrong");
        return 0;
    }

    int32_t       match_len    = 0;
    const auto&   cache_config = cache_manager_->cacheConfig();
    const int32_t input_len    = static_cast<int32_t>(cache_keys.size());

    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markMatchBeginUs(metrics);

    auto stop = std::make_shared<std::atomic<bool>>(false);
    auto task =
        [weak_this = weak_from_this(), &cache_keys, ignore_block_num, request_id, &extra_metas, stop, &locations_map]()
        -> int32_t {
        if (weak_this.expired()) {
            RTP_LLM_LOG_WARNING("match for all rank failed, dist kv cache has been expired, request_id: %ld",
                                request_id);
            return 0;
        }
        auto shared_this = weak_this.lock();
        if (!(shared_this->is_legacy_)) {
            return shared_this->match(cache_keys, locations_map, ignore_block_num, request_id, extra_metas, stop);
        }
        int32_t match_len = static_cast<int32_t>(cache_keys.size());
        auto    metas     = extra_metas;
        for (int i = 0; i < shared_this->gpt_init_params_.tp_size_; i++) {
            metas["TP_RANK"] = std::to_string(i);
            auto ret         = shared_this->match(cache_keys, locations_map, ignore_block_num, request_id, metas, stop);
            if (ret < match_len) {
                match_len = ret;
            }
            if (match_len == 0) {
                break;
            }
        }
        return match_len;
    };

    auto future = wait_match_thread_pool_->async(task);
    if (future.wait_for(std::chrono::milliseconds(init_params_.match_timeout_ms)) == std::future_status::ready) {
        match_len = future.get();
    } else {
        RTP_LLM_LOG_WARNING(
            "match for all rank timeout, timeout: %d ms, request: %ld, input block num: %zu, ignore block num: %zu",
            init_params_.match_timeout_ms,
            request_id,
            cache_keys.size(),
            ignore_block_num);
        if (stop) {
            stop->store(true);
        }
    }

    DistKvCacheMetrics::markMatchDoneUs(metrics);
    DistKvCacheMetrics::setMatchQps(metrics, true);
    DistKvCacheMetrics::setMatchFailedQps(metrics, match_len <= 0);
    DistKvCacheMetrics::setCacheInputLength(metrics, input_len * cache_config.seq_size_per_block);
    if (match_len > 0) {
        total_match_len_.fetch_add(match_len);
        int64_t total_match_len = total_match_len_.load();
        total_input_len_.fetch_add(input_len);
        int64_t total_input_len = total_input_len_.load();

        DistKvCacheMetrics::setCacheMatchLength(metrics, match_len * cache_config.seq_size_per_block);
        DistKvCacheMetrics::setCacheHitRate(metrics, match_len * 100.0 / input_len);
        DistKvCacheMetrics::setTotalCacheMatchLength(metrics, total_match_len * cache_config.seq_size_per_block);
        DistKvCacheMetrics::setTotalCacheInputLength(metrics, total_input_len * cache_config.seq_size_per_block);
        DistKvCacheMetrics::setTotalCacheHitRate(metrics, total_match_len * 100.0 / total_input_len);
    }
    return match_len;
}

int32_t DistKvCache::match(const std::vector<int64_t>&               cache_keys,
                           kv_cache_manager::LocationsMap&           locations_map,
                           size_t                                    ignore_block_num,
                           int64_t                                   request_id,
                           std::map<std::string, std::string>        extra_metas,
                           const std::shared_ptr<std::atomic<bool>>& stop) const {
    if ((stop && stop->load()) || cache_keys.empty()) {
        return 0;
    }

    RTP_LLM_LOG_DEBUG("cache_keys size: %zu", cache_keys.size());
    RTP_LLM_LOG_DEBUG("ignore_block_num: %zu", ignore_block_num);

    if (!is_legacy_) {
        std::string trace_id   = "match_" + std::to_string(request_id);
        auto        query_type = kv_cache_manager::QueryType::QT_PREFIX_MATCH;
        // TODO 这接口后面几个参数不需要，要改一下，block_mask要放前面去。
        auto [success, result] = kvcm_client_wrapper_->match(
            genUniqueId(extra_metas), trace_id, query_type, cache_keys, ignore_block_num, {});
        if (!success) {
            RTP_LLM_LOG_ERROR("remote match failed");
            return 0;
        }
        locations_map = std::move(result);
        return ignore_block_num + (locations_map.empty() ? 0 : locations_map.begin()->second.size());
    }

    for (auto& [key, value] : default_metas_) {
        if (extra_metas.count(key) == 0) {
            extra_metas[key] = value;
        }
    }

    RTP_LLM_LOG_DEBUG("extra_metas: %s", extra_metas.at("TP_RANK").c_str());

    // get layout at the granularity of items i.e. blocks collection
    auto layout_items = planner_->layout(cache_keys, {}, ignore_block_num, extra_metas);

    if (layout_items.empty()) {
        return 0;
    }

    if (storage_->lookup(layout_items.back())) {
        RTP_LLM_LOG_DEBUG("request[%ld] match all", request_id);
        return cache_keys.size();
    }

    int32_t left      = 0;
    int32_t right     = static_cast<int32_t>(layout_items.size()) - 2;
    int32_t mid       = 0;
    int32_t match_len = 0;

    while (left <= right) {
        if (stop && stop->load()) {
            return 0;
        }
        mid = left + (right - left) / 2;
        if (storage_->lookup(layout_items[mid])) {
            RTP_LLM_LOG_DEBUG("request[%ld] match index: %d", request_id, mid);
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    match_len = (right + 1) * init_params_.max_block_size_per_item;
    DistStorage::Item item;
    // deep search partial block
    item       = DistStorage::Item();
    item.type  = DistStorage::ST_3FS;
    item.metas = extra_metas;

    int start_key = (right + 1) * init_params_.max_block_size_per_item;
    int end_key   = start_key + init_params_.max_block_size_per_item - 1;
    end_key       = std::min(end_key, static_cast<int>(cache_keys.size()) - 1);
    for (int i = end_key; i >= start_key; i--) {
        item.key               = std::to_string(cache_keys[start_key]) + "_" + std::to_string(cache_keys[i]);
        item.metas["ITEM_KEY"] = item.key;
        if (storage_->lookup(item)) {
            RTP_LLM_LOG_DEBUG(
                "request[%ld] deep match item: %s, len: %d", request_id, item.key.c_str(), i - start_key + 1);
            match_len += (i - start_key + 1);
            break;
        }
    }

    return match_len;
}

bool DistKvCache::getForAllRank(const std::vector<int64_t>&           cache_keys,
                                const std::vector<int32_t>&           block_indices,
                                const kv_cache_manager::LocationsMap& locations_map,
                                size_t                                ignore_block_num,
                                int64_t                               request_id,
                                std::map<std::string, std::string>    extra_metas) const {
    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);

    const auto input_len        = static_cast<int32_t>(cache_keys.size());
    const auto remote_match_len = input_len - ignore_block_num;
    if (remote_match_len != block_indices.size()) {
        DistKvCacheMetrics::setGetCacheFailedQps(metrics, true);
        RTP_LLM_LOG_WARNING(
            "get cache for all rank failed, cache key size %d not equal to block size %zu, request id: %lld",
            remote_match_len,
            block_indices.size(),
            request_id);
        return false;
    }

    DistKvCacheMetrics::markTotalGetCacheBeginUs(metrics);

    bool result = syncCallAllRank(
        cache_keys, block_indices, locations_map, ignore_block_num, request_id, extra_metas, OpType::OP_GET);

    DistKvCacheMetrics::markTotalGetCacheDoneUs(metrics);
    DistKvCacheMetrics::setGetCacheFailedQps(metrics, result == false);
    if (result) {
        const auto& cache_config = cache_manager_->cacheConfig();
        DistKvCacheMetrics::setCacheGetLength(metrics, remote_match_len * cache_config.seq_size_per_block);
    }

    return result;
}

bool DistKvCache::get(const std::vector<int64_t>&        cache_keys,
                      const std::vector<int32_t>&        block_indices,
                      const kv_cache_manager::Locations& locations,
                      const kv_cache_manager::BlockMask& block_mask,
                      int64_t                            request_id,
                      std::map<std::string, std::string> extra_metas) const {
    if (is_legacy_) {
        for (auto& [key, value] : default_metas_) {
            if (extra_metas.count(key) == 0) {
                extra_metas[key] = value;
            }
        }
    }

    auto layout_items = planner_->layout(cache_keys, block_indices, block_mask, extra_metas);
    if (layout_items.empty()) {
        RTP_LLM_LOG_WARNING("dist kv cache get cache, layout iovs is empty");
        return false;
    }

    for (auto& item : layout_items) {
        RTP_LLM_LOG_DEBUG("layout item: %s", item.key.c_str());
    }

    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markGetCacheBeginUs(metrics);

    std::vector<autil::ThreadPoolBase::Future<bool>> get_futures;
    if (is_legacy_) {
        get_futures.reserve(layout_items.size());
        for (auto& it : layout_items) {
            auto get_task = [this, it]() mutable -> bool { return storage_->get(it); };
            get_futures.emplace_back(io_thread_pool_->async(get_task));
        }
    } else {
        // TODO 先在这里转换为client认识的内存地址
        kv_cache_manager::BlockBuffers block_buffers;
        for (auto& item : layout_items) {
            kv_cache_manager::BlockBuffer block_buffer;
            for (auto item_iov : item.iovs) {
                kv_cache_manager::Iov iov = {
                    kv_cache_manager::MemoryType::GPU, item_iov.data.get(), item_iov.len, item_iov.ignore};
                block_buffer.iovs.push_back(iov);
            }
            block_buffers.push_back(std::move(block_buffer));
        }
        RTP_LLM_LOG_INFO("kvcm_match: key_size=%zu,match_len=%zu,first=%s,buffer_size=%zu",
                         cache_keys.size(),
                         locations.size(),
                         locations[0].c_str(),
                         block_buffers.size());

        get_futures.reserve(1);
        auto get_task = [this, &locations, x = std::move(block_buffers)]() mutable -> bool {
            return kvcm_client_wrapper_->loadKvCaches(locations, x);
        };
        get_futures.emplace_back(io_thread_pool_->async(get_task));
    }

    bool any_failed = false;
    for (auto& future : get_futures) {
        if (future.valid() && !future.get()) {
            any_failed = true;
            break;
        }
    }
    if (any_failed) {
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

bool DistKvCache::putForAllRank(const std::vector<int64_t>&        cache_keys,
                                const std::vector<int32_t>&        block_indices,
                                size_t                             ignore_block_num,
                                int64_t                            request_id,
                                std::map<std::string, std::string> extra_metas) const {
    std::map<std::string, std::string> rpc_extra_metas;
    if (is_legacy_) {
        rpc_extra_metas = extra_metas;  // 拷贝一份用于rpc传输, 否则rpc可能会传比较多冗余信息
        for (auto& [key, value] : default_metas_) {
            if (extra_metas.count(key) == 0) {
                extra_metas[key] = value;
            }
        }
    }
    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markTotalPutCacheBeginUs(metrics);
    kv_cache_manager::WriteLocation write_location;
    if (!is_legacy_) {
        std::string trace_id = "put_start_" + std::to_string(request_id);
        // TODO timeout不需要传递了，每个instance一个。
        // 有个session id还挺烦的，能自己指定一个吗？这样就可以用这次所有key的hash + requestid？
        auto [success, res] =
            kvcm_client_wrapper_->getWriteLocation(genUniqueId(extra_metas), trace_id, cache_keys, {}, 1800, {});
        if (!success) {
            RTP_LLM_LOG_WARNING("kvcm getWriteLocation failed, [%s]", trace_id.c_str());
            return false;
        }
        if (res.locations_map.empty()) {
            RTP_LLM_LOG_WARNING("kvcm getWriteLocation empty, [%s]", trace_id.c_str());
            return true;
        }
        write_location = std::move(res);
    }
    bool result = syncCallAllRank(cache_keys,
                                  block_indices,
                                  write_location.locations_map,
                                  is_legacy_ ? ignore_block_num : write_location.block_mask,
                                  request_id,
                                  is_legacy_ ? rpc_extra_metas : extra_metas,
                                  OpType::OP_PUT);

    if (!is_legacy_) {
        bool        all_block_succeed = result;
        size_t      succeed_block     = all_block_succeed ? write_location.locations_map.begin()->second.size() : 0;
        std::string trace_id          = "put_finish_" + std::to_string(request_id);
        // TODO write_location里面有个session_id貌似不太合理
        result = kvcm_client_wrapper_->finishWrite(
            genUniqueId(extra_metas), trace_id, write_location.write_session_id, succeed_block, {});
        if (!result) {
            RTP_LLM_LOG_WARNING("kvcm finishWrite failed, [%s]", trace_id.c_str());
        }
    }

    const auto& cache_config = cache_manager_->cacheConfig();
    DistKvCacheMetrics::markTotalPutCacheDoneUs(metrics);
    DistKvCacheMetrics::setPutCacheFailedQps(metrics, result == false);
    DistKvCacheMetrics::setCachePutLength(metrics, cache_keys.size() * cache_config.seq_size_per_block);

    return result;
}

bool DistKvCache::put(const std::vector<int64_t>&        cache_keys,
                      const std::vector<int32_t>&        block_indices,
                      const kv_cache_manager::Locations& locations,
                      const kv_cache_manager::BlockMask& block_mask,
                      int64_t                            request_id,
                      std::map<std::string, std::string> extra_metas) const {
    if (is_legacy_) {
        for (auto& [key, value] : default_metas_) {
            if (extra_metas.count(key) == 0) {
                extra_metas[key] = value;
            }
        }
    }

    auto layout_items = planner_->layout(cache_keys, block_indices, block_mask, extra_metas);
    if (layout_items.empty()) {
        RTP_LLM_LOG_WARNING("dist kv cache put cache, layout iovs is empty");
        return false;
    }

    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markPutCacheBeginUs(metrics);

    std::vector<autil::ThreadPoolBase::Future<bool>> put_futures;
    if (is_legacy_) {
        put_futures.reserve(layout_items.size());
        for (auto& it : layout_items) {
            auto put_task = [this, it]() mutable -> bool { return storage_->putIfNotExist(it); };
            put_futures.emplace_back(io_thread_pool_->async(put_task));
        }
    } else {
        // TODO 先在这里转换为client认识的内存地址
        kv_cache_manager::BlockBuffers block_buffers;
        for (auto& item : layout_items) {
            kv_cache_manager::BlockBuffer block_buffer;
            for (const auto& item_iov : item.iovs) {
                kv_cache_manager::Iov iov = {
                    kv_cache_manager::MemoryType::GPU, item_iov.data.get(), item_iov.len, item_iov.ignore};
                block_buffer.iovs.push_back(iov);
            }
            block_buffers.push_back(std::move(block_buffer));
        }

        RTP_LLM_LOG_INFO("kvcm_get_write_location:keys_size=%zu,location_size=%zu,buffer_size=%zu,first_location=%s\n",
                         cache_keys.size(),
                         locations.size(),
                         block_buffers.size(),
                         locations[0].c_str());
        put_futures.reserve(1);
        auto put_task = [this, &locations, x = std::move(block_buffers)]() mutable -> bool {
            // TODO(qisa.cb)
            return kvcm_client_wrapper_->saveKvCaches(locations, x).first;
        };
        put_futures.emplace_back(io_thread_pool_->async(put_task));
    }

    bool any_failed = false;
    for (auto& future : put_futures) {
        if (future.valid() && !future.get()) {
            any_failed = true;
            break;
        }
    }
    if (any_failed) {
        RTP_LLM_LOG_WARNING("SaveKvCaches failed");
        DistKvCacheMetrics::markPutCacheDoneUs(metrics);
        return false;
    }

    DistKvCacheMetrics::markPutCacheDoneUs(metrics);
    return true;
}

struct WorkerRpcContext {
    WorkerRpcContext() {
        client_context = std::make_shared<grpc::ClientContext>();
    }
    grpc::Status                         status;
    std::shared_ptr<RpcService::Stub>    stub;
    std::shared_ptr<grpc::ClientContext> client_context;
    std::string                          server_addr;
    DistKvCacheRequestPB                 request;
    DistKvCacheResponsePB                response;
    grpc::CompletionQueue                completion_queue;
};

// TODO: sync call all rank 的逻辑用到的地方比较多, 抽一下
bool DistKvCache::syncCallAllRank(const std::vector<int64_t>&               cache_keys,
                                  const std::vector<int32_t>&               block_indices,
                                  const kv_cache_manager::LocationsMap&     locations_map,
                                  const kv_cache_manager::BlockMask&        block_mask,
                                  int64_t                                   request_id,
                                  const std::map<std::string, std::string>& extra_metas,
                                  DistKvCache::OpType                       op_type) const {
    const auto& grpc_workers = gpt_init_params_.worker_grpc_addrs_;
    if (grpc_workers.empty()) {
        RTP_LLM_LOG_WARNING("rpc get cache failed, grpc workers is empty, request: %ld", request_id);
        return false;
    }
    if (!rpc_pool_) {
        RTP_LLM_LOG_WARNING("rpc get cache failed, rpc pool is nullptr, request: %ld", request_id);
        return false;
    }

    const int                     worker_size = static_cast<int>(grpc_workers.size());
    std::vector<WorkerRpcContext> worker_rpc_contexts(worker_size);

    for (int rank = 0; rank < worker_size; ++rank) {
        RTP_LLM_LOG_DEBUG("send get cache rpc request to rank: %d, addr: %s", rank, grpc_workers[rank].c_str());
        const auto& worker_addr    = grpc_workers[rank];
        auto        connect_status = rpc_pool_->getConnection(worker_addr);
        if (!connect_status.ok()) {
            RTP_LLM_LOG_WARNING("rpc get cache failed, get grpc connection failed for rank: %d, request: %ld, addr: %s",
                                rank,
                                request_id,
                                worker_addr.c_str());
            return false;
        }
        auto& rpc_context = worker_rpc_contexts[rank];
        auto& request     = rpc_context.request;

        request.set_request_id(request_id);
        for (const auto cache_key : cache_keys) {
            request.add_cache_keys(cache_key);
        }
        for (const auto block_index : block_indices) {
            request.add_block_ids(block_index);
        }
        request.set_ignore_block_num(is_legacy_ ? std::get<size_t>(block_mask) : 0);  // TODO : deprecated
        request.set_op(op_type == OpType::OP_GET ? DistKvCacheOp::GET : DistKvCacheOp::PUT);
        for (const auto& extra_meta : extra_metas) {
            auto* meta = request.add_extra_metas();
            meta->set_key(extra_meta.first);
            meta->set_value(extra_meta.second);
        }
        if (!is_legacy_) {
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
        }
        kvcmBlockMaskToPB(block_mask, request.mutable_kvcm_block_mask());

        rpc_context.stub        = connect_status.value().stub;
        rpc_context.server_addr = worker_addr;

        std::unique_ptr<::grpc::ClientAsyncResponseReader<DistKvCacheResponsePB>> reader(
            rpc_context.stub->AsyncDistKvCache(
                rpc_context.client_context.get(), request, &(rpc_context.completion_queue)));
        reader->Finish(&(rpc_context.response), &rpc_context.status, reinterpret_cast<void*>(rank));
    }

    std::vector<int> success_ranks(worker_size, 0);
    int              finished_count      = 0;
    bool             all_request_success = true;
    const int        timeout_ms =
        op_type == OpType::OP_GET ? init_params_.rpc_get_cache_timeout_ms : init_params_.rpc_put_cache_timeout_ms;
    const auto start_time = std::chrono::steady_clock::now();

    while (true) {
        if (finished_count == worker_size) {
            break;
        }

        if (std::chrono::steady_clock::now() - start_time >= std::chrono::milliseconds(timeout_ms)) {
            std::string timeout_ranks_str;
            for (int rank = 0; rank < worker_size; ++rank) {
                if (success_ranks[rank] == 0) {
                    timeout_ranks_str += std::to_string(rank) + ",";
                }
            }
            RTP_LLM_LOG_WARNING("request timeout: %d ms, timeout rank: [%s], request: %ld",
                                timeout_ms,
                                timeout_ranks_str.c_str(),
                                request_id);
            all_request_success = false;
            break;
        }

        const int  once_timeout_ms = 10;
        const auto once_deadline   = std::chrono::system_clock::now() + std::chrono::milliseconds(once_timeout_ms);
        for (uint32_t i = 0; i < worker_rpc_contexts.size(); i++) {
            if (success_ranks[i] == 1) {
                continue;
            }
            auto& rpc_context = worker_rpc_contexts.at(i);

            void*      got_tag     = nullptr;
            bool       ok          = false;
            const auto next_status = rpc_context.completion_queue.AsyncNext(&got_tag, &ok, once_deadline);
            if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
                continue;
            }
            if (!ok) {
                RTP_LLM_LOG_WARNING(
                    "request failed, grpc completion queue failed, status: %d, request: %ld", next_status, request_id);
                all_request_success = false;
                break;
            }
            ++finished_count;

            const int   rank   = reinterpret_cast<intptr_t>(got_tag);
            const auto& status = rpc_context.status;
            if (!status.ok()) {
                RTP_LLM_LOG_WARNING("request failed, rank %d failed, error: %d(%s), addr: %s, request: %ld",
                                    rank,
                                    status.error_code(),
                                    status.error_message().c_str(),
                                    rpc_context.server_addr.c_str(),
                                    request_id);
                all_request_success = false;
                break;
            }
            success_ranks[rank] = 1;
            RTP_LLM_LOG_DEBUG("rank %d request success, request: %ld", rank, request_id);
        }
        if (!all_request_success) {
            break;
        }
    }
    return all_request_success;
}

std::string DistKvCache::genUniqueId(const std::map<std::string, std::string>& extra_metas) const {
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

}  // namespace rtp_llm
