#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <unistd.h>
#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/cache/DistKvCache.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

using namespace std;

namespace rtp_llm {

CacheManager::CacheManager(const CacheConfig&                 config,
                           rtp_llm::DeviceBase*               device,
                           bool                               warmup,
                           const kmonitor::MetricsReporterPtr metrics_reporter,
                           const GptInitParameter&            params):
    config_(config),
    seq_size_per_block_(config.seq_size_per_block),
    block_cache_(config.seq_size_per_block),
    device_(device),
    metrics_reporter_(metrics_reporter),
    params_(params) {

    if (warmup) {
        config_.block_nums = 1;
    } else {
        allocateAndSync();
    }
    config_.refresh();
    RTP_LLM_LOG_INFO("cache config: %s", config.debugString().c_str());

    allocator_ = std::make_shared<KVCacheAllocator>(config_, device_, AllocationType::DEVICE);
    if (!allocator_->init()) {
        RTP_LLM_FAIL("kvcache allocator init failed");
    }
    available_blocks_ = allocator_->totalBlocks();

    if (params_.kv_cache_config.enable_3fs) {
        enable_dist_kvcache_ = initDistKvCache();
        if (!enable_dist_kvcache_) {
            RTP_LLM_FAIL("dist kv cache init failed");
        }
    }
}

void CacheManager::regUserMr(size_t model_id) {
    allocator_->regUserMr(model_id);
}

CacheManager::~CacheManager() {
    dist_kvcache_.reset();

    stop_ = true;
    if (metrics_reporter_thread_.joinable()) {
        metrics_reporter_thread_.join();
    }

    allocator_.reset();
}

uint32_t CacheManager::totalBlocks() const {
    return allocator_->totalBlocks();
}

uint32_t CacheManager::maxSeqLen() const {
    return totalBlocks() * seq_size_per_block_;
}

void CacheManager::reportMetricsLoop() {
    while (!stop_) {
        if (metrics_reporter_) {
            RtpLLMCacheMetricsCollector collector;
            {
                std::lock_guard<std::mutex> guard(mutex_);
                collector.kv_cache_item_num         = block_cache_.size();
                auto available_blocks               = availableBlockNums();
                collector.kv_cache_left_seq         = available_blocks * seq_size_per_block_;
                collector.kv_cache_available_blocks = available_blocks;
                collector.kv_cache_free_blocks      = freeBlockNums();
                collector.kv_cache_used_ratio       = 100.0 * (totalBlocks() - available_blocks) / totalBlocks();
                collector.mr_cost_time_ms           = allocator_->getMrCostTimeMs();
            }
            metrics_reporter_->report<RtpLLMCacheMetrics, RtpLLMCacheMetricsCollector>(nullptr, &collector);
            std::this_thread::sleep_for(std::chrono::seconds(1));  // 1s
        }
    }
}

void CacheManager::allocateAndSync() {
    const auto properties = device_->getDeviceProperties();
    size_t     world_size = properties.tp_size * properties.dp_size;
    if (world_size > 1) {
        size_t    local_rank = properties.tp_size * properties.dp_rank + properties.tp_rank;
        BufferPtr block_num_infos =
            device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {world_size}, rtp_llm::AllocationType::HOST});
        auto block_num_ptr        = block_num_infos->data<int>();
        block_num_ptr[local_rank] = config_.block_nums;
        device_->allGather({{block_num_infos}, ParallelMode::DP_AND_TP});
        device_->syncCommunication(false);
        device_->syncAndCheck();
        if (properties.ffn_as_service) {
            config_.block_nums = 1;
        } else {
            config_.block_nums = *std::min_element(block_num_ptr, block_num_ptr + world_size);
        }
    }
    RTP_LLM_LOG_INFO("block nums is %d after tp sync", config_.block_nums);
}

const CacheConfig& CacheManager::cacheConfig() const {
    return config_;
}

const BlockCache& CacheManager::blockCache() const {
    return block_cache_;
}

size_t CacheManager::freeBlockNums() const {
    return allocator_->freeBlockNums();
}

size_t CacheManager::availableBlockNums() const {
    return available_blocks_;
}

KVCacheInfo CacheManager::getKVCacheInfo(int64_t latest_version, bool need_cache_keys) const {
    auto                 snapshot = block_cache_.cacheSnapshot(latest_version);
    std::vector<int64_t> cachekeys;
    if (need_cache_keys) {
        std::unordered_set<int64_t> seen_keys;
        for (const auto& cacheItem : snapshot.values) {
            for (auto& key_part : cacheItem.cache_key) {
                if (seen_keys.insert(key_part).second) {
                    cachekeys.push_back(key_part);
                }
            }
        }
    }
    KVCacheInfo info{(size_t)availableBlockNums() * seq_size_per_block_,
                     (size_t)totalBlocks() * seq_size_per_block_,
                     (size_t)seq_size_per_block_,
                     std::move(cachekeys),
                     snapshot.version};
    return info;
}

size_t CacheManager::cacheItemNum() const {
    return block_cache_.size();
}

const KVCacheAllocator::KVCacheBuffer& CacheManager::kvCacheBuffer() const {
    return allocator_->kvCacheBuffer();
}

CacheManager::MatchInfo CacheManager::mallocWithCache(const AdvancedMallocInfo& malloc_info) {
    if (malloc_info.token_ids.size() < config_.seq_size_per_block + 1) {
        return MatchInfo{};
    }

    std::lock_guard<std::mutex> guard(mutex_);
    auto                        match_begin_time_us = currentTimeUs();
    auto                        match_info          = matchImpl(malloc_info);
    auto                        match_cost_time_us  = currentTimeUs() - match_begin_time_us;
    if (match_info.loss.empty() && malloc_info.need_loss) {
        freeWithoutLock(match_info.cache_blocks);
        return {0, {}, {}};
    }
    if (metrics_reporter_) {
        RtpLLMCacheReuseMetricsCollector collector;
        collector.kv_cache_reuse_length = match_info.reuse_length;
        collector.match_cost_time_us    = match_cost_time_us;
        collector.gpu_input_length   = static_cast<int32_t>(malloc_info.cache_keys.size()) * config_.seq_size_per_block;
        collector.gpu_reuse_length   = match_info.local_reuse_length;
        collector.gpu_cache_hit_rate = collector.gpu_reuse_length * 100 / collector.gpu_input_length;
        metrics_reporter_->report<RtpLLMCacheReuseMetrics, RtpLLMCacheReuseMetricsCollector>(nullptr, &collector);
    }
    return match_info;
}

// This function must not hold mutex_
CacheManager::MatchInfo CacheManager::matchImpl(const AdvancedMallocInfo& malloc_info) {
    // match in gpu
    auto match_result = block_cache_.match(malloc_info.cache_keys);
    incrRefCounter(match_result.block_indices);
    auto local_match_blocks = match_result.block_indices.size();

    // match in dist kvcache if cache keys not fully matched
    if (enable_dist_kvcache_ && malloc_info.enable_3fs && !malloc_info.need_loss
        && local_match_blocks < malloc_info.cache_keys.size()) {
        matchInDistKvCache(malloc_info, match_result);
    }

    int cache_block_num = match_result.block_indices.size();
    int reuse_block_num =
        std::min(cache_block_num, static_cast<int>((malloc_info.token_ids.size() - 1) / config_.seq_size_per_block));
    // common length must large than reuse_length, when need calculate loss
    if ((!match_result.loss.empty()) && reuse_block_num) {
        reuse_block_num -= 1;
    }
    int reuse_length = reuse_block_num * config_.seq_size_per_block;
    for (int i = malloc_info.mm_bounds.size() - 1; i >= 0; --i) {
        auto& bound = malloc_info.mm_bounds[i];
        if (reuse_length > bound[0] && reuse_length < bound[0] + bound[1]) {
            reuse_length = bound[0] / config_.seq_size_per_block * config_.seq_size_per_block;
        }
    }
    reuse_block_num = reuse_length / config_.seq_size_per_block;
    if (reuse_block_num < cache_block_num) {
        std::vector<int> need_decref_blocks(match_result.block_indices.begin() + reuse_block_num,
                                            match_result.block_indices.end());
        freeWithoutLock(need_decref_blocks);
    }

    RTP_LLM_CHECK_WITH_INFO((reuse_block_num <= cache_block_num),
                            "reuse block nums[%d] is greater than need block nums[%d]",
                            reuse_block_num,
                            cache_block_num);
    RTP_LLM_CHECK_WITH_INFO((match_result.loss.empty() || match_result.loss.size() >= reuse_length),
                            "reuse loss nums [%d] is less than need loss nums[%d]",
                            match_result.loss.size(),
                            reuse_length);

    local_match_blocks          = local_match_blocks <= reuse_block_num ? local_match_blocks : reuse_block_num;
    const auto local_reuse_len  = local_match_blocks * config_.seq_size_per_block;
    const auto remote_reuse_len = (reuse_block_num - local_match_blocks) * config_.seq_size_per_block;

    return {(size_t)reuse_length,
            local_reuse_len,
            remote_reuse_len,
            vector<int>(match_result.block_indices.begin(), match_result.block_indices.begin() + reuse_block_num),
            vector<float>(match_result.loss.begin(),
                          match_result.loss.begin() + std::min((int)match_result.loss.size(), reuse_length))};
}

void CacheManager::incrBlockRefCounter(const std::vector<int>& indices) {
    allocator_->incrBlockRefCounter(indices);
}

void CacheManager::incrQueryRefCounter(const std::vector<int>& blocks) {
    std::set<int> unique_blocks(blocks.begin(), blocks.end());
    for (auto block : unique_blocks) {
        // it is okey to use getRefCounterUnchecked here, as the ref counters are about to increase
        if (query_ref_counter_.getRefCounterUnchecked(block) == 0) {
            available_blocks_--;
        }
    }

    query_ref_counter_.incrementRefCounter(blocks);
}

void CacheManager::decrQueryRefCounter(const std::vector<int>& blocks) {
    query_ref_counter_.decrementRefCounter(blocks);

    std::set<int> unique_blocks(blocks.begin(), blocks.end());
    for (auto block : unique_blocks) {
        if (query_ref_counter_.getRefCounter(block) == 0) {
            available_blocks_++;
        }
    }
}

std::tuple<bool, KVCacheResource> CacheManager::malloc(const KVCacheAllocator::SimpleMallocInfo& malloc_info) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto [success, block_indices] = mallocIndex(malloc_info);
    return {success, {block_indices}};
}

std::tuple<bool, std::vector<int>> CacheManager::mallocIndex(const KVCacheAllocator::SimpleMallocInfo& malloc_info) {
    maybeFreeBlockFromCache(malloc_info.block_nums);
    auto [success, resources] = allocator_->malloc(malloc_info);
    incrQueryRefCounter(resources.block_id);
    return {success, resources.block_id};
}

void CacheManager::reserveBlocks(int nums) {
    maybeFreeBlockFromCache(nums);
}

void CacheManager::free(const std::vector<KVCacheResource>& resource) {
    for (const auto& kv_block : resource) {
        free(kv_block.block_id);
    }
}

void CacheManager::free(const std::vector<int>& block_indices) {
    std::lock_guard<std::mutex> guard(mutex_);
    decrQueryRefCounter(block_indices);
    allocator_->free(block_indices);
}

void CacheManager::freeWithoutLock(const std::vector<int>& block_indices) {
    decrQueryRefCounter(block_indices);
    allocator_->free(block_indices);
}

size_t CacheManager::newFreeBlocks(const std::vector<int>& indices) {
    std::unordered_map<int, int> decrement_counts;
    for (int i : indices) {
        decrement_counts[i]++;
    }
    size_t new_free_blocks = 0;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        for (const auto& pair : decrement_counts) {
            int block_id          = pair.first;
            int decrement_count   = pair.second;
            int current_ref_count = query_ref_counter_.getRefCounter(block_id);
            if (current_ref_count == decrement_count) {
                new_free_blocks++;
            }
        }
    }
    return new_free_blocks;
}

void CacheManager::maybeFreeBlockFromCache(int nums) {
    while (int(freeBlockNums()) < nums && !block_cache_.empty()) {
        std::vector<int> indices = block_cache_.pop();
        if (indices.empty()) {
            // avoid infinite loop
            break;
        }
        allocator_->free(indices);
    }
}

void CacheManager::freeWithCache(FreeInfo& free_info) {
    std::lock_guard<std::mutex> guard(mutex_);
    decrQueryRefCounter(free_info.block_indices);
    free_info.is_resident = false;
    insertIntoCache(free_info);
}

void CacheManager::insertResidentCache(FreeInfo& free_info) {
    free_info.is_resident = true;
    insertIntoCache(free_info);
}

void CacheManager::insertIntoCache(FreeInfo& free_info) {
    if (free_info.token_ids.size() > 1) {
        size_t token_len = free_info.token_ids.size() - 1;
        size_t block_len = std::min(std::min(free_info.block_indices.size(), free_info.cache_keys.size()),
                                    token_len / seq_size_per_block_);
        token_len        = block_len * seq_size_per_block_;
        CacheItem        item{{free_info.token_ids.begin(), free_info.token_ids.begin() + token_len},
                              {free_info.block_indices.begin(), free_info.block_indices.begin() + block_len},
                              {free_info.cache_keys.begin(), free_info.cache_keys.begin() + block_len},
                       free_info.loss.empty() ?
                                  free_info.loss :
                                  std::vector<float>{free_info.loss.begin(), free_info.loss.begin() + token_len},
                       free_info.is_resident};
        std::vector<int> indices = block_cache_.put(item);
        if (enable_dist_kvcache_ && free_info.enable_3fs && free_info.loss.empty()) {
            putToDistKvCache(item.cache_key, item.block_indices, 0, free_info.request_id, free_info.adapter_name);
        }
        allocator_->free(indices);
        allocator_->free(std::vector<int>(free_info.block_indices.begin() + block_len, free_info.block_indices.end()));
    } else {
        allocator_->free(free_info.block_indices);
    }
}

void CacheManager::incrRefCounter(const std::vector<int>& indices) {
    incrBlockRefCounter(indices);
    incrQueryRefCounter(indices);
}

void CacheManager::setKVBlockValue(int              block_index,
                                   int              layer_id,
                                   rtp_llm::Buffer& k_buffer,
                                   rtp_llm::Buffer& v_buffer) {
    allocator_->setKVBlockValue(block_index, layer_id, k_buffer, v_buffer);
}

void CacheManager::setKVBlockValue(int block_index, rtp_llm::Buffer& k_buffer, rtp_llm::Buffer& v_buffer) {
    allocator_->setKVBlockValue(block_index, k_buffer, v_buffer);
}

std::tuple<rtp_llm::BufferPtr, rtp_llm::BufferPtr> CacheManager::getKVBlockValue(int block_index, int layer_id) {
    auto [success, k_buffer, v_buffer] = allocator_->getKVBlockValue(block_index, layer_id);
    return {k_buffer, v_buffer};
}

std::tuple<rtp_llm::BufferPtr, rtp_llm::BufferPtr> CacheManager::getKVBlockValue(int block_index) {
    auto [success, k_buffer, v_buffer] = allocator_->getKVBlockValue(block_index);
    return {k_buffer, v_buffer};
}

void CacheManager::blockCopy(int src_block_index, int dest_block_index) {
    allocator_->blockCopy(src_block_index, dest_block_index);
}

void CacheManager::blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping) {
    allocator_->blockBatchCopy(copy_mapping);
}

void CacheManager::blockBatchCopy(const Buffer& copy_mapping) {
    return allocator_->blockBatchCopy(copy_mapping);
}

void CacheManager::blockBatchCopy(const BlockIdPair* begin_ptr, const BlockIdPair* end_ptr) {
    return allocator_->blockBatchCopy(begin_ptr, end_ptr);
}

KVCacheAllocator::BlockAddrInfo CacheManager::convertIndexToAddr(int block_index, int layer_id) const {
    return allocator_->convertIndexToAddr(block_index, layer_id);
}

bool CacheManager::initDistKvCache() {
    DistKvCacheInitParams init_params;
    init_params.match_timeout_ms         = params_.kv_cache_config.match_timeout_ms;
    init_params.rpc_get_cache_timeout_ms = params_.kv_cache_config.rpc_get_cache_timeout_ms;
    init_params.rpc_put_cache_timeout_ms = params_.kv_cache_config.rpc_put_cache_timeout_ms;
    init_params.max_block_size_per_item  = params_.kv_cache_config.max_block_size_per_item;
    if (params_.kv_cache_config.enable_3fs) {
        DistStorage3FSInitParams init_params_3fs;
        init_params_3fs.read_iov_size                      = params_.kv_cache_config.threefs_read_iov_size;
        init_params_3fs.write_iov_size                     = params_.kv_cache_config.threefs_write_iov_size;
        init_params_3fs.read_timeout_ms                    = params_.kv_cache_config.threefs_read_timeout_ms;
        init_params_3fs.write_timeout_ms                   = params_.kv_cache_config.threefs_write_timeout_ms;
        init_params.storage_manager_params.init_params_3fs = init_params_3fs;
    }
    auto dist_kvcache = std::make_shared<DistKvCache>(this, params_, metrics_reporter_);
    if (!dist_kvcache->init(init_params)) {
        RTP_LLM_LOG_WARNING("dist kvcache init failed!!!");
        return false;
    }
    dist_kvcache_  = std::move(dist_kvcache);
    lora_info_map_ = getLoraInfo();
    return true;
}

void CacheManager::matchInDistKvCache(const AdvancedMallocInfo& malloc_info, BlockCache::MatchResult& match_result) {
    const auto cache_keys           = malloc_info.cache_keys;
    const auto request_id           = malloc_info.request_id;
    const auto local_matched_blocks = match_result.block_indices.size();

    if (local_matched_blocks >= cache_keys.size()) {
        return;
    }
    if (!dist_kvcache_) {
        RTP_LLM_LOG_WARNING("match in dist kvcache failed, dist kvcache is nullptr");
        return;
    }

    std::map<std::string, std::string> extra_metas;
    extra_metas["LORA_CKPT_PATH"] = getLoraCkptPath(malloc_info.adapter_name);

    auto matched_blocks = dist_kvcache_->matchForAllRank(cache_keys, local_matched_blocks, request_id, extra_metas);
    if (matched_blocks <= 0) {
        return;
    }

    int64_t need_block_num = static_cast<int64_t>(matched_blocks) - static_cast<int64_t>(local_matched_blocks);
    if (need_block_num <= 0) {
        return;
    }
    auto [success, block_id] =
        mallocIndex(KVCacheAllocator::SimpleMallocInfo(request_id, static_cast<uint32_t>(need_block_num), true));
    if (!success) {
        RTP_LLM_LOG_WARNING(
            "prefix matched in dist kvcache but free block index not enough, need block num: %d, free blocks num: %lu",
            need_block_num,
            freeBlockNums());
        return;
    }

    std::vector<int64_t> matched_cache_keys(cache_keys.begin(), cache_keys.begin() + matched_blocks);
    if (!dist_kvcache_->getForAllRank(matched_cache_keys, block_id, local_matched_blocks, request_id, extra_metas)) {
        freeWithoutLock(block_id);
        return;
    }

    match_result.block_indices.insert(match_result.block_indices.end(), block_id.begin(), block_id.end());
}

bool CacheManager::putToDistKvCache(const std::vector<int64_t>& cache_keys,
                                    const std::vector<int32_t>& block_indices,
                                    size_t                      ignore_block_num,
                                    int64_t                     request_id,
                                    const std::string&          adapter_name) const {
    if (cache_keys.empty()) {
        return true;
    }
    if (cache_keys.size() != block_indices.size()) {
        RTP_LLM_LOG_WARNING(
            "cache key size %d not equal to block index size %d", cache_keys.size(), block_indices.size());
        return false;
    }
    if (dist_kvcache_) {
        std::map<std::string, std::string> extra_metas;
        extra_metas["LORA_CKPT_PATH"] = getLoraCkptPath(adapter_name);
        return dist_kvcache_->putForAllRank(cache_keys, block_indices, ignore_block_num, request_id, extra_metas);
    }
    return false;
}

bool CacheManager::getCacheForRank(const std::vector<int64_t>&               cache_keys,
                                   const std::vector<int32_t>&               block_indices,
                                   size_t                                    ignore_block_num,
                                   int64_t                                   request_id,
                                   const std::map<std::string, std::string>& extra_metas) const {
    if (dist_kvcache_) {
        return dist_kvcache_->get(cache_keys, block_indices, ignore_block_num, request_id, extra_metas);
    }
    return false;
}

bool CacheManager::putCacheForRank(const std::vector<int64_t>&               cache_keys,
                                   const std::vector<int32_t>&               block_indices,
                                   size_t                                    ignore_block_num,
                                   int64_t                                   request_id,
                                   const std::map<std::string, std::string>& extra_metas) const {
    if (dist_kvcache_) {
        return dist_kvcache_->put(cache_keys, block_indices, ignore_block_num, request_id, extra_metas);
    }
    return false;
}

class LoraInfo: public autil::legacy::Jsonizable {
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("", lora_info_map, {});
    }

public:
    // {adapter_name: lora_ckpt_path}
    std::map<std::string, std::string> lora_info_map;
};

std::map<std::string, std::string> CacheManager::getLoraInfo() const {
    const auto lora_info_str = autil::EnvUtil::getEnv("LORA_INFO", std::string(""));
    RTP_LLM_LOG_INFO("lora info: %s", lora_info_str.c_str());
    if (lora_info_str.empty()) {
        return {};
    }
    try {
        LoraInfo lora_info;
        autil::legacy::FromJsonString(lora_info, lora_info_str);
        return lora_info.lora_info_map;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING(
            "found exception when parse lora info. lora info: %s, exception: [%s]", lora_info_str.c_str(), e.what());
    }
    return {};
}

inline std::size_t hashString(const std::string& str) {
    std::hash<std::string> hasher;
    return hasher(str);
}

std::string CacheManager::getLoraCkptPath(const std::string& adapter_name) const {
    std::string lora_ckpt_path = "no_lora";
    if (lora_info_map_.count(adapter_name) != 0 && !lora_info_map_.at(adapter_name).empty()) {
        lora_ckpt_path = std::to_string(hashString(lora_info_map_.at(adapter_name)));
    }
    return lora_ckpt_path;
}

const std::shared_ptr<KVCacheAllocator>& CacheManager::kvCacheAllocator() const {
    return allocator_;
}

}  // namespace rtp_llm