#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <unistd.h>
#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

#ifdef ENABLE_3FS
#include "rtp_llm/cpp/cache/ThreeFSCacheManager.h"
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
    device_(device),
    metrics_reporter_(metrics_reporter),
    params_(params) {
    RTP_LLM_LOG_INFO("cache config: %s", config.debugString().c_str());

    if (warmup) {
        initFakeKVCache();
    } else {
        allocateAndSync();
        RTP_LLM_LOG_INFO("block nums is %d after tp sync", config_.block_nums);
        initFreeBlock();
        initKvCache();
        if (metrics_reporter_) {
            metrics_reporter_thread_ = std::thread(&CacheManager::reportMetricsLoop, this);
        }
    }

#ifdef ENABLE_3FS
    if (params_.enable_3fs_) {
        enable_3fs_ = init3FS();
    }
    RTP_LLM_LOG_INFO("enable 3fs: %d", enable_3fs_);
#endif
}

void CacheManager::regUserMr() {
    if (device_->cacheStore() && !kvcache_reg_mr_) {
        RTP_LLM_LOG_INFO("start to register user mr");
        auto memory_util = static_pointer_cast<NormalCacheStore>(device_->cacheStore())->getMemoryUtil();
        
        auto start_time_us = currentTimeUs();
        if (!memory_util->regUserMr(cache_base_ptr_, config_.k_total_size, true, config_.k_block_stride)) {
            RTP_LLM_FAIL("register user mr for k buffer failed");
        }
        auto cost_time_ms = (currentTimeUs() - start_time_us) / 1000;
        RTP_LLM_LOG_INFO("register user mr for k buffer success: cost %ld ms, cache base address %p, len %lu, end address %p",
            cost_time_ms, cache_base_ptr_, config_.k_total_size, (int8_t*)cache_base_ptr_ + config_.k_total_size);
        mr_cost_time_ms_ += cost_time_ms;

        start_time_us = currentTimeUs();
        if (!memory_util->regUserMr((int8_t*)cache_base_ptr_ + config_.k_total_size,
                config_.v_total_size, true, config_.v_block_stride)) {
            RTP_LLM_FAIL("register user mr for v buffer failed");
        }
        cost_time_ms = (currentTimeUs() - start_time_us) / 1000;
        RTP_LLM_LOG_INFO("register user mr for v buffer success: cost %ld ms, cache base address %p, len %lu, end address %p",
            cost_time_ms, (int8_t*)cache_base_ptr_ + config_.k_total_size, config_.v_total_size,
            (int8_t*)cache_base_ptr_ + config_.total_size);
        
        // TODO(xinfei.sxf) reg kv cache scale

        mr_cost_time_ms_ += cost_time_ms;

        kvcache_reg_mr_ = true;
    }
}

void CacheManager::deregUserMr() {
    if (device_->cacheStore() && kvcache_reg_mr_) {
        RTP_LLM_LOG_INFO("start to deregUserMr user mr");
        auto memory_util = static_pointer_cast<NormalCacheStore>(device_->cacheStore())->getMemoryUtil();
        if (!memory_util->deregUserMr(cache_base_ptr_, true)) {
            RTP_LLM_FAIL("deregUserMr user mr for k buffer failed");
        }
        RTP_LLM_LOG_INFO("deregUserMr user mr for k buffer success");
        if (!memory_util->deregUserMr((int8_t*)cache_base_ptr_ + config_.k_total_size, true)) {
            RTP_LLM_FAIL("deregUserMr user mr for v buffer failed");
        }
        RTP_LLM_LOG_INFO("deregUserMr user mr for v buffer success");
    }
}

CacheManager::~CacheManager() {
    deregUserMr();

    stop_ = true;
    if (metrics_reporter_thread_.joinable()) {
        metrics_reporter_thread_.join();
    }
    cache_aligned_buffer_.reset();
}

uint32_t CacheManager::totalBlocks() const {
    // block 0 is reserved
    return config_.block_nums - 1;
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
                collector.kv_cache_item_num = block_cache_.size();
                auto available_blocks = availableBlockNums();
                collector.kv_cache_left_seq = available_blocks * seq_size_per_block_;
                collector.kv_cache_available_blocks = available_blocks;
                collector.kv_cache_free_blocks = freeBlockNums();
                collector.kv_cache_used_ratio = 100.0 * (totalBlocks() - available_blocks) / totalBlocks();
                collector.mr_cost_time_ms = mr_cost_time_ms_;
            }
            metrics_reporter_->report<RtpLLMCacheMetrics, RtpLLMCacheMetricsCollector>(nullptr, &collector);
            std::this_thread::sleep_for(std::chrono::seconds(1));  // 1s
        }
    }
}

void CacheManager::initFreeBlock() {
    free_blocks_index_ = std::set<int>();
    // block 0 is reserved for tmp or padding use
    for (int i = 1; i < int(config_.block_nums); ++i) {
        free_blocks_index_.insert(i);
    }
    available_blocks_ = totalBlocks();

    block_ref_counter_ = BlockRefCounter(config_.block_nums);
}

void CacheManager::allocateAndSync() {
    const auto properties = device_->getDeviceProperties();
    size_t world_size = properties.tp_size * properties.dp_size;
    if (world_size > 1) {
        size_t local_rank = properties.tp_size * properties.dp_rank + properties.tp_rank;
        BufferPtr block_num_infos =
            device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {world_size}, rtp_llm::AllocationType::HOST});
        auto block_num_ptr                = block_num_infos->data<int>();
        block_num_ptr[local_rank] = config_.block_nums;
        device_->allGather({{block_num_infos}, ParallelMode::DP_AND_TP});
        device_->syncCommunication(false);
        device_->syncAndCheck();
        config_.block_nums = *std::min_element(block_num_ptr, block_num_ptr + world_size);
    }
    config_.refresh();
    cache_aligned_buffer_ =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_INT8, {config_.total_size}});
    // temp hack for mla, since other devices not impl bufMemset
    if (config_.use_mla) {
        device_->bufMemset(*cache_aligned_buffer_, 0);
    }

    cache_base_ptr_ = cache_aligned_buffer_->data();
}

void CacheManager::initFakeKVCache() {
    auto k_block_size = getKBlockSize();
    auto v_block_size = getVBlockSize();
    cache_aligned_buffer_ = device_->allocateBuffer({rtp_llm::DataType::TYPE_INT8, {k_block_size + v_block_size}});
    cache_base_ptr_ = cache_aligned_buffer_->data();
    initKvCache();
}

size_t CacheManager::getKBlockSize() const {
    if (config_.use_mla) {
        return getTypeSize(config_.dtype) * (size_t)config_.layer_num * (size_t)config_.block_nums
            * (size_t)config_.seq_size_per_block * (size_t)config_.kv_lora_rank;
    } else {
        return getTypeSize(config_.dtype) * (size_t)config_.layer_num *  (size_t)config_.block_nums
            * (size_t)config_.local_head_num_kv * (size_t)config_.seq_size_per_block * (size_t)config_.size_per_head;
    }
}

size_t CacheManager::getVBlockSize() const {
    if (config_.use_mla) {
        return config_.dtype * (size_t)config_.layer_num * (size_t)config_.block_nums                                                 
            * (size_t)config_.seq_size_per_block * (size_t)config_.rope_head_dim;
    } else {
        return getTypeSize(config_.dtype) * (size_t)config_.layer_num *  (size_t)config_.block_nums
        * (size_t)config_.local_head_num_kv * (size_t)config_.seq_size_per_block * (size_t)config_.size_per_head;
    }
}

void CacheManager::initKvCache() {
    if (config_.use_mla) {
        initKvCacheMla();
    } else {
        initKvCacheNormal();
    }

    initKVCacheScale();

    regUserMr();
}

void CacheManager::initKVCacheScale() {
    if (config_.dtype == rtp_llm::DataType::TYPE_INT8) {
        kv_cache_.k_scale =
            std::make_unique<rtp_llm::Buffer>(rtp_llm::MemoryType::MEMORY_GPU,
                                         rtp_llm::DataType::TYPE_FP32,
                                         std::vector<size_t>{(size_t)config_.layer_num,
                                                             (size_t)config_.block_nums,
                                                             (size_t)config_.local_head_num_kv,
                                                             (size_t)config_.seq_size_per_block},
                                         (int8_t*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes() * 2);
        kv_cache_.v_scale = std::make_unique<rtp_llm::Buffer>(rtp_llm::MemoryType::MEMORY_GPU,
                                                         rtp_llm::DataType::TYPE_FP32,
                                                         std::vector<size_t>{(size_t)config_.layer_num,
                                                                             (size_t)config_.block_nums,
                                                                             (size_t)config_.local_head_num_kv,
                                                                             (size_t)config_.seq_size_per_block},
                                                         (int8_t*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes() * 2
                                                             + kv_cache_.k_scale->sizeBytes());
    }

#ifdef ENABLE_FP8
    else if (config_.dtype == rtp_llm::DataType::TYPE_FP8_E4M3) {
        kv_cache_.k_scale = std::make_unique<rtp_llm::Buffer>(
                rtp_llm::MemoryType::MEMORY_GPU,
                rtp_llm::DataType::TYPE_FP32,
                std::vector<size_t>{(size_t)config_.layer_num,
                    (size_t)config_.block_nums,
                    (size_t)config_.local_head_num_kv,
                    (size_t)config_.seq_size_per_block},
                (__nv_fp8_e4m3*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes() * 2);
        kv_cache_.v_scale = std::make_unique<rtp_llm::Buffer>(
                rtp_llm::MemoryType::MEMORY_GPU,
                rtp_llm::DataType::TYPE_FP32,
                std::vector<size_t>{(size_t)config_.layer_num,
                    (size_t)config_.block_nums,
                    (size_t)config_.local_head_num_kv,
                    (size_t)config_.seq_size_per_block},
                (__nv_fp8_e4m3*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes() * 2 + kv_cache_.k_scale->sizeBytes());
    }
#endif
}

void CacheManager::initKvCacheMla() {
    RTP_LLM_LOG_INFO("init mla kv cache");
    kv_cache_.k_blocks = std::make_unique<rtp_llm::Buffer>(rtp_llm::MemoryType::MEMORY_GPU,
                                                      config_.dtype,
                                                      std::vector<size_t>{(size_t)config_.layer_num,
                                                                          (size_t)config_.block_nums,
                                                                          (size_t)config_.seq_size_per_block,
                                                                          (size_t)config_.kv_lora_rank},
                                                      cache_base_ptr_);
    kv_cache_.v_blocks = std::make_unique<rtp_llm::Buffer>(rtp_llm::MemoryType::MEMORY_GPU,
                                                      config_.dtype,
                                                      std::vector<size_t>{(size_t)config_.layer_num,
                                                                          (size_t)config_.block_nums,                                                      
                                                                          (size_t)config_.seq_size_per_block,
                                                                          (size_t)config_.rope_head_dim},
                                                      (int8_t*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes());
}

void CacheManager::initKvCacheNormal() {
    RTP_LLM_LOG_INFO("init normal kv cache");
    kv_cache_.k_blocks = std::make_unique<rtp_llm::Buffer>(rtp_llm::MemoryType::MEMORY_GPU,
                                                      config_.dtype,
                                                      std::vector<size_t>{(size_t)config_.layer_num,
                                                                          (size_t)config_.block_nums,
                                                                          (size_t)config_.local_head_num_kv,
                                                                          (size_t)config_.seq_size_per_block,
                                                                          (size_t)config_.size_per_head},
                                                      cache_base_ptr_);
    kv_cache_.v_blocks = std::make_unique<rtp_llm::Buffer>(rtp_llm::MemoryType::MEMORY_GPU,
                                                      config_.dtype,
                                                      std::vector<size_t>{(size_t)config_.layer_num,
                                                                          (size_t)config_.block_nums,
                                                                          (size_t)config_.local_head_num_kv,
                                                                          (size_t)config_.seq_size_per_block,
                                                                          (size_t)config_.size_per_head},
                                                      (int8_t*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes());
}

const CacheConfig& CacheManager::cacheConfig() const {
    return config_;
}

const BlockRefCounter& CacheManager::blockRefCounter() const {
    return block_ref_counter_;
}

const BlockCache& CacheManager::blockCache() const {
    return block_cache_;
}

size_t CacheManager::freeBlockNums() const {
    return free_blocks_index_.size();
}

size_t CacheManager::availableBlockNums() const {
    return available_blocks_;
}

KVCacheInfo CacheManager::getKVCacheInfo() const {
    return {availableBlockNums() * seq_size_per_block_, totalBlocks() * seq_size_per_block_};
}

size_t CacheManager::cacheItemNum() const {
    return block_cache_.size();
}

const CacheManager::KVCacheBuffer& CacheManager::kvCacheBuffer() const {
    return kv_cache_;
}

CacheManager::MatchInfo CacheManager::mallocWithCache(const AdvancedMallocInfo& malloc_info) {
    auto match_begin_time_us = currentTimeUs();
    auto match_info = matchImpl(malloc_info);
    auto match_cost_time_us = currentTimeUs() - match_begin_time_us;
    if (match_info.loss.empty() && malloc_info.need_loss) {
        free(match_info.cache_blocks);
        return {0, {}, {}};
    }
    if (metrics_reporter_) {
        RtpLLMCacheReuseMetricsCollector collector;
        collector.kv_cache_reuse_length = match_info.reuse_length;
        collector.match_cost_time_us = match_cost_time_us;
        metrics_reporter_->report<RtpLLMCacheReuseMetrics, RtpLLMCacheReuseMetricsCollector>(nullptr, &collector);
    }
    return match_info;
}

CacheManager::MatchInfo CacheManager::matchImpl(const AdvancedMallocInfo& malloc_info) {
    // match in gpu
    auto match_result = block_cache_.match(malloc_info.cache_keys);
    incrRefCounter(match_result.block_indices);

#ifdef ENABLE_3FS
    // match in 3fs if cache keys not fully matched
    if (enable_3fs_ && match_result.matched_len < malloc_info.cache_keys.size()) {
        std::vector<int64_t> need_match_cache_keys(malloc_info.cache_keys.begin() + match_result.matched_len,
                                                   malloc_info.cache_keys.end());
        const auto           threefs_match_result = matchIn3FS(need_match_cache_keys, malloc_info.request_id);
        if (threefs_match_result.matched_len != 0) {
            match_result.matched_len += threefs_match_result.matched_len;
            match_result.block_indices.insert(match_result.block_indices.end(),
                                              threefs_match_result.block_indices.begin(),
                                              threefs_match_result.block_indices.end());
        }
    }
#endif

    int cache_block_num = match_result.block_indices.size();
    int reuse_block_num = std::min(match_result.matched_len, static_cast<size_t>((malloc_info.token_ids.size()) - 1) / config_.seq_size_per_block);
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
        free(need_decref_blocks);
    }

    RTP_LLM_CHECK_WITH_INFO((reuse_block_num <= cache_block_num),
                       "reuse block nums[%d] is less than need block nums[%d]",
                       reuse_block_num,
                       cache_block_num);
    RTP_LLM_CHECK_WITH_INFO((match_result.loss.empty() || match_result.loss.size() >= reuse_length),
                       "reuse loss nums [%d] is less than need loss nums[%d]",
                       match_result.loss.size(),
                       reuse_length);
    return {(size_t)reuse_length,
            vector<int>(match_result.block_indices.begin(), match_result.block_indices.begin() + reuse_block_num),
            vector<float>(match_result.loss.begin(), match_result.loss.begin() + std::min((int)match_result.loss.size(), reuse_length))};
}

void CacheManager::incrBlockRefCounter(const std::vector<int>& indices) {
    block_ref_counter_.incrementRefCounter(indices);
}

void CacheManager::incrQueryRefCounter(const std::vector<int>& blocks) {
    query_ref_counter_.incrementRefCounter(blocks);
    for (auto block : blocks) {
        if (query_ref_counter_.getRefCounter(block) == 1) {
            available_blocks_--;
        }
    }
}

void CacheManager::decrQueryRefCounter(const std::vector<int>& blocks) {
    query_ref_counter_.decrementRefCounter(blocks);
    for (auto block : blocks) {
        if (query_ref_counter_.getRefCounter(block) == 0) {
            available_blocks_++;
        }
    }
}

std::tuple<bool, KVCacheResource> CacheManager::malloc(const SimpleMallocInfo& malloc_info) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto [success, block_indices] = mallocIndex(malloc_info);
    return {success, {block_indices}};
}

std::tuple<bool, std::vector<int>> CacheManager::mallocIndex(const SimpleMallocInfo& malloc_info) {
    maybeFreeBlockFromCache(malloc_info.block_nums);
    return mallocImpl(malloc_info);
}

std::tuple<bool, std::vector<int>> CacheManager::mallocImpl(const SimpleMallocInfo& malloc_info) {
    if (free_blocks_index_.size() < static_cast<size_t>(malloc_info.block_nums)) {
        if (malloc_info.verbose) {
            std::string error_msg = "request " + std::to_string(malloc_info.request_id)
                                    + " failed to malloc " + std::to_string(malloc_info.block_nums)
                                    + " blocks, only " + std::to_string(free_blocks_index_.size()) + " blocks left";
            RTP_LLM_LOG_ERROR("%s", error_msg.c_str());
        }
        return {false, {}};
    } else {
        std::vector<int> result;
        result.reserve(malloc_info.block_nums);
        for (int i = 0; i < malloc_info.block_nums; ++i) {
            int block = *free_blocks_index_.begin();
            free_blocks_index_.erase(free_blocks_index_.begin());
            result.push_back(block);
        }
        block_ref_counter_.incrementRefCounter(result);
        incrQueryRefCounter(result);
        return {true, result};
    }
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
    freeImpl(block_indices);
}

void CacheManager::freeImpl(const std::vector<int>& block_indices) {
    block_ref_counter_.decrementRefCounter(block_indices);
    for (auto block : block_indices) {
        int ref_count = block_ref_counter_.getRefCounter(block);
        if (ref_count == 0) {
            free_blocks_index_.insert(block);
        }
    }
}

void CacheManager::maybeFreeBlockFromCache(int nums) {
    while (int(freeBlockNums()) < nums && !block_cache_.empty()) {
        std::vector<int> indices = block_cache_.pop();
        if (indices.empty()) {
            // avoid infinite loop
            break;
        }
        freeImpl(indices);
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
        size_t block_len = std::min(
            std::min(free_info.block_indices.size(), free_info.cache_keys.size()), token_len / seq_size_per_block_);
        token_len        = block_len * seq_size_per_block_;
        CacheItem item{{free_info.token_ids.begin(), free_info.token_ids.begin() + token_len},
                       {free_info.block_indices.begin(), free_info.block_indices.begin() + block_len},
                       {free_info.cache_keys.begin(), free_info.cache_keys.begin() + block_len},
                        free_info.loss.empty() ?
                        free_info.loss : std::vector<float>{free_info.loss.begin(), free_info.loss.begin() + token_len},
                        free_info.is_resident};
        std::vector<int> indices = block_cache_.put(item);
#ifdef ENABLE_3FS
        if (enable_3fs_) {
            putCacheTo3FSForAllRank(item.cache_key, item.block_indices, free_info.request_id);
        }
#endif
        freeImpl(indices);
        freeImpl(std::vector<int>(free_info.block_indices.begin() + block_len, free_info.block_indices.end()));
    } else {
        freeImpl(free_info.block_indices);
    }
}

void CacheManager::incrRefCounter(const std::vector<int>& indices) {
    incrBlockRefCounter(indices);
    incrQueryRefCounter(indices);
}

void CacheManager::setKVBlockValue(int block_index, int layer_id, rtp_llm::Buffer& k_buffer, rtp_llm::Buffer& v_buffer) {
    auto k_offset = config_.getKeyOffset(block_index, layer_id);
    auto v_offset = config_.getValueOffset(block_index, layer_id);
    auto k_shape = config_.getKeyShape();
    auto v_shape = config_.getValueShape();

    auto copyFunc = [&](rtp_llm::Buffer& src_buffer, rtp_llm::BufferPtr& dst_blocks, size_t offset, size_t shape){
        auto dst_data = (char*)dst_blocks->data() + offset;
        auto dst_buffer = Buffer(
            dst_blocks->where(), src_buffer.type(), {shape}, dst_data);
        device_->copy({dst_buffer, src_buffer});
    };

    copyFunc(k_buffer, kv_cache_.k_blocks, k_offset, k_shape);
    copyFunc(v_buffer, kv_cache_.v_blocks, v_offset, v_shape);
}

void CacheManager::setKVBlockValue(int block_index, rtp_llm::Buffer& k_buffer, rtp_llm::Buffer& v_buffer) {
    for (uint32_t layer_id = 0; layer_id < config_.layer_num; layer_id++) {
        auto layer_k_data = (char*)(k_buffer.data()) + layer_id * config_.getKeyBlockStride();
        auto layer_k_buffer = Buffer(
                k_buffer.where(), k_buffer.type(), {config_.getKeyShape()}, layer_k_data);
        auto layer_v_data = (char*)(v_buffer.data()) + layer_id * config_.getValueBlockStride();
        auto layer_v_buffer = Buffer(
                v_buffer.where(), v_buffer.type(), {config_.getValueShape()}, layer_v_data);
        setKVBlockValue(block_index, layer_id, layer_k_buffer, layer_v_buffer);
    }
}

std::tuple<rtp_llm::BufferPtr, rtp_llm::BufferPtr> CacheManager::getKVBlockValue(int block_index, int layer_id) {
    auto k_offset = config_.getKeyOffset(block_index, layer_id);
    auto v_offset = config_.getValueOffset(block_index, layer_id);
    auto k_shape = config_.getKeyShape();
    auto v_shape = config_.getValueShape();

    auto kdst_buffer = device_->allocateBuffer(
            {config_.dtype, {k_shape}, rtp_llm::AllocationType::DEVICE});
    auto vdst_buffer = device_->allocateBuffer(
            {config_.dtype, {v_shape}, rtp_llm::AllocationType::DEVICE});

    auto copyFunc = [&](rtp_llm::BufferPtr& src_blocks, rtp_llm::BufferPtr& dst_buffer, size_t offset, size_t shape){
        auto src_data = (char*)(src_blocks->data()) + offset;
        auto src_buffer = Buffer(
            src_blocks->where(), config_.dtype, {shape}, src_data);
        device_->copy({*dst_buffer, src_buffer});
    };

    copyFunc(kv_cache_.k_blocks, kdst_buffer, k_offset, k_shape);
    copyFunc(kv_cache_.v_blocks, vdst_buffer, v_offset, v_shape);

    return {kdst_buffer, vdst_buffer};
}

std::tuple<rtp_llm::BufferPtr, rtp_llm::BufferPtr> CacheManager::getKVBlockValue(int block_index) {
    auto k_shape = config_.getKeyShape();
    auto v_shape = config_.getValueShape();
    auto kdst_buffer = device_->allocateBuffer(
            {config_.dtype, {config_.layer_num, k_shape}, rtp_llm::AllocationType::DEVICE});
    auto vdst_buffer = device_->allocateBuffer(
            {config_.dtype, {config_.layer_num, v_shape}, rtp_llm::AllocationType::DEVICE});

    for (uint32_t layer_id = 0; layer_id < config_.layer_num; layer_id++) {
        auto k_offset = config_.getKeyOffset(block_index, layer_id);
        auto v_offset = config_.getValueOffset(block_index, layer_id);
        auto copyFunc = [&](rtp_llm::BufferPtr& src_blocks, rtp_llm::BufferPtr& dst_buffer, size_t offset, size_t shape){
        auto src_data = (char*)(src_blocks->data()) + offset;
        auto src_buffer = Buffer(
            src_blocks->where(), config_.dtype, {shape}, src_data);
        device_->copy({dst_buffer->view(layer_id, 1)[0], src_buffer});
    };
        copyFunc(kv_cache_.k_blocks, kdst_buffer, k_offset, k_shape);
        copyFunc(kv_cache_.v_blocks, vdst_buffer, v_offset, v_shape);
    }

    return {kdst_buffer, vdst_buffer};
}

void CacheManager::blockCopy(int src_block_index, int dest_block_index) {
    auto k_shape = config_.getKeyShape();
    auto v_shape = config_.getValueShape();

    for (uint32_t layer_id = 0; layer_id < config_.layer_num; layer_id++) {
        auto dst_k_offset = config_.getKeyOffset(dest_block_index, layer_id);
        auto src_k_offset = config_.getKeyOffset(src_block_index, layer_id);
        auto dst_v_offset = config_.getValueOffset(dest_block_index, layer_id);
        auto src_v_offset = config_.getValueOffset(src_block_index, layer_id);
        auto copyFunc = [&](rtp_llm::BufferPtr& buffer_blocks, size_t dst_offset, size_t src_offset, size_t shape){
            auto dst_data = (char*)(buffer_blocks->data()) + dst_offset;
            auto src_data = (char*)(buffer_blocks->data()) + src_offset;
            auto dst_buffer = Buffer(
                buffer_blocks->where(), config_.dtype, {shape}, dst_data);
            auto src_buffer = Buffer(
                buffer_blocks->where(), config_.dtype, {shape}, src_data);
            device_->copy({dst_buffer, src_buffer});
        };
        copyFunc(kv_cache_.k_blocks, dst_k_offset, src_k_offset, k_shape);
        copyFunc(kv_cache_.v_blocks, dst_v_offset, src_v_offset, v_shape);
    }
}

CacheManager::BlockAddrInfo CacheManager::convertIndexToAddr(int block_index, int layer_id) const {
    BlockAddrInfo addr_info;
    size_t total_blocks_num = (size_t)(layer_id * config_.block_nums + block_index);
    auto k_offset = config_.getKeyOffset(block_index, layer_id);
    auto v_offset = config_.getValueOffset(block_index, layer_id);
    auto scale_offset = total_blocks_num * (size_t)config_.kv_scale_block_stride;
    addr_info.k_addr = (int8_t*)kv_cache_.k_blocks->data() + k_offset;
    addr_info.v_addr = (int8_t*)kv_cache_.v_blocks->data() + v_offset;

    // TODO(yinzhi): check scale
    if (kv_cache_.k_scale) {
        addr_info.k_scale_addr = (int8_t*)kv_cache_.k_scale->data() + scale_offset;
        addr_info.v_scale_addr = (int8_t*)kv_cache_.v_scale->data() + scale_offset;
    }
    return addr_info;
}

void CacheManager::copyKvCacheFromSeqIdxs(const std::vector<int>& block_indice_list,
                                          const std::vector<int>& src_index,
                                          const std::vector<int>& target_index) {
    if (src_index.size() != target_index.size()) {
        RTP_LLM_FAIL("src index and target index length should equal");
    }
    std::vector<SeqPosition> src_seq_positions;
    std::vector<SeqPosition> target_seq_positions;

    for (size_t i = 0; i < src_index.size(); ++i) {
        src_seq_positions.push_back(getSeqPosition(block_indice_list, src_index[i]));
        target_seq_positions.push_back(getSeqPosition(block_indice_list, target_index[i]));
    }

    for (size_t i = 0; i < src_seq_positions.size(); ++i) {
        copyKvCacheFromSeqPosition(src_seq_positions[i], target_seq_positions[i]);
    }
}

CacheManager::SeqPosition CacheManager::getSeqPosition(const std::vector<int>& block_indice_list, int idx) {
    int block_idx = idx / seq_size_per_block_;
    if (block_idx >= static_cast<int>(block_indice_list.size())) {
        RTP_LLM_FAIL("block idx should not >= len(block_indice_list)");
    }
    return SeqPosition{block_indice_list[block_idx], idx % seq_size_per_block_};
}

void CacheManager::copyKvCacheFromSeqPosition(const SeqPosition& src_seq_position,
                                              const SeqPosition& dst_seq_position) {
    // kv_cache_.k_blocks.index({"...", dst_seq_position.index, "...", dst_seq_position.offset, "..."}).copy_(
    //     kv_cache_.k_blocks.index({"...", src_seq_position.index, "...", src_seq_position.offset, "..."}),
    //     /*non_blocking=*/true);
    // kv_cache_.v_blocks.index({"...", dst_seq_position.index, "...", dst_seq_position.offset, "..."}).copy_(
    //     kv_cache_.v_blocks.index({"...", src_seq_position.index, "...", src_seq_position.offset, "..."}),
    //     /*non_blocking=*/true);
}


// src_block_offset and target_block_offset has same shape.
void CacheManager::beamSearchKvUpdate(rtp_llm::BufferPtr src_block_offset,
                                      rtp_llm::BufferPtr target_block_offset)
{
    // TODO(yinzhi): not available for mla yet
    auto k_blocks_tensor = Buffer2torchTensor(kv_cache_.k_blocks, false);
    auto v_blocks_tensor = Buffer2torchTensor(kv_cache_.v_blocks, false);

    auto org_kv_cache_offset_tensor = Buffer2torchTensor(src_block_offset, false);
    auto target_kv_cache_offset_tensor = Buffer2torchTensor(target_block_offset, false);

    auto k_tmp = k_blocks_tensor.index_select(1, target_kv_cache_offset_tensor.to(device_->getTorchDevice()));
    auto v_tmp = v_blocks_tensor.index_select(1, target_kv_cache_offset_tensor.to(device_->getTorchDevice()));

    for (int i = 0; i < org_kv_cache_offset_tensor.size(0); i++) {
        auto src_index = org_kv_cache_offset_tensor[i].item<int>();
        k_blocks_tensor.index({torch::indexing::Slice(), src_index}).copy_(k_tmp.index({torch::indexing::Slice(), i}).to(device_->getTorchDevice()));
        v_blocks_tensor.index({torch::indexing::Slice(), src_index}).copy_(v_tmp.index({torch::indexing::Slice(), i}).to(device_->getTorchDevice()));
    }
};

#ifdef ENABLE_3FS
bool CacheManager::init3FS() {
    auto threefs_cache_manager = std::make_shared<ThreeFSCacheManager>(
        kv_cache_.k_blocks, kv_cache_.v_blocks, config_, params_, metrics_reporter_);
    if (!threefs_cache_manager->init()) {
        RTP_LLM_LOG_WARNING("3fs init failed, 3fs cache manager init failed");
        return false;
    }
    threefs_cache_manager_ = threefs_cache_manager;
    return true;
}

BlockCache::MatchResult CacheManager::matchIn3FS(const std::vector<int64_t>& cache_keys, int64_t request_id) {
    BlockCache::MatchResult match_result;
    match_result.matched_len = 0;

    if (cache_keys.empty()) {
        return match_result;
    }
    if (!threefs_cache_manager_) {
        RTP_LLM_LOG_WARNING("match in 3fs failed, 3fs cache manager is nullptr");
        return match_result;
    }

    auto matched_len = threefs_cache_manager_->matchCache(cache_keys);
    if (matched_len <= 0) {
        return match_result;
    }

    auto [success, resource] = malloc(SimpleMallocInfo(-1, matched_len, true));
    if (!success) {
        RTP_LLM_LOG_WARNING(
            "prefix matched in 3fs but free block index not enough, matched len: %d, free block index len: %lu",
            matched_len,
            freeBlockNums());
        return match_result;
    }

    std::vector<int64_t> matched_cache_keys(cache_keys.begin(), cache_keys.begin() + matched_len);
    const auto           input_len = static_cast<int32_t>(cache_keys.size());
    if (!threefs_cache_manager_->getCacheForAllRank(matched_cache_keys, resource.block_id, input_len, request_id)) {
        free(resource.block_id);
        return match_result;
    }

    match_result.matched_len   = matched_len;
    match_result.block_indices = resource.block_id;
    return match_result;
}

bool CacheManager::putCacheTo3FSForAllRank(const std::vector<int64_t>& cache_keys,
                                           const std::vector<int32_t>& block_indices,
                                           int64_t                     request_id) const {
    if (!threefs_cache_manager_) {
        RTP_LLM_LOG_WARNING("put cache to 3fs failed, 3fs cache manager is nullptr");
        return false;
    }
    return threefs_cache_manager_->putCacheForAllRank(cache_keys, block_indices, request_id);
}
#endif

bool CacheManager::getCacheFrom3FSForRank(const std::vector<int64_t>& cache_keys,
                                          const std::vector<int32_t>& block_indices,
                                          int64_t                     request_id) const {
#ifdef ENABLE_3FS
    if (!threefs_cache_manager_) {
        RTP_LLM_LOG_WARNING("get cache from 3fs for rank failed, 3fs cache manager is nullptr, request: %ld",
                            request_id);
        return false;
    }
    return threefs_cache_manager_->getCacheForRank(cache_keys, block_indices, request_id);
#else
    return false;
#endif
}

bool CacheManager::putCacheTo3FSForRank(const std::vector<int64_t>& cache_keys,
                                        const std::vector<int32_t>& block_indices,
                                        int64_t                     request_id) const {
#ifdef ENABLE_3FS
    if (!threefs_cache_manager_) {
        RTP_LLM_LOG_WARNING("put cache to 3fs for rank failed, 3fs cache manager is nullptr, request: %ld", request_id);
        return false;
    }
    return threefs_cache_manager_->putCacheForRank(cache_keys, block_indices, request_id);
#else
    return false;
#endif
}

}  // namespace rtp_llm
