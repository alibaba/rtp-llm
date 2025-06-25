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

    if (params_.kv_cache_config.enable_3fs) {
        enable_dist_kvcache_ = initDistKvCache();
        if (!enable_dist_kvcache_) {
            RTP_LLM_FAIL("dist kv cache init failed");
        }
    }
}

void CacheManager::regUserMr(size_t model_id) {
    if (device_->cacheStore() && !kvcache_reg_mr_) {
        RTP_LLM_LOG_INFO("start to register user mr");
        auto memory_util = static_pointer_cast<NormalCacheStore>(device_->cacheStore())->getMemoryUtil();

        auto start_time_us = currentTimeUs();
        if (!memory_util->regUserMr(cache_base_ptr_, config_.k_total_size, true, config_.k_block_stride)) {
            RTP_LLM_FAIL("register user mr for k buffer failed");
        }
        auto cost_time_ms = (currentTimeUs() - start_time_us) / 1000;
        RTP_LLM_LOG_INFO(
            "register user mr for k buffer success: cost %ld ms, cache base address %p, len %lu, end address %p",
            cost_time_ms,
            cache_base_ptr_,
            config_.k_total_size,
            (int8_t*)cache_base_ptr_ + config_.k_total_size);
        mr_cost_time_ms_ += cost_time_ms;

        start_time_us = currentTimeUs();
        if (!memory_util->regUserMr(
                (int8_t*)cache_base_ptr_ + config_.k_total_size, config_.v_total_size, true, config_.v_block_stride)) {
            RTP_LLM_FAIL("register user mr for v buffer failed");
        }
        cost_time_ms = (currentTimeUs() - start_time_us) / 1000;
        RTP_LLM_LOG_INFO(
            "register user mr for v buffer success: cost %ld ms, cache base address %p, len %lu, end address %p",
            cost_time_ms,
            (int8_t*)cache_base_ptr_ + config_.k_total_size,
            config_.v_total_size,
            (int8_t*)cache_base_ptr_ + config_.total_size);

        // TODO(xinfei.sxf) reg kv cache scale

        mr_cost_time_ms_ += cost_time_ms;

        kvcache_reg_mr_ = true;

        auto k_block_size       = getKBlockSize() / (size_t)config_.layer_num / (size_t)config_.block_nums;
        auto v_block_size       = getVBlockSize() / (size_t)config_.layer_num / (size_t)config_.block_nums;
        auto k_scale_block_size = (size_t)config_.kv_scale_block_stride;
        auto v_scale_block_size = (size_t)config_.kv_scale_block_stride;
        std::vector<std::shared_ptr<BlockBuffer>> buffers;
        for (int block_index = 0; block_index < config_.block_nums; ++block_index) {
            for (int layer_index = 0; layer_index < config_.layer_num; ++layer_index) {
                auto block_key = makeCacheKey(model_id, std::to_string(block_index), layer_index);
                auto addr_info = convertIndexToAddr(block_index, layer_index);
                auto k_buffer  = std::make_shared<BlockBuffer>(
                    "k_" + block_key, std::shared_ptr<void>(addr_info.k_addr, [](void*) {}), k_block_size, true, true);
                buffers.push_back(k_buffer);
                auto v_buffer = std::make_shared<BlockBuffer>(
                    "v_" + block_key, std::shared_ptr<void>(addr_info.v_addr, [](void*) {}), v_block_size, true, true);
                buffers.push_back(v_buffer);

                if (addr_info.k_scale_addr) {
                    auto k_scale_buffer =
                        std::make_shared<BlockBuffer>("k_scale_" + block_key,
                                                      std::shared_ptr<void>(addr_info.k_scale_addr, [](void*) {}),
                                                      k_scale_block_size,
                                                      true,
                                                      true);
                    buffers.push_back(k_scale_buffer);
                }

                if (addr_info.v_scale_addr) {
                    auto v_scale_buffer =
                        std::make_shared<BlockBuffer>("v_scale_" + block_key,
                                                      std::shared_ptr<void>(addr_info.v_scale_addr, [](void*) {}),
                                                      v_scale_block_size,
                                                      true,
                                                      true);
                    buffers.push_back(v_scale_buffer);
                }
            }
        }
        device_->cacheStore()->regUserBuffers(buffers);
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
    dist_kvcache_.reset();
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
                collector.kv_cache_item_num         = block_cache_.size();
                auto available_blocks               = availableBlockNums();
                collector.kv_cache_left_seq         = available_blocks * seq_size_per_block_;
                collector.kv_cache_available_blocks = available_blocks;
                collector.kv_cache_free_blocks      = freeBlockNums();
                collector.kv_cache_used_ratio       = 100.0 * (totalBlocks() - available_blocks) / totalBlocks();
                collector.mr_cost_time_ms           = mr_cost_time_ms_;
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
    config_.refresh();
    cache_aligned_buffer_ = device_->allocateBuffer({rtp_llm::DataType::TYPE_INT8, {config_.total_size}});
    // temp hack for mla, since other devices not impl bufMemset
    if (config_.use_mla) {
        device_->bufMemset(*cache_aligned_buffer_, 0);
    }

    cache_base_ptr_ = cache_aligned_buffer_->data();
}

void CacheManager::initFakeKVCache() {
    auto k_block_size     = getKBlockSize();
    auto v_block_size     = getVBlockSize();
    cache_aligned_buffer_ = device_->allocateBuffer({rtp_llm::DataType::TYPE_INT8, {k_block_size + v_block_size}});
    cache_base_ptr_       = cache_aligned_buffer_->data();
    initKvCache();
}

size_t CacheManager::getKBlockSize() const {
    if (config_.use_mla) {
        return getTypeSize(config_.dtype) * (size_t)config_.layer_num * (size_t)config_.block_nums
               * (size_t)config_.seq_size_per_block * (size_t)config_.kv_lora_rank;
    } else {
        return getTypeSize(config_.dtype) * (size_t)config_.layer_num * (size_t)config_.block_nums
               * (size_t)config_.local_head_num_kv * (size_t)config_.seq_size_per_block * (size_t)config_.size_per_head;
    }
}

size_t CacheManager::getVBlockSize() const {
    if (config_.use_mla) {
        return config_.dtype * (size_t)config_.layer_num * (size_t)config_.block_nums
               * (size_t)config_.seq_size_per_block * (size_t)config_.rope_head_dim;
    } else {
        return getTypeSize(config_.dtype) * (size_t)config_.layer_num * (size_t)config_.block_nums
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
        kv_cache_.v_scale = std::make_unique<rtp_llm::Buffer>(
            rtp_llm::MemoryType::MEMORY_GPU,
            rtp_llm::DataType::TYPE_FP32,
            std::vector<size_t>{(size_t)config_.layer_num,
                                (size_t)config_.block_nums,
                                (size_t)config_.local_head_num_kv,
                                (size_t)config_.seq_size_per_block},
            (int8_t*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes() * 2 + kv_cache_.k_scale->sizeBytes());
    }

#ifdef ENABLE_FP8
    else if (config_.dtype == rtp_llm::DataType::TYPE_FP8_E4M3) {
        kv_cache_.k_scale =
            std::make_unique<rtp_llm::Buffer>(rtp_llm::MemoryType::MEMORY_GPU,
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
    // memset k_blocks and v_blocks
#ifdef USING_ROCM
    device_->bufMemset(*kv_cache_.k_blocks, 0);
    device_->bufMemset(*kv_cache_.v_blocks, 0);
#endif
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
    // memset k_blocks and v_blocks
#ifdef USING_ROCM
    device_->bufMemset(*kv_cache_.k_blocks, 0);
    device_->bufMemset(*kv_cache_.v_blocks, 0);
#endif
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

KVCacheInfo CacheManager::getKVCacheInfo(int64_t latest_version) const {
    auto snapshot = block_cache_.cacheSnapshot(latest_version);
    return {(size_t)availableBlockNums() * seq_size_per_block_,
            (size_t)totalBlocks() * seq_size_per_block_,
            (size_t)seq_size_per_block_,
            std::move(snapshot.keys),
            snapshot.version};
}

size_t CacheManager::cacheItemNum() const {
    return block_cache_.size();
}

const CacheManager::KVCacheBuffer& CacheManager::kvCacheBuffer() const {
    return kv_cache_;
}

CacheManager::MatchInfo CacheManager::mallocWithCache(const AdvancedMallocInfo& malloc_info) {
    if (malloc_info.token_ids.empty() || malloc_info.token_ids.size() < config_.seq_size_per_block + 1) {
        return MatchInfo{};
    }

    auto match_begin_time_us = currentTimeUs();
    auto match_info          = matchImpl(malloc_info);
    auto match_cost_time_us  = currentTimeUs() - match_begin_time_us;
    if (match_info.loss.empty() && malloc_info.need_loss) {
        free(match_info.cache_blocks);
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

CacheManager::MatchInfo CacheManager::matchImpl(const AdvancedMallocInfo& malloc_info) {
    // match in gpu
    auto match_result = block_cache_.match(malloc_info.cache_keys);
    incrRefCounter(match_result.block_indices);
    auto local_match_len = match_result.matched_len;

    // match in dist kvcache if cache keys not fully matched
    if (enable_dist_kvcache_ && malloc_info.enable_3fs && match_result.matched_len < malloc_info.cache_keys.size()) {
        matchInDistKvCache(malloc_info, match_result);
    }

    int cache_block_num = match_result.block_indices.size();
    int reuse_block_num = std::min(
        match_result.matched_len, static_cast<size_t>((malloc_info.token_ids.size()) - 1) / config_.seq_size_per_block);
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

    local_match_len             = local_match_len <= reuse_block_num ? local_match_len : reuse_block_num;
    const auto local_reuse_len  = local_match_len * config_.seq_size_per_block;
    const auto remote_reuse_len = (reuse_block_num - local_match_len) * config_.seq_size_per_block;

    return {(size_t)reuse_length,
            local_reuse_len,
            remote_reuse_len,
            vector<int>(match_result.block_indices.begin(), match_result.block_indices.begin() + reuse_block_num),
            vector<float>(match_result.loss.begin(),
                          match_result.loss.begin() + std::min((int)match_result.loss.size(), reuse_length))};
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
            std::string error_msg = "request " + std::to_string(malloc_info.request_id) + " failed to malloc "
                                    + std::to_string(malloc_info.block_nums) + " blocks, only "
                                    + std::to_string(free_blocks_index_.size()) + " blocks left";
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
        if (enable_dist_kvcache_ && free_info.enable_3fs) {
            putCacheForAllRank(item.cache_key, item.block_indices, 0, free_info.request_id, free_info.adapter_name);
        }
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

void CacheManager::setKVBlockValue(int              block_index,
                                   int              layer_id,
                                   rtp_llm::Buffer& k_buffer,
                                   rtp_llm::Buffer& v_buffer) {
    auto k_offset = config_.getKeyOffset(block_index, layer_id);
    auto v_offset = config_.getValueOffset(block_index, layer_id);
    auto k_shape  = config_.getKeyShape();
    auto v_shape  = config_.getValueShape();

    auto copyFunc = [&](rtp_llm::Buffer& src_buffer, rtp_llm::BufferPtr& dst_blocks, size_t offset, size_t shape) {
        auto dst_data   = (char*)dst_blocks->data() + offset;
        auto dst_buffer = Buffer(dst_blocks->where(), src_buffer.type(), {shape}, dst_data);
        device_->copy({dst_buffer, src_buffer});
    };

    copyFunc(k_buffer, kv_cache_.k_blocks, k_offset, k_shape);
    copyFunc(v_buffer, kv_cache_.v_blocks, v_offset, v_shape);
}

void CacheManager::setKVBlockValue(int block_index, rtp_llm::Buffer& k_buffer, rtp_llm::Buffer& v_buffer) {
    for (uint32_t layer_id = 0; layer_id < config_.layer_num; layer_id++) {
        auto layer_k_data   = (char*)(k_buffer.data()) + layer_id * config_.getKeyBlockStride();
        auto layer_k_buffer = Buffer(k_buffer.where(), k_buffer.type(), {config_.getKeyShape()}, layer_k_data);
        auto layer_v_data   = (char*)(v_buffer.data()) + layer_id * config_.getValueBlockStride();
        auto layer_v_buffer = Buffer(v_buffer.where(), v_buffer.type(), {config_.getValueShape()}, layer_v_data);
        setKVBlockValue(block_index, layer_id, layer_k_buffer, layer_v_buffer);
    }
}

std::tuple<rtp_llm::BufferPtr, rtp_llm::BufferPtr> CacheManager::getKVBlockValue(int block_index, int layer_id) {
    auto k_offset = config_.getKeyOffset(block_index, layer_id);
    auto v_offset = config_.getValueOffset(block_index, layer_id);
    auto k_shape  = config_.getKeyShape();
    auto v_shape  = config_.getValueShape();

    auto kdst_buffer = device_->allocateBuffer({config_.dtype, {k_shape}, rtp_llm::AllocationType::DEVICE});
    auto vdst_buffer = device_->allocateBuffer({config_.dtype, {v_shape}, rtp_llm::AllocationType::DEVICE});

    auto copyFunc = [&](rtp_llm::BufferPtr& src_blocks, rtp_llm::BufferPtr& dst_buffer, size_t offset, size_t shape) {
        auto src_data   = (char*)(src_blocks->data()) + offset;
        auto src_buffer = Buffer(src_blocks->where(), config_.dtype, {shape}, src_data);
        device_->copy({*dst_buffer, src_buffer});
    };

    copyFunc(kv_cache_.k_blocks, kdst_buffer, k_offset, k_shape);
    copyFunc(kv_cache_.v_blocks, vdst_buffer, v_offset, v_shape);

    return {kdst_buffer, vdst_buffer};
}

std::tuple<rtp_llm::BufferPtr, rtp_llm::BufferPtr> CacheManager::getKVBlockValue(int block_index) {
    auto k_shape = config_.getKeyShape();
    auto v_shape = config_.getValueShape();
    auto kdst_buffer =
        device_->allocateBuffer({config_.dtype, {config_.layer_num, k_shape}, rtp_llm::AllocationType::DEVICE});
    auto vdst_buffer =
        device_->allocateBuffer({config_.dtype, {config_.layer_num, v_shape}, rtp_llm::AllocationType::DEVICE});

    for (uint32_t layer_id = 0; layer_id < config_.layer_num; layer_id++) {
        auto k_offset = config_.getKeyOffset(block_index, layer_id);
        auto v_offset = config_.getValueOffset(block_index, layer_id);
        auto copyFunc =
            [&](rtp_llm::BufferPtr& src_blocks, rtp_llm::BufferPtr& dst_buffer, size_t offset, size_t shape) {
                auto src_data   = (char*)(src_blocks->data()) + offset;
                auto src_buffer = Buffer(src_blocks->where(), config_.dtype, {shape}, src_data);
                device_->copy({dst_buffer->view(layer_id, 1)[0], src_buffer});
            };
        copyFunc(kv_cache_.k_blocks, kdst_buffer, k_offset, k_shape);
        copyFunc(kv_cache_.v_blocks, vdst_buffer, v_offset, v_shape);
    }

    return {kdst_buffer, vdst_buffer};
}

void CacheManager::blockCopy(int src_block_index, int dest_block_index) {
    BlockIdPair copy_mapping{src_block_index, dest_block_index};
    blockBatchCopy(&copy_mapping, &copy_mapping + 1);
}

void CacheManager::blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping) {
    blockBatchCopy(copy_mapping.data(), copy_mapping.data() + copy_mapping.size());
}

void CacheManager::blockBatchCopy(const Buffer& copy_mapping) {
    RTP_LLM_CHECK(copy_mapping.dim() == 2 && copy_mapping.shape()[1] == 2);
    const auto* begin_ptr = (const BlockIdPair*)copy_mapping.data();
    size_t      copy_num  = copy_mapping.shape()[0];
    blockBatchCopy(begin_ptr, begin_ptr + copy_num);
}

void CacheManager::blockBatchCopy(const BlockIdPair* begin_ptr, const BlockIdPair* end_ptr) {
    if (end_ptr == begin_ptr) {
        return;
    }

    auto& k_blocks = *kv_cache_.k_blocks;
    auto& v_blocks = *kv_cache_.v_blocks;

    BatchCopyParams copy_params;

    auto k_copy_type = BatchCopyParams::get_copy_type(k_blocks.where(), k_blocks.where());
    auto v_copy_type = BatchCopyParams::get_copy_type(v_blocks.where(), v_blocks.where());

    const size_t copy_num = (end_ptr - begin_ptr) * config_.layer_num;
    if (k_copy_type == v_copy_type) {
        copy_params.reserve(k_copy_type, 2 * copy_num);
    } else {
        copy_params.reserve(k_copy_type, copy_num);
        copy_params.reserve(v_copy_type, copy_num);
    }

    const auto copy_blocks =
        [&](Buffer& buffer_blocks, size_t block_bytes, BatchCopyParams::CopyType copy_type, auto get_offset) {
            auto blocks_data = (char*)(buffer_blocks.data());
            for (auto it = begin_ptr; it != end_ptr; ++it) {
                auto [src_block_index, dest_block_index] = *it;
                for (uint32_t layer_id = 0; layer_id < config_.layer_num; layer_id++) {
                    auto dst_offset = get_offset(dest_block_index, layer_id);
                    auto dst_data   = blocks_data + dst_offset;

                    auto src_offset = get_offset(src_block_index, layer_id);
                    auto src_data   = blocks_data + src_offset;

                    copy_params.add(dst_data, src_data, block_bytes, copy_type);
                }
            }
        };

    // copy k blocks
    auto k_copy_bytes = config_.getKeyBlockStride();
    copy_blocks(k_blocks, k_copy_bytes, k_copy_type, [&](int block_index, int layer_id) {
        return config_.getKeyOffset(block_index, layer_id);
    });

    // copy v blocks
    auto v_copy_bytes = config_.getValueBlockStride();
    copy_blocks(v_blocks, v_copy_bytes, v_copy_type, [&](int block_index, int layer_id) {
        return config_.getValueOffset(block_index, layer_id);
    });

    device_->batchCopy(copy_params);
}

CacheManager::BlockAddrInfo CacheManager::convertIndexToAddr(int block_index, int layer_id) const {
    if (block_index < 0 || block_index >= config_.block_nums || layer_id < 0 || layer_id >= config_.layer_num) {
        RTP_LLM_FAIL("block index or layer id out of range, block_index: %d, layer_id: %d");
    }

    BlockAddrInfo addr_info;
    size_t        total_blocks_num = (size_t)(layer_id * config_.block_nums + block_index);
    auto          k_offset         = config_.getKeyOffset(block_index, layer_id);
    auto          v_offset         = config_.getValueOffset(block_index, layer_id);
    auto          scale_offset     = total_blocks_num * (size_t)config_.kv_scale_block_stride;
    addr_info.k_addr               = (int8_t*)kv_cache_.k_blocks->data() + k_offset;
    addr_info.v_addr               = (int8_t*)kv_cache_.v_blocks->data() + v_offset;

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
    const auto cache_keys        = malloc_info.cache_keys;
    const auto request_id        = malloc_info.request_id;
    const auto local_matched_len = match_result.matched_len;

    if (local_matched_len >= cache_keys.size()) {
        return;
    }
    if (!dist_kvcache_) {
        RTP_LLM_LOG_WARNING("match in dist kvcache failed, dist kvcache is nullptr");
        return;
    }

    std::map<std::string, std::string> extra_metas;
    extra_metas["LORA_CKPT_PATH"] = getLoraCkptPath(malloc_info.adapter_name);

    auto matched_len = dist_kvcache_->matchForAllRank(cache_keys, local_matched_len, request_id, extra_metas);
    if (matched_len <= 0) {
        return;
    }

    int64_t need_block_num = static_cast<int64_t>(matched_len) - static_cast<int64_t>(local_matched_len);
    if (need_block_num <= 0) {
        return;
    }
    auto [success, resource] = malloc(SimpleMallocInfo(-1, static_cast<uint32_t>(need_block_num), true));
    if (!success) {
        RTP_LLM_LOG_WARNING(
            "prefix matched in dist kvcache but free block index not enough, need block num: %d, free block index len: %lu",
            need_block_num,
            freeBlockNums());
        return;
    }

    std::vector<int64_t> matched_cache_keys(cache_keys.begin(), cache_keys.begin() + matched_len);
    if (!dist_kvcache_->getForAllRank(
            matched_cache_keys, resource.block_id, local_matched_len, request_id, extra_metas)) {
        free(resource.block_id);
        return;
    }

    match_result.matched_len = matched_len;
    match_result.block_indices.insert(
        match_result.block_indices.end(), resource.block_id.begin(), resource.block_id.end());
}

bool CacheManager::putCacheForAllRank(const std::vector<int64_t>& cache_keys,
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

}  // namespace rtp_llm
