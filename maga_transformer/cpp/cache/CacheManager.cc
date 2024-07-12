#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unistd.h>
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "maga_transformer/cpp/common/fatal_util.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/Types.h"

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

CacheManager::CacheManager(const CacheConfig& config, ft::DeviceBase* device,
                           const kmonitor::MetricsReporterPtr metrics_reporter):
    config_(config),
    seq_size_per_block_(config.seq_size_per_block),
    device_(device),
    metrics_reporter_(metrics_reporter)
{
    FT_LOG_INFO("cache config: %s", config.debugString().c_str());
    allocateAndTpSync();
    FT_LOG_INFO("block nums is %d after tp sync", config_.block_nums);
    initFreeBlock();
    initKvCache();
    if (metrics_reporter_) {
        metrics_reporter_thread_ = std::thread(&CacheManager::reportMetricsLoop, this);
    }
}

CacheManager::~CacheManager() {
    stop_ = true;
    if (metrics_reporter_thread_.joinable()) {
        metrics_reporter_thread_.join();
    }
    cache_aligned_buffer_.reset();
}

void CacheManager::reportMetricsLoop() {
    while (!stop_) {
        if (metrics_reporter_) {
            RtpLLMCacheMetricsCollector collector;
            collector.kv_cache_item_num = block_cache_.size();
            collector.kv_cache_left_seq = freeBlockNums() * seq_size_per_block_;
            collector.kv_cache_used_ratio = 100.0 * (config_.block_nums - freeBlockNums()) / config_.block_nums;
            metrics_reporter_->report<RtpLLMCacheMetrics, RtpLLMCacheMetricsCollector>(nullptr, &collector);
            std::this_thread::sleep_for(std::chrono::seconds(1)); // 1s
        }
    }
}

void CacheManager::initFreeBlock() {
    free_blocks_index_ = std::set<int>();
    // block 0 is reserved for tmp or padding use
    for (int i = 1; i < config_.block_nums; ++i) {
        free_blocks_index_.insert(i);
    }

    block_ref_counter_ = BlockRefCounter(config_.block_nums);
    block_cache_       = BlockCache();
}

void CacheManager::allocateAndTpSync() {
    const auto properties = device_->getDeviceProperties();
    if (properties.tp_size > 1) {
        BufferPtr block_num_infos = device_->allocateBuffer({ft::DataType::TYPE_INT32, {properties.tp_size}, ft::AllocationType::HOST});
        auto block_num_ptr = block_num_infos->data<int>();
        block_num_ptr[properties.tp_rank] = config_.block_nums;
        device_->allGather({{block_num_infos}});
        device_->syncCommunication(false);
        device_->syncAndCheck();
        config_.block_nums = *std::min_element(block_num_ptr, block_num_ptr + properties.tp_size);
    }
    cache_aligned_buffer_ = device_->allocateBuffer({ft::DataType::TYPE_INT8, {config_.block_size * config_.block_nums}});
    cache_base_ptr_ = cache_aligned_buffer_->data();
}

void CacheManager::initKvCache() {
    kv_cache_.k_blocks = std::make_unique<ft::Buffer>(
            ft::MemoryType::MEMORY_GPU,
            config_.dtype,
            std::vector<size_t>{(size_t)config_.layer_num,
                (size_t)config_.block_nums,
                (size_t)config_.local_head_num_kv,
                (size_t)config_.seq_size_per_block,
                (size_t)config_.size_per_head},
            cache_base_ptr_);
    kv_cache_.v_blocks = std::make_unique<ft::Buffer>(
        ft::MemoryType::MEMORY_GPU,
        config_.dtype,
        std::vector<size_t>{(size_t)config_.layer_num,
            (size_t)config_.block_nums,
            (size_t)config_.local_head_num_kv,
            (size_t)config_.seq_size_per_block,
            (size_t)config_.size_per_head},
        (int8_t*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes());
    if (config_.dtype == ft::DataType::TYPE_INT8) {
        kv_cache_.k_scale = std::make_unique<ft::Buffer>(
                ft::MemoryType::MEMORY_GPU,
                ft::DataType::TYPE_FP32,
                std::vector<size_t>{(size_t)config_.layer_num,
                    (size_t)config_.block_nums,
                    (size_t)config_.local_head_num_kv,
                    (size_t)config_.seq_size_per_block},
                (int8_t*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes() * 2);
        kv_cache_.v_scale = std::make_unique<ft::Buffer>(
                ft::MemoryType::MEMORY_GPU,
                ft::DataType::TYPE_FP32,
                std::vector<size_t>{(size_t)config_.layer_num,
                    (size_t)config_.block_nums,
                    (size_t)config_.local_head_num_kv,
                    (size_t)config_.seq_size_per_block},
                (int8_t*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes() * 2 + kv_cache_.k_scale->sizeBytes());
    }
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

KVCacheInfo CacheManager::getKVCacheInfo() const {
    // block 0 is reserved, so total block num need - 1
    return {freeBlockNums() * seq_size_per_block_, (config_.block_nums - 1) * seq_size_per_block_};
}

size_t CacheManager::cacheItemNum() const {
    return block_cache_.size();
}

const CacheManager::KVCacheBuffer& CacheManager::kvCacheBuffer() const {
    return kv_cache_;
}

std::tuple<bool, KVCacheBlockAddr, int> CacheManager::mallocWithCache(int                     want_block_nums,
                                                                      const std::vector<int>& token_ids) {
    auto [success, block_indices, reuse_length] = mallocWithCacheImpl(want_block_nums, token_ids);
    return {success, {block_indices}, reuse_length};
}

std::tuple<bool, std::vector<int>, int> CacheManager::mallocWithCacheImpl(int                     want_block_nums,
                                                                          const std::vector<int>& token_ids) {
    auto [cache_blocks, common_length] = block_cache_.match(token_ids);

    int cache_block_num  = cache_blocks.size();
    int reuse_length     = std::min(common_length, static_cast<size_t>(token_ids.size()) - 1);
    int old_reuse_length = reuse_length;
    int reuse_block_num  = reuse_length / config_.seq_size_per_block;
    reuse_length         = reuse_block_num * config_.seq_size_per_block;

    if (reuse_block_num > want_block_nums || reuse_block_num > cache_block_num) {
        FT_LOG_ERROR("reuse_block_num[%d] should not be greater than want_block_nums[%d], "
                    "and reuse_block_num[%d] should not be greater than cache_block_num[%d]",
                    reuse_block_num, want_block_nums, reuse_block_num, cache_block_num);
        return {false, {}, 0};
    }
    FT_CHECK_WITH_INFO(
        (reuse_block_num <= want_block_nums),
        "reuse block nums[%d] is less than need block nums[%d]", reuse_block_num, want_block_nums);

    FT_CHECK_WITH_INFO(
        (reuse_block_num <= cache_block_num),
        "reuse block nums[%d] is less than need block nums[%d]", reuse_block_num, cache_block_num);

    std::vector<int> reuse_blocks(cache_blocks.begin(), cache_blocks.begin() + reuse_block_num);
    block_ref_counter_.incrementRefCounter(reuse_blocks);
    maybeFreeBlockFromCache(want_block_nums - reuse_block_num);

    auto [success, new_blocks] = mallocImpl(want_block_nums - reuse_block_num);
    if (success) {
        reuse_blocks.insert(reuse_blocks.end(), new_blocks.begin(), new_blocks.end());
        if (metrics_reporter_) {
            RtpLLMCacheReuseMetricsCollector collector;
            collector.kv_cache_reuse_length = reuse_length;
            metrics_reporter_->report<RtpLLMCacheReuseMetrics, RtpLLMCacheReuseMetricsCollector>(nullptr, &collector);
        }

        return {true, reuse_blocks, reuse_length};
    } else {
        free(reuse_blocks);
        return {false, {}, 0};
    }
}

std::tuple<bool, KVCacheBlockAddr> CacheManager::malloc(int nums) {
    auto [success, block_indices] = mallocIndex(nums);
    return {success, {block_indices}};
}

std::tuple<bool, std::vector<int>> CacheManager::mallocIndex(int nums) {
    maybeFreeBlockFromCache(nums);
    return mallocImpl(nums);
}

std::tuple<bool, std::vector<int>> CacheManager::mallocImpl(int nums) {
    if (free_blocks_index_.size() < static_cast<size_t>(nums)) {
        std::string error_msg = "Failed to malloc " + std::to_string(nums) + " blocks, only "
                                + std::to_string(free_blocks_index_.size()) + " blocks left";
        FT_LOG_ERROR("%s", error_msg.c_str());
        return {false, {}};
    } else {
        std::vector<int> result;
        result.reserve(nums);
        for (int i = 0; i < nums; ++i) {
            int block = *free_blocks_index_.begin();
            free_blocks_index_.erase(free_blocks_index_.begin());
            result.push_back(block);
        }
        block_ref_counter_.incrementRefCounter(result);
        return {true, result};
    }
}

void CacheManager::reserveBlocks(int nums) {
    maybeFreeBlockFromCache(nums);
}

void CacheManager::free(const std::vector<int>& block_indices) {
    block_ref_counter_.decrementRefCounter(block_indices);
    for (auto block : block_indices) {
        int ref_count = block_ref_counter_.getRefCounter(block);
        if (ref_count == 0) {
            free_blocks_index_.insert(block);
        }
    }
}

void CacheManager::maybeFreeBlockFromCache(int nums) {
    while (freeBlockNums() < nums && !block_cache_.empty()) {
        std::vector<int> indices = block_cache_.pop();
        if (indices.empty()) {
            // Avoid infinite loop
            break;
        }
        free(indices);
    }
}

void CacheManager::free(const std::vector<KVCacheBlockAddr>& resource) {
    for (const auto& kv_block : resource) {
        free(kv_block.offset);
    }
}

void CacheManager::freeWithCache(const std::vector<int>& block_indices,
                                 const std::vector<int>& token_ids) {
    insertIntoCache(block_indices, token_ids, false);
}

void CacheManager::insertResidentCache(const std::vector<int>& block_indices, const std::vector<int>& token_ids) {
    insertIntoCache(block_indices, token_ids, true);
}

void CacheManager::insertIntoCache(const std::vector<int>& block_indices,
                                   const std::vector<int>& token_ids,
                                   bool                    is_resident) {
    if (token_ids.size() > 1) {
        size_t                  cache_len   = token_ids.size() - 1;
        size_t                  block_len   = std::min(block_indices.size(), cache_len / seq_size_per_block_);
        std::vector<int>        indices =
            block_cache_.put(std::vector<int>(token_ids.begin(), token_ids.begin() + cache_len),
                             std::vector<int>(block_indices.begin(), block_indices.begin() + block_len),
                             is_resident);
        free(indices);
        free(std::vector<int>(block_indices.begin() + block_len, block_indices.end()));
    } else {
        free(block_indices);
    }
}

void CacheManager::incrBlockRefCounter(const std::vector<int>& indices) {
    block_ref_counter_.incrementRefCounter(indices);
}

void CacheManager::setKVBlockValue(int kindex, int vindex, ft::BufferPtr& k_value, ft::BufferPtr& v_value) {
    auto layer_stride = config_.block_nums * config_.kv_block_stride;
    for (uint32_t layer_num = 0; layer_num < config_.layer_num; layer_num++) {
        // k
        auto kdst = (int8_t*)kv_cache_.k_blocks->data() + layer_num * layer_stride + kindex * config_.kv_block_stride;
        auto ksrc = (int8_t*)k_value->data() + layer_num * config_.kv_block_stride;
        auto kdst_buffer = Buffer(
            kv_cache_.k_blocks->where(), k_value->type(), {config_.kv_block_stride/ft::getTypeSize(config_.dtype)}, kdst);
        auto ksrc_buffer = Buffer(
            k_value->where(), k_value->type(), {config_.kv_block_stride/ft::getTypeSize(config_.dtype)}, ksrc);
        device_->copy({kdst_buffer, ksrc_buffer});
        // v
        auto vdst = (int8_t*)kv_cache_.v_blocks->data() + layer_num * layer_stride + vindex * config_.kv_block_stride;
        auto vsrc = (int8_t*)v_value->data() + layer_num * config_.kv_block_stride;
        auto vdst_buffer = Buffer(
            kv_cache_.v_blocks->where(), v_value->type(), {config_.kv_block_stride/ft::getTypeSize(config_.dtype)}, vdst);
        auto vsrc_buffer = Buffer(
            v_value->where(), v_value->type(), {config_.kv_block_stride/ft::getTypeSize(config_.dtype)}, vsrc);
        device_->copy({vdst_buffer, vsrc_buffer});
    }
}

void CacheManager::blockCopy(int src_block_index, int dest_block_index) {
    // kv_cache_.k_blocks.index({torch::indexing::Slice(), dest_block_index, torch::indexing::Ellipsis}) =
    //     kv_cache_.k_blocks.index({torch::indexing::Slice(), src_block_index, torch::indexing::Ellipsis});
    // kv_cache_.v_blocks.index({torch::indexing::Slice(), dest_block_index, torch::indexing::Ellipsis}) =
    //     kv_cache_.v_blocks.index({torch::indexing::Slice(), src_block_index, torch::indexing::Ellipsis});
}

void CacheManager::copyKvCacheFromSeqIdxs(const std::vector<int>& block_indice_list,
                                          const std::vector<int>& src_index,
                                          const std::vector<int>& tgt_index) {
    if (src_index.size() != tgt_index.size()) {
        RAISE_FATAL_ERROR(std::string("src and tgt length should equal"));
    }
    std::vector<SeqPosition> src_seq_positions;
    std::vector<SeqPosition> tgt_seq_positions;

    for (size_t i = 0; i < src_index.size(); ++i) {
        src_seq_positions.push_back(getSeqPosition(block_indice_list, src_index[i]));
        tgt_seq_positions.push_back(getSeqPosition(block_indice_list, tgt_index[i]));
    }

    for (size_t i = 0; i < src_seq_positions.size(); ++i) {
        copyKvCacheFromSeqPosition(src_seq_positions[i], tgt_seq_positions[i]);
    }
}

CacheManager::SeqPosition CacheManager::getSeqPosition(const std::vector<int>& block_indice_list, int idx) {
    int block_idx = idx / seq_size_per_block_;
    if (block_idx >= static_cast<int>(block_indice_list.size())) {
        RAISE_FATAL_ERROR(std::string("block idx should not >= len(block_indice_list)"));
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

}  // namespace rtp_llm
