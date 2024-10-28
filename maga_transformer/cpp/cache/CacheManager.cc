#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <unistd.h>
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "maga_transformer/cpp/common/fatal_util.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "maga_transformer/cpp/utils/StringUtil.h"
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

CacheManager::CacheManager(const CacheConfig&                 config,
                           ft::DeviceBase*                    device,
                           const kmonitor::MetricsReporterPtr metrics_reporter):
    config_(config),
    seq_size_per_block_(config.seq_size_per_block),
    device_(device),
    metrics_reporter_(metrics_reporter) {
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
            collector.kv_cache_item_num   = block_cache_.size();
            auto available_blocks         = availableBlockNums();
            collector.kv_cache_left_seq   = available_blocks * seq_size_per_block_;
            collector.kv_cache_used_ratio = 100.0 * (config_.block_nums - available_blocks) / config_.block_nums;
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
    available_blocks_ = config_.block_nums - 1;

    block_ref_counter_ = BlockRefCounter(config_.block_nums);
}

void CacheManager::allocateAndTpSync() {
    const auto properties = device_->getDeviceProperties();
    if (properties.tp_size > 1) {
        BufferPtr block_num_infos =
            device_->allocateBuffer({ft::DataType::TYPE_INT32, {properties.tp_size}, ft::AllocationType::HOST});
        auto block_num_ptr                = block_num_infos->data<int>();
        block_num_ptr[properties.tp_rank] = config_.block_nums;
        device_->allGather({{block_num_infos}});
        device_->syncCommunication(false);
        device_->syncAndCheck();
        config_.block_nums = *std::min_element(block_num_ptr, block_num_ptr + properties.tp_size);
    }
    cache_aligned_buffer_ =
        device_->allocateBuffer({ft::DataType::TYPE_INT8, {config_.block_size * config_.block_nums}});
    cache_base_ptr_ = cache_aligned_buffer_->data();
}

void CacheManager::initKvCache() {
    kv_cache_.k_blocks = std::make_unique<ft::Buffer>(ft::MemoryType::MEMORY_GPU,
                                                      config_.dtype,
                                                      std::vector<size_t>{(size_t)config_.layer_num,
                                                                          (size_t)config_.block_nums,
                                                                          (size_t)config_.local_head_num_kv,
                                                                          (size_t)config_.seq_size_per_block,
                                                                          (size_t)config_.size_per_head},
                                                      cache_base_ptr_);
    kv_cache_.v_blocks = std::make_unique<ft::Buffer>(ft::MemoryType::MEMORY_GPU,
                                                      config_.dtype,
                                                      std::vector<size_t>{(size_t)config_.layer_num,
                                                                          (size_t)config_.block_nums,
                                                                          (size_t)config_.local_head_num_kv,
                                                                          (size_t)config_.seq_size_per_block,
                                                                          (size_t)config_.size_per_head},
                                                      (int8_t*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes());
    if (config_.dtype == ft::DataType::TYPE_INT8) {
        kv_cache_.k_scale =
            std::make_unique<ft::Buffer>(ft::MemoryType::MEMORY_GPU,
                                         ft::DataType::TYPE_FP32,
                                         std::vector<size_t>{(size_t)config_.layer_num,
                                                             (size_t)config_.block_nums,
                                                             (size_t)config_.local_head_num_kv,
                                                             (size_t)config_.seq_size_per_block},
                                         (int8_t*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes() * 2);
        kv_cache_.v_scale = std::make_unique<ft::Buffer>(ft::MemoryType::MEMORY_GPU,
                                                         ft::DataType::TYPE_FP32,
                                                         std::vector<size_t>{(size_t)config_.layer_num,
                                                                             (size_t)config_.block_nums,
                                                                             (size_t)config_.local_head_num_kv,
                                                                             (size_t)config_.seq_size_per_block},
                                                         (int8_t*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes() * 2
                                                             + kv_cache_.k_scale->sizeBytes());
    }
#ifdef ENABLE_FP8
    else if (config_.dtype == ft::DataType::TYPE_FP8_E4M3) {
        kv_cache_.k_scale = std::make_unique<ft::Buffer>(
                ft::MemoryType::MEMORY_GPU,
                ft::DataType::TYPE_FP32,
                std::vector<size_t>{(size_t)config_.layer_num,
                    (size_t)config_.block_nums,
                    (size_t)config_.local_head_num_kv,
                    (size_t)config_.seq_size_per_block},
                (__nv_fp8_e4m3*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes() * 2);
        kv_cache_.v_scale = std::make_unique<ft::Buffer>(
                ft::MemoryType::MEMORY_GPU,
                ft::DataType::TYPE_FP32,
                std::vector<size_t>{(size_t)config_.layer_num,
                    (size_t)config_.block_nums,
                    (size_t)config_.local_head_num_kv,
                    (size_t)config_.seq_size_per_block},
                (__nv_fp8_e4m3*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes() * 2 + kv_cache_.k_scale->sizeBytes());
    }
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

KVCacheInfo CacheManager::getKVCacheInfo() const {
    // block 0 is reserved, so total block num need - 1
    return {availableBlockNums() * seq_size_per_block_, (config_.block_nums - 1) * seq_size_per_block_};
}

size_t CacheManager::cacheItemNum() const {
    return block_cache_.size();
}

const CacheManager::KVCacheBuffer& CacheManager::kvCacheBuffer() const {
    return kv_cache_;
}

CacheManager::MatchInfo CacheManager::mallocWithCache(const std::vector<int>& token_ids, const std::vector<std::vector<int>>& mm_bounds, bool need_loss) {
    return mallocWithCacheImpl(token_ids, mm_bounds, need_loss);
}

CacheManager::MatchInfo CacheManager::matchImpl(const std::vector<int>& token_ids, const std::vector<std::vector<int>>& mm_bounds) {
    auto match_result = block_cache_.match(token_ids);
    int cache_block_num = match_result.block_indices.size();
    int reuse_length    = std::min(match_result.matched_len, static_cast<size_t>(token_ids.size()) - 1);
    int reuse_block_num = reuse_length / config_.seq_size_per_block;
    // common length must large than reuse_length, when need calculate loss
    if ((!match_result.loss.empty()) && reuse_block_num && match_result.matched_len % config_.seq_size_per_block == 0) {
        reuse_block_num -= 1;
    }
    reuse_length        = reuse_block_num * config_.seq_size_per_block;
    for (int i = mm_bounds.size() - 1; i >= 0; --i) {
        auto& bound = mm_bounds[i];
        if (reuse_length > bound[0] && reuse_length < bound[0] + bound[1]) {
            reuse_length = bound[0] / config_.seq_size_per_block * config_.seq_size_per_block;
        }
    }
    reuse_block_num = reuse_length / config_.seq_size_per_block;

    FT_CHECK_WITH_INFO((reuse_block_num <= cache_block_num),
                       "reuse block nums[%d] is less than need block nums[%d]",
                       reuse_block_num,
                       cache_block_num);
    FT_CHECK_WITH_INFO((match_result.loss.empty() || match_result.loss.size() >= reuse_length),
                       "reuse loss nums [%d] is less than need loss nums[%d]",
                       match_result.loss.size(),
                       reuse_length);
    return {(size_t)reuse_length,
            vector<int>(match_result.block_indices.begin(), match_result.block_indices.begin() + reuse_block_num),
            vector<float>(match_result.loss.begin(), match_result.loss.begin() + std::min((int)match_result.loss.size(), reuse_length))};
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

CacheManager::MatchInfo CacheManager::mallocWithCacheImpl(const std::vector<int>& token_ids, const std::vector<std::vector<int>>& mm_bounds, bool need_loss) {
    auto match_info = matchImpl(token_ids, mm_bounds);
    if (match_info.loss.empty() && need_loss) {
        return {0, {}, {}};
    }
    block_ref_counter_.incrementRefCounter(match_info.cache_blocks);
    incrQueryRefCounter(match_info.cache_blocks);
    if (metrics_reporter_) {
        RtpLLMCacheReuseMetricsCollector collector;
        collector.kv_cache_reuse_length = match_info.reuse_length;
        metrics_reporter_->report<RtpLLMCacheReuseMetrics, RtpLLMCacheReuseMetricsCollector>(nullptr, &collector);
    }
    return match_info;
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
        incrQueryRefCounter(result);
        return {true, result};
    }
}

void CacheManager::reserveBlocks(int nums) {
    maybeFreeBlockFromCache(nums);
}

void CacheManager::free(const std::vector<KVCacheBlockAddr>& resource) {
    for (const auto& kv_block : resource) {
        free(kv_block.offset);
    }
}

void CacheManager::free(const std::vector<int>& block_indices) {
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
            // Avoid infinite loop
            break;
        }
        freeImpl(indices);
    }
}

void CacheManager::freeWithCache(const std::vector<int>&   block_indices,
                                 const std::vector<int>&   token_ids,
                                 const std::vector<float>& loss) {
    decrQueryRefCounter(block_indices);
    insertIntoCache(block_indices, token_ids, loss, false);
}

void CacheManager::insertResidentCache(const std::vector<int>& block_indices, const std::vector<int>& token_ids) {
    insertIntoCache(block_indices, token_ids, {}, true);
}

void CacheManager::insertIntoCache(const std::vector<int>&   block_indices,
                                   const std::vector<int>&   token_ids,
                                   const std::vector<float>& loss,
                                   bool                      is_resident) {
    if (token_ids.size() > 1) {
        size_t cache_len = token_ids.size() - 1;
        size_t block_len = std::min(block_indices.size(), cache_len / seq_size_per_block_);
        cache_len        = block_len * seq_size_per_block_;
        std::vector<int> indices =
            block_cache_.put(std::vector<int>(token_ids.begin(), token_ids.begin() + cache_len),
                             std::vector<int>(block_indices.begin(), block_indices.begin() + block_len),
                             loss.empty() ? loss : std::vector<float>(loss.begin(), loss.begin() + cache_len),
                             is_resident);
        freeImpl(indices);
        freeImpl(std::vector<int>(block_indices.begin() + block_len, block_indices.end()));
    } else {
        freeImpl(block_indices);
    }
}

void CacheManager::incrBlockRefCounter(const std::vector<int>& indices) {
    block_ref_counter_.incrementRefCounter(indices);
}

void CacheManager::setKVBlockValue(int kindex, int vindex, ft::BufferPtr& k_value, ft::BufferPtr& v_value) {
    auto layer_stride = config_.block_nums * config_.kv_block_stride;
    for (uint32_t layer_num = 0; layer_num < config_.layer_num; layer_num++) {

        auto copyFunc = [&](ft::BufferPtr& src_value, ft::BufferPtr& dst_blocks) {
            auto dst_data   = (char*)(dst_blocks->data()) + layer_num * layer_stride + kindex * config_.kv_block_stride;
            auto src_data   = (char*)(src_value->data()) + layer_num * config_.kv_block_stride;
            auto dst_buffer = Buffer(dst_blocks->where(),
                                     src_value->type(),
                                     {config_.kv_block_stride / ft::getTypeSize(config_.dtype)},
                                     dst_data);
            auto src_buffer = Buffer(src_value->where(),
                                     src_value->type(),
                                     {config_.kv_block_stride / ft::getTypeSize(config_.dtype)},
                                     src_data);
            device_->copy({dst_buffer, src_buffer});
        };

        copyFunc(k_value, kv_cache_.k_blocks);
        copyFunc(v_value, kv_cache_.v_blocks);
    }
}

std::tuple<ft::BufferPtr, ft::BufferPtr> CacheManager::getKVBlockValue(int block_index) {
    auto layer_stride = config_.block_nums * config_.kv_block_stride;
    auto kdst_buffer =
        device_->allocateBuffer({config_.dtype,
                                 {config_.layer_num, config_.kv_block_stride / ft::getTypeSize(config_.dtype)},
                                 ft::AllocationType::DEVICE});
    auto vdst_buffer =
        device_->allocateBuffer({config_.dtype,
                                 {config_.layer_num, config_.kv_block_stride / ft::getTypeSize(config_.dtype)},
                                 ft::AllocationType::DEVICE});
    for (uint32_t layer_num = 0; layer_num < config_.layer_num; layer_num++) {

        auto copyFunc = [&](ft::BufferPtr& src_blocks, ft::BufferPtr& dst_buffer) {
            auto src_data =
                (char*)(src_blocks->data()) + layer_num * layer_stride + block_index * config_.kv_block_stride;
            auto src_buffer = Buffer(src_blocks->where(),
                                     config_.dtype,
                                     {config_.kv_block_stride / ft::getTypeSize(config_.dtype)},
                                     src_data);
            device_->copy({dst_buffer->view(layer_num, 1)[0], src_buffer});
        };

        copyFunc(kv_cache_.k_blocks, kdst_buffer);
        copyFunc(kv_cache_.v_blocks, vdst_buffer);
    }
    return {kdst_buffer, vdst_buffer};
}

void CacheManager::blockCopy(int src_block_index, int dest_block_index) {
    auto layer_stride = config_.block_nums * config_.kv_block_stride;
    for (uint32_t layer_num = 0; layer_num < config_.layer_num; layer_num++) {
        auto copyFunc = [&](ft::BufferPtr& buffer_blocks) {
            auto dst_data =
                (char*)(buffer_blocks->data()) + layer_num * layer_stride + dest_block_index * config_.kv_block_stride;
            auto src_data =
                (char*)(buffer_blocks->data()) + layer_num * layer_stride + src_block_index * config_.kv_block_stride;
            auto dst_buffer = Buffer(buffer_blocks->where(),
                                     config_.dtype,
                                     {config_.kv_block_stride / ft::getTypeSize(config_.dtype)},
                                     dst_data);
            auto src_buffer = Buffer(buffer_blocks->where(),
                                     config_.dtype,
                                     {config_.kv_block_stride / ft::getTypeSize(config_.dtype)},
                                     src_data);
            device_->copy({dst_buffer, src_buffer});
        };

        // copy value
        copyFunc(kv_cache_.k_blocks);
        copyFunc(kv_cache_.v_blocks);
    }
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


// src_block_offset and target_block_offset has same shape.
void CacheManager::beamSearchKvUpdate(ft::BufferPtr src_block_offset,
                                      ft::BufferPtr target_block_offset)
{
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


}  // namespace rtp_llm
