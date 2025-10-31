#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"

using namespace std;

namespace rtp_llm {

KVCacheAllocator::KVCacheAllocator(const CacheConfig& config, rtp_llm::DeviceBase* device, AllocationType atype):
    config_(config), device_(device), atype_(atype) {
    RTP_LLM_LOG_INFO("kvcache allocator cache config: %s, atype: %s",
                     config.debugString().c_str(),
                     atype == AllocationType::DEVICE ? "DEVICE" : "HOST");
}

KVCacheAllocator::~KVCacheAllocator() {
    deregUserMr();
    cache_aligned_buffer_.reset();
}

bool KVCacheAllocator::init() {
    cache_aligned_buffer_ = device_->allocateBuffer({rtp_llm::TYPE_INT8, {config_.total_size}, atype_});
    cache_base_ptr_       = cache_aligned_buffer_->data();
    if (cache_aligned_buffer_ == nullptr || cache_base_ptr_ == nullptr) {
        RTP_LLM_LOG_ERROR("kvcache allocator allocate cache aligned buffer is null");
        return false;
    }

    // temp hack for mla, since other devices not impl bufMemset
    if (config_.use_mla) {
        device_->bufMemset(*cache_aligned_buffer_, 0);
    }

    if (config_.block_nums > 1) {  // for warmup
        initFreeBlock();
    }

    initKvCache();

    RTP_LLM_LOG_INFO("kvcache allocator init success");
    return true;
}

void KVCacheAllocator::initFreeBlock() {
    free_blocks_index_ = std::set<int>();
    // block 0 is reserved for tmp or padding use
    for (int i = 1; i < int(config_.block_nums); ++i) {
        free_blocks_index_.insert(i);
    }
    block_ref_counter_.init(config_.block_nums);
}

void KVCacheAllocator::initKvCache() {
    if (config_.use_mla) {
        initKvCacheMla();
    } else {
        initKvCacheNormal();
    }

    initKVCacheScale();
}

void KVCacheAllocator::initKVCacheScale() {
    if (config_.dtype == rtp_llm::DataType::TYPE_INT8) {
        kv_cache_.k_scale =
            std::make_unique<rtp_llm::Buffer>(rtp_llm::MemoryType::MEMORY_GPU,
                                              rtp_llm::DataType::TYPE_FP32,
                                              std::vector<size_t>{(size_t)config_.layer_num,
                                                                  (size_t)config_.block_nums,
                                                                  (size_t)config_.local_head_num_kv,
                                                                  (size_t)config_.seq_size_per_block},
                                              (int8_t*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes());
        kv_cache_.v_scale = std::make_unique<rtp_llm::Buffer>(rtp_llm::MemoryType::MEMORY_GPU,
                                                              rtp_llm::DataType::TYPE_FP32,
                                                              std::vector<size_t>{(size_t)config_.layer_num,
                                                                                  (size_t)config_.block_nums,
                                                                                  (size_t)config_.local_head_num_kv,
                                                                                  (size_t)config_.seq_size_per_block},
                                                              (int8_t*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes()
                                                                  + kv_cache_.k_scale->sizeBytes());
    }

    else if (config_.dtype == rtp_llm::DataType::TYPE_FP8_E4M3) {
        kv_cache_.k_scale =
            std::make_unique<rtp_llm::Buffer>(rtp_llm::MemoryType::MEMORY_GPU,
                                              rtp_llm::DataType::TYPE_FP32,
                                              std::vector<size_t>{(size_t)config_.layer_num,
                                                                  (size_t)config_.block_nums,
                                                                  (size_t)config_.local_head_num_kv,
                                                                  (size_t)config_.seq_size_per_block},
                                              (int8_t*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes());
        kv_cache_.v_scale = std::make_unique<rtp_llm::Buffer>(rtp_llm::MemoryType::MEMORY_GPU,
                                                              rtp_llm::DataType::TYPE_FP32,
                                                              std::vector<size_t>{(size_t)config_.layer_num,
                                                                                  (size_t)config_.block_nums,
                                                                                  (size_t)config_.local_head_num_kv,
                                                                                  (size_t)config_.seq_size_per_block},
                                                              (int8_t*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes()
                                                                  + kv_cache_.k_scale->sizeBytes());
        Buffer2torchTensor(kv_cache_.k_scale, false).fill_(1.0);
        Buffer2torchTensor(kv_cache_.v_scale, false).fill_(1.0);
    }
}

void KVCacheAllocator::initKvCacheMla() {
    RTP_LLM_LOG_INFO("init mla kv cache");
    kv_cache_.k_blocks =
        std::make_unique<rtp_llm::Buffer>(rtp_llm::MemoryType::MEMORY_GPU,
                                          config_.dtype,
                                          std::vector<size_t>{(size_t)config_.layer_num,
                                                              (size_t)config_.block_nums,
                                                              (size_t)config_.seq_size_per_block,
                                                              (size_t)config_.kv_lora_rank + config_.rope_head_dim},
                                          cache_base_ptr_);
    kv_cache_.v_blocks = std::make_unique<rtp_llm::Buffer>(
        rtp_llm::MemoryType::MEMORY_GPU,
        config_.dtype,
        std::vector<size_t>{
            (size_t)config_.layer_num, (size_t)config_.block_nums, (size_t)config_.seq_size_per_block, (size_t)0},
        (int8_t*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes());
// memset k_blocks and v_blocks for cuda or rocm
// since warmup produce nan maybe influence kvcache
#if defined(USING_ROCM) || defined(USING_CUDA)
    device_->bufMemset(*kv_cache_.k_blocks, 0);
    device_->bufMemset(*kv_cache_.v_blocks, 0);
#endif
}

void KVCacheAllocator::initKvCacheNormal() {
    RTP_LLM_LOG_INFO("init normal kv cache");
    kv_cache_.k_blocks = std::make_unique<rtp_llm::Buffer>(rtp_llm::MemoryType::MEMORY_GPU,
                                                           config_.dtype,
                                                           std::vector<size_t>{(size_t)config_.layer_num,
                                                                               (size_t)config_.block_nums,
                                                                               (size_t)2,
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
                                                                               (size_t)0},
                                                           (int8_t*)cache_base_ptr_ + kv_cache_.k_blocks->sizeBytes());
    // memset k_blocks and v_blocks
#if defined(USING_ROCM) || USING_CUDA
    device_->bufMemset(*kv_cache_.k_blocks, 0);
    device_->bufMemset(*kv_cache_.v_blocks, 0);
#endif
}

size_t KVCacheAllocator::totalBlocks() const {
    return config_.block_nums - 1;
}

size_t KVCacheAllocator::freeBlockNums() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return free_blocks_index_.size();
}

const KVCacheAllocator::KVCacheBuffer& KVCacheAllocator::kvCacheBuffer() const {
    return kv_cache_;
}

std::tuple<bool, KVCacheResource> KVCacheAllocator::malloc(const SimpleMallocInfo& malloc_info) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (free_blocks_index_.size() < static_cast<size_t>(malloc_info.block_nums)) {
        if (malloc_info.verbose) {
            std::string error_msg = "request " + std::to_string(malloc_info.request_id) + " failed to malloc "
                                    + std::to_string(malloc_info.block_nums) + " blocks, only "
                                    + std::to_string(free_blocks_index_.size()) + " blocks left";
            RTP_LLM_LOG_ERROR("%s", error_msg.c_str());
        }
        return {false, {{}}};
    } else {
        std::vector<int> result;
        result.reserve(malloc_info.block_nums);
        for (int i = 0; i < malloc_info.block_nums; ++i) {
            int block = *free_blocks_index_.begin();
            free_blocks_index_.erase(free_blocks_index_.begin());
            result.push_back(block);
        }
        block_ref_counter_.incrementRefCounter(result);
        return {true, {result}};
    }
}

void KVCacheAllocator::free(const std::vector<KVCacheResource>& resource) {
    for (const auto& kv_block : resource) {
        free(kv_block.block_id);
    }
}

void KVCacheAllocator::free(const std::vector<int>& block_indices) {
    std::lock_guard<std::mutex> guard(mutex_);
    block_ref_counter_.decrementRefCounter(block_indices);
    for (auto block : block_indices) {
        if (block_ref_counter_.getRefCounter(block) == 0) {
            free_blocks_index_.insert(block);
        }
    }
}

bool KVCacheAllocator::setKVBlockValue(int              block_index,
                                       int              layer_id,
                                       rtp_llm::Buffer& k_buffer,
                                       rtp_llm::Buffer& v_buffer) {
    // 检查block_index是否有效
    if (block_index < 0 || block_index >= config_.block_nums) {
        RTP_LLM_LOG_WARNING("Invalid block_index: %d, valid range: [0, %d)", block_index, config_.block_nums);
        return false;
    }

    // 检查layer_id是否有效
    if (layer_id < 0 || layer_id >= config_.layer_num) {
        RTP_LLM_LOG_WARNING("Invalid layer_id: %d, valid range: [0, %d)", layer_id, config_.layer_num);
        return false;
    }

    auto k_offset = config_.getKeyOffset(block_index, layer_id);
    auto v_offset = config_.getValueOffset(block_index, layer_id);
    auto k_shape  = config_.getKeyShape();
    auto v_shape  = config_.getValueShape();

    auto copyFunc = [&](rtp_llm::Buffer& src_buffer, rtp_llm::BufferPtr& dst_blocks, size_t offset, size_t shape) {
        if (shape == 0) {
            return;
        }
        auto dst_data   = (char*)dst_blocks->data() + offset;
        auto dst_buffer = Buffer(dst_blocks->where(), src_buffer.type(), {shape}, dst_data);
        device_->copy({dst_buffer, src_buffer});
    };

    copyFunc(k_buffer, kv_cache_.k_blocks, k_offset, k_shape);
    copyFunc(v_buffer, kv_cache_.v_blocks, v_offset, v_shape);

    return true;
}

bool KVCacheAllocator::setKVBlockValue(int block_index, rtp_llm::Buffer& k_buffer, rtp_llm::Buffer& v_buffer) {
    // 检查block_index是否有效
    if (block_index < 0 || block_index >= config_.block_nums) {
        RTP_LLM_LOG_WARNING("Invalid block_index: %d, valid range: [0, %d)", block_index, config_.block_nums);
        return false;
    }

    bool all_success = true;
    for (uint32_t layer_id = 0; layer_id < config_.layer_num; layer_id++) {
        auto layer_k_data   = (char*)(k_buffer.data()) + layer_id * config_.getKeyBlockStride();
        auto layer_k_buffer = Buffer(k_buffer.where(), k_buffer.type(), {config_.getKeyShape()}, layer_k_data);
        auto layer_v_data   = (char*)(v_buffer.data()) + layer_id * config_.getValueBlockStride();
        auto layer_v_buffer = Buffer(v_buffer.where(), v_buffer.type(), {config_.getValueShape()}, layer_v_data);
        if (!setKVBlockValue(block_index, layer_id, layer_k_buffer, layer_v_buffer)) {
            all_success = false;
        }
    }
    return all_success;
}

std::tuple<bool, rtp_llm::BufferPtr, rtp_llm::BufferPtr> KVCacheAllocator::getKVBlockValue(int block_index,
                                                                                           int layer_id) {
    // 检查block_index是否有效
    if (block_index < 0 || block_index >= config_.block_nums) {
        RTP_LLM_LOG_WARNING("Invalid block_index: %d, valid range: [0, %d)", block_index, config_.block_nums);
        return {false, nullptr, nullptr};
    }

    // 检查layer_id是否有效
    if (layer_id < 0 || layer_id >= config_.layer_num) {
        RTP_LLM_LOG_WARNING("Invalid layer_id: %d, valid range: [0, %d)", layer_id, config_.layer_num);
        return {false, nullptr, nullptr};
    }

    auto k_offset = config_.getKeyOffset(block_index, layer_id);
    auto v_offset = config_.getValueOffset(block_index, layer_id);
    auto k_shape  = config_.getKeyShape();
    auto v_shape  = config_.getValueShape();

    auto kdst_buffer = device_->allocateBuffer({config_.dtype, {k_shape}, atype_});
    auto vdst_buffer = device_->allocateBuffer({config_.dtype, {v_shape}, atype_});

    auto copyFunc = [&](rtp_llm::BufferPtr& src_blocks, rtp_llm::BufferPtr& dst_buffer, size_t offset, size_t shape) {
        auto src_data   = (char*)(src_blocks->data()) + offset;
        auto src_buffer = Buffer(src_blocks->where(), config_.dtype, {shape}, src_data);
        device_->copy({*dst_buffer, src_buffer});
    };

    copyFunc(kv_cache_.k_blocks, kdst_buffer, k_offset, k_shape);
    copyFunc(kv_cache_.v_blocks, vdst_buffer, v_offset, v_shape);

    return {true, kdst_buffer, vdst_buffer};
}

std::tuple<bool, rtp_llm::BufferPtr, rtp_llm::BufferPtr> KVCacheAllocator::getKVBlockValue(int block_index) {
    // 检查block_index是否有效
    if (block_index < 0 || block_index >= config_.block_nums) {
        RTP_LLM_LOG_WARNING("Invalid block_index: %d, valid range: [0, %d)", block_index, config_.block_nums);
        return {false, nullptr, nullptr};
    }

    auto k_shape     = config_.getKeyShape();
    auto v_shape     = config_.getValueShape();
    auto kdst_buffer = device_->allocateBuffer({config_.dtype, {config_.layer_num, k_shape}, atype_});
    auto vdst_buffer = device_->allocateBuffer({config_.dtype, {config_.layer_num, v_shape}, atype_});

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

    return {true, kdst_buffer, vdst_buffer};
}

std::tuple<bool, rtp_llm::BufferPtr, rtp_llm::BufferPtr> KVCacheAllocator::getKVBlockValueRef(int block_index,
                                                                                              int layer_id) {
    if (block_index < 0 || block_index >= config_.block_nums) {
        RTP_LLM_LOG_WARNING("Invalid block_index: %d, valid range: [0, %d)", block_index, config_.block_nums);
        return {false, nullptr, nullptr};
    }

    if (layer_id < 0 || layer_id >= config_.layer_num) {
        RTP_LLM_LOG_WARNING("Invalid layer_id: %d, valid range: [0, %d)", layer_id, config_.layer_num);
        return {false, nullptr, nullptr};
    }

    auto k_offset = config_.getKeyOffset(block_index, layer_id);
    auto v_offset = config_.getValueOffset(block_index, layer_id);
    auto k_shape  = config_.getKeyShape();
    auto v_shape  = config_.getValueShape();

    std::shared_ptr<rtp_llm::Buffer> k_buffer(new rtp_llm::Buffer(
        kv_cache_.k_blocks->where(), config_.dtype, {k_shape}, (void*)((char*)kv_cache_.k_blocks->data() + k_offset)));
    std::shared_ptr<rtp_llm::Buffer> v_buffer(new rtp_llm::Buffer(
        kv_cache_.v_blocks->where(), config_.dtype, {v_shape}, (void*)((char*)kv_cache_.v_blocks->data() + v_offset)));
    return {true, k_buffer, v_buffer};
}

void KVCacheAllocator::blockCopy(int src_block_index, int dest_block_index) {
    BlockIdPair copy_mapping{src_block_index, dest_block_index};
    blockBatchCopy(&copy_mapping, &copy_mapping + 1);
}

void KVCacheAllocator::blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping) {
    blockBatchCopy(copy_mapping.data(), copy_mapping.data() + copy_mapping.size());
}

void KVCacheAllocator::blockBatchCopy(const Buffer& copy_mapping) {
    RTP_LLM_CHECK(copy_mapping.dim() == 2 && copy_mapping.shape()[1] == 2);
    const auto* begin_ptr = (const BlockIdPair*)copy_mapping.data();
    size_t      copy_num  = copy_mapping.shape()[0];
    blockBatchCopy(begin_ptr, begin_ptr + copy_num);
}

void KVCacheAllocator::blockBatchCopy(const BlockIdPair* begin_ptr, const BlockIdPair* end_ptr) {
    using CopyType = BatchCopyParams::CopyType;

    if (end_ptr == begin_ptr) {
        return;
    }

    auto& k_blocks = *kv_cache_.k_blocks;
    auto& v_blocks = *kv_cache_.v_blocks;
    auto* k_scale  = kv_cache_.k_scale.get();
    auto* v_scale  = kv_cache_.v_scale.get();

    BatchCopyParams copy_params;

    // reserve space for each copy type
    size_t copy_nums[CopyType::TYPE_SIZE] = {};

    const size_t copy_num = (end_ptr - begin_ptr) * config_.layer_num;

    auto k_copy_type = BatchCopyParams::get_copy_type(k_blocks.where(), k_blocks.where());
    copy_nums[k_copy_type] += copy_num;
    auto v_copy_type = BatchCopyParams::get_copy_type(v_blocks.where(), v_blocks.where());
    copy_nums[v_copy_type] += copy_num;
    CopyType k_scale_copy_type;
    if (k_scale != nullptr) {
        k_scale_copy_type = BatchCopyParams::get_copy_type(k_scale->where(), k_scale->where());
        copy_nums[k_scale_copy_type] += copy_num;
    }
    CopyType v_scale_copy_type;
    if (v_scale != nullptr) {
        v_scale_copy_type = BatchCopyParams::get_copy_type(v_scale->where(), v_scale->where());
        copy_nums[v_scale_copy_type] += copy_num;
    }

    for (size_t i = 0; i < CopyType::TYPE_SIZE; ++i) {
        copy_params.reserve(static_cast<CopyType>(i), copy_nums[i]);
    }

    // construct batch copy params
    const auto copy_blocks = [&](Buffer& buffer_blocks, size_t block_bytes, CopyType copy_type, auto get_offset) {
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

    // copy k scales
    if (k_scale != nullptr) {
        auto k_scale_copy_bytes = config_.getKVScaleBlockStride();
        copy_blocks(*k_scale, k_scale_copy_bytes, k_scale_copy_type, [&](int block_index, int layer_id) {
            return config_.getKVScaleOffset(block_index, layer_id);
        });
    }

    // copy v scales
    if (v_scale != nullptr) {
        auto v_scale_copy_bytes = config_.getKVScaleBlockStride();
        copy_blocks(*v_scale, v_scale_copy_bytes, v_scale_copy_type, [&](int block_index, int layer_id) {
            return config_.getKVScaleOffset(block_index, layer_id);
        });
    }

    device_->batchCopy(copy_params);
}

KVCacheAllocator::BlockAddrInfo KVCacheAllocator::convertIndexToAddr(int block_index, int layer_id) const {
    if (block_index < 0 || block_index >= config_.block_nums) {
        RTP_LLM_LOG_WARNING("Invalid block_index: %d, valid range: [0, %d)", block_index, config_.block_nums);
        return {nullptr, nullptr, nullptr, nullptr};
    }

    if (layer_id < 0 || layer_id >= config_.layer_num) {
        RTP_LLM_LOG_WARNING("Invalid layer_id: %d, valid range: [0, %d)", layer_id, config_.layer_num);
        return {nullptr, nullptr, nullptr, nullptr};
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

void KVCacheAllocator::regUserMr(size_t model_id) {
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

        auto k_block_size       = config_.getKBlockSize() / (size_t)config_.layer_num / (size_t)config_.block_nums;
        auto v_block_size       = config_.getVBlockSize() / (size_t)config_.layer_num / (size_t)config_.block_nums;
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

void KVCacheAllocator::deregUserMr() {
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

int64_t KVCacheAllocator::getMrCostTimeMs() const {
    return mr_cost_time_ms_;
}

const CacheConfig& KVCacheAllocator::cacheConfig() const {
    return config_;
}

void KVCacheAllocator::incrBlockRefCounter(const std::vector<int>& blocks) {
    block_ref_counter_.incrementRefCounter(blocks);
}

void KVCacheAllocator::decrBlockRefCounter(const std::vector<int>& blocks) {
    block_ref_counter_.decrementRefCounter(blocks);
}

const BlockRefCounter& KVCacheAllocator::blockRefCounter() const {
    return block_ref_counter_;
}

}  // namespace rtp_llm