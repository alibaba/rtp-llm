#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

size_t BlockIds::blocksNum() const {
    return block_indices.size();
}

const BlockIndicesType& BlockIds::blocks() const {
    return block_indices;
}

const BlockIndicesType& BlockIds::kernelBlocks() const {
    return kernel_block_indices_;
}

size_t BlockIds::kernelBlocksPerKvBlock() const {
    return kernel_blocks_per_kv_block_;
}

BlockIdxType BlockIds::popBack() {
    RTP_LLM_CHECK(!block_indices.empty());
    const BlockIdxType val = block_indices.back();
    block_indices.pop_back();
    kernel_block_indices_.resize(block_indices.size() * kernel_blocks_per_kv_block_);
    return val;
}

void BlockIds::add(const BlockIndicesType& ids) {
    const size_t old_size = block_indices.size();
    block_indices.insert(block_indices.end(), ids.begin(), ids.end());
    kernel_block_indices_.resize((old_size + ids.size()) * kernel_blocks_per_kv_block_);
    for (size_t i = 0; i < ids.size(); ++i) {
        updateKernelSlotAt(old_size + i, ids[i]);
    }
}

void BlockIds::remove(const std::vector<size_t>& indices) {
    for (auto idx : indices) {
        RTP_LLM_CHECK(idx < block_indices.size());
        block_indices[idx] = NULL_BLOCK_IDX;
        updateKernelSlotAt(idx, NULL_BLOCK_IDX);
    }
}

void BlockIds::swap(size_t pos_a, size_t pos_b) {
    RTP_LLM_CHECK(pos_a < block_indices.size());
    RTP_LLM_CHECK(pos_b < block_indices.size());
    if (pos_a == pos_b) {
        return;
    }
    std::swap(block_indices[pos_a], block_indices[pos_b]);
    updateKernelSlotAt(pos_a, block_indices[pos_a]);
    updateKernelSlotAt(pos_b, block_indices[pos_b]);
}

void BlockIds::assign(const BlockIndicesType& new_block_indices) {
    block_indices = new_block_indices;
    syncKernelBlocks();
}

void BlockIds::assign(BlockIndicesType&& new_block_indices) {
    block_indices = std::move(new_block_indices);
    syncKernelBlocks();
}

void BlockIds::setAt(size_t pos, BlockIdxType val) {
    RTP_LLM_CHECK(pos < block_indices.size());
    block_indices[pos] = val;
    updateKernelSlotAt(pos, val);
}

void BlockIds::resize(size_t new_size, BlockIdxType value) {
    const size_t old_size = block_indices.size();
    block_indices.resize(new_size, value);
    kernel_block_indices_.resize(new_size * kernel_blocks_per_kv_block_);
    for (size_t i = old_size; i < new_size; ++i) {
        updateKernelSlotAt(i, value);
    }
}

void BlockIds::updateKernelSlotAt(size_t pos, BlockIdxType val) {
    const size_t bpk      = kernel_blocks_per_kv_block_;
    const size_t base_pos = pos * bpk;
    RTP_LLM_CHECK_WITH_INFO(base_pos + bpk <= kernel_block_indices_.size(),
                            "OOB: base_pos=%zu + bpk=%zu > kernel size=%zu (physical_blocks=%zu)",
                            base_pos,
                            bpk,
                            kernel_block_indices_.size(),
                            block_indices.size());
    if (isNullBlockIdx(val)) {
        for (size_t j = 0; j < bpk; ++j) {
            kernel_block_indices_[base_pos + j] = NULL_BLOCK_IDX;
        }
    } else {
        const BlockIdxType base = val * static_cast<BlockIdxType>(bpk);
        for (size_t j = 0; j < bpk; ++j) {
            kernel_block_indices_[base_pos + j] = base + static_cast<BlockIdxType>(j);
        }
    }
}

void BlockIds::syncKernelBlocks() {
    const size_t n   = block_indices.size();
    const size_t bpk = kernel_blocks_per_kv_block_;
    kernel_block_indices_.resize(n * bpk);
    for (size_t i = 0; i < n; ++i) {
        updateKernelSlotAt(i, block_indices[i]);
    }
}

void KVCacheResource::initGroups(int                                group_num,
                                 int                                layer_num,
                                 const std::vector<int>&            layer_to_group_id,
                                 size_t                             kernel_blocks_per_kv_block,
                                 const std::vector<CacheGroupType>& group_types) {
    group_block_ids.clear();
    layer_block_ids.clear();

    group_block_ids.reserve(static_cast<size_t>(group_num));
    for (int i = 0; i < group_num; i++) {
        const bool   is_full = group_types.empty() || group_types[static_cast<size_t>(i)] == CacheGroupType::FULL;
        const size_t group_kernel_blocks_per_kv_block = is_full ? kernel_blocks_per_kv_block : 1;
        auto         bid                              = std::make_shared<BlockIds>(group_kernel_blocks_per_kv_block);
        group_block_ids.push_back(std::move(bid));
    }

    if (!group_block_ids.empty()) {
        RTP_LLM_CHECK_WITH_INFO(layer_to_group_id.empty() || layer_to_group_id.size() >= static_cast<size_t>(layer_num),
                                "KVCacheResource::initGroups: layer_to_group_id size %zu < layer_num %d",
                                layer_to_group_id.size(),
                                layer_num);
        layer_block_ids.resize(layer_num);
        for (int i = 0; i < layer_num; ++i) {
            int gid = layer_to_group_id.empty() ? 0 : layer_to_group_id[i];
            RTP_LLM_CHECK_WITH_INFO(gid >= 0 && gid < group_num,
                                    "KVCacheResource::initGroups: invalid group id %d for layer %d (group_num=%d)",
                                    gid,
                                    i,
                                    group_num);
            layer_block_ids[i] = group_block_ids[gid];
        }
    }
}

void KVCacheResource::resizeBlocks(int reserver_blocks, int value) {
    for (auto& group : group_block_ids) {
        group->resize(reserver_blocks, value);
    }
}

int KVCacheResource::blocksNum(int group_id) const {
    RTP_LLM_CHECK(group_block_ids.size() > static_cast<size_t>(group_id));
    return static_cast<int>(group_block_ids[group_id]->blocksNum());
}

const BlockIndicesType& KVCacheResource::blocks(int group_id) const {
    RTP_LLM_CHECK(group_block_ids.size() > static_cast<size_t>(group_id));
    return group_block_ids[group_id]->blocks();
}

const BlockIndicesType& KVCacheResource::kernelBlocks(int group_id) const {
    RTP_LLM_CHECK(group_block_ids.size() > static_cast<size_t>(group_id));
    return group_block_ids[group_id]->kernelBlocks();
}

BlockIds& KVCacheResource::mutableBlockIds(int group_id) const {
    RTP_LLM_CHECK(group_block_ids.size() > static_cast<size_t>(group_id));
    return *group_block_ids[group_id];
}

int KVCacheResource::groupNums() const {
    return static_cast<int>(group_block_ids.size());
}

GroupBlockIds& KVCacheResource::groupBlocks() {
    return group_block_ids;
}

const GroupBlockIds& KVCacheResource::groupBlocks() const {
    return group_block_ids;
}

const LayerBlockIds& KVCacheResource::layerBlocks() const {
    return layer_block_ids;
}

CacheKeysType& KVCacheResource::cacheKeys() {
    return cache_keys;
}

const CacheKeysType& KVCacheResource::cacheKeys() const {
    return cache_keys;
}

size_t KVCacheResource::reuseBlockNum() const {
    return device_reuse_block_num_ + memory_reuse_block_num_ + remote_reuse_block_num_;
}

size_t KVCacheResource::deviceReuseBlockNum() const {
    return device_reuse_block_num_;
}

void KVCacheResource::setDeviceReuseBlockNum(size_t device_reuse_blocks_num) {
    device_reuse_block_num_ = device_reuse_blocks_num;
}

size_t KVCacheResource::memoryReuseBlockNum() const {
    return memory_reuse_block_num_;
}

void KVCacheResource::setMemoryReuseBlockNum(size_t memory_reuse_blocks_num) {
    memory_reuse_block_num_ = memory_reuse_blocks_num;
}

size_t KVCacheResource::remoteReuseBlockNum() const {
    return remote_reuse_block_num_;
}

void KVCacheResource::setRemoteReuseBlockNum(size_t remote_reuse_blocks_num) {
    remote_reuse_block_num_ = remote_reuse_blocks_num;
}

bool KVCacheResource::lastBlockAligned() const {
    return last_block_aligned_;
}

void KVCacheResource::setLastBlockAligned(bool last_block_aligned) {
    last_block_aligned_ = last_block_aligned;
}

std::string KVCacheResource::debugString() const {
    std::stringstream debug_string;
    const int         group_nums = static_cast<int>(group_block_ids.size());
    for (int group_id = 0; group_id < group_nums; group_id++) {
        debug_string << "group:[" << group_id << "], block:[";
        auto& block_indices = blocks(group_id);
        for (auto& block : block_indices) {
            debug_string << block << ", ";
        }
        debug_string << "], ";
    }

    return debug_string.str();
}

void KVCacheResource::swapBlocks(size_t group_id, size_t rhs, size_t lhs) {
    group_block_ids[group_id]->swap(rhs, lhs);
}

KVCacheResource KVCacheResource::deepCopy() const {
    KVCacheResource copy;
    copy.cache_keys              = cache_keys;
    copy.device_reuse_block_num_ = device_reuse_block_num_;
    copy.memory_reuse_block_num_ = memory_reuse_block_num_;
    copy.remote_reuse_block_num_ = remote_reuse_block_num_;
    copy.last_block_aligned_     = last_block_aligned_;

    // Deep copy group blocks (new BlockIds objects, not shared_ptr aliases)
    copy.group_block_ids.reserve(group_block_ids.size());
    for (const auto& group : group_block_ids) {
        copy.group_block_ids.push_back(std::make_shared<BlockIds>(*group));
    }

    // Rebuild layer→group pointer mapping
    copy.layer_block_ids.resize(layer_block_ids.size());
    for (size_t l = 0; l < layer_block_ids.size(); ++l) {
        for (size_t g = 0; g < group_block_ids.size(); ++g) {
            if (layer_block_ids[l].get() == group_block_ids[g].get()) {
                copy.layer_block_ids[l] = copy.group_block_ids[g];
                break;
            }
        }
    }
    return copy;
}

KVCacheResource KVCacheResource::prefixCopy(size_t n_blocks) const {
    KVCacheResource copy;
    // Slice cache_keys to the first n_blocks
    const size_t key_n = std::min(n_blocks, cache_keys.size());
    copy.cache_keys.assign(cache_keys.begin(), cache_keys.begin() + key_n);

    copy.device_reuse_block_num_ = device_reuse_block_num_;
    copy.memory_reuse_block_num_ = memory_reuse_block_num_;
    copy.remote_reuse_block_num_ = remote_reuse_block_num_;
    // Treat the trimmed tail as aligned; downstream dropLastPartialBlock will
    // see this flag and skip popping. We've already chosen the exact prefix.
    copy.last_block_aligned_ = true;

    // Per-group: copy only the first n_blocks of each group's block_indices.
    copy.group_block_ids.reserve(group_block_ids.size());
    for (const auto& group : group_block_ids) {
        auto         bid   = std::make_shared<BlockIds>(group->kernelBlocksPerKvBlock());
        const auto&  src   = group->blocks();
        const size_t blk_n = std::min(n_blocks, src.size());
        if (blk_n > 0) {
            BlockIndicesType slice(src.begin(), src.begin() + blk_n);
            bid->assign(std::move(slice));
        }
        copy.group_block_ids.push_back(std::move(bid));
    }
    // layer_block_ids intentionally left empty — insertIntoCache downstream
    // (SingleType/HybridType allocators) only reads cacheKeys()/blocks(gid)
    // and never walks the layer mapping.
    return copy;
}

}  // namespace rtp_llm
