#include "rtp_llm/cpp/cache/KVCacheResource.h"

#include <algorithm>

#include "rtp_llm/cpp/cache/CacheTopology.h"

namespace rtp_llm {

void KVCacheResource::initGroups(std::shared_ptr<const CacheTopology> topology) {
    RTP_LLM_CHECK_WITH_INFO(topology != nullptr, "KVCacheResource::initGroups requires a topology");
    topology_ = std::move(topology);

    group_block_ids_.clear();
    for (const auto& group : topology_->groups()) {
        RTP_LLM_CHECK_WITH_INFO(
            group_block_ids_.emplace(group.tag, std::make_shared<BlockIds>(storedKernelBlocksPerKvBlock(group))).second,
            "KVCacheResource has duplicate tag=%s",
            group.tag.c_str());
    }
}

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
    if (pos_a >= block_indices.size() || pos_b >= block_indices.size()) {
        RTP_LLM_LOG_ERROR("BlockIds::swap: pos_a=%zu or pos_b=%zu is out of range, block_indices.size()=%zu",
                          pos_a,
                          pos_b,
                          block_indices.size());
        RTP_LLM_CHECK_WITH_INFO(false,
                                "BlockIds::swap: pos_a=%zu or pos_b=%zu is out of range, block_indices.size()=%zu",
                                pos_a,
                                pos_b,
                                block_indices.size());
    }

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

void KVCacheResource::resizeBlocks(int reserver_blocks, int value) {
    for (auto& [tag, block_ids] : group_block_ids_) {
        (void)tag;
        block_ids->resize(reserver_blocks, value);
    }
}

int KVCacheResource::blocksNum(std::string_view tag) const {
    return static_cast<int>(blockIds(tag).blocksNum());
}

const BlockIndicesType& KVCacheResource::blocks(std::string_view tag) const {
    return blockIds(tag).blocks();
}

const BlockIndicesType& KVCacheResource::blocksForLayer(int layer_id, std::string_view tag) const {
    return mutableBlockIdsForLayer(layer_id, tag).blocks();
}

const BlockIndicesType& KVCacheResource::kernelBlocks(std::string_view tag) const {
    return blockIds(tag).kernelBlocks();
}

const BlockIndicesType& KVCacheResource::kernelBlocksForLayer(int layer_id, std::string_view tag) const {
    return mutableBlockIdsForLayer(layer_id, tag).kernelBlocks();
}

BlockIds& KVCacheResource::mutableBlockIds(std::string_view tag) const {
    const auto it = group_block_ids_.find(tag);
    RTP_LLM_CHECK_WITH_INFO(it != group_block_ids_.end(), "KVCacheResource missing tag=%s", std::string(tag).c_str());
    return *it->second;
}

BlockIds& KVCacheResource::mutableBlockIdsForLayer(int layer_id, std::string_view tag) const {
    RTP_LLM_CHECK_WITH_INFO(layerContainsTag(layer_id, tag),
                            "KVCacheResource layer=%d does not own tag=%s",
                            layer_id,
                            std::string(tag).c_str());
    return mutableBlockIds(tag);
}

const BlockIds& KVCacheResource::blockIds(std::string_view tag) const {
    return mutableBlockIds(tag);
}

const BlockIds& KVCacheResource::blockIdsForLayer(int layer_id, std::string_view tag) const {
    return mutableBlockIdsForLayer(layer_id, tag);
}

bool KVCacheResource::layerContainsTag(int layer_id, std::string_view tag) const {
    const auto& tags  = groupTagsForLayer(layer_id);
    const auto  value = std::string(tag);
    return std::find(tags.begin(), tags.end(), value) != tags.end();
}

const std::vector<std::string>& KVCacheResource::groupTagsForLayer(int layer_id) const {
    RTP_LLM_CHECK_WITH_INFO(topology_ != nullptr, "KVCacheResource groups are not initialized");
    return topology_->layer(layer_id).group_tags;
}

int KVCacheResource::layerNum() const {
    RTP_LLM_CHECK_WITH_INFO(topology_ != nullptr, "KVCacheResource groups are not initialized");
    return static_cast<int>(topology_->layers().size());
}

int KVCacheResource::groupNums() const {
    return static_cast<int>(group_block_ids_.size());
}

bool KVCacheResource::groupsInitialized() const {
    return topology_ != nullptr;
}

const GroupBlockIds& KVCacheResource::groupBlocks() const {
    return group_block_ids_;
}

const CacheKeysType& KVCacheResource::cacheKeys() const {
    return cache_keys;
}

void KVCacheResource::setCacheKeysAndBlockDependencies(CacheKeysType keys, BlockDependenciesType dependencies) {
    RTP_LLM_CHECK_WITH_INFO(keys.size() == dependencies.size(),
                            "cache timeline size mismatch: keys=%zu dependencies=%zu",
                            keys.size(),
                            dependencies.size());
    cache_keys                   = std::move(keys);
    block_dependencies           = std::move(dependencies);
    cache_keys_are_cp_canonical_ = false;
}

void KVCacheResource::setCacheKeys(CacheKeysType keys) {
    cache_keys                   = std::move(keys);
    cache_keys_are_cp_canonical_ = false;
    rebuildLinearBlockDependencies();
}

bool KVCacheResource::cacheKeysAreCpCanonical() const {
    return cache_keys_are_cp_canonical_;
}

void KVCacheResource::setCacheKeysAreCpCanonical(bool cache_keys_are_cp_canonical) {
    cache_keys_are_cp_canonical_ = cache_keys_are_cp_canonical;
}

void KVCacheResource::rebuildLinearBlockDependencies() {
    block_dependencies.clear();
    block_dependencies.reserve(cache_keys.size());
    for (size_t i = 0; i < cache_keys.size(); ++i) {
        BlockDependency dependency;
        dependency.ordinal = static_cast<uint32_t>(i);
        if (i > 0) {
            dependency.has_parent = true;
            dependency.parent_key = cache_keys[i - 1];
        }
        block_dependencies.push_back(dependency);
    }
}

void KVCacheResource::appendCacheKey(CacheKeyType key) {
    BlockDependency dependency;
    dependency.ordinal = static_cast<uint32_t>(cache_keys.size());
    if (!cache_keys.empty()) {
        dependency.has_parent = true;
        dependency.parent_key = cache_keys.back();
    }
    cache_keys.push_back(key);
    block_dependencies.push_back(dependency);
}

void KVCacheResource::popBackCacheKey() {
    if (cache_keys.empty()) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(block_dependencies.size() == cache_keys.size(),
                            "cache key/dependency timeline diverged before pop: keys=%zu dependencies=%zu",
                            cache_keys.size(),
                            block_dependencies.size());
    cache_keys.pop_back();
    block_dependencies.pop_back();
}

void KVCacheResource::clearCacheKeys() {
    cache_keys.clear();
    block_dependencies.clear();
}

const BlockDependenciesType& KVCacheResource::blockDependencies() const {
    return block_dependencies;
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
    for (const auto& [tag, block_ids] : group_block_ids_) {
        debug_string << "group:[" << tag << "], block:[";
        const auto& block_indices = block_ids->blocks();
        for (auto& block : block_indices) {
            debug_string << block << ", ";
        }
        debug_string << "], ";
    }

    return debug_string.str();
}

void KVCacheResource::swapBlocks(std::string_view tag, size_t rhs, size_t lhs) {
    mutableBlockIds(tag).swap(rhs, lhs);
}

}  // namespace rtp_llm
