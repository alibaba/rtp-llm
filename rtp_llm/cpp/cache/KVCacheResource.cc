#include "rtp_llm/cpp/cache/KVCacheResource.h"

#include <algorithm>

#include "rtp_llm/cpp/cache/CacheConfig.h"

namespace rtp_llm {

namespace {

BlockIndicesType projectKernelBlockIds(const BlockIndicesType& block_ids, size_t kernel_blocks_per_block) {
    RTP_LLM_CHECK_WITH_INFO(kernel_blocks_per_block > 0, "kernel blocks per block must be positive");
    BlockIndicesType kernel_block_ids;
    kernel_block_ids.reserve(block_ids.size() * kernel_blocks_per_block);
    for (const auto block_id : block_ids) {
        if (isNullBlockIdx(block_id)) {
            kernel_block_ids.insert(kernel_block_ids.end(), kernel_blocks_per_block, 0);
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(block_id >= 0, "invalid physical block id=%d", block_id);
        const BlockIdxType base = block_id * static_cast<BlockIdxType>(kernel_blocks_per_block);
        for (size_t i = 0; i < kernel_blocks_per_block; ++i) {
            kernel_block_ids.push_back(base + static_cast<BlockIdxType>(i));
        }
    }
    return kernel_block_ids;
}

}  // namespace

void KVCacheResource::initGroups(const CacheConfig& config) {

    layer_group_tags_.clear();
    blocks_by_group_.clear();

    const auto& groups = config.groups();

    for (const auto& group : groups) {
        RTP_LLM_CHECK_WITH_INFO(!group.tag.empty(), "KVCacheResource requires a non-empty cache group tag");

        RTP_LLM_CHECK_WITH_INFO(
            blocks_by_group_.emplace(group.tag, BlockIds(group.storedKernelBlocksPerKvBlock())).second,
            "KVCacheResource has duplicate tag=%s",
            group.tag.c_str());
    }

    const auto& layers = config.layers();
    layer_group_tags_.reserve(layers.size());
    for (size_t layer_id = 0; layer_id < layers.size(); ++layer_id) {
        const auto& layer = layers[layer_id];
        for (const auto& tag : layer) {
            config.groupForLayer(static_cast<int>(layer_id), tag);
        }
        layer_group_tags_.push_back(layer);
    }
}

BlockIds::BlockIds(size_t kernel_blocks_per_block): kernel_blocks_per_block_(kernel_blocks_per_block) {
    RTP_LLM_CHECK_WITH_INFO(kernel_blocks_per_block_ > 0, "kernel blocks per block must be positive");
}

size_t BlockIds::blocksNum() const {
    return block_ids_.size();
}
size_t BlockIds::kernelBlocksNum() const {
    return block_ids_.size() * kernel_blocks_per_block_;
}

const BlockIndicesType& BlockIds::blocks() const {
    return block_ids_;
}

BlockIndicesType BlockIds::kernelBlocks() const {
    return projectKernelBlockIds(block_ids_, kernel_blocks_per_block_);
}

size_t BlockIds::writeKernelBlocks(BlockIdxType* data, size_t num) const {
    const size_t required = block_ids_.size() * kernel_blocks_per_block_;
    RTP_LLM_CHECK_WITH_INFO(
        num >= required, "kernel block table capacity is insufficient: capacity=%zu required=%zu", num, required);
    if (required == 0) {
        return 0;
    }
    RTP_LLM_CHECK_WITH_INFO(data != nullptr, "kernel block table output buffer must not be null");

    size_t output_index = 0;
    for (const auto block_id : block_ids_) {
        if (isNullBlockIdx(block_id)) {
            std::fill_n(data + output_index, kernel_blocks_per_block_, 0);
            output_index += kernel_blocks_per_block_;
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(block_id >= 0, "invalid physical block id=%d", block_id);
        const BlockIdxType base = block_id * static_cast<BlockIdxType>(kernel_blocks_per_block_);
        for (size_t i = 0; i < kernel_blocks_per_block_; ++i) {
            data[output_index++] = base + static_cast<BlockIdxType>(i);
        }
    }
    return output_index;
}

BlockIdxType BlockIds::popBack() {
    RTP_LLM_CHECK(!block_ids_.empty());
    const BlockIdxType val = block_ids_.back();
    block_ids_.pop_back();
    return val;
}

void BlockIds::add(const BlockIndicesType& ids) {
    block_ids_.insert(block_ids_.end(), ids.begin(), ids.end());
}

void BlockIds::remove(const std::vector<size_t>& positions) {
    for (auto pos : positions) {
        RTP_LLM_CHECK(pos < block_ids_.size());
        block_ids_[pos] = NULL_BLOCK_IDX;
    }
}

void BlockIds::swap(size_t pos_a, size_t pos_b) {
    if (pos_a >= block_ids_.size() || pos_b >= block_ids_.size()) {
        RTP_LLM_LOG_ERROR("BlockIds::swap: pos_a=%zu or pos_b=%zu is out of range, block_ids.size()=%zu",
                          pos_a,
                          pos_b,
                          block_ids_.size());
        RTP_LLM_CHECK_WITH_INFO(false,
                                "BlockIds::swap: pos_a=%zu or pos_b=%zu is out of range, block_ids.size()=%zu",
                                pos_a,
                                pos_b,
                                block_ids_.size());
    }

    if (pos_a == pos_b) {
        return;
    }
    std::swap(block_ids_[pos_a], block_ids_[pos_b]);
}

void BlockIds::assign(const BlockIndicesType& new_block_ids) {
    block_ids_ = new_block_ids;
}

void BlockIds::assign(BlockIndicesType&& new_block_ids) {
    block_ids_ = std::move(new_block_ids);
}

void BlockIds::setAt(size_t pos, BlockIdxType val) {
    RTP_LLM_CHECK(pos < block_ids_.size());
    block_ids_[pos] = val;
}

void BlockIds::resize(size_t new_size, BlockIdxType value) {
    block_ids_.resize(new_size, value);
}

void KVCacheResource::resizeBlocks(int reserver_blocks, int value) {
    for (auto& [tag, block_ids] : blocks_by_group_) {
        (void)tag;
        block_ids.resize(reserver_blocks, value);
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

BlockIds& KVCacheResource::mutableBlockIds(std::string_view tag) const {
    const auto value = std::string(tag);
    const auto it    = blocks_by_group_.find(value);
    RTP_LLM_CHECK_WITH_INFO(it != blocks_by_group_.end(), "KVCacheResource missing tag=%s", value.c_str());
    return it->second;
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
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layer_group_tags_.size(),
                            "KVCacheResource invalid layer_id=%d size=%zu",
                            layer_id,
                            layer_group_tags_.size());
    return layer_group_tags_[static_cast<size_t>(layer_id)];
}

const std::string& KVCacheResource::soleGroupTagForLayer(int layer_id) const {
    const auto& tags = groupTagsForLayer(layer_id);
    RTP_LLM_CHECK_WITH_INFO(
        tags.size() == 1, "KVCacheResource layer=%d requires exactly one group, got %zu", layer_id, tags.size());
    return tags.front();
}

int KVCacheResource::layerNum() const {
    return static_cast<int>(layer_group_tags_.size());
}

int KVCacheResource::groupNums() const {
    return static_cast<int>(blocks_by_group_.size());
}

const std::map<std::string, BlockIds>& KVCacheResource::blocksByGroup() const {
    return blocks_by_group_;
}

bool KVCacheResource::layerOwnsTag(int layer_id, std::string_view tag) const {
    if (tag.empty() || blocks_by_group_.find(std::string(tag)) == blocks_by_group_.end()) {
        return false;
    }
    return layerContainsTag(layer_id, tag);
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
    BlockDependenciesType dependencies;
    dependencies.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        BlockDependency dependency;
        dependency.ordinal = static_cast<uint32_t>(i);
        if (i > 0) {
            dependency.has_parent = true;
            dependency.parent_key = keys[i - 1];
        }
        dependencies.push_back(dependency);
    }
    setCacheKeysAndBlockDependencies(std::move(keys), std::move(dependencies));
}

bool KVCacheResource::cacheKeysAreCpCanonical() const {
    return cache_keys_are_cp_canonical_;
}

void KVCacheResource::setCacheKeysAreCpCanonical(bool cache_keys_are_cp_canonical) {
    cache_keys_are_cp_canonical_ = cache_keys_are_cp_canonical;
}

void KVCacheResource::appendCacheKey(CacheKeyType key) {
    RTP_LLM_CHECK_WITH_INFO(block_dependencies.size() == cache_keys.size(),
                            "cache key/dependency timeline diverged before append: keys=%zu dependencies=%zu",
                            cache_keys.size(),
                            block_dependencies.size());
    BlockDependency dependency;
    dependency.ordinal = static_cast<uint32_t>(cache_keys.size());
    if (!cache_keys.empty()) {
        dependency.has_parent = true;
        dependency.parent_key = cache_keys.back();
    }
    const size_t new_size = cache_keys.size() + 1;
    if (cache_keys.capacity() < new_size || block_dependencies.capacity() < new_size) {
        CacheKeysType         new_keys         = cache_keys;
        BlockDependenciesType new_dependencies = block_dependencies;
        new_keys.push_back(key);
        new_dependencies.push_back(dependency);
        cache_keys.swap(new_keys);
        block_dependencies.swap(new_dependencies);
        return;
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
    for (const auto& [tag, block_ids] : blocks_by_group_) {
        debug_string << "group:[" << tag << "], block:[";
        const auto& block_indices = block_ids.blocks();
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
