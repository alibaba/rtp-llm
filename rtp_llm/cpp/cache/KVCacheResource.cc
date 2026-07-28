#include "rtp_llm/cpp/cache/KVCacheResource.h"

#include <algorithm>

#include "rtp_llm/cpp/cache/CacheTopology.h"

namespace rtp_llm {

void KVCacheResource::initGroups(std::shared_ptr<const CacheTopology> topology) {
    RTP_LLM_CHECK_WITH_INFO(topology != nullptr, "KVCacheResource::initGroups requires a topology");
    tag_to_group_id_.clear();
    layer_group_tags_.clear();
    group_block_ids.clear();
    layer_group_block_ids.clear();

    const auto& groups = topology->groups();
    group_block_ids.reserve(groups.size());
    for (size_t group_id = 0; group_id < groups.size(); ++group_id) {
        const auto& group = groups[group_id];
        tag_to_group_id_.emplace(group.tag, static_cast<int>(group_id));

        const size_t blocks_per_kv_block = group.seq_size_per_block / group.kernel_seq_size_per_block;
        const size_t stored_blocks_per_kv_block =
            group.policy.group_type == CacheGroupType::FULL ? std::max<size_t>(1, blocks_per_kv_block) : 1;
        group_block_ids.push_back(std::make_shared<BlockIds>(stored_blocks_per_kv_block));
    }

    const auto& layers = topology->layers();
    layer_group_tags_.reserve(layers.size());
    layer_group_block_ids.resize(layers.size());
    for (const auto& layer : layers) {
        layer_group_tags_.push_back(layer.group_tags);
        auto& group_blocks = layer_group_block_ids[static_cast<size_t>(layer.layer_id)];
        group_blocks.assign(groups.size(), nullptr);
        for (const auto& tag : layer.group_tags) {
            const auto group_id    = topology->groupIdForTag(tag);
            group_blocks[group_id] = group_block_ids[group_id];
        }
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
    for (auto& group : group_block_ids) {
        group->resize(reserver_blocks, value);
    }
}

int KVCacheResource::blocksNum(int group_id) const {
    RTP_LLM_CHECK(group_block_ids.size() > static_cast<size_t>(group_id));
    return static_cast<int>(group_block_ids[group_id]->blocksNum());
}

int KVCacheResource::blocksNum(std::string_view tag) const {
    return blocksNum(groupIdForTag(tag));
}

const BlockIndicesType& KVCacheResource::blocks(int group_id) const {
    RTP_LLM_CHECK(group_block_ids.size() > static_cast<size_t>(group_id));
    return group_block_ids[group_id]->blocks();
}

const BlockIndicesType& KVCacheResource::blocks(std::string_view tag) const {
    return blocks(groupIdForTag(tag));
}

const BlockIndicesType& KVCacheResource::blocks(int layer_id, int group_id) const {
    return mutableBlockIds(layer_id, group_id).blocks();
}

const BlockIndicesType& KVCacheResource::blocksForLayer(int layer_id, std::string_view tag) const {
    return mutableBlockIdsForLayer(layer_id, tag).blocks();
}

const BlockIndicesType& KVCacheResource::kernelBlocks(int group_id) const {
    RTP_LLM_CHECK(group_block_ids.size() > static_cast<size_t>(group_id));
    return group_block_ids[group_id]->kernelBlocks();
}

const BlockIndicesType& KVCacheResource::kernelBlocks(std::string_view tag) const {
    return kernelBlocks(groupIdForTag(tag));
}

const BlockIndicesType& KVCacheResource::kernelBlocks(int layer_id, int group_id) const {
    return mutableBlockIds(layer_id, group_id).kernelBlocks();
}

const BlockIndicesType& KVCacheResource::kernelBlocksForLayer(int layer_id, std::string_view tag) const {
    return mutableBlockIdsForLayer(layer_id, tag).kernelBlocks();
}

BlockIds& KVCacheResource::mutableBlockIds(int group_id) const {
    RTP_LLM_CHECK(group_block_ids.size() > static_cast<size_t>(group_id));
    return *group_block_ids[group_id];
}

BlockIds& KVCacheResource::mutableBlockIds(std::string_view tag) const {
    return mutableBlockIds(groupIdForTag(tag));
}

BlockIds& KVCacheResource::mutableBlockIds(int layer_id, int group_id) const {
    RTP_LLM_CHECK(static_cast<size_t>(layer_id) < layer_group_block_ids.size());
    RTP_LLM_CHECK(static_cast<size_t>(group_id) < layer_group_block_ids[static_cast<size_t>(layer_id)].size());
    auto block_ids = layer_group_block_ids[static_cast<size_t>(layer_id)][static_cast<size_t>(group_id)];
    RTP_LLM_CHECK_WITH_INFO(
        block_ids != nullptr, "KVCacheResource: missing block ids for layer %d group_id %d", layer_id, group_id);
    return *block_ids;
}

BlockIds& KVCacheResource::mutableBlockIdsForLayer(int layer_id, std::string_view tag) const {
    const int group_id = groupIdForLayerTag(layer_id, tag);
    return mutableBlockIds(layer_id, group_id);
}

const BlockIds& KVCacheResource::blockIds(std::string_view tag) const {
    return mutableBlockIds(tag);
}

const BlockIds& KVCacheResource::blockIdsForLayer(int layer_id, std::string_view tag) const {
    return mutableBlockIdsForLayer(layer_id, tag);
}

int KVCacheResource::groupIdForTag(std::string_view tag) const {
    const auto value = std::string(tag);
    const auto it    = tag_to_group_id_.find(value);
    RTP_LLM_CHECK_WITH_INFO(it != tag_to_group_id_.end(), "KVCacheResource missing tag=%s", value.c_str());
    return it->second;
}

int KVCacheResource::groupIdForLayerTag(int layer_id, std::string_view tag) const {
    const auto& tags  = groupTagsForLayer(layer_id);
    const auto  value = std::string(tag);
    RTP_LLM_CHECK_WITH_INFO(std::find(tags.begin(), tags.end(), value) != tags.end(),
                            "KVCacheResource layer=%d does not own tag=%s",
                            layer_id,
                            value.c_str());
    return groupIdForTag(tag);
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

bool KVCacheResource::hasOneGroupPerLayer() const {
    return std::all_of(
        layer_group_tags_.begin(), layer_group_tags_.end(), [](const auto& tags) { return tags.size() == 1; });
}

int KVCacheResource::layerNum() const {
    return static_cast<int>(layer_group_tags_.size());
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

LayerBlockIds KVCacheResource::layerBlocks() const {
    RTP_LLM_CHECK_WITH_INFO(hasOneGroupPerLayer(),
                            "KVCacheResource::layerBlocks is a deprecated single-group-per-layer projection; "
                            "use blockIdsForLayer(layer, tag) for multi-group layers");
    LayerBlockIds layer_blocks;
    layer_blocks.reserve(layer_group_block_ids.size());
    for (size_t layer = 0; layer < layer_group_block_ids.size(); ++layer) {
        const auto&               group_blocks = layer_group_block_ids[layer];
        std::shared_ptr<BlockIds> selected_blocks;
        int                       selected_group_id = -1;
        int                       mapped_group_num  = 0;
        for (size_t group_id = 0; group_id < group_blocks.size(); ++group_id) {
            if (group_blocks[group_id] == nullptr) {
                continue;
            }
            selected_blocks   = group_blocks[group_id];
            selected_group_id = static_cast<int>(group_id);
            ++mapped_group_num;
        }
        RTP_LLM_CHECK_WITH_INFO(mapped_group_num == 1 && selected_blocks != nullptr,
                                "KVCacheResource::layerBlocks requires exactly one group per layer, layer=%zu "
                                "mapped_group_num=%d selected_group=%d group_num=%zu",
                                layer,
                                mapped_group_num,
                                selected_group_id,
                                group_blocks.size());
        layer_blocks.push_back(std::move(selected_blocks));
    }
    return layer_blocks;
}

const LayerAttnBlockIds& KVCacheResource::layerGroupBlocks() const {
    return layer_group_block_ids;
}

int KVCacheResource::groupId(int layer_id, int group_id) const {
    RTP_LLM_CHECK(static_cast<size_t>(layer_id) < layer_group_block_ids.size());
    if (group_id < 0 || static_cast<size_t>(group_id) >= layer_group_block_ids[static_cast<size_t>(layer_id)].size()
        || !layer_group_block_ids[static_cast<size_t>(layer_id)][static_cast<size_t>(group_id)]) {
        return -1;
    }
    return group_id;
}

CacheKeysType& KVCacheResource::cacheKeys() {
    return cache_keys;
}

const CacheKeysType& KVCacheResource::cacheKeys() const {
    return cache_keys;
}

void KVCacheResource::setCacheKeys(const CacheKeysType& keys) {
    cache_keys                   = keys;
    cache_keys_are_cp_canonical_ = false;
    rebuildLinearBlockDependencies();
}

void KVCacheResource::setCacheKeys(CacheKeysType&& keys) {
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

BlockDependenciesType& KVCacheResource::blockDependencies() {
    return block_dependencies;
}

const BlockDependenciesType& KVCacheResource::blockDependencies() const {
    return block_dependencies;
}

void KVCacheResource::setBlockDependencies(const BlockDependenciesType& dependencies) {
    block_dependencies = dependencies;
}

void KVCacheResource::setBlockDependencies(BlockDependenciesType&& dependencies) {
    block_dependencies = std::move(dependencies);
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

void KVCacheResource::ensureLinearBlockDependencies() {
    rebuildLinearBlockDependencies();
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

}  // namespace rtp_llm
