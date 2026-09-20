#include "rtp_llm/cpp/cache/KVCacheResource.h"

#include <algorithm>

#include "rtp_llm/cpp/cache/CacheTopology.h"

namespace rtp_llm {

size_t GroupBlockIds::size() const {
    return rows_.size();
}

void GroupBlockIds::validate() const {
    RTP_LLM_CHECK_WITH_INFO(tag_to_index_.size() == rows_.size(), "GroupBlockIds tag/row count mismatch");
    std::vector<bool> seen(rows_.size(), false);
    for (const auto& [tag, index] : tag_to_index_) {
        RTP_LLM_CHECK_WITH_INFO(!tag.empty(), "GroupBlockIds requires non-empty tags");
        RTP_LLM_CHECK_WITH_INFO(index < rows_.size(), "GroupBlockIds row index out of range for tag=%s", tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(!seen[index], "GroupBlockIds duplicate row index for tag=%s", tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(rows_[index] != nullptr, "GroupBlockIds null holder for tag=%s", tag.c_str());
        seen[index] = true;
    }
}

const BlockIds& GroupBlockIds::blockIds(std::string_view tag) const {
    const auto value = std::string(tag);
    const auto it    = tag_to_index_.find(value);
    RTP_LLM_CHECK_WITH_INFO(it != tag_to_index_.end(), "GroupBlockIds missing tag=%s", value.c_str());
    RTP_LLM_CHECK_WITH_INFO(it->second < rows_.size() && rows_[it->second] != nullptr,
                            "GroupBlockIds invalid row for tag=%s",
                            value.c_str());
    return *rows_[it->second];
}

BlockIds& GroupBlockIds::mutableBlockIds(std::string_view tag) {
    return const_cast<BlockIds&>(blockIds(tag));
}

int GroupBlockIds::blocksNum(std::string_view tag) const {
    return static_cast<int>(blockIds(tag).blocksNum());
}

const BlockIndicesType& GroupBlockIds::blocks(std::string_view tag) const {
    return blockIds(tag).blocks();
}

const BlockIndicesType& GroupBlockIds::kernelBlocks(std::string_view tag) const {
    return blockIds(tag).kernelBlocks();
}

std::vector<std::string> GroupBlockIds::orderedTags() const {
    validate();
    std::vector<std::string> tags(rows_.size());
    for (const auto& [tag, index] : tag_to_index_) {
        tags[index] = tag;
    }
    return tags;
}

void KVCacheResource::initGroups(std::shared_ptr<const CacheTopology> topology) {
    RTP_LLM_CHECK_WITH_INFO(topology != nullptr, "KVCacheResource::initGroups requires a topology");
    GroupBlockIds candidate;
    const auto& groups = topology->groups();
    candidate.rows_.reserve(groups.size());
    candidate.tag_to_index_.reserve(groups.size());
    for (const auto& group : groups) {
        RTP_LLM_CHECK_WITH_INFO(candidate.tag_to_index_.emplace(group.tag, candidate.rows_.size()).second,
                                "KVCacheResource duplicate tag=%s",
                                group.tag.c_str());
        const size_t blocks_per_kv_block = group.kernelBlocksPerKvBlock();
        const size_t stored_blocks_per_kv_block =
            group.policy.group_type == CacheGroupType::FULL ? std::max<size_t>(1, blocks_per_kv_block) : 1;
        candidate.rows_.push_back(std::make_shared<BlockIds>(stored_blocks_per_kv_block));
    }
    candidate.validate();

    std::vector<std::vector<std::string>> layer_tags;
    layer_tags.reserve(topology->layers().size());
    for (const auto& layer : topology->layers()) {
        RTP_LLM_CHECK_WITH_INFO(layer.layer_id >= 0 && static_cast<size_t>(layer.layer_id) == layer_tags.size(),
                                "KVCacheResource invalid layer_id=%d",
                                layer.layer_id);
        for (const auto& tag : layer.group_tags) {
            (void)candidate.blockIds(tag);
        }
        layer_tags.push_back(layer.group_tags);
    }

    // All throwing construction/validation finished; publish both views together.
    group_block_ids.tag_to_index_.swap(candidate.tag_to_index_);
    group_block_ids.rows_.swap(candidate.rows_);
    layer_group_tags_.swap(layer_tags);
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
    for (auto& group : group_block_ids.rows_) {
        group->resize(reserver_blocks, value);
    }
}

int KVCacheResource::blocksNum(std::string_view tag) const {
    return group_block_ids.blocksNum(tag);
}

const BlockIndicesType& KVCacheResource::blocks(int group_id) const {
    RTP_LLM_CHECK(group_id >= 0 && group_block_ids.rows_.size() > static_cast<size_t>(group_id));
    return group_block_ids.rows_[static_cast<size_t>(group_id)]->blocks();
}

const BlockIndicesType& KVCacheResource::blocks(std::string_view tag) const {
    return group_block_ids.blocks(tag);
}

const BlockIndicesType& KVCacheResource::blocksForLayer(int layer_id, std::string_view tag) const {
    checkLayerTag(layer_id, tag);
    return group_block_ids.blocks(tag);
}

const BlockIndicesType& KVCacheResource::kernelBlocks(int group_id) const {
    RTP_LLM_CHECK(group_id >= 0 && group_block_ids.rows_.size() > static_cast<size_t>(group_id));
    return group_block_ids.rows_[static_cast<size_t>(group_id)]->kernelBlocks();
}

const BlockIndicesType& KVCacheResource::kernelBlocks(std::string_view tag) const {
    return group_block_ids.kernelBlocks(tag);
}

const BlockIndicesType& KVCacheResource::kernelBlocksForLayer(int layer_id, std::string_view tag) const {
    checkLayerTag(layer_id, tag);
    return group_block_ids.kernelBlocks(tag);
}

BlockIds& KVCacheResource::mutableBlockIds(std::string_view tag) const {
    // Preserve the shared mutable pointee contract without mutating the identity container.
    return const_cast<BlockIds&>(group_block_ids.blockIds(tag));
}

BlockIds& KVCacheResource::mutableBlockIdsForLayer(int layer_id, std::string_view tag) const {
    checkLayerTag(layer_id, tag);
    return mutableBlockIds(tag);
}

const BlockIds& KVCacheResource::blockIds(std::string_view tag) const {
    return group_block_ids.blockIds(tag);
}

const BlockIds& KVCacheResource::blockIdsForLayer(int layer_id, std::string_view tag) const {
    checkLayerTag(layer_id, tag);
    return group_block_ids.blockIds(tag);
}

std::shared_ptr<BlockIds> KVCacheResource::groupBlockIds(std::string_view tag) const {
    (void)group_block_ids.blockIds(tag);
    return group_block_ids.rows_[group_block_ids.tag_to_index_.at(std::string(tag))];
}

const GroupBlockIds& KVCacheResource::groupBlockIds() const {
    return group_block_ids;
}

void KVCacheResource::checkLayerTag(int layer_id, std::string_view tag) const {
    const auto& tags  = groupTagsForLayer(layer_id);
    const auto  value = std::string(tag);
    RTP_LLM_CHECK_WITH_INFO(std::find(tags.begin(), tags.end(), value) != tags.end(),
                            "KVCacheResource layer=%d does not own tag=%s",
                            layer_id,
                            value.c_str());
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

std::vector<std::string> KVCacheResource::groupTags() const {
    return group_block_ids.orderedTags();
}

int KVCacheResource::maxBlocksNum() const {
    int count = 0;
    for (const auto& row : group_block_ids.rows_) {
        count = std::max(count, static_cast<int>(row->blocksNum()));
    }
    return count;
}

size_t KVCacheResource::firstNonEmptyBlocksNum() const {
    for (const auto& row : group_block_ids.rows_) {
        if (row->blocksNum() != 0) {
            return row->blocksNum();
        }
    }
    return 0;
}

LayerBlockIds KVCacheResource::layerBlocks() const {
    RTP_LLM_CHECK_WITH_INFO(hasOneGroupPerLayer(),
                            "KVCacheResource::layerBlocks is a deprecated single-group-per-layer projection; "
                            "use blockIdsForLayer(layer, tag) for multi-group layers");
    LayerBlockIds layer_blocks;
    layer_blocks.reserve(layer_group_tags_.size());
    for (const auto& tags : layer_group_tags_) {
        layer_blocks.push_back(groupBlockIds(tags.front()));
    }
    return layer_blocks;
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
    return device_reuse_block_num_ + memory_reuse_block_num_ + disk_reuse_block_num_ + storage_backend_reuse_block_num_;
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

size_t KVCacheResource::diskReuseBlockNum() const {
    return disk_reuse_block_num_;
}

void KVCacheResource::setDiskReuseBlockNum(size_t disk_reuse_blocks_num) {
    disk_reuse_block_num_ = disk_reuse_blocks_num;
}

size_t KVCacheResource::storageBackendReuseBlockNum() const {
    return storage_backend_reuse_block_num_;
}

void KVCacheResource::setStorageBackendReuseBlockNum(size_t storage_backend_reuse_blocks_num) {
    storage_backend_reuse_block_num_ = storage_backend_reuse_blocks_num;
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
        const auto& block_indices = group_block_ids.rows_[static_cast<size_t>(group_id)]->blocks();
        for (auto& block : block_indices) {
            debug_string << block << ", ";
        }
        debug_string << "], ";
    }

    return debug_string.str();
}

void KVCacheResource::swapBlocks(std::string_view group_tag, size_t rhs, size_t lhs) {
    mutableBlockIds(group_tag).swap(rhs, lhs);
}

}  // namespace rtp_llm
