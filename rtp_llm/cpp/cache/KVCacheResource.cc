#include "rtp_llm/cpp/cache/KVCacheResource.h"

#include <algorithm>

#include "rtp_llm/cpp/cache/CacheTopology.h"

namespace rtp_llm {

void KVCacheResource::initGroups(std::shared_ptr<const CacheTopology> topology) {
    RTP_LLM_CHECK_WITH_INFO(topology != nullptr, "KVCacheResource::initGroups requires a topology");
    topology_ = std::move(topology);
    request_prefix_.configure(*topology_);

    group_resources_.clear();
    group_resources_.reserve(topology_->groups().size());
    group_offset_by_tag_.clear();
    group_offset_by_tag_.reserve(topology_->groups().size());
    for (const auto& group : topology_->groups()) {
        const size_t blocks_per_kv_block = group.seq_size_per_block / group.kernel_seq_size_per_block;
        const size_t stored_blocks_per_kv_block =
            group.policy.group_type == CacheGroupType::FULL ? std::max<size_t>(1, blocks_per_kv_block) : 1;
        const size_t offset = group_resources_.size();
        group_resources_.push_back({group.tag, std::make_shared<BlockIds>(stored_blocks_per_kv_block)});
        RTP_LLM_CHECK_WITH_INFO(group_offset_by_tag_.emplace(group.tag, offset).second,
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
    for (auto& group : group_resources_) {
        group.block_ids->resize(reserver_blocks, value);
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
    return *group_resources_[groupOffset(tag)].block_ids;
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
    return static_cast<int>(group_resources_.size());
}

size_t KVCacheResource::physicalBlockSpan(std::string_view tag) const {
    RTP_LLM_CHECK_WITH_INFO(topology_ != nullptr, "KVCacheResource groups are not initialized");
    return topology_->group(tag).seq_size_per_block;
}

const std::vector<CacheGroupResource>& KVCacheResource::groupResources() const {
    return group_resources_;
}

CacheGroupResource& KVCacheResource::groupResource(std::string_view tag) {
    return group_resources_[groupOffset(tag)];
}

const CacheGroupResource& KVCacheResource::groupResource(std::string_view tag) const {
    return group_resources_[groupOffset(tag)];
}

const std::string& KVCacheResource::strictSingleGroupTag() const {
    RTP_LLM_CHECK_WITH_INFO(group_resources_.size() == 1,
                            "legacy cache resource adapter requires exactly one tagged group, got %zu",
                            group_resources_.size());
    return group_resources_.front().tag;
}

CacheKeysType& KVCacheResource::cacheKeys(std::string_view tag) {
    return groupResource(tag).cache_keys;
}

const CacheKeysType& KVCacheResource::cacheKeys(std::string_view tag) const {
    return groupResource(tag).cache_keys;
}

CacheKeysType& KVCacheResource::cacheKeys() {
    return cacheKeys(strictSingleGroupTag());
}

const CacheKeysType& KVCacheResource::cacheKeys() const {
    return cacheKeys(strictSingleGroupTag());
}

void KVCacheResource::setCacheKeys(std::string_view tag, const CacheKeysType& keys) {
    auto& resource                       = groupResource(tag);
    resource.cache_keys                  = keys;
    resource.cache_keys_are_cp_canonical = false;
    rebuildLinearBlockDependencies(tag);
}

void KVCacheResource::setCacheKeys(std::string_view tag, CacheKeysType&& keys) {
    auto& resource                       = groupResource(tag);
    resource.cache_keys                  = std::move(keys);
    resource.cache_keys_are_cp_canonical = false;
    rebuildLinearBlockDependencies(tag);
}

void KVCacheResource::setCacheKeys(const CacheKeysType& keys) {
    setCacheKeys(strictSingleGroupTag(), keys);
}

void KVCacheResource::setCacheKeys(CacheKeysType&& keys) {
    setCacheKeys(strictSingleGroupTag(), std::move(keys));
}

bool KVCacheResource::cacheKeysAreCpCanonical(std::string_view tag) const {
    return groupResource(tag).cache_keys_are_cp_canonical;
}

void KVCacheResource::setCacheKeysAreCpCanonical(std::string_view tag, bool value) {
    groupResource(tag).cache_keys_are_cp_canonical = value;
}

BlockDependenciesType& KVCacheResource::blockDependencies(std::string_view tag) {
    return groupResource(tag).block_dependencies;
}

const BlockDependenciesType& KVCacheResource::blockDependencies(std::string_view tag) const {
    return groupResource(tag).block_dependencies;
}

BlockDependenciesType& KVCacheResource::blockDependencies() {
    return blockDependencies(strictSingleGroupTag());
}

const BlockDependenciesType& KVCacheResource::blockDependencies() const {
    return blockDependencies(strictSingleGroupTag());
}

void KVCacheResource::setBlockDependencies(std::string_view tag, const BlockDependenciesType& dependencies) {
    groupResource(tag).block_dependencies = dependencies;
}

void KVCacheResource::setBlockDependencies(std::string_view tag, BlockDependenciesType&& dependencies) {
    groupResource(tag).block_dependencies = std::move(dependencies);
}

void KVCacheResource::rebuildLinearBlockDependencies(std::string_view tag) {
    auto& resource = groupResource(tag);
    RTP_LLM_CHECK_WITH_INFO(topology_ != nullptr, "KVCacheResource groups are not initialized");
    resource.block_dependencies.clear();
    resource.block_dependencies.reserve(resource.cache_keys.size());
    for (size_t i = 0; i < resource.cache_keys.size(); ++i) {
        BlockDependency dependency;
        dependency.ordinal = static_cast<uint32_t>(i);
        if (i > 0) {
            dependency.has_parent = true;
            dependency.parent_key = resource.cache_keys[i - 1];
        }
        resource.block_dependencies.push_back(dependency);
    }
}

void KVCacheResource::rebuildLinearBlockDependencies() {
    rebuildLinearBlockDependencies(strictSingleGroupTag());
}

void KVCacheResource::ensureLinearBlockDependencies(std::string_view tag) {
    const auto& resource = groupResource(tag);
    if (resource.block_dependencies.size() == resource.cache_keys.size()) {
        return;
    }
    rebuildLinearBlockDependencies(tag);
}

size_t KVCacheResource::reuseTokenNum() const {
    return request_prefix_.reuseTokens();
}

size_t KVCacheResource::deviceReuseTokenNum() const {
    return request_prefix_.deviceReuseTokens();
}

size_t KVCacheResource::memoryReuseTokenNum() const {
    return request_prefix_.memoryReuseTokens();
}

size_t KVCacheResource::remoteReuseTokenNum() const {
    return request_prefix_.remoteReuseTokens();
}

size_t KVCacheResource::reuseBlockNum() const {
    return reuseTokenNum() / physicalBlockSpan(strictSingleGroupTag());
}

size_t KVCacheResource::deviceReuseBlockNum() const {
    return deviceReuseTokenNum() / physicalBlockSpan(strictSingleGroupTag());
}

size_t KVCacheResource::memoryReuseBlockNum() const {
    return memoryReuseTokenNum() / physicalBlockSpan(strictSingleGroupTag());
}

size_t KVCacheResource::remoteReuseBlockNum() const {
    return remoteReuseTokenNum() / physicalBlockSpan(strictSingleGroupTag());
}

void KVCacheResource::setDeviceReuseTokenNum(size_t tokens) {
    request_prefix_.setDeviceReuseTokens(tokens);
}

void KVCacheResource::setMemoryReuseTokenNum(size_t tokens) {
    request_prefix_.setMemoryReuseTokens(tokens);
}

void KVCacheResource::setRemoteReuseTokenNum(size_t tokens) {
    request_prefix_.setRemoteReuseTokens(tokens);
}

void KVCacheResource::setDeviceReuseBlockNum(size_t blocks) {
    setDeviceReuseTokenNum(blocks * physicalBlockSpan(strictSingleGroupTag()));
}

void KVCacheResource::setMemoryReuseBlockNum(size_t blocks) {
    setMemoryReuseTokenNum(blocks * physicalBlockSpan(strictSingleGroupTag()));
}

void KVCacheResource::setRemoteReuseBlockNum(size_t blocks) {
    setRemoteReuseTokenNum(blocks * physicalBlockSpan(strictSingleGroupTag()));
}

bool KVCacheResource::lastBlockAligned(std::string_view tag) const {
    return groupResource(tag).last_block_aligned;
}

void KVCacheResource::setLastBlockAligned(std::string_view tag, bool value) {
    groupResource(tag).last_block_aligned = value;
}

bool KVCacheResource::lastBlockAligned() const {
    return lastBlockAligned(strictSingleGroupTag());
}

void KVCacheResource::setLastBlockAligned(bool value) {
    setLastBlockAligned(strictSingleGroupTag(), value);
}

std::string KVCacheResource::debugString() const {
    std::stringstream debug_string;
    for (const auto& group : group_resources_) {
        debug_string << "group:[" << group.tag << "], block:[";
        const auto& block_indices = group.block_ids->blocks();
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

LayerBlockIdsView KVCacheResource::blockIdsForLayer(int layer_id) const {
    groupTagsForLayer(layer_id);
    return LayerBlockIdsView(this, layer_id);
}

const BlockIds& LayerBlockIdsView::at(std::string_view tag) const {
    return resource_->blockIdsForLayer(layer_id_, tag);
}

bool LayerBlockIdsView::contains(std::string_view tag) const {
    const auto& tags = resource_->groupTagsForLayer(layer_id_);
    return std::any_of(
        tags.begin(), tags.end(), [tag](const std::string& candidate) { return std::string_view(candidate) == tag; });
}

size_t LayerBlockIdsView::size() const {
    return resource_->groupTagsForLayer(layer_id_).size();
}

LayerBlockIdsView::Iterator LayerBlockIdsView::begin() const {
    return Iterator(resource_, layer_id_, tags().begin());
}

LayerBlockIdsView::Iterator LayerBlockIdsView::end() const {
    return Iterator(resource_, layer_id_, tags().end());
}

LayerBlockIdsView::Iterator::value_type LayerBlockIdsView::Iterator::operator*() const {
    return {*tag_it_, std::cref(resource_->blockIds(*tag_it_))};
}

const std::vector<std::string>& LayerBlockIdsView::tags() const {
    return resource_->groupTagsForLayer(layer_id_);
}

size_t KVCacheResource::groupOffset(std::string_view tag) const {
    const std::string value(tag);
    const auto        it = group_offset_by_tag_.find(value);
    RTP_LLM_CHECK_WITH_INFO(it != group_offset_by_tag_.end(), "KVCacheResource missing tag=%s", value.c_str());
    return it->second;
}

}  // namespace rtp_llm
