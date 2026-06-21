#include "rtp_llm/cpp/cache/KVCacheResource.h"

#include <algorithm>

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
    if (pos_a >= block_indices.size() || pos_b >= block_indices.size()) {
        RTP_LLM_LOG_ERROR("BlockIds::swap: pos_a=%d or pos_b=%d is out of range, block_indices.size()=%d",
                          pos_a,
                          pos_b,
                          block_indices.size());
        RTP_LLM_CHECK_WITH_INFO(false,
                                "BlockIds::swap: pos_a=%d or pos_b=%d is out of range, block_indices.size()=%d",
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

void KVCacheResource::initGroups(int                                  group_num,
                                 int                                  layer_num,
                                 const std::vector<int>&              layer_to_group_id,
                                 size_t                               kernel_blocks_per_kv_block,
                                 const std::vector<CacheGroupType>&   group_types,
                                 const std::vector<std::vector<int>>& layer_to_group_ids) {
    group_block_ids.clear();
    layer_block_ids.clear();
    layer_group_block_ids.clear();

    if (!group_types.empty()) {
        RTP_LLM_CHECK_WITH_INFO(group_types.size() >= static_cast<size_t>(group_num),
                                "KVCacheResource::initGroups: group_types size %zu < group_num %d",
                                group_types.size(),
                                group_num);
    }

    group_block_ids.reserve(static_cast<size_t>(group_num));
    for (int i = 0; i < group_num; i++) {
        const bool   is_full_group = group_types.empty() || group_types[static_cast<size_t>(i)] == CacheGroupType::FULL;
        const size_t bpk           = is_full_group ? std::max<size_t>(1, kernel_blocks_per_kv_block) : 1;
        auto         bid           = std::make_shared<BlockIds>(bpk);
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
            if (gid < 0 && !layer_to_group_ids.empty()) {
                RTP_LLM_CHECK_WITH_INFO(layer_to_group_ids.size() >= static_cast<size_t>(layer_num),
                                        "KVCacheResource::initGroups: layer_to_group_ids size %zu < layer_num %d",
                                        layer_to_group_ids.size(),
                                        layer_num);
                const auto& dense_groups = layer_to_group_ids[static_cast<size_t>(i)];
                RTP_LLM_CHECK_WITH_INFO(!dense_groups.empty(),
                                        "KVCacheResource::initGroups: empty group ids for layer %d",
                                        i);
                gid = dense_groups.back();
            }
            RTP_LLM_CHECK_WITH_INFO(gid >= 0 && gid < group_num,
                                    "KVCacheResource::initGroups: invalid group id %d for layer %d (group_num=%d)",
                                    gid,
                                    i,
                                    group_num);
            layer_block_ids[i] = group_block_ids[gid];
        }

        layer_group_block_ids.resize(static_cast<size_t>(layer_num));
        for (int layer = 0; layer < layer_num; ++layer) {
            auto& group_blocks = layer_group_block_ids[static_cast<size_t>(layer)];
            group_blocks.assign(static_cast<size_t>(group_num), nullptr);

            if (!layer_to_group_ids.empty()) {
                RTP_LLM_CHECK_WITH_INFO(layer_to_group_ids.size() >= static_cast<size_t>(layer_num),
                                        "KVCacheResource::initGroups: layer_to_group_ids size %zu < layer_num %d",
                                        layer_to_group_ids.size(),
                                        layer_num);
                const auto& dense_groups = layer_to_group_ids[static_cast<size_t>(layer)];
                for (int gid : dense_groups) {
                    RTP_LLM_CHECK_WITH_INFO(
                        gid >= 0 && gid < group_num,
                        "KVCacheResource::initGroups: invalid group id %d for layer %d (group_num=%d)",
                        gid,
                        layer,
                        group_num);
                    group_blocks[static_cast<size_t>(gid)] = group_block_ids[static_cast<size_t>(gid)];
                }
            } else {
                const int gid = layer_to_group_id.empty() ? 0 : layer_to_group_id[static_cast<size_t>(layer)];
                group_blocks[static_cast<size_t>(gid)] = layer_block_ids[static_cast<size_t>(layer)];
            }
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

const BlockIndicesType& KVCacheResource::blocks(int layer_id, int group_id) const {
    return mutableBlockIds(layer_id, group_id).blocks();
}

const BlockIndicesType& KVCacheResource::kernelBlocks(int group_id) const {
    RTP_LLM_CHECK(group_block_ids.size() > static_cast<size_t>(group_id));
    return group_block_ids[group_id]->kernelBlocks();
}

const BlockIndicesType& KVCacheResource::kernelBlocks(int layer_id, int group_id) const {
    return mutableBlockIds(layer_id, group_id).kernelBlocks();
}

BlockIds& KVCacheResource::mutableBlockIds(int group_id) const {
    RTP_LLM_CHECK(group_block_ids.size() > static_cast<size_t>(group_id));
    return *group_block_ids[group_id];
}

BlockIds& KVCacheResource::mutableBlockIds(int layer_id, int group_id) const {
    RTP_LLM_CHECK(static_cast<size_t>(layer_id) < layer_group_block_ids.size());
    RTP_LLM_CHECK(static_cast<size_t>(group_id) < layer_group_block_ids[static_cast<size_t>(layer_id)].size());
    auto block_ids = layer_group_block_ids[static_cast<size_t>(layer_id)][static_cast<size_t>(group_id)];
    RTP_LLM_CHECK_WITH_INFO(
        block_ids != nullptr, "KVCacheResource: missing block ids for layer %d group_id %d", layer_id, group_id);
    return *block_ids;
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
    cache_keys = keys;
    cache_keys_are_cp_canonical_ = false;
    rebuildLinearBlockDependencies();
}

void KVCacheResource::setCacheKeys(CacheKeysType&& keys) {
    cache_keys = std::move(keys);
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
    if (block_dependencies.size() != cache_keys.size()) {
        rebuildLinearBlockDependencies();
    }
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
