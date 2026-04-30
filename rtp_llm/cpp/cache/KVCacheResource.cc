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

void KVCacheResource::initGroups(int                                  group_num,
                                 int                                  layer_num,
                                 const std::vector<int>&              layer_to_group_id,
                                 size_t                               kernel_blocks_per_kv_block,
                                 const std::vector<CacheGroupType>&   group_types,
                                 const std::vector<std::vector<int>>& layer_region_to_group_id) {
    group_block_ids.clear();
    layer_block_ids.clear();
    layer_region_block_ids.clear();

    RTP_LLM_CHECK_WITH_INFO(group_types.empty() || group_types.size() >= static_cast<size_t>(group_num),
                            "KVCacheResource::initGroups: group_types size %zu < group_num %d",
                            group_types.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(!group_types.empty() || layer_region_to_group_id.empty(),
                            "KVCacheResource::initGroups: group_types must be explicit for typed layer-region mapping");

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

        const size_t region_name_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
        layer_region_block_ids.resize(static_cast<size_t>(layer_num));
        for (int layer = 0; layer < layer_num; ++layer) {
            auto& attn_blocks = layer_region_block_ids[static_cast<size_t>(layer)];
            attn_blocks.assign(region_name_count, nullptr);

            if (!layer_region_to_group_id.empty()) {
                RTP_LLM_CHECK_WITH_INFO(layer_region_to_group_id.size() >= static_cast<size_t>(layer_num),
                                        "KVCacheResource::initGroups: layer_region_to_group_id size %zu < layer_num %d",
                                        layer_region_to_group_id.size(),
                                        layer_num);
                const auto&  dense_groups = layer_region_to_group_id[static_cast<size_t>(layer)];
                const size_t n            = std::min(region_name_count, dense_groups.size());
                bool         has_region   = false;
                for (size_t attn = 0; attn < n; ++attn) {
                    const int gid = dense_groups[attn];
                    if (gid < 0) {
                        continue;
                    }
                    RTP_LLM_CHECK_WITH_INFO(
                        gid < group_num,
                        "KVCacheResource::initGroups: invalid group id %d for layer %d region_name %zu (group_num=%d)",
                        gid,
                        layer,
                        attn,
                        group_num);
                    attn_blocks[attn] = group_block_ids[static_cast<size_t>(gid)];
                    has_region        = true;
                }
                RTP_LLM_CHECK_WITH_INFO(
                    has_region, "KVCacheResource::initGroups: missing layer-region mapping for layer %d", layer);
            } else {
                attn_blocks[static_cast<size_t>(KVCacheRegionName::DEFAULT)] =
                    layer_block_ids[static_cast<size_t>(layer)];
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

const BlockIndicesType& KVCacheResource::blocks(int layer_id, KVCacheRegionName region_name) const {
    return mutableBlockIds(layer_id, region_name).blocks();
}

const BlockIndicesType& KVCacheResource::kernelBlocks(int group_id) const {
    RTP_LLM_CHECK(group_block_ids.size() > static_cast<size_t>(group_id));
    return group_block_ids[group_id]->kernelBlocks();
}

const BlockIndicesType& KVCacheResource::kernelBlocks(int layer_id, KVCacheRegionName region_name) const {
    return mutableBlockIds(layer_id, region_name).kernelBlocks();
}

BlockIds& KVCacheResource::mutableBlockIds(int group_id) const {
    RTP_LLM_CHECK(group_block_ids.size() > static_cast<size_t>(group_id));
    return *group_block_ids[group_id];
}

BlockIds& KVCacheResource::mutableBlockIds(int layer_id, KVCacheRegionName region_name) const {
    const auto attn_id = static_cast<size_t>(region_name);
    RTP_LLM_CHECK(static_cast<size_t>(layer_id) < layer_region_block_ids.size());
    RTP_LLM_CHECK(attn_id < layer_region_block_ids[static_cast<size_t>(layer_id)].size());
    auto block_ids = layer_region_block_ids[static_cast<size_t>(layer_id)][attn_id];
    RTP_LLM_CHECK_WITH_INFO(
        block_ids != nullptr, "KVCacheResource: missing block ids for layer %d region_name %zu", layer_id, attn_id);
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

const LayerRegionBlockIds& KVCacheResource::layerRegionBlocks() const {
    return layer_region_block_ids;
}

int KVCacheResource::groupId(int layer_id, KVCacheRegionName region_name) const {
    const auto attn_id = static_cast<size_t>(region_name);
    RTP_LLM_CHECK(static_cast<size_t>(layer_id) < layer_region_block_ids.size());
    RTP_LLM_CHECK(attn_id < layer_region_block_ids[static_cast<size_t>(layer_id)].size());
    const auto& block_ids = layer_region_block_ids[static_cast<size_t>(layer_id)][attn_id];
    if (!block_ids) {
        return -1;
    }
    for (size_t gid = 0; gid < group_block_ids.size(); ++gid) {
        if (group_block_ids[gid] == block_ids) {
            return static_cast<int>(gid);
        }
    }
    return -1;
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

}  // namespace rtp_llm
