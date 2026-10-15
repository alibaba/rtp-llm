#include "rtp_llm/cpp/cache/KVCacheResource.h"

#include <algorithm>

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/CacheDomainConfig.h"

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

void KVCacheResource::initGroups(std::shared_ptr<const CacheTopology> topology, bool materialize_layer_views) {
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
    if (materialize_layer_views) {
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
    }

    // All throwing construction/validation finished; publish both views together.
    modules_.resize(1);
    active_module_count_ = 1;
    modules_.front().groups.tag_to_index_.swap(candidate.tag_to_index_);
    modules_.front().groups.rows_.swap(candidate.rows_);
    modules_.front().layer_group_tags.swap(layer_tags);
    modules_.front().topology = materialize_layer_views ? nullptr : std::move(topology);
}

void KVCacheResource::initGroups(const CacheDomainConfig& domain, bool materialize_layer_views) {
    std::vector<ModuleRows> candidate;
    candidate.reserve(domain.moduleCount());
    for (size_t index = 0; index < domain.moduleCount(); ++index) {
        KVCacheResource leaf;
        leaf.initGroups(domain.module(CacheModuleId(index)).topologyPtr(), materialize_layer_views);
        candidate.push_back(std::move(leaf.modules_.front()));
    }
    modules_.swap(candidate);
    active_module_count_ = domain.activeModuleCount();
}

size_t KVCacheResource::moduleCount() const {
    return modules_.size();
}

size_t KVCacheResource::activeModuleCount() const {
    return active_module_count_;
}

int KVCacheResource::layerNum(CacheModuleId module_id) const {
    const auto& rows = moduleRows(module_id);
    return static_cast<int>(rows.topology ? rows.topology->layers().size() : rows.layer_group_tags.size());
}

const KVCacheResource::ModuleRows& KVCacheResource::moduleRows(CacheModuleId module_id) const {
    RTP_LLM_CHECK_WITH_INFO(module_id.value() < modules_.size(),
                            "KVCacheResource module=%zu out of range [0,%zu)",
                            module_id.value(),
                            modules_.size());
    return modules_[module_id.value()];
}

std::vector<ModuleGroupKey> KVCacheResource::groupKeys() const {
    std::vector<ModuleGroupKey> keys;
    for (size_t index = 0; index < modules_.size(); ++index) {
        for (auto& tag : modules_[index].groups.orderedTags()) {
            keys.push_back({CacheModuleId(index), std::move(tag)});
        }
    }
    return keys;
}

const GroupBlockIds& KVCacheResource::groupBlockIds(CacheModuleId module_id) const {
    return moduleRows(module_id).groups;
}

const BlockIds& KVCacheResource::blockIds(const ModuleGroupKey& key) const {
    const auto& groups = groupBlockIds(key.module_id);
    RTP_LLM_CHECK_WITH_INFO(groups.tag_to_index_.count(key.tag) != 0,
                            "KVCacheResource missing module=%zu tag=%s",
                            key.module_id.value(),
                            key.tag.c_str());
    return groups.blockIds(key.tag);
}

BlockIds& KVCacheResource::mutableBlockIds(const ModuleGroupKey& key) const {
    return const_cast<BlockIds&>(blockIds(key));
}

std::shared_ptr<BlockIds> KVCacheResource::groupBlockIds(const ModuleGroupKey& key) const {
    (void)blockIds(key);
    const auto& groups = groupBlockIds(key.module_id);
    return groups.rows_[groups.tag_to_index_.at(key.tag)];
}

int KVCacheResource::blocksNum(const ModuleGroupKey& key) const {
    return static_cast<int>(blockIds(key).blocksNum());
}

const BlockIndicesType& KVCacheResource::blocks(const ModuleGroupKey& key) const {
    return blockIds(key).blocks();
}

const BlockIndicesType& KVCacheResource::kernelBlocks(const ModuleGroupKey& key) const {
    return blockIds(key).kernelBlocks();
}

const std::vector<std::string>& KVCacheResource::groupTagsForLayer(CacheModuleId module_id, int local_layer_id) const {
    const auto& rows  = moduleRows(module_id);
    const auto  count = rows.topology ? rows.topology->layers().size() : rows.layer_group_tags.size();
    RTP_LLM_CHECK_WITH_INFO(local_layer_id >= 0 && static_cast<size_t>(local_layer_id) < count,
                            "KVCacheResource module=%zu layer=%d out of range [0,%zu)",
                            module_id.value(),
                            local_layer_id,
                            count);
    return rows.topology ? rows.topology->layer(local_layer_id).group_tags : rows.layer_group_tags[local_layer_id];
}

const BlockIds&
KVCacheResource::blockIdsForLayer(CacheModuleId module_id, int local_layer_id, std::string_view tag) const {
    const auto& tags = groupTagsForLayer(module_id, local_layer_id);
    RTP_LLM_CHECK_WITH_INFO(std::find(tags.begin(), tags.end(), tag) != tags.end(),
                            "KVCacheResource module=%zu layer=%d does not own tag=%.*s",
                            module_id.value(),
                            local_layer_id,
                            static_cast<int>(tag.size()),
                            tag.data());
    return blockIds({module_id, std::string(tag)});
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
    for (auto& module : modules_) {
        for (auto& group : module.groups.rows_) {
            group->resize(reserver_blocks, value);
        }
    }
}

int KVCacheResource::blocksNum(std::string_view tag) const {
    return modules_.front().groups.blocksNum(tag);
}

const BlockIndicesType& KVCacheResource::blocks(std::string_view tag) const {
    return modules_.front().groups.blocks(tag);
}

const BlockIndicesType& KVCacheResource::blocksForLayer(int layer_id, std::string_view tag) const {
    checkLayerTag(layer_id, tag);
    return modules_.front().groups.blocks(tag);
}

const BlockIndicesType& KVCacheResource::kernelBlocks(std::string_view tag) const {
    return modules_.front().groups.kernelBlocks(tag);
}

const BlockIndicesType& KVCacheResource::kernelBlocksForLayer(int layer_id, std::string_view tag) const {
    checkLayerTag(layer_id, tag);
    return modules_.front().groups.kernelBlocks(tag);
}

BlockIds& KVCacheResource::mutableBlockIds(std::string_view tag) const {
    // Preserve the shared mutable pointee contract without mutating the identity container.
    return const_cast<BlockIds&>(modules_.front().groups.blockIds(tag));
}

BlockIds& KVCacheResource::mutableBlockIdsForLayer(int layer_id, std::string_view tag) const {
    checkLayerTag(layer_id, tag);
    return mutableBlockIds(tag);
}

const BlockIds& KVCacheResource::blockIds(std::string_view tag) const {
    return modules_.front().groups.blockIds(tag);
}

const BlockIds& KVCacheResource::blockIdsForLayer(int layer_id, std::string_view tag) const {
    checkLayerTag(layer_id, tag);
    return modules_.front().groups.blockIds(tag);
}

std::shared_ptr<BlockIds> KVCacheResource::groupBlockIds(std::string_view tag) const {
    (void)modules_.front().groups.blockIds(tag);
    return modules_.front().groups.rows_[modules_.front().groups.tag_to_index_.at(std::string(tag))];
}

const GroupBlockIds& KVCacheResource::groupBlockIds() const {
    return modules_.front().groups;
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
    if (modules_.front().topology) {
        return modules_.front().topology->layer(layer_id).group_tags;
    }
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < modules_.front().layer_group_tags.size(),
                            "KVCacheResource invalid layer_id=%d size=%zu",
                            layer_id,
                            modules_.front().layer_group_tags.size());
    return modules_.front().layer_group_tags[static_cast<size_t>(layer_id)];
}

const std::string& KVCacheResource::soleGroupTagForLayer(int layer_id) const {
    const auto& tags = groupTagsForLayer(layer_id);
    RTP_LLM_CHECK_WITH_INFO(
        tags.size() == 1, "KVCacheResource layer=%d requires exactly one group, got %zu", layer_id, tags.size());
    return tags.front();
}

bool KVCacheResource::hasOneGroupPerLayer() const {
    if (modules_.front().topology) {
        return modules_.front().topology->hasOneGroupPerLayer();
    }
    return std::all_of(modules_.front().layer_group_tags.begin(),
                       modules_.front().layer_group_tags.end(),
                       [](const auto& tags) { return tags.size() == 1; });
}

int KVCacheResource::layerNum() const {
    return modules_.front().topology ? static_cast<int>(modules_.front().topology->layers().size()) :
                                       static_cast<int>(modules_.front().layer_group_tags.size());
}

int KVCacheResource::groupNums() const {
    return static_cast<int>(modules_.front().groups.size());
}

std::vector<std::string> KVCacheResource::groupTags() const {
    return modules_.front().groups.orderedTags();
}

int KVCacheResource::maxBlocksNum() const {
    int count = 0;
    for (const auto& module : modules_) {
        for (const auto& row : module.groups.rows_) {
            count = std::max(count, static_cast<int>(row->blocksNum()));
        }
    }
    return count;
}

size_t KVCacheResource::firstNonEmptyBlocksNum() const {
    for (const auto& module : modules_) {
        for (const auto& row : module.groups.rows_) {
            if (row->blocksNum() != 0) {
                return row->blocksNum();
            }
        }
    }
    return 0;
}

LayerBlockIds KVCacheResource::layerBlocks() const {
    RTP_LLM_CHECK_WITH_INFO(hasOneGroupPerLayer(),
                            "KVCacheResource::layerBlocks is a deprecated single-group-per-layer projection; "
                            "use blockIdsForLayer(layer, tag) for multi-group layers");
    LayerBlockIds layer_blocks;
    layer_blocks.reserve(layerNum());
    for (int layer = 0; layer < layerNum(); ++layer) {
        layer_blocks.push_back(groupBlockIds(soleGroupTagForLayer(layer)));
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
    for (const auto& key : groupKeys()) {
        debug_string << key.debugString() << ", blocks:[";
        for (const auto block : blocks(key)) {
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
