#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"

#include "rtp_llm/cpp/cache/KVCacheGroup.h"

#include <algorithm>
#include <sstream>
#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

std::shared_ptr<const BlockPoolConfigBase> makeAdapterConfig(const BlockPoolPtr& block_pool) {
    RTP_LLM_CHECK(block_pool != nullptr);
    auto config                  = std::make_shared<BlockPoolConfigBase>();
    config->pool_type            = BlockPoolType::DEVICE;
    config->pool_name            = block_pool->poolName();
    config->physical_block_count = block_pool->totalBlocksNum() + 1;
    return config;
}

}  // namespace

DeviceBlockPool::DeviceBlockPool(BlockPoolPtr block_pool, std::shared_ptr<KVCacheGroup> address_view):
    IBlockPool(makeAdapterConfig(block_pool)),
    block_pool_(std::move(block_pool)),
    address_view_(std::move(address_view)) {
    RTP_LLM_CHECK(address_view_ == nullptr || address_view_->blockPool() == block_pool_);
}

bool DeviceBlockPool::init() {
    std::lock_guard<std::mutex> lock(mutex_);
    RTP_LLM_CHECK(!initialized_);
    initialized_ = true;
    return true;
}

const std::string& DeviceBlockPool::poolName() const {
    return block_pool_->poolName();
}

std::optional<BlockIdxType> DeviceBlockPool::malloc() {
    auto blocks = malloc(1);
    if (!blocks.has_value()) {
        return std::nullopt;
    }
    return blocks->front();
}

std::optional<BlockIdList> DeviceBlockPool::malloc(size_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    if (n == 0) {
        return BlockIdList{};
    }
    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return std::nullopt;
    }

    auto blocks = block_pool_->malloc(static_cast<int>(n));
    if (blocks.size() != n) {
        if (!blocks.empty()) {
            block_pool_->requestFree(blocks);
        }
        return std::nullopt;
    }

    // Target malloc establishes request ownership. Convert it into exactly one
    // tree-cache hold without allocating a second backing.
    block_pool_->blockCacheReference(blocks);
    block_pool_->requestFree(blocks);
    for (const auto block : blocks) {
        const auto inserted = tree_refs_.emplace(block, 0).second;
        RTP_LLM_CHECK(inserted);
    }
    return blocks;
}

void DeviceBlockPool::free(BlockIdxType block) {
    free(BlockIdList{block});
}

void DeviceBlockPool::free(const BlockIdList& blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    checkUniqueNoLock(blocks);
    for (const auto block : blocks) {
        const auto it = tree_refs_.find(block);
        RTP_LLM_CHECK_WITH_INFO(it != tree_refs_.end(), "device block [%d] has no tree-cache hold", block);
        RTP_LLM_CHECK_WITH_INFO(
            it->second <= 1, "cannot free device block [%d] with tree refcount [%u]", block, it->second);
    }
    for (const auto block : blocks) {
        const auto it = tree_refs_.find(block);
        block_pool_->blockCacheFree(block);
        tree_refs_.erase(it);
    }
}

void DeviceBlockPool::incRef(BlockIdxType block) {
    incRef(BlockIdList{block});
}

void DeviceBlockPool::incRef(const BlockIdList& blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    checkUniqueNoLock(blocks);
    for (const auto block : blocks) {
        checkValidNoLock(block);
    }
    for (const auto block : blocks) {
        auto it = tree_refs_.find(block);
        if (it == tree_refs_.end()) {
            // Adopt an allocator-owned block without disturbing its request hold.
            block_pool_->blockCacheReference(block);
            tree_refs_.emplace(block, 1);
            continue;
        }
        if (it->second > 0) {
            block_pool_->blockCacheReference(block);
        }
        it->second += 1;
    }
}

void DeviceBlockPool::decRef(BlockIdxType block) {
    decRef(BlockIdList{block});
}

void DeviceBlockPool::decRef(const BlockIdList& blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    checkUniqueNoLock(blocks);
    for (const auto block : blocks) {
        const auto it = tree_refs_.find(block);
        RTP_LLM_CHECK_WITH_INFO(
            it != tree_refs_.end() && it->second > 0, "device block [%d] has no releasable tree-cache hold", block);
    }
    for (const auto block : blocks) {
        auto it = tree_refs_.find(block);
        it->second -= 1;
        block_pool_->blockCacheFree(block);
        if (it->second == 0) {
            tree_refs_.erase(it);
        }
    }
}

uint32_t DeviceBlockPool::refCount(BlockIdxType block) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = tree_refs_.find(block);
    RTP_LLM_CHECK_WITH_INFO(it != tree_refs_.end(), "device block [%d] has no tree-cache hold", block);
    return it->second;
}

bool DeviceBlockPool::validBlock(BlockIdxType block) const {
    return block > 0 && static_cast<size_t>(block) <= block_pool_->totalBlocksNum() && !isNullBlockIdx(block);
}

bool DeviceBlockPool::isAllocated(BlockIdxType block) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tree_refs_.find(block) != tree_refs_.end();
}

bool DeviceBlockPool::hasRequestHold(BlockIdxType block) const {
    return validBlock(block) && block_pool_->requestRefCount(block) > 0;
}

bool DeviceBlockPool::hasExternalHolderNoLock(BlockIdxType block, uint32_t tree_refs) const {
    const BlockHolderRefCounts holders                  = block_pool_->blockHolderRefCounts(block);
    const int                  adapter_owned_cache_refs = static_cast<int>(std::max<uint32_t>(1, tree_refs));
    return holders.request_connector > 0 || holders.block_cache > adapter_owned_cache_refs;
}

bool DeviceBlockPool::hasExternalHolder(BlockIdxType block) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = tree_refs_.find(block);
    return it != tree_refs_.end() && hasExternalHolderNoLock(block, it->second);
}

bool DeviceBlockPool::isCacheOnly(BlockIdxType block) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = tree_refs_.find(block);
    return it != tree_refs_.end() && it->second == 1 && !hasExternalHolderNoLock(block, it->second);
}

size_t DeviceBlockPool::totalBlocksNum() const {
    return block_pool_->totalBlocksNum();
}

size_t DeviceBlockPool::freeBlocksNum() const {
    return block_pool_->freeBlocksNum();
}

size_t DeviceBlockPool::usedBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tree_refs_.size();
}

size_t DeviceBlockPool::unreferencedBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<size_t>(
        std::count_if(tree_refs_.begin(), tree_refs_.end(), [](const auto& entry) { return entry.second == 0; }));
}

size_t DeviceBlockPool::treeCachedBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<size_t>(
        std::count_if(tree_refs_.begin(), tree_refs_.end(), [](const auto& entry) { return entry.second > 0; }));
}

size_t DeviceBlockPool::activeTreeCachedBlocksNumNoLock() const {
    return static_cast<size_t>(std::count_if(tree_refs_.begin(), tree_refs_.end(), [this](const auto& entry) {
        return entry.second > 1 || hasExternalHolderNoLock(entry.first, entry.second);
    }));
}

size_t DeviceBlockPool::activeTreeCachedBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return activeTreeCachedBlocksNumNoLock();
}

int DeviceBlockPool::deviceIndex() const {
    const auto layers = block_pool_->allLayerCacheBase();
    if (!layers.empty() && layers.front().defined() && layers.front().is_cuda()) {
        return static_cast<int>(layers.front().get_device());
    }
    return -1;
}

MemoryType DeviceBlockPool::where() const {
    return block_pool_->where();
}

std::vector<torch::Tensor> DeviceBlockPool::allLayerCacheBase() const {
    return block_pool_->allLayerCacheBase();
}

std::vector<torch::Tensor> DeviceBlockPool::allLayerScaleCacheBase() const {
    return block_pool_->allLayerScaleCacheBase();
}

void DeviceBlockPool::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    block_pool_->regUserMr(model_id, std::move(cache_store));
}

void DeviceBlockPool::deregUserMr() {
    block_pool_->deregUserMr();
}

int64_t DeviceBlockPool::getMrCostTimeMs() const {
    return block_pool_->getMrCostTimeMs();
}

BlockAddrInfo DeviceBlockPool::convertIndexToAddr(int layer_id, BlockIdxType block) const {
    return address_view_ ? address_view_->convertIndexToAddr(layer_id, block) :
                           block_pool_->convertIndexToAddr(layer_id, block);
}

std::vector<BlockInfo> DeviceBlockPool::convertIndexToBuffer(int layer_id, BlockIdxType block) const {
    return address_view_ ? address_view_->convertIndexToBuffer(layer_id, block) :
                           block_pool_->convertIndexToBuffer(layer_id, block);
}

std::vector<BlockInfo>
DeviceBlockPool::convertIndexToBuffer(int layer_id, BlockIdxType block, int partition_count, int partition_id) const {
    return address_view_ ? address_view_->convertIndexToBuffer(layer_id, block, partition_count, partition_id) :
                           block_pool_->convertIndexToBuffer(layer_id, block, partition_count, partition_id);
}

void* DeviceBlockPool::getBaseAddress() const {
    return block_pool_->getBaseAddress();
}

size_t DeviceBlockPool::getTotalSizeBytes() const {
    return block_pool_->getTotalSizeBytes();
}

std::string DeviceBlockPool::debugString() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream          oss;
    oss << "DeviceBlockPoolAdapter{name=" << block_pool_->poolName()
        << ", target_total=" << block_pool_->totalBlocksNum() << ", tree_holds=" << tree_refs_.size()
        << ", active_tree_holds=" << activeTreeCachedBlocksNumNoLock()
        << ", target_free=" << block_pool_->freeBlocksNum() << "}";
    return oss.str();
}

void DeviceBlockPool::checkInitializedNoLock() const {
    RTP_LLM_CHECK_WITH_INFO(initialized_, "device block pool adapter [%s] is not initialized", poolName().c_str());
}

void DeviceBlockPool::checkValidNoLock(BlockIdxType block) const {
    RTP_LLM_CHECK_WITH_INFO(validBlock(block), "invalid device block id [%d] for pool [%s]", block, poolName().c_str());
}

void DeviceBlockPool::checkUniqueNoLock(const BlockIdList& blocks) const {
    BlockIdList sorted_blocks(blocks);
    std::sort(sorted_blocks.begin(), sorted_blocks.end());
    RTP_LLM_CHECK_WITH_INFO(std::adjacent_find(sorted_blocks.begin(), sorted_blocks.end()) == sorted_blocks.end(),
                            "duplicate device block id for pool [%s]",
                            poolName().c_str());
}

}  // namespace rtp_llm
