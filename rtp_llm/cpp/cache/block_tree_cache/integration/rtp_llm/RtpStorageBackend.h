#pragma once

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"

#include <unordered_set>

namespace rtp_llm {

// Native storage drivers may resolve CUDA buffers and register an entire
// device pool at startup. The common backend only sees page holds and policy.
class RtpStorageBackend: public StorageBackend {
public:
    using StorageBackend::StorageBackend;
    using PoolsByTag     = std::unordered_map<std::string, DeviceBlockPoolPtr>;
    using BufferResolver = std::function<std::vector<BlockInfo>(int layer_id, const std::string&, int block_id)>;

    bool init(std::shared_ptr<const CacheTopology> topology, PoolsByTag pools, BufferResolver resolver) {
        return initEngine(*this, std::move(topology), std::move(pools), std::move(resolver));
    }

    static bool initEngine(StorageBackend&                      backend,
                           std::shared_ptr<const CacheTopology> topology,
                           PoolsByTag                           pools,
                           BufferResolver                       resolver) {
        auto* native = dynamic_cast<RtpStorageBackend*>(&backend);
        if (native && native->initializationAttempted()) {
            return false;
        }
        if (!topology || !resolver || pools.size() != topology->groups().size()) {
            throw std::invalid_argument("RTP storage requires one pool per topology group and a buffer resolver");
        }
        Groups                                groups;
        std::unordered_set<const IBlockPool*> seen;
        for (const auto& [tag, pool] : pools) {
            (void)topology->group(tag);
            if (!pool || !seen.emplace(pool.get()).second) {
                throw std::invalid_argument("RTP storage group has a null or shared device pool: " + tag);
            }
            groups.emplace(tag,
                           GroupAccess{
                               {pool, [pool](BlockIdxType block) { return pool->pinForStorageIo(block); }},
                               {},
                               [topology, tag](size_t index, size_t matched) {
                                   return matched - index <= topology->group(tag).reuseBlockCount(matched);
                               },
                           });
        }
        if (native) {
            native->topology_ = std::move(topology);
            native->pools_    = std::move(pools);
            native->resolver_ = std::move(resolver);
        }
        if (backend.init(std::move(groups))) {
            return true;
        }
        if (native) {
            native->resolver_ = {};
            native->pools_.clear();
            native->topology_.reset();
        }
        return false;
    }

protected:
    const CacheTopology& topology() const {
        return *topology_;
    }
    const DeviceBlockPoolPtr& devicePool(const std::string& tag) const {
        return pools_.at(tag);
    }
    std::vector<BlockInfo> convertIndexToBuffer(int layer, const std::string& tag, int block) const {
        (void)topology().group(tag);
        return resolver_(layer, tag, block);
    }

private:
    std::shared_ptr<const CacheTopology> topology_;
    PoolsByTag                           pools_;
    BufferResolver                       resolver_;
};

}  // namespace rtp_llm
