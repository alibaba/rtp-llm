#pragma once

#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/cache_new/KVCacheConnector.h"
#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

#include <map>

namespace rtp_llm {

class BlockCacheV1;
class BlockPool;
class DeviceBase;
class KVCacheAllocator;
class TpBroadcastManager;
class TPBroadcastResult;

class MemoryConnectorAsyncContext: public KVCacheConnector::AsyncContext {
public:
    MemoryConnectorAsyncContext(const std::shared_ptr<TPBroadcastResult>& broadcast_result,
                                const std::function<void(bool)>&          done_callback):
        broadcast_result_(broadcast_result), done_callback_(done_callback) {}
    MemoryConnectorAsyncContext(bool already_done, bool success): already_done_(already_done), success_(success) {}
    ~MemoryConnectorAsyncContext() override = default;

public:
    bool success() const override;
    void cancel() override;
    void waitDone() override;

private:
    bool allResponseSuccess() const;

private:
    std::shared_ptr<TPBroadcastResult> broadcast_result_;
    std::function<void(bool)>          done_callback_;
    bool                               already_done_{false};
    bool                               success_{false};
};

class MemoryConnector final: public KVCacheConnector {
public:
    MemoryConnector(const CacheConfig&                       cache_config,
                    const std::shared_ptr<KVCacheAllocator>& allocator,
                    rtp_llm::DeviceBase*                     device,
                    const std::vector<std::string>&          tp_addrs);
    ~MemoryConnector() override;

public:
    struct GroupCopyInfo {
        int              group_id{-1};
        std::vector<int> gpu_block_indices;
        std::vector<int> memory_block_indices;
    };
    enum class CopyDirection {
        H2D,
        D2H,
    };

    bool                          init() override;
    std::shared_ptr<AsyncContext> asyncRead(const std::shared_ptr<KVCacheResourceV1>& resource,
                                            const std::shared_ptr<Meta>&              meta) override;
    std::shared_ptr<AsyncContext> asyncWrite(const std::shared_ptr<KVCacheResourceV1>& resource,
                                             const std::shared_ptr<Meta>&              meta) override;
    std::shared_ptr<AsyncContext> asyncWriteByLayer(int                                       layer_id,
                                                    const std::shared_ptr<KVCacheResourceV1>& resource,
                                                    const std::shared_ptr<Meta>&              meta) override {
        throw std::runtime_error("MemoryConnector asyncWriteByLayer is not implemented");
    }

    // 拷贝KVCache(单TP)
    void copyCache(const MemoryBroadcastTpRequestPB& request, MemoryBroadcastTpResponsePB& response) const;

private:
    // 异步拷贝KVCache(多TP)
    std::shared_ptr<TPBroadcastResult> asyncCopyCache(const std::vector<GroupCopyInfo>& group_copy_infos,
                                                      CopyDirection                     direction) const;
    // 拷贝KVCache(单TP)
    bool copyCache(const std::vector<GroupCopyInfo>& group_copy_infos, CopyDirection direction) const;

    size_t match(const std::vector<size_t>& keys) const;
    size_t prefixMatch(const std::vector<size_t>& keys) const;
    size_t prefixMatch(const std::shared_ptr<BlockCacheV1>& block_cache, const std::vector<size_t>& keys) const;
    std::vector<bool> hashMatch(const std::vector<size_t>& keys) const;
    std::vector<bool> hashMatch(const std::shared_ptr<BlockCacheV1>& block_cache,
                                const std::vector<size_t>&           keys) const;

    void copyBuffers(const std::vector<BufferPtr>& dst, const std::vector<BufferPtr>& src) const;
    bool ensureEnoughFreeBlocks(const std::shared_ptr<BlockPool>& block_pool, int need_blocks) const;

private:
    const CacheConfig&                  cache_config_;
    std::shared_ptr<KVCacheAllocator>   allocator_;
    rtp_llm::DeviceBase*                device_{nullptr};
    const std::vector<std::string>      tp_addrs_;
    std::shared_ptr<TpBroadcastManager> broadcast_manager_;

    // 与 allocator 中的分组保持一致
    struct Group {
        GroupType                  type{GroupType::Invalid};
        std::map<int, int>         global_layer_to_local_layer;
        std::shared_ptr<BlockPool> block_pool;
    };
    // group_id->group
    std::map<int, Group> groups_;
    // layer_id->group_id, 表示这个层属于哪个group
    std::map<int, int> layer_to_group_;

    // full每1个block存一个cache, 如果linear每3个block存一个cache, 则group_block_stride_为3
    int group_block_stride_{1};
};

}  // namespace rtp_llm
