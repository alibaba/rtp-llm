#pragma once

#include <memory>
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache_new/AsyncContext.h"

namespace rtp_llm {

class KVCacheConnector;

class HybridReadAsyncContext: public AsyncContext {
public:
    HybridReadAsyncContext(int64_t                                   request_id,
                           const std::shared_ptr<KVCacheResourceV1>& resource,
                           const std::shared_ptr<KVCacheConnector>&  memory_connector,
                           const std::shared_ptr<KVCacheConnector>&  remote_connector);
    ~HybridReadAsyncContext() override = default;

    void waitDone() override;
    void cancel() override {
        // TODO
    }
    bool done() const override;
    bool success() const override;

    inline size_t memory_reuse_block_num() const {
        return memory_reuse_block_num_;
    }

    inline size_t remote_reuse_block_num() const {
        return remote_reuse_block_num_;
    }

private:
    void genRemoteContext() const;

private:
    int64_t                               request_id_;
    std::shared_ptr<KVCacheResourceV1>    resource_;
    std::shared_ptr<KVCacheConnector>     memory_connector_;
    mutable std::shared_ptr<AsyncContext> memory_context_;
    std::shared_ptr<KVCacheConnector>     remote_connector_;
    mutable std::shared_ptr<AsyncContext> remote_context_;

    // metric:
    mutable size_t memory_reuse_block_num_ = 0;  // TODO
    mutable size_t remote_reuse_block_num_ = 0;
};

}  // namespace rtp_llm