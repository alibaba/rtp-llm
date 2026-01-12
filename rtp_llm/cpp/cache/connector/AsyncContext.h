#pragma once

#include <memory>
#include <vector>

namespace rtp_llm {

class KVCacheResource;
struct KVCacheConnectorMeta;

enum class ConnectorType {
    Memory = 0,
    Remote = 1,
    P2P    = 2
};

class AsyncContext {
public:
    AsyncContext()          = default;
    virtual ~AsyncContext() = default;

public:
    virtual bool done() const    = 0;
    virtual bool success() const = 0;
};

class AsyncMatchContext: public AsyncContext {
public:
    AsyncMatchContext()                             = default;
    ~AsyncMatchContext() override                   = default;
    virtual size_t        matchedBlockCount() const = 0;
    virtual ConnectorType connectorType() const     = 0;
};

class FusedAsyncContext: public AsyncContext {
public:
    FusedAsyncContext(const std::vector<std::shared_ptr<AsyncContext>>& contexts);
    ~FusedAsyncContext() override = default;

public:
    bool done() const override;
    bool success() const override;

    const std::vector<std::shared_ptr<AsyncContext>>& contexts() const {
        return contexts_;
    }

private:
    std::vector<std::shared_ptr<AsyncContext>> contexts_;
};

class FusedAsyncReadContext: public AsyncContext {
public:
    FusedAsyncReadContext(const std::shared_ptr<FusedAsyncContext>&    fused_match_context,
                          const std::shared_ptr<KVCacheResource>&      resource,
                          const std::shared_ptr<KVCacheConnectorMeta>& meta);
    ~FusedAsyncReadContext() override = default;

public:
    bool done() const override;
    bool success() const override;
    void setFusedReadContext(const std::shared_ptr<FusedAsyncContext>& fused_read_context);
    const std::shared_ptr<FusedAsyncContext>& fusedMatchContext() const {
        return fused_match_context_;
    }
    const std::shared_ptr<FusedAsyncContext>& fusedReadContext() const {
        return fused_read_context_;
    }
    const std::shared_ptr<KVCacheResource>& resource() const {
        return resource_;
    }
    const std::shared_ptr<KVCacheConnectorMeta>& meta() const {
        return meta_;
    }

    int reuseBlockNum() const {
        return reuse_block_num_;
    }

    void setReuseBlockNum(int reuse_block_num) {
        reuse_block_num_ = reuse_block_num;
    }

private:
    std::shared_ptr<FusedAsyncContext>    fused_match_context_;
    std::shared_ptr<FusedAsyncContext>    fused_read_context_;
    std::shared_ptr<KVCacheResource>      resource_;
    std::shared_ptr<KVCacheConnectorMeta> meta_;

    int reuse_block_num_ = 0;
};

}  // namespace rtp_llm