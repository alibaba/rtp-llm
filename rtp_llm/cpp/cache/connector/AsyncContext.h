#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

namespace rtp_llm {

class Meta;
class KVCacheResource;

class AsyncContext {
public:
    AsyncContext()          = default;
    virtual ~AsyncContext() = default;

public:
    virtual void waitDone()      = 0;
    virtual bool done() const    = 0;
    virtual bool success() const = 0;
};

class AsyncMatchContext: public AsyncContext {
public:
    ~AsyncMatchContext() override = default;

public:
    virtual size_t matchedBlockCount() const = 0;
};

class FusedAsyncContext: public AsyncContext {
public:
    FusedAsyncContext(const std::vector<std::shared_ptr<AsyncContext>>& contexts);
    ~FusedAsyncContext() override = default;

public:
    void waitDone() override;
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
    FusedAsyncReadContext(const std::shared_ptr<FusedAsyncContext>& fused_match_context,
                          const std::shared_ptr<KVCacheResource>&   resource,
                          const std::shared_ptr<Meta>&              meta);
    ~FusedAsyncReadContext() override;

public:
    void waitDone() override;
    bool done() const override;
    bool success() const override;
    void setFusedReadContext(const std::shared_ptr<FusedAsyncContext>& fused_read_context);
    const std::shared_ptr<FusedAsyncContext>  fusedReadContext() const;
    const std::shared_ptr<FusedAsyncContext>& fusedMatchContext() const;
    const std::shared_ptr<KVCacheResource>&   resource() const;
    const std::shared_ptr<Meta>&              meta() const;

private:
    std::shared_ptr<FusedAsyncContext> fused_match_context_;
    std::shared_ptr<FusedAsyncContext> fused_read_context_;
    std::shared_ptr<KVCacheResource>   resource_;
    std::shared_ptr<Meta>              meta_;

    mutable std::mutex              read_ctx_mutex_;
    mutable std::condition_variable read_ctx_cv_;
    std::atomic<bool>               read_ctx_set_{false};
};

}  // namespace rtp_llm