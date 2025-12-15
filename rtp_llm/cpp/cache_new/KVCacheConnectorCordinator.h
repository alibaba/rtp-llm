#pragma once

namespace rtp_llm {

class FusedAsyncContext: public AsyncContext {
public:
    FusedAsyncContext(const std::vector<std::shared_ptr<AsyncContext>>& contexts);
    ~FusedAsyncContext() override;

public:
    bool done() const override {
        for (const auto& context : contexts_) {
            if (context && !context->done()) {
                return false;
            }
        }
        return true;
    }
    bool success() const override {
        for (const auto& context : contexts_) {
            if (context && !context->success()) {
                return false;
            }
        }
        return true;
    }

    const std::shared_ptr<KVCacheResourceV1>& resource() const {
        return resource_;
    }

private:
    std::shared_ptr<KVCacheResourceV1> resource_;
};

class FusedAsyncReadContext: public AsyncContext {
public:
    FusedAsyncReadContext(const std::shared_ptr<FusedAsyncContext>& fused_match_context);
    ~FusedAsyncReadContext() override;

public:
    bool done() const override {
        if (!fused_match_context) {
            return true;
        }
        if (!fused_match_context->done()) {
            return false;
        }
        if (!fused_match_context->success()) {
            return true;
        }
        return fused_read_context_ && fused_read_context_->done();
    }
    bool success() const override {
        return done() && fused_match_context_->success() && (!fused_read_context_ || fused_read_context_->success());
    }

    const std::shared_ptr<FusedAsyncContext>& fused_match_context() const {
        return fused_match_context_;
    }
    const std::shared_ptr<FusedAsyncContext>& fused_read_context() const {
        return fused_read_context_;
    }

private:
    std::shared_ptr<FusedAsyncContext> fused_match_context_;
    std::shared_ptr<FusedAsyncContext> fused_read_context_;
};

class KVCacheConnectorCordinator {
public:
    KVCacheConnectorCordinator();
    ~KVCacheConnectorCordinator();

public:
    std::shared_ptr<AsyncContext> asyncRead(const std::shared_ptr<KVCacheResourceV1>& resource,
                                            const std::shared_ptr<Meta>&              meta);
    std::shared_ptr<AsyncContext> asyncWrite(const std::shared_ptr<KVCacheResourceV1>& resource,
                                             const std::shared_ptr<Meta>&              meta);
    std::shared_ptr<AsyncContext> asyncWriteByLayer(int                                       layer_id,
                                                    const std::shared_ptr<KVCacheResourceV1>& resource,
                                                    const std::shared_ptr<Meta>&              meta);

private:
    bool updateOnce();

private:
    std::shared_ptr<MemoryConnector> memory_connector_;
    std::shared_ptr<RemoteConnector> remote_connector_;
    std::shared_ptr<P2PConnector>    p2p_connector_;

    std::vector<std::shared_ptr<KVCacheConnector>> connectors_;

    mutable std::mutex                                update_mutex_;
    std::list<std::shared_ptr<FusedAsyncReadContext>> fused_async_read_context_list_;
    std::list<std::shared_ptr<FusedAsyncContext>>     fused_async_write_context_list_;
    autil::LoopThreadPtr                              update_thread_;
};

}  // namespace rtp_llm