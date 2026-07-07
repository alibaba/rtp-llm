#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/connector/kvs/KVSObjectStore.h"
#include "rtp_llm/cpp/cache/connector/kvs/KVSConnectorConfig.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

struct KVSConnectorTaskState;

class KVSAsyncContext: public AsyncMatchContext {
public:
    enum class State {
        INIT,
        RUNNING,
        SUCCESS,
        FAILED,
    };

    KVSAsyncContext();

    void      waitDone() override;
    bool      done() const override;
    bool      success() const override;
    size_t    matchedBlockCount() const override;
    ErrorInfo errorInfo() const override;

    void markRunning();
    void markSuccess(size_t matched_block_count = 0);
    void markFailed(const std::string& message);

private:
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    State                   state_{State::INIT};
    size_t                  matched_block_count_{0};
    ErrorInfo               error_info_;
};

class KVSMatchContext: public KVSAsyncContext {
public:
    KVSMatchContext() = default;
    ~KVSMatchContext();

    void setMatchResult(std::shared_ptr<KVSObjectStore>                 store,
                        KVSReadHandle                                  handle,
                        std::vector<std::vector<KVSObjectBuffer>>       block_objects,
                        std::string                                    trace_id,
                        size_t                                         matched_block_count);
    bool hasHandle() const;
    const KVSReadHandle& handle() const;
    const std::vector<std::vector<KVSObjectBuffer>>& blockObjects() const;
    void releaseHandle();

private:
    std::shared_ptr<KVSObjectStore>           store_;
    KVSReadHandle                            handle_;
    std::vector<std::vector<KVSObjectBuffer>> block_objects_;
    std::string                              trace_id_;
    bool                                     has_handle_{false};
    bool                                     released_{false};
};

class KVSConnector: public KVCacheConnector {
public:
    enum class Operation {
        READ,
        WRITE,
    };

    struct KVSBufferSpec {
        int          layer_id{0};
        int          group_id{0};
        BlockIdxType block_id{NULL_BLOCK_IDX};
        size_t       object_offset{0};
    };

    struct KVSObjectPlan {
        std::string                object_key;
        std::vector<KVSBufferSpec> buffers;
        size_t                     rank_bytes{0};
    };

    using BlockBufferResolver = std::function<std::vector<BlockInfo>(int layer_id, int group_id, BlockIdxType block_id)>;
    using KVSPlanSender = std::function<bool(Operation                         operation,
                                             const std::vector<KVSObjectPlan>& objects,
                                             const std::string&                trace_id)>;

    KVSConnector(const CacheConfig&                 cache_config,
                 KVSConnectorConfig                config,
                 std::shared_ptr<KVSObjectStore>   store,
                 BlockBufferResolver               block_buffer_resolver,
                 std::vector<std::string>          tp_addrs         = {},
                 size_t                            tp_size          = 1,
                 KVSPlanSender                     plan_sender      = nullptr);
    ~KVSConnector() override;

    bool init();

    bool executeWorkerPlan(const KVSOperationRequestPB& request, KVSOperationResponsePB& response);

    std::shared_ptr<AsyncMatchContext> asyncMatch(const std::shared_ptr<KVCacheResource>& resource,
                                                  const std::shared_ptr<Meta>&            meta) override;
    std::shared_ptr<AsyncContext>      asyncRead(const std::shared_ptr<KVCacheResource>&   resource,
                                                 const std::shared_ptr<Meta>&              meta,
                                                 const std::shared_ptr<AsyncMatchContext>& match_context,
                                                 int                                       start_read_block_index,
                                                 int                                       read_block_num) override;
    std::shared_ptr<AsyncContext>      asyncWrite(const std::shared_ptr<KVCacheResource>& resource,
                                                  const std::shared_ptr<Meta>&            meta) override;
    std::shared_ptr<AsyncContext>
    asyncWriteByLayer(int layer_id, const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) override;

private:
    bool submitTask(const std::shared_ptr<KVSAsyncContext>& context, std::function<void()> task) const;
    bool sendTpPlan(Operation operation, const std::vector<KVSObjectPlan>& objects, const std::string& trace_id) const;

private:
    CacheConfig                                 cache_config_;
    KVSConnectorConfig                          config_;
    std::shared_ptr<KVSObjectStore>             store_;
    std::shared_ptr<const KVSConnectorTaskState> task_state_;
    std::vector<std::string>                    tp_addrs_;
    size_t                                      tp_size_{1};
    KVSPlanSender                               plan_sender_;
    std::shared_ptr<class BroadcastManager>     broadcast_manager_;
    std::shared_ptr<autil::LockFreeThreadPool>  worker_thread_pool_;
    std::mutex                                  worker_plan_mutex_;
};

}  // namespace rtp_llm
