#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/RemoteStoreTask.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreMetricsCollector.h"

#include <mutex>
#include <condition_variable>

namespace rtp_llm {

class RemoteStoreTaskImpl: public RemoteStoreTask, public std::enable_shared_from_this<RemoteStoreTaskImpl> {
public:
    RemoteStoreTaskImpl(const std::shared_ptr<RemoteStoreRequest>&                    request,
                        const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>& collector,
                        CheckCancelFunc                                               check_cancel_func);
    ~RemoteStoreTaskImpl() = default;

    void waitDone() override;
    bool success() const override;
    bool setDoneCallback(DoneCallback callback) override;
    bool setDrainedCallback(DrainedCallback callback) override;
    bool cancel(const ErrorInfo& error) override;
    bool done() const;

    // A returned request is already counted as in flight. The caller must
    // dispatch it or invoke its callback with failure; it must not be abandoned.
    std::shared_ptr<TransferRequest>
    makeAvailableRequest(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer);
    std::shared_ptr<TransferRequest> makeAvailableRequest(const std::vector<std::shared_ptr<BlockBuffer>>& blocks);
    void notifyRequestDone(const std::map<std::string, std::string>& block_keys, bool success);

private:
    void            completeRpc(uint64_t id, bool success);
    void            finishLocked(const ErrorInfo& error);
    void            notifyCallbacks();
    DrainedCallback takeDrainedCallback();

    // Admission, terminal state, callback registration and drain accounting share
    // one mutex. No caller callback or cancellation predicate runs under it.
    mutable std::mutex                                     mutex_;
    std::condition_variable                                cond_;
    std::map<std::string, std::string>                     to_load_buffers_;
    std::map<uint64_t, std::map<std::string, std::string>> pending_rpcs_;
    uint64_t                                               next_rpc_id_                 = 0;
    size_t                                                 outstanding_rpcs_            = 0;
    size_t                                                 done_buffer_count_           = 0;
    size_t                                                 expect_done_buffer_count_    = 0;
    bool                                                   done_                        = false;
    bool                                                   done_callback_registered_    = false;
    bool                                                   drained_callback_registered_ = false;
    bool                                                   done_callback_running_       = false;
    DoneCallback                                           done_callback_;
    DrainedCallback                                        drained_callback_;
    std::shared_ptr<CacheStoreRemoteStoreMetricsCollector> collector_;
};

}  // namespace rtp_llm
