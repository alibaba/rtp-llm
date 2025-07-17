#include "rtp_llm/cpp/disaggregate/cache_store/RemoteStoreTask.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreMetricsCollector.h"

#include <mutex>
#include <condition_variable>

namespace rtp_llm {

class RemoteStoreTaskImpl: public RemoteStoreTask, public std::enable_shared_from_this<RemoteStoreTaskImpl> {
public:
    RemoteStoreTaskImpl(const std::shared_ptr<RemoteStoreRequest>& request, 
                        const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector >& collector,
                        CheckCancelFunc check_cancel_func);
    ~RemoteStoreTaskImpl();

public:
    void waitDone() override;
    bool success() const override;

public:
    std::shared_ptr<TransferRequest>
    makeAvailableRequest(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer);
    std::shared_ptr<TransferRequest> makeAvailableRequest(const std::vector<std::shared_ptr<BlockBuffer>>& blocks);
    void notifyRequestDone(const std::map<std::string, std::string>& block_keys, bool success);
    bool done() const {
        return done_;
    }

private:
    std::shared_mutex                  buffers_mutex_;
    std::map<std::string, std::string> to_load_buffers_;
    std::map<std::string, std::string> loading_buffers_;
    std::map<std::string, std::string> done_buffers_;

    std::shared_ptr<CacheStoreRemoteStoreMetricsCollector> collector_;

    mutable std::mutex      mutex_;
    std::condition_variable cond_;

    int  expect_done_buffer_count_ = 0;
    bool all_success_              = true;
    bool done_                     = false;
};

}  // namespace rtp_llm