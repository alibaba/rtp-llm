#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/LockedBlockBufferManager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TransferConnection.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"

namespace rtp_llm {

class CacheTransferServiceImplContext: public std::enable_shared_from_this<CacheTransferServiceImplContext> {
public:
    CacheTransferServiceImplContext(const ::CacheTransferRequest*                        request,
                                    ::CacheTransferResponse*                             response,
                                    ::google::protobuf::Closure*                         done,
                                    const std::vector<std::shared_ptr<BlockBuffer>>&     local_blocks,
                                    const std::vector<std::shared_ptr<BlockBufferInfo>>& remote_blocks,
                                    const std::shared_ptr<LockedBlockBufferManager>&     locked_block_buffer_manager,
                                    const std::shared_ptr<MemoryUtil>&                   memory_util,
                                    const std::shared_ptr<TransferConnection>&           transfer_connection);

    ~CacheTransferServiceImplContext() = default;

public:
    void run();

protected:
    void
    notifyDone(bool success, CacheStoreErrorCode error_code, const std::vector<std::shared_ptr<BlockBuffer>>& blocks);
    void                         runSuccess();
    void                         runFailed(CacheStoreErrorCode ec);
    KvCacheStoreServiceErrorCode toProtoEc(CacheStoreErrorCode ec);

protected:
    const ::CacheTransferRequest* request_;
    ::CacheTransferResponse*      response_;
    ::google::protobuf::Closure*  done_;

    std::string request_id_;
    std::string client_ip_;

    std::vector<std::shared_ptr<BlockBuffer>>     local_blocks_;
    std::vector<std::shared_ptr<BlockBufferInfo>> remote_blocks_;

    std::shared_ptr<LockedBlockBufferManager> locked_block_buffer_manager_;
    std::shared_ptr<MemoryUtil>               memory_util_;
    std::shared_ptr<TransferConnection>       transfer_connection_;

    std::mutex mutex_;
    bool       finished_{false};
    int        unfinished_count_{0};
};

}  // namespace rtp_llm