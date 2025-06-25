#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"

namespace rtp_llm {

class TransferConnection {
public:
    typedef std::function<void(bool, CacheStoreErrorCode, const std::vector<std::shared_ptr<BlockBuffer>>&)>
        ReadDoneCallback;

public:
    virtual void read(const std::vector<std::shared_ptr<BlockBuffer>>&     local_blocks,
                      const std::vector<std::shared_ptr<BlockBufferInfo>>& remote_blocks,
                      ReadDoneCallback                                     callback,
                      uint32_t                                             timeout_ms) = 0;
};

}  // namespace rtp_llm