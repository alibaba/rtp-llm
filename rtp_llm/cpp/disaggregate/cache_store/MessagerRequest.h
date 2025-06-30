#pragma once

#include "autil/TimeUtility.h"

#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"

namespace rtp_llm {

struct LoadRequest {
    std::string                         ip;
    uint32_t                            port;
    uint32_t                            rdma_port;
    std::shared_ptr<RequestBlockBuffer> request_block_buffer;
    CacheStoreLoadDoneCallback          callback;
    uint32_t                            timeout_ms;
    int                                 partition_count;
    int                                 partition_id;
    LoadRequest(const std::string&                         ip,
                uint32_t                                   port,
                uint32_t                                   rdma_port,
                const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
                CacheStoreLoadDoneCallback                 callback,
                uint32_t                                   timeout_ms,
                int                                        partition_count,
                int                                        partition_id):
        ip(ip),
        port(port),
        rdma_port(rdma_port),
        request_block_buffer(request_block_buffer),
        callback(callback),
        timeout_ms(timeout_ms),
        partition_count(partition_count),
        partition_id(partition_id) {}
};

}  // namespace rtp_llm