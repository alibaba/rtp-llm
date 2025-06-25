#pragma once

#include "rtp_llm/cpp/utils/TimeUtil.h"

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

struct TransferRequest {
    int         id;          // request id in task
    std::string request_id;  // associated request id
    std::string ip;          // ip to transfer to
    uint32_t    port;        // port to transfer to
    uint32_t    rdma_port;   // rdma port to transfer to

    uint64_t timeout_ms;  // transfer timeout in ms

    // partition info for blocks to transfer
    uint32_t local_partition_count  = 1;
    uint32_t local_partition_id     = 0;
    uint32_t remote_partition_count = 1;
    uint32_t remote_partition_id    = 0;

    std::map<std::string, std::string> buffer_pairs;

    CacheStoreRemoteStoreDoneCallback callback;

    TransferRequest(const std::string& request_id,
                    const std::string& remote_addr,
                    uint32_t           remote_port,
                    uint32_t           remote_rdma_port,
                    uint64_t           timeout_ms):
        request_id(request_id),
        ip(remote_addr),
        port(remote_port),
        rdma_port(remote_rdma_port),
        timeout_ms(timeout_ms) {}

    TransferRequest(const std::shared_ptr<RemoteStoreRequest>& store_request):
        request_id(store_request->request_id),
        ip(store_request->remote_addr),
        port(store_request->remote_port),
        rdma_port(store_request->remote_rdma_port),
        timeout_ms((store_request->deadline_us - currentTimeUs()) / 1000 - 10),
        local_partition_count(store_request->local_partition_count),
        local_partition_id(store_request->local_partition_id),
        remote_partition_count(store_request->remote_partition_count),
        remote_partition_id(store_request->remote_partition_id) {}
};

}  // namespace rtp_llm