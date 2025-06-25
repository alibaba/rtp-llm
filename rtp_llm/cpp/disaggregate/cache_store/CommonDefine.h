#pragma once

#include <string>
#include <functional>
#include <stdint.h>
#include <unistd.h>
#include <map>
#include <sstream>

namespace rtp_llm {

enum class CacheStoreErrorCode {
    None = 0,

    // common error
    InvalidParams        = 1,
    PushWorkerItemFailed = 2,

    // load failed error
    LoadConnectFailed     = 3,
    LoadSendRequestFailed = 4,
    CallPrefillTimeout    = 5,
    LoadRdmaConnectFailed = 6,
    LoadRdmaWriteFailed   = 7,
    LoadBufferTimeout     = 8,
    LoadErrorUnknown      = 9,

    // store failed
    StoreFailed = 10,
};

inline std::string CacheStoreErrorCodeToString(CacheStoreErrorCode code) {
    switch (code) {
        case CacheStoreErrorCode::None:
            return "None";
        case CacheStoreErrorCode::InvalidParams:
            return "InvalidParams";
        case CacheStoreErrorCode::PushWorkerItemFailed:
            return "PushWorkerItemFailed";
        case CacheStoreErrorCode::LoadConnectFailed:
            return "LoadConnectFailed";
        case CacheStoreErrorCode::LoadSendRequestFailed:
            return "LoadSendRequestFailed";
        case CacheStoreErrorCode::CallPrefillTimeout:
            return "CallPrefillTimeout";
        case CacheStoreErrorCode::LoadRdmaConnectFailed:
            return "LoadRdmaConnectFailed";
        case CacheStoreErrorCode::LoadRdmaWriteFailed:
            return "LoadRdmaWriteFailed";
        case CacheStoreErrorCode::LoadBufferTimeout:
            return "LoadBufferTimeout";
        case CacheStoreErrorCode::LoadErrorUnknown:
            return "LoadErrorUnknown";
        case CacheStoreErrorCode::StoreFailed:
            return "StoreFailed";
        default:
            return "Error: Unrecognized ErrorCode";
    }
}

typedef std::function<void(bool, CacheStoreErrorCode)> CacheStoreStoreDoneCallback;
typedef std::function<void(bool, CacheStoreErrorCode)> CacheStoreLoadDoneCallback;
typedef std::function<void(bool)>                      WriteBlockDoneCallback;
typedef std::function<void(bool, CacheStoreErrorCode, const std::map<std::string, std::string>&)>
                  CacheStoreRemoteStoreDoneCallback;
const std::string kEnvRdmaMode             = "CACHE_STORE_RDMA_MODE";
const std::string kEnvRdmaWriteBlockConcat = "CACHE_STORE_RDMA_WRITE_BLOCK_CONCAT";
const uint32_t    kTcpRdmaPortDiff         = 100;

struct RemoteStoreRequest {
    std::string                        client_id;
    std::string                        request_id;
    std::string                        remote_addr;
    uint32_t                           remote_port;
    uint32_t                           remote_rdma_port;
    int64_t                            deadline_us;
    uint32_t                           local_partition_count  = 1;
    uint32_t                           local_partition_id     = 0;
    uint32_t                           remote_partition_count = 1;
    uint32_t                           remote_partition_id    = 0;
    std::map<std::string, std::string> buffer_pairs;

    RemoteStoreRequest(const std::string& clientid,
                       const std::string& request_id,
                       const std::string& remote_addr,
                       uint32_t           remote_port,
                       uint32_t           remote_rdma_port,
                       int64_t            deadline_us,
                       uint32_t           local_partition_count,
                       uint32_t           local_partition_id,
                       uint32_t           remote_partition_count,
                       uint32_t           remote_partition_id):
        client_id(clientid),
        request_id(request_id),
        remote_addr(remote_addr),
        remote_port(remote_port),
        remote_rdma_port(remote_rdma_port),
        deadline_us(deadline_us),
        local_partition_count(local_partition_count),
        local_partition_id(local_partition_id),
        remote_partition_count(remote_partition_count),
        remote_partition_id(remote_partition_id) {}

    std::string toString() const {
        std::stringstream ss;
        ss << "client_id: " << client_id << ", request_id: " << request_id << ", remote_addr: " << remote_addr
           << ", remote_port: " << remote_port << ", remote_rdma_port: " << remote_rdma_port
           << ", deadline_us: " << deadline_us << ", local_partition_count: " << local_partition_count
           << ", local_partition_id: " << local_partition_id << ", remote_partition_count: " << remote_partition_count
           << ", remote_partition_id: " << remote_partition_id;
        ss << ", buffer_pair: ";
        for (const auto& buffer_pair : buffer_pairs) {
            ss << buffer_pair.first << ":" << buffer_pair.second << "; ";
        }
        return ss.str();
    }
};

}  // namespace rtp_llm