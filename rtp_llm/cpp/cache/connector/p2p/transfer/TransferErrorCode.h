#pragma once

#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {
namespace transfer {

/// @brief Transfer-layer error codes, scoped to the transport abstraction.
/// Callers outside the transfer layer (e.g. P2PConnectorWorker) convert to
/// rtp_llm::ErrorCode via toErrorCode().
enum class TransferErrorCode {
    OK = 0,

    // Task lifecycle (IKVCacheRecvTask)
    CANCELLED,  // cancel() / forceCancel() called explicitly
    TIMEOUT,    // task exceeded its deadline

    // Sender preparation (IKVCacheSender callback)
    BUILD_REQUEST_FAILED,  // CUDA host-copy failed or RDMA MR not registered

    // Transport channel
    CONNECTION_FAILED,  // cannot acquire RPC channel to remote peer
    RPC_FAILED,         // network-level RPC failure (controller->Failed())

    // Data semantics
    BUFFER_MISMATCH,  // block size / structure mismatch reported by receiver

    // RDMA-specific
    RDMA_FAILED,  // RDMA connect / read operation failed

    // Catch-all
    UNKNOWN,  // service not initialised, thread-pool error, context destroyed, etc.
};

/// @brief Convert a transfer-layer error code to the global rtp_llm::ErrorCode.
/// This function is the single authoritative mapping and must only be called
/// at the p2p-connector layer boundary (e.g. P2PConnectorWorker).
inline ErrorCode toErrorCode(TransferErrorCode ec) {
    switch (ec) {
        case TransferErrorCode::OK:
            return ErrorCode::NONE_ERROR;
        case TransferErrorCode::CANCELLED:
            return ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED;
        case TransferErrorCode::TIMEOUT:
            return ErrorCode::P2P_CONNECTOR_WORKER_READ_TIMEOUT;
        case TransferErrorCode::BUILD_REQUEST_FAILED:
            return ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_FAILED;
        case TransferErrorCode::CONNECTION_FAILED:
            return ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_FAILED;
        case TransferErrorCode::RPC_FAILED:
            return ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_FAILED;
        case TransferErrorCode::BUFFER_MISMATCH:
            return ErrorCode::P2P_CONNECTOR_WORKER_READ_BUFFER_MISMATCH;
        case TransferErrorCode::RDMA_FAILED:
            return ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_RDMA_FAILED;
        case TransferErrorCode::UNKNOWN:
            return ErrorCode::P2P_CONNECTOR_WORKER_READ_FAILED;
        default:
            return ErrorCode::P2P_CONNECTOR_WORKER_READ_FAILED;
    }
}

}  // namespace transfer
}  // namespace rtp_llm
