#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

inline const char* blockIOStatusName(BlockIOStatus status) {
    switch (status) {
        case BlockIOStatus::OK:
            return "OK";
        case BlockIOStatus::INVALID_BLOCK:
            return "INVALID_BLOCK";
        case BlockIOStatus::INVALID_SIZE:
            return "INVALID_SIZE";
        case BlockIOStatus::ALIGNMENT_ERROR:
            return "ALIGNMENT_ERROR";
        case BlockIOStatus::IO_ERROR:
            return "IO_ERROR";
        case BlockIOStatus::PARTIAL_FAILURE:
            return "PARTIAL_FAILURE";
    }
    return "UNKNOWN";
}

inline TransferStatus blockIOStatusToTransferStatus(BlockIOStatus status) {
    switch (status) {
        case BlockIOStatus::OK:
            return TransferStatus::OK;
        case BlockIOStatus::INVALID_BLOCK:
        case BlockIOStatus::INVALID_SIZE:
        case BlockIOStatus::ALIGNMENT_ERROR:
            return TransferStatus::INVALID_ARGS;
        case BlockIOStatus::IO_ERROR:
        case BlockIOStatus::PARTIAL_FAILURE:
            return TransferStatus::DISK_IO_ERROR;
    }
    return TransferStatus::DISK_IO_ERROR;
}

}  // namespace rtp_llm
