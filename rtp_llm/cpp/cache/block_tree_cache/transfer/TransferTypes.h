#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

struct TreeNode;

enum class TaskCompletion { UNKNOWN, PENDING, FINISHED };

enum class TransferStatus {
    OK,
    INVALID_ARGS,
    DEVICE_IO_ERROR,
    DISK_IO_ERROR,
    RESOURCE_EXHAUSTED,
    CRC_MISMATCH,
    CRC_COMPUTE_ERROR,
};

// Diagnostic categories for the existing cache-copy metric. They do not change
// the business error or recovery policy. Higher values take precedence when a
// logical copy contains multiple failures; confirmed CRC evidence survives RPC
// failures on other ranks.
enum class CacheCopyError : uint8_t {
    NONE,
    NOT_EXECUTED,
    COPY_FAILED,
    INVALID_REQUEST,
    IO_FAILED,
    RPC_FAILED,
    CRC_COMPUTE_FAILED,
    CRC_MISMATCH,
};

inline const char* cacheCopyErrorName(CacheCopyError error) {
    switch (error) {
        case CacheCopyError::NONE:
            return "NONE";
        case CacheCopyError::NOT_EXECUTED:
            return "NOT_EXECUTED";
        case CacheCopyError::COPY_FAILED:
            return "COPY_FAILED";
        case CacheCopyError::INVALID_REQUEST:
            return "INVALID_REQUEST";
        case CacheCopyError::IO_FAILED:
            return "IO_FAILED";
        case CacheCopyError::RPC_FAILED:
            return "RPC_FAILED";
        case CacheCopyError::CRC_COMPUTE_FAILED:
            return "CRC_COMPUTE_FAILED";
        case CacheCopyError::CRC_MISMATCH:
            return "CRC_MISMATCH";
    }
    return "COPY_FAILED";
}

inline CacheCopyError cacheCopyErrorFor(TransferStatus status) {
    switch (status) {
        case TransferStatus::OK:
            return CacheCopyError::NONE;
        case TransferStatus::INVALID_ARGS:
            return CacheCopyError::INVALID_REQUEST;
        case TransferStatus::DISK_IO_ERROR:
            return CacheCopyError::IO_FAILED;
        case TransferStatus::CRC_COMPUTE_ERROR:
            return CacheCopyError::CRC_COMPUTE_FAILED;
        case TransferStatus::CRC_MISMATCH:
            return CacheCopyError::CRC_MISMATCH;
        default:
            return CacheCopyError::COPY_FAILED;
    }
}

inline ErrorInfo transferStatusToErrorInfo(TransferStatus status) {
    switch (status) {
        case TransferStatus::OK:
            return ErrorInfo::OkStatus();
        case TransferStatus::INVALID_ARGS:
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid block transfer request");
        case TransferStatus::DEVICE_IO_ERROR:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "device block transfer failed");
        case TransferStatus::DISK_IO_ERROR:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "disk block transfer failed");
        case TransferStatus::CRC_COMPUTE_ERROR:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "cache CRC computation failed");
        case TransferStatus::CRC_MISMATCH:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "cache integrity verification failed");
        case TransferStatus::RESOURCE_EXHAUSTED:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "device-disk staging pool exhausted");
    }
    return ErrorInfo(ErrorCode::UNKNOWN_ERROR, "unknown block transfer status");
}
// Results are ordered like the task's descriptors; NONE means the target copy completed,
// and NOT_EXECUTED means no target was committed to the caller. The copy may
// have been skipped or discarded after a batch CRC failure; target bytes can change.
// An empty block_errors vector represents a uniform result until the task boundary expands it.
// copy_error summarizes diagnostic priority. Unknown completion takes priority in
// status.code() so CRC evidence cannot hide outstanding I/O; status keeps all failure messages.
struct TransferResult {
    TransferResult() = default;
    explicit TransferResult(size_t block_count): block_errors(block_count, CacheCopyError::NOT_EXECUTED) {}
    TransferResult(TransferStatus status):
        TransferResult(transferStatusToErrorInfo(status), cacheCopyErrorFor(status)) {}
    TransferResult(ErrorInfo error):
        status(std::move(error)),
        copy_error(status.ok()                                ? CacheCopyError::NONE :
                   status.code() == ErrorCode::INVALID_PARAMS ? CacheCopyError::INVALID_REQUEST :
                                                                CacheCopyError::COPY_FAILED) {}
    TransferResult(ErrorInfo error, CacheCopyError copy_error): status(std::move(error)), copy_error(copy_error) {}

    bool ok() const {
        return status.ok();
    }

    // Append every failure message; keep the first error code at equal priority.
    void merge(TransferResult other) {
        const auto unknown = ErrorCode::TRANSFER_COMPLETION_UNKNOWN;
        if ((status.code() == unknown && recovery_task_ids.empty())
            || (other.status.code() == unknown && other.recovery_task_ids.empty())) {
            recovery_task_ids.clear();
        } else {
            recovery_task_ids.insert(recovery_task_ids.end(), other.recovery_task_ids.begin(), other.recovery_task_ids.end());
        }
        if (other.ok()) {
            return;
        }
        if (ok()) {
            status     = std::move(other.status);
            copy_error = other.copy_error;
            return;
        }
        const bool completion_unknown = status.code() == ErrorCode::TRANSFER_COMPLETION_UNKNOWN
                                        || other.status.code() == ErrorCode::TRANSFER_COMPLETION_UNKNOWN;
        const auto code = completion_unknown            ? ErrorCode::TRANSFER_COMPLETION_UNKNOWN :
                          other.copy_error > copy_error ? other.status.code() :
                                                          status.code();
        status          = ErrorInfo(code, status.ToString() + "; " + other.status.ToString());
        copy_error      = std::max(copy_error, other.copy_error);
    }

    CacheCopyError blockError(size_t index) const {
        return block_errors.empty() ? copy_error : block_errors[index];
    }

    void expandBlockErrors(size_t block_count) {
        if (block_errors.empty()) {
            block_errors.assign(block_count, copy_error);
        }
    }

    ErrorInfo                   status{ErrorInfo::OkStatus()};
    CacheCopyError              copy_error{CacheCopyError::NONE};
    std::vector<CacheCopyError> block_errors;
    // Empty on UNKNOWN means unrecoverable; otherwise query every listed execution.
    std::vector<std::string> recovery_task_ids;
};

struct DeviceHostCopyOptions {
    size_t staged_sm_min_tile_count{16};
    size_t staged_sm_min_bytes{64 * 1024};
    bool   staged_sm_copy_enabled{true};
    bool   cuda_batch_copy_enabled{true};
};

struct HostBufferView {
    void*  base{nullptr};
    size_t payload_bytes{0};
    // Safe access range from base; disk I/O may require capacity beyond the logical payload.
    size_t capacity_bytes{0};
};

inline bool
isValidHostBufferView(const HostBufferView& view, size_t required_payload_bytes, size_t required_access_bytes) {
    return view.base != nullptr && view.payload_bytes <= view.capacity_bytes
           && view.payload_bytes >= required_payload_bytes && view.capacity_bytes >= required_access_bytes;
}

// Unified operation descriptor for load, eviction, and transfer execution.
// Business-only fields are ignored by transfer executors and RPC conversion.
struct TransferDescriptor {
    TransferDescriptor() = default;
    TransferDescriptor(TreeNode*                 node,
                       size_t                    group_set_id,
                       size_t                    path_index,
                       Tier                      source_tier,
                       Tier                      target_tier,
                       std::vector<BlockIdxType> source_blocks):
        node(node),
        group_set_id(group_set_id),
        path_index(path_index),
        source_tier(source_tier),
        target_tier(target_tier),
        source_blocks(std::move(source_blocks)) {}

    static TransferDescriptor
    deviceToHost(size_t group_set_id, std::vector<BlockIdxType> device_blocks, BlockIdxType host_block) {
        TransferDescriptor desc;
        desc.group_set_id  = group_set_id;
        desc.source_tier   = Tier::DEVICE;
        desc.target_tier   = Tier::HOST;
        desc.source_blocks = std::move(device_blocks);
        desc.target_blocks = {host_block};
        return desc;
    }

    static TransferDescriptor
    hostToDevice(size_t group_set_id, BlockIdxType host_block, std::vector<BlockIdxType> device_blocks) {
        TransferDescriptor desc;
        desc.group_set_id  = group_set_id;
        desc.source_tier   = Tier::HOST;
        desc.target_tier   = Tier::DEVICE;
        desc.source_blocks = {host_block};
        desc.target_blocks = std::move(device_blocks);
        return desc;
    }

    static TransferDescriptor hostToDisk(size_t group_set_id, BlockIdxType host_block, BlockIdxType disk_block) {
        TransferDescriptor desc;
        desc.group_set_id  = group_set_id;
        desc.source_tier   = Tier::HOST;
        desc.target_tier   = Tier::DISK;
        desc.source_blocks = {host_block};
        desc.target_blocks = {disk_block};
        return desc;
    }

    static TransferDescriptor diskToHost(size_t group_set_id, BlockIdxType disk_block, BlockIdxType host_block) {
        TransferDescriptor desc;
        desc.group_set_id  = group_set_id;
        desc.source_tier   = Tier::DISK;
        desc.target_tier   = Tier::HOST;
        desc.source_blocks = {disk_block};
        desc.target_blocks = {host_block};
        return desc;
    }

    static TransferDescriptor
    deviceToDisk(size_t group_set_id, std::vector<BlockIdxType> device_blocks, BlockIdxType disk_block) {
        TransferDescriptor desc;
        desc.group_set_id  = group_set_id;
        desc.source_tier   = Tier::DEVICE;
        desc.target_tier   = Tier::DISK;
        desc.source_blocks = std::move(device_blocks);
        desc.target_blocks = {disk_block};
        return desc;
    }

    static TransferDescriptor
    diskToDevice(size_t group_set_id, BlockIdxType disk_block, std::vector<BlockIdxType> device_blocks) {
        TransferDescriptor desc;
        desc.group_set_id  = group_set_id;
        desc.source_tier   = Tier::DISK;
        desc.target_tier   = Tier::DEVICE;
        desc.source_blocks = {disk_block};
        desc.target_blocks = std::move(device_blocks);
        return desc;
    }

    bool needsTransfer() const {
        return target_tier != Tier::NONE && source_tier != target_tier;
    }

    const std::vector<BlockIdxType>& blocksAt(Tier tier) const {
        return source_tier == tier ? source_blocks : target_blocks;
    }

    BlockIdxType singleBlockAt(Tier tier) const {
        return blocksAt(tier)[0];
    }

    bool isExecutable() const {
        const bool supported_direction =
            (source_tier == Tier::DEVICE && (target_tier == Tier::HOST || target_tier == Tier::DISK))
            || (source_tier == Tier::HOST && (target_tier == Tier::DEVICE || target_tier == Tier::DISK))
            || (source_tier == Tier::DISK && (target_tier == Tier::DEVICE || target_tier == Tier::HOST));
        return supported_direction && endpointResolved(source_tier, source_blocks)
               && endpointResolved(target_tier, target_blocks);
    }

    std::string debugString() const {
        return "TransferDescriptor{group_set_id=" + std::to_string(group_set_id)
               + ", direction=" + tierName(source_tier) + "->" + tierName(target_tier) + ", source_blocks=["
               + blocksDebugString(source_blocks) + "], target_blocks=[" + blocksDebugString(target_blocks) + "]}";
    }

    // Null for node-independent descriptors, including DEVICE-source reuse
    // whose blocks may outlive eviction of the originating tree node.
    TreeNode*                 node{nullptr};
    size_t                    group_set_id{0};
    size_t                    path_index{0};
    Tier                      source_tier{Tier::NONE};
    Tier                      target_tier{Tier::NONE};
    std::vector<BlockIdxType> source_blocks;
    std::vector<BlockIdxType> target_blocks;

private:
    static bool endpointResolved(Tier tier, const std::vector<BlockIdxType>& blocks) {
        if (tier != Tier::DEVICE && tier != Tier::HOST && tier != Tier::DISK) {
            return false;
        }
        if (blocks.empty() || (tier != Tier::DEVICE && blocks.size() != 1)) {
            return false;
        }
        return std::none_of(blocks.begin(), blocks.end(), [](BlockIdxType block) { return isNullBlockIdx(block); });
    }

    static std::string blocksDebugString(const std::vector<BlockIdxType>& blocks) {
        std::string result;
        for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
            result += (block_index == 0 ? "" : ",") + std::to_string(blocks[block_index]);
        }
        return result;
    }
};

class TransferTask {
public:
    using Clock = std::chrono::steady_clock;

    TransferTask(std::vector<TransferDescriptor> descriptors, std::chrono::milliseconds timeout, std::string task_id = {}):
        descriptors_(std::move(descriptors)), deadline_(Clock::now() + timeout), task_id_(std::move(task_id)) {}

    const std::string& taskId() const { return task_id_; }

    const std::vector<TransferDescriptor>& descriptors() const {
        return descriptors_;
    }

    Clock::time_point deadline() const {
        return deadline_;
    }

    void addDescriptor(TransferDescriptor descriptor) {
        descriptors_.push_back(std::move(descriptor));
    }

    // Business task builders may fill or resolve descriptors before admission.
    std::vector<TransferDescriptor>& mutableDescriptorsForPreparation() {
        return descriptors_;
    }

    std::optional<std::chrono::milliseconds> remainingTimeout() const {
        const auto remaining = deadline_ - Clock::now();
        if (remaining <= Clock::duration::zero()) {
            return std::nullopt;
        }
        return std::chrono::ceil<std::chrono::milliseconds>(remaining);
    }

    bool expired() const {
        return !remainingTimeout().has_value();
    }

    TransferTask subtask(std::vector<TransferDescriptor> descriptors) const {
        auto task = TransferTask(std::move(descriptors), deadline_);
        task.task_id_ = task_id_;
        return task;
    }

private:
    TransferTask(std::vector<TransferDescriptor> descriptors, Clock::time_point deadline):
        descriptors_(std::move(descriptors)), deadline_(deadline) {}

    std::vector<TransferDescriptor> descriptors_;
    Clock::time_point               deadline_;
    std::string                    task_id_;
};

}  // namespace rtp_llm
