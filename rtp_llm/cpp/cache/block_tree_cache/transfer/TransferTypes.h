#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBufferSafety.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/AsyncProfilingScope.h"

namespace rtp_llm {

struct TreeNode;

// Transfer lanes describe cache movement, independently of business priority.
enum class TransferDirection {
    LOAD,
    EVICT,
};

// Disk -> Device uses host staging internally without exposing Disk -> Host.
inline bool isLoadTransfer(Tier source, Tier target) {
    return (source == Tier::DISK && target == Tier::DEVICE) || (source == Tier::HOST && target == Tier::DEVICE);
}

enum class TransferStatus {
    OK,
    INVALID_ARGS,
    DEVICE_IO_ERROR,
    DISK_IO_ERROR,
    RESOURCE_EXHAUSTED,
    CRC_MISMATCH,
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
        case TransferStatus::CRC_MISMATCH:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "cache integrity verification failed");
        case TransferStatus::RESOURCE_EXHAUSTED:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "device-disk staging pool exhausted");
    }
    return ErrorInfo(ErrorCode::UNKNOWN_ERROR, "unknown block transfer status");
}
enum class TransferRpcDispatchState { NOT_SUBMITTED, POSSIBLY_SUBMITTED, SUBMITTED };

// Preserve the existing execution identity and its namespace: ordinary transfer
// uses a numeric trace, while KVCM READ uses the original string from its request.
using TransferExecutionTrace = std::variant<uint64_t, std::string>;

inline bool validTransferExecutionTrace(const TransferExecutionTrace& trace) noexcept {
    if (const auto* numeric = std::get_if<uint64_t>(&trace)) {
        return *numeric != 0;
    }
    const auto* remote = std::get_if<std::string>(&trace);
    return remote && !remote->empty();
}

// A snapshot of one participant in one actual execution. A missing status is
// unknown, never an implicit successful RPC. Local executions have remote=false.
struct TransferRankSnapshot {
    size_t                   rank{0};
    std::string              server_address;
    bool                     remote{true};
    TransferRpcDispatchState dispatch_state{TransferRpcDispatchState::NOT_SUBMITTED};
    std::string              dispatch_error;
    bool                     cq_received{false};
    bool                     cq_ok{false};
    std::optional<int>       grpc_code;
    std::string              grpc_message;
    bool                     response_received{false};
    bool                     response_valid{false};
    std::string              response_error;
    std::optional<int64_t>   worker_error_code;
    std::string              worker_error_message;
    std::optional<CacheCopyError> worker_copy_error;
    std::vector<CacheCopyError>   block_errors;
    std::optional<bool>           buffer_safe;
    std::optional<int>            worker_transfer_status;
    std::optional<bool>           sdk_failed;
};

struct TransferExecutionDiagnostics {
    TransferExecutionTrace            trace_id{uint64_t{0}};
    std::string                       failure_reason;
    std::vector<TransferRankSnapshot> ranks;
};

// Results are ordered like the task's descriptors; NONE means the target copy completed,
// and NOT_EXECUTED means no target was committed to the caller. The copy may
// have been skipped or discarded after a batch CRC failure; target bytes can change.
// An empty block_errors vector represents a uniform result until the task boundary expands it.
// status.code() preserves unknown completion; copy_error summarizes the highest-priority diagnostic.
// status keeps all failure messages.
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

    // Identity for combining independent operations, not proof for an executed
    // operation. Every contributing result must explicitly prove buffer safety.
    static TransferResult emptyAggregate(size_t block_count = 0) {
        TransferResult result(block_count);
        result.buffer_safe = true;
        return result;
    }

    bool ok() const {
        return status.ok();
    }

    // Completion evidence must survive failure to allocate the diagnostic itself.
    static TransferResult executionException(bool safe, const char* message = nullptr) noexcept {
        TransferResult result;
        result.status.setErrorCode(ErrorCode::EXECUTION_EXCEPTION);
        result.copy_error = CacheCopyError::COPY_FAILED;
        result.buffer_safe = safe;
        if (message) {
            try {
                result.status = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, message);
            } catch (...) {}
        }
        return result;
    }

    // Append every failure message; keep unknown completion ahead of known failures
    // without losing higher-priority copy diagnostics such as confirmed CRC errors.
    void merge(TransferResult other) {
        // A successful business result can still lack completion proof. Merge
        // safety before either early return, independently of error precedence.
        buffer_safe = buffer_safe && other.buffer_safe;
        if (other.ok()) {
            return;
        }
        if (ok()) {
            status     = std::move(other.status);
            copy_error = other.copy_error;
            return;
        }
        auto code = other.copy_error > copy_error ? other.status.code() : status.code();
        if (status.code() == ErrorCode::TRANSFER_COMPLETION_UNKNOWN
            || other.status.code() == ErrorCode::TRANSFER_COMPLETION_UNKNOWN) {
            code = ErrorCode::TRANSFER_COMPLETION_UNKNOWN;
        }
        try {
            status = ErrorInfo(code, status.ToString() + "; " + other.status.ToString());
        } catch (...) {
            status.setErrorCode(code);
        }
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
    // True only when this operation can no longer access any of its buffers.
    // False means unproven, not necessarily in-flight. Business success, error
    // codes and NOT_EXECUTED diagnostics never implicitly grant this proof.
    bool buffer_safe{false};
    // Leaf execution diagnostics belong to this exact trace. The coordinator
    // captures them before merging into a parent task; merge() never conflates
    // snapshots belonging to different executions.
    std::shared_ptr<const TransferExecutionDiagnostics> execution_diagnostics;
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
            (source_tier == Tier::DEVICE && target_tier == Tier::HOST)
            || (source_tier == Tier::HOST && (target_tier == Tier::DEVICE || target_tier == Tier::DISK))
            || (source_tier == Tier::DISK && target_tier == Tier::DEVICE);
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

// Shared task description across preparation, RPC transport and local batching.
// Builders resolve descriptors before submission; execution entrances validate
// their batch constraints. Executors own their copy and do not mutate it.
// Results use the current descriptor order; PerRank owns the parent-index maps.
class TransferTask {
public:
    using Clock = std::chrono::steady_clock;

    TransferTask(std::vector<TransferDescriptor> descriptors,
                 std::chrono::milliseconds       timeout,
                 std::optional<int64_t>          request_id = {}):
        descriptors_(std::move(descriptors)), deadline_(Clock::now() + timeout), request_id_(request_id) {}

    std::optional<int64_t> requestId() const {
        return request_id_;
    }

    uint64_t traceId() const {
        if (AsyncProfileSession::enabled())
            AsyncProfileSession::taskLink(trace_id_, workflow_id_, parent_trace_id_, request_id_.value_or(-1));
        return trace_id_;
    }
    uint64_t workflowId() const {
        return workflow_id_;
    }
    uint64_t parentTraceId() const {
        return parent_trace_id_;
    }
    void setWorkflow(uint64_t workflow, uint64_t parent = 0) {
        workflow_id_     = workflow ? workflow : trace_id_;
        parent_trace_id_ = parent;
    }
    void setTraceId(uint64_t id) {
        if (id != 0) {
            if (workflow_id_ == trace_id_)
                workflow_id_ = id;
            trace_id_ = id;
        }
    }

    const std::shared_ptr<TransferBufferSafety>& completionProof() const noexcept {
        return completion_proof_;
    }
    void setCompletionProof(std::shared_ptr<TransferBufferSafety> proof) noexcept {
        completion_proof_ = std::move(proof);
    }

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
        // The proof covers this complete physical batch, not an arbitrary
        // subset or intermediate stage. Ownership must be transferred explicitly.
        auto task      = TransferTask(std::move(descriptors), *this);
        task.completion_proof_.reset();
        task.trace_id_ = trace_id_;
        task.workflow_id_     = workflow_id_;
        task.parent_trace_id_ = parent_trace_id_;
        return task;
    }

    // Construction gives each independently submitted RPC batch a new trace ID
    // as its execution identity. Nested worker batches inherit that trace ID
    // through subtask(), while the workflow links independent RPC batches.
    TransferTask rpcSubtask(std::vector<TransferDescriptor> descriptors) const {
        auto task = TransferTask(std::move(descriptors), *this);
        task.completion_proof_.reset();
        task.trace_id_ = next_trace_id_.fetch_add(1, std::memory_order_relaxed);
        task.setWorkflow(workflow_id_, trace_id_);
        return task;
    }

private:
    TransferTask(std::vector<TransferDescriptor> descriptors, const TransferTask& parent):
        descriptors_(std::move(descriptors)),
        deadline_(parent.deadline_),
        request_id_(parent.request_id_),
        trace_id_(parent.trace_id_),
        workflow_id_(parent.workflow_id_),
        parent_trace_id_(parent.parent_trace_id_) {}

    std::vector<TransferDescriptor>       descriptors_;
    Clock::time_point                     deadline_;
    std::optional<int64_t>                request_id_;
    std::shared_ptr<TransferBufferSafety> completion_proof_;
    inline static std::atomic<uint64_t> next_trace_id_{1};
    uint64_t                            trace_id_{next_trace_id_.fetch_add(1, std::memory_order_relaxed)};
    uint64_t                            workflow_id_{trace_id_};
    uint64_t                            parent_trace_id_{0};
};

}  // namespace rtp_llm
