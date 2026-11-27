#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"

#include <algorithm>
#include <exception>
#include <iterator>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

const GroupSet* resolveGroupSet(const MemoryOperationRequestPB::CopyItem& item,
                                const std::vector<GroupSetPtr>&           group_sets) {
    if (item.group_tags_size() == 0) {
        return nullptr;
    }
    std::unordered_set<std::string> requested_tags;
    for (const auto& tag : item.group_tags()) {
        if (tag.empty() || !requested_tags.emplace(tag).second) {
            return nullptr;
        }
    }
    const GroupSet* match = nullptr;
    for (const auto& candidate : group_sets) {
        if (!candidate || candidate->groupTags().size() != requested_tags.size()) {
            continue;
        }
        const bool same_members = std::all_of(
            candidate->groupTags().begin(), candidate->groupTags().end(), [&requested_tags](const std::string& tag) {
                return requested_tags.find(tag) != requested_tags.end();
            });
        if (!same_members) {
            continue;
        }
        if (match != nullptr) {
            return nullptr;
        }
        match = candidate.get();
    }
    return match;
}

}  // namespace

bool BlockTransferRequestConverter::directionFor(const TransferDescriptor&                descriptor,
                                                 MemoryOperationRequestPB::CopyDirection& request_direction) {
    if (descriptor.source_tier == Tier::DEVICE && descriptor.target_tier == Tier::HOST) {
        request_direction = MemoryOperationRequestPB::D2H;
        return true;
    }
    if (descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DEVICE) {
        request_direction = MemoryOperationRequestPB::H2D;
        return true;
    }
    if (descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DISK) {
        request_direction = MemoryOperationRequestPB::H2DISK;
        return true;
    }
    if (descriptor.source_tier == Tier::DISK && descriptor.target_tier == Tier::DEVICE) {
        request_direction = MemoryOperationRequestPB::DISK2D;
        return true;
    }
    return false;
}

bool BlockTransferRequestConverter::decodeDeviceBlocks(const CopyItem&            item,
                                                       const GroupSet&            group_set,
                                                       std::vector<BlockIdxType>& blocks) {
    const auto&                                   device_pools = group_set.devicePools();
    const auto&                                   group_tags   = group_set.groupTags();
    std::unordered_map<std::string, BlockIdxType> blocks_by_tag;
    for (const auto& group_block : item.group_blocks()) {
        if (group_block.tag().empty() || !blocks_by_tag.emplace(group_block.tag(), group_block.block_id()).second) {
            return false;
        }
    }
    if (blocks_by_tag.size() != group_tags.size() || device_pools.size() != group_tags.size()) {
        return false;
    }
    blocks.reserve(group_tags.size());
    for (size_t i = 0; i < group_tags.size(); ++i) {
        const auto it = blocks_by_tag.find(group_tags[i]);
        if (it == blocks_by_tag.end() || !device_pools[i]->validBlock(it->second)) {
            return false;
        }
        blocks.push_back(it->second);
    }
    return true;
}

bool BlockTransferRequestConverter::encodeTransfer(MemoryOperationRequestPB&       request,
                                                   const TransferTask&             task,
                                                   const std::vector<GroupSetPtr>& group_sets) {
    const auto remaining = task.remainingTimeout();
    if (!remaining) {
        return false;
    }
    request.set_timeout_ms(remaining->count());
    request.set_transfer_trace_id(task.traceId());
    request.set_transfer_workflow_id(task.workflowId());
    request.set_transfer_parent_id(task.parentTraceId());
    request.set_has_transfer_request_id(task.requestId().has_value());
    request.set_transfer_request_id(task.requestId().value_or(0));
    const auto& descriptors = task.descriptors();
    if (descriptors.empty()) {
        return false;
    }
    const TransferDescriptor&               first = descriptors.front();
    MemoryOperationRequestPB::CopyDirection request_direction;
    if (!directionFor(first, request_direction)) {
        return false;
    }
    request.set_copy_direction(request_direction);

    for (const TransferDescriptor& descriptor : descriptors) {
        if (!descriptor.isExecutable() || descriptor.group_set_id >= group_sets.size()
            || !group_sets[descriptor.group_set_id] || descriptor.source_tier != first.source_tier
            || descriptor.target_tier != first.target_tier) {
            return false;
        }
        const GroupSet& group_set = *group_sets[descriptor.group_set_id];
        const auto&     group_tags = group_set.groupTags();
        const bool      has_device = descriptor.source_tier == Tier::DEVICE || descriptor.target_tier == Tier::DEVICE;
        if (has_device && descriptor.blocksAt(Tier::DEVICE).size() != group_tags.size()) {
            return false;
        }
        auto* item = request.add_copy_items();
        for (const auto& tag : group_tags) {
            item->add_group_tags(tag);
        }

        if (descriptor.source_tier == Tier::HOST || descriptor.target_tier == Tier::HOST) {
            item->set_mem_block(descriptor.singleBlockAt(Tier::HOST));
        }
        if (descriptor.source_tier == Tier::DISK || descriptor.target_tier == Tier::DISK) {
            item->set_disk_block(descriptor.singleBlockAt(Tier::DISK));
        }
        if (has_device) {
            const auto& blocks = descriptor.blocksAt(Tier::DEVICE);
            for (size_t i = 0; i < blocks.size(); ++i) {
                auto* group_block = item->add_group_blocks();
                group_block->set_tag(group_tags[i]);
                group_block->set_block_id(blocks[i]);
            }
        }
    }
    return true;
}

bool BlockTransferRequestConverter::decodeTransfer(const MemoryOperationRequestPB&  request,
                                                   std::vector<TransferDescriptor>& descriptors,
                                                   const std::vector<GroupSetPtr>&  group_sets,
                                                   std::vector<size_t>*             wire_item_indices) {
    // This boundary is shared by the real RPC service and native smoke workers.
    // group_set links in-process ranks to worker_init and dispatcher_init logs.
    const auto log_decode = [&](const char* event, const char* result) noexcept {
        try {
            RTP_LLM_LOG_INFO("transfer_lifecycle event=%s trace=%lu request=%ld group_set=%p timeout_ms=%ld "
                             "items=%d result=%s",
                             event,
                             request.transfer_trace_id(),
                             request.transfer_request_id(),
                             group_sets.empty() ? nullptr : static_cast<const void*>(group_sets.front().get()),
                             request.timeout_ms(),
                             request.copy_items_size(),
                             result);
        } catch (...) {}
    };
    bool      decoded_ok         = false;
    const int initial_exceptions = std::uncaught_exceptions();
    log_decode("decode_enter", "received");
    block_tree_cache_detail::ScopeRollback log_decode_exit([&] {
        log_decode("decode_exit",
                   decoded_ok                                      ? "accepted" :
                   std::uncaught_exceptions() > initial_exceptions ? "exception" :
                                                                     "rejected");
    });
    if (request.copy_items_size() == 0) {
        return false;
    }

    std::vector<TransferDescriptor> decoded;
    std::vector<size_t>             decoded_item_indices;
    decoded.reserve(static_cast<size_t>(request.copy_items_size()));
    if (wire_item_indices) {
        decoded_item_indices.reserve(static_cast<size_t>(request.copy_items_size()));
    }
    for (int item_index = 0; item_index < request.copy_items_size(); ++item_index) {
        const auto&     item          = request.copy_items(item_index);
        const GroupSet* group_set_ptr = resolveGroupSet(item, group_sets);
        if (group_set_ptr == nullptr) {
            RTP_LLM_LOG_WARNING("cannot resolve BlockTree GroupSet membership");
            return false;
        }
        const GroupSet&    group_set = *group_set_ptr;
        TransferDescriptor descriptor;
        switch (request.copy_direction()) {
            case MemoryOperationRequestPB::D2H: {
                std::vector<BlockIdxType> device_blocks;
                if (!group_set.hostPool() || !group_set.hostPool()->validBlock(item.mem_block())
                    || !decodeDeviceBlocks(item, group_set, device_blocks)) {
                    return false;
                }
                descriptor = TransferDescriptor::deviceToHost(
                    group_set.groupSetId(), std::move(device_blocks), item.mem_block());
                break;
            }
            case MemoryOperationRequestPB::H2D: {
                std::vector<BlockIdxType> device_blocks;
                if (!group_set.hostPool() || !group_set.hostPool()->validBlock(item.mem_block())
                    || !decodeDeviceBlocks(item, group_set, device_blocks)) {
                    return false;
                }
                descriptor = TransferDescriptor::hostToDevice(
                    group_set.groupSetId(), item.mem_block(), std::move(device_blocks));
                break;
            }
            case MemoryOperationRequestPB::H2DISK:
                if (!group_set.hostPool() || !group_set.hostPool()->validBlock(item.mem_block())
                    || !group_set.diskPool() || !group_set.diskPool()->validBlock(item.disk_block())) {
                    return false;
                }
                descriptor =
                    TransferDescriptor::hostToDisk(group_set.groupSetId(), item.mem_block(), item.disk_block());
                break;
            case MemoryOperationRequestPB::DISK2D: {
                std::vector<BlockIdxType> device_blocks;
                if (!group_set.diskPool() || !group_set.diskPool()->validBlock(item.disk_block())
                    || !decodeDeviceBlocks(item, group_set, device_blocks)) {
                    return false;
                }
                descriptor = TransferDescriptor::diskToDevice(
                    group_set.groupSetId(), item.disk_block(), std::move(device_blocks));
                break;
            }
            default:
                return false;
        }
        if (group_set.payloadBytes() == 0) {
            continue;
        }
        decoded.push_back(std::move(descriptor));
        if (wire_item_indices) {
            decoded_item_indices.push_back(static_cast<size_t>(item_index));
        }
    }
    if (wire_item_indices) {
        wire_item_indices->reserve(wire_item_indices->size() + decoded_item_indices.size());
    }
    descriptors.insert(
        descriptors.end(), std::make_move_iterator(decoded.begin()), std::make_move_iterator(decoded.end()));
    if (wire_item_indices) {
        wire_item_indices->insert(wire_item_indices->end(), decoded_item_indices.begin(), decoded_item_indices.end());
    }
    decoded_ok = true;
    return true;
}

MemoryOperationResponsePB::Code BlockTransferRequestConverter::encodeCode(CacheCopyError error) {
    switch (error) {
        case CacheCopyError::NONE:
            return MemoryOperationResponsePB::OK;
        case CacheCopyError::NOT_EXECUTED:
            return MemoryOperationResponsePB::NOT_EXECUTED;
        case CacheCopyError::INVALID_REQUEST:
            return MemoryOperationResponsePB::INVALID_REQUEST;
        case CacheCopyError::IO_FAILED:
            return MemoryOperationResponsePB::IO_FAILED;
        case CacheCopyError::CRC_MISMATCH:
            return MemoryOperationResponsePB::CRC_MISMATCH;
        default:
            return MemoryOperationResponsePB::COPY_FAILED;
    }
}

CacheCopyError BlockTransferRequestConverter::decodeCode(MemoryOperationResponsePB::Code error) {
    switch (error) {
        case MemoryOperationResponsePB::OK:
            return CacheCopyError::NONE;
        case MemoryOperationResponsePB::NOT_EXECUTED:
            return CacheCopyError::NOT_EXECUTED;
        case MemoryOperationResponsePB::INVALID_REQUEST:
            return CacheCopyError::INVALID_REQUEST;
        case MemoryOperationResponsePB::IO_FAILED:
            return CacheCopyError::IO_FAILED;
        case MemoryOperationResponsePB::CRC_MISMATCH:
            return CacheCopyError::CRC_MISMATCH;
        default:
            return CacheCopyError::COPY_FAILED;
    }
}

std::optional<TransferTask>
BlockTransferRequestConverter::decodeTransferTask(const MemoryOperationRequestPB&  request,
                                                  const std::vector<GroupSetPtr>& group_sets,
                                                  MemoryOperationResponsePB&      response,
                                                  std::vector<size_t>*             wire_item_indices) {
    const auto reject = [&](const char* message) {
        TransferResult result(ErrorInfo(ErrorCode::INVALID_PARAMS, message));
        result.buffer_safe = true;
        result.expandBlockErrors(request.copy_items_size());
        encodeTransferResult(response, result);
        RTP_LLM_LOG_WARNING("%s", message);
        return std::nullopt;
    };
    if (request.transfer_trace_id() == 0) {
        return reject("transfer execution trace ID is required");
    }
    std::vector<TransferDescriptor> descriptors;
    if (!decodeTransfer(request, descriptors, group_sets, wire_item_indices)) {
        return reject("invalid grouped transfer request");
    }
    if (descriptors.empty()) {
        auto result = TransferResult::emptyAggregate();
        result.expandBlockErrors(request.copy_items_size());
        encodeTransferResult(response, result);
        return std::nullopt;
    }
    if (request.timeout_ms() > std::numeric_limits<int>::max()) {
        return reject("transfer timeout exceeds supported range");
    }
    const auto timeout = request.timeout_ms() > 0 ? std::chrono::milliseconds(request.timeout_ms()) :
                                                    BlockTreeTaskPool::kDefaultQueueWaitTimeout;
    const auto request_id = request.has_transfer_request_id() ? std::optional<int64_t>(request.transfer_request_id()) :
                                                               std::nullopt;
    TransferTask task(std::move(descriptors), timeout, request_id);
    task.setTraceId(request.transfer_trace_id());
    task.setWorkflow(request.transfer_workflow_id(), request.transfer_parent_id());
    return task;
}

void BlockTransferRequestConverter::encodeTransferResult(MemoryOperationResponsePB& response,
                                                         const TransferResult&      result) {
    response.set_code(encodeCode(result.copy_error));
    response.set_buffer_safe(result.buffer_safe);
    response.mutable_error_info()->set_error_code(static_cast<int64_t>(result.status.code()));
    response.mutable_error_info()->set_error_message(result.status.ToString());
    response.clear_block_errors();
    for (const auto error : result.block_errors) {
        response.add_block_errors(encodeCode(error));
    }
}

TransferResult BlockTransferRequestConverter::decodeTransferResult(const MemoryOperationResponsePB& response,
                                                                   size_t                           block_count) {
    TransferResult result;
    if (static_cast<size_t>(response.block_errors_size()) != block_count
        || !MemoryOperationResponsePB::Code_IsValid(response.code())
        || std::any_of(response.block_errors().begin(), response.block_errors().end(), [](int code) {
               return !MemoryOperationResponsePB::Code_IsValid(code);
           })) {
        result = {ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "invalid block transfer response"),
                  CacheCopyError::RPC_FAILED};
        result.expandBlockErrors(block_count);
        return result;
    }
    result.buffer_safe = response.buffer_safe();
    result.copy_error  = decodeCode(response.code());
    for (int index = 0; index < response.block_errors_size(); ++index) {
        const auto block_error = decodeCode(response.block_errors(index));
        result.block_errors.push_back(block_error);
        result.copy_error = std::max(result.copy_error, block_error);
    }
    const auto& error = response.error_info();
    result.status     = ErrorInfo(static_cast<ErrorCode>(error.error_code()), error.error_message());
    if (result.copy_error != CacheCopyError::NONE && result.ok()) {
        result.status = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "remote block transfer failed");
    }
    return result;
}

}  // namespace rtp_llm
