#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaReader.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <utility>

#include <torch/python.h>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/rdma_transport/RdmaTransport.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool assembleMMRdmaOutput(const std::vector<torch::Tensor>&        mm_tensors,
                          const std::vector<MMRdmaSlotPB::Role>&   roles,
                          const MultimodalOutputPB*                output_pb,
                          MultimodalOutput*                        mm_output) {
    try {
        if (mm_output == nullptr || output_pb == nullptr || mm_tensors.size() != roles.size()) {
            return false;
        }
        std::vector<torch::Tensor> embedding_chunks;
        torch::Tensor              mm_position_id;
        bool                       has_pos_id = false;
        std::vector<torch::Tensor> extra_inputs;
        for (size_t i = 0; i < roles.size(); ++i) {
            switch (roles[i]) {
                case MMRdmaSlotPB::EMBEDDING:
                    embedding_chunks.emplace_back(mm_tensors[i]);
                    break;
                case MMRdmaSlotPB::POS_ID:
                    if (has_pos_id) {
                        return false;
                    }
                    mm_position_id = mm_tensors[i].to(torch::kCPU);
                    has_pos_id = true;
                    break;
                case MMRdmaSlotPB::EXTRA_INPUT:
                    extra_inputs.emplace_back(mm_tensors[i]);
                    break;
                case MMRdmaSlotPB::ROLE_UNSPECIFIED:
                    RTP_LLM_LOG_WARNING("rdma manifest tensor %zu has unset role", i);
                    return false;
                default:
                    return false;
            }
        }
        if (embedding_chunks.empty()) {
            return false;
        }
        auto embedding =
            embedding_chunks.size() == 1 ? embedding_chunks[0] : torch::cat(embedding_chunks, 0);
        std::vector<int64_t> split_sizes(output_pb->split_size().begin(), output_pb->split_size().end());
        const int64_t split_total = std::accumulate(split_sizes.begin(), split_sizes.end(), int64_t{0});
        if (split_sizes.empty() || split_total != embedding.size(0)) {
            return false;
        }
        if (has_pos_id && split_total != mm_position_id.size(0)) {
            return false;
        }
        if (!extra_inputs.empty() && extra_inputs.size() != split_sizes.size()) {
            return false;
        }
        MultimodalOutput assembled;
        assembled.mm_features = embedding.split(split_sizes, 0);
        if (has_pos_id) {
            assembled.mm_position_ids = mm_position_id.split(split_sizes, 0);
        }
        if (!extra_inputs.empty()) {
            assembled.mm_extra_input = std::move(extra_inputs);
        }
        *mm_output = std::move(assembled);
        return true;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("rdma output materialization failed: %s", e.what());
        return false;
    }
}

namespace {

class SlotLease {
public:
    SlotLease(DeliveryContext& context, std::vector<std::string> handles):
        context_(context), handles_(std::move(handles)) {}
    ~SlotLease() {
        if (!released_) {
            context_.control.release(context_.endpoint, handles_, context_.budget);
        }
    }
    void releaseAsync() {
        context_.control.releaseAsync(context_.endpoint, std::move(handles_));
        released_ = true;
    }
    SlotLease(const SlotLease&)            = delete;
    SlotLease& operator=(const SlotLease&) = delete;

private:
    DeliveryContext&         context_;
    std::vector<std::string> handles_;
    bool                     released_ = false;
};

}  // namespace

bool MMRdmaReader::advertise(const std::string& /*endpoint*/, MultimodalInputsPB& request_pb) {
    if (reader_ == nullptr) {
        return false;
    }
    request_pb.set_support_rdma(true);
    return true;
}

bool MMRdmaReader::matches(const MultimodalOutputPB& receipt) const {
    return receipt.output_rdma_slots_size() > 0;
}

ConsumeResult MMRdmaReader::consume(const MultimodalOutputPB& receipt, DeliveryContext& context) {
    if (reader_ == nullptr) {
        return ConsumeResult::failure(
            ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "rdma receipt reached an adapter with no RDMA reader"));
    }

    std::vector<std::string>       handles = handlesOf(receipt);
    std::vector<torch::Tensor>      mm_tensors;
    std::vector<MMRdmaSlotPB::Role> roles;
    SlotLease                       lease(context, std::move(handles));
    bool                            deadline_exhausted = false;
    const bool read_ok = readAllSlots(receipt, context, &mm_tensors, &roles, &deadline_exhausted);

    if (!read_ok) {
        if (deadline_exhausted) {
            lease.releaseAsync();
        }
        RTP_LLM_LOG_WARNING("rdma read of multimodal embedding failed (%zu slot(s))",
                            static_cast<size_t>(receipt.output_rdma_slots_size()));
        return ConsumeResult::failure(
            ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "failed to read multimodal output over RDMA"));
    }

    MultimodalOutput mm_output;
    if (!assembleMMRdmaOutput(mm_tensors, roles, &receipt, &mm_output)) {
        RTP_LLM_LOG_WARNING("rdma manifest of multimodal embedding is inconsistent (%zu slot(s))",
                            static_cast<size_t>(receipt.output_rdma_slots_size()));
        return ConsumeResult::failure(
            ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "invalid multimodal RDMA output manifest"));
    }

    RTP_LLM_LOG_INFO("[MM-RDMA-HIT] multimodal embedding read over rdma, %d slot(s)",
                     receipt.output_rdma_slots_size());
    lease.releaseAsync();
    return ConsumeResult::success(std::move(mm_output));
}

bool MMRdmaReader::readAllSlots(const MultimodalOutputPB&        receipt,
                                DeliveryContext&                 context,
                                std::vector<torch::Tensor>*      mm_tensors,
                                std::vector<MMRdmaSlotPB::Role>* roles,
                                bool*                            deadline_exhausted) {
    *deadline_exhausted = false;
    constexpr size_t kMaxDescriptorsPerReceipt    = 1024;
    constexpr size_t kMaxManifestEntriesPerReceipt = 16384;
    const size_t descriptor_count = static_cast<size_t>(receipt.output_rdma_slots_size());
    if (descriptor_count == 0 || descriptor_count > kMaxDescriptorsPerReceipt) {
        RTP_LLM_LOG_WARNING("rdma receipt has invalid descriptor count %zu", descriptor_count);
        return false;
    }

    uint64_t split_total = 0;
    for (int64_t split_size : receipt.split_size()) {
        if (split_size <= 0
            || split_total > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - split_size)) {
            RTP_LLM_LOG_WARNING("rdma receipt has invalid or overflowing split_size");
            return false;
        }
        split_total += static_cast<uint64_t>(split_size);
    }
    if (receipt.split_size().empty()) {
        RTP_LLM_LOG_WARNING("rdma receipt has no split_size");
        return false;
    }

    std::vector<rdma_transport::RdmaDescriptor> descriptors;
    std::vector<MMRdmaSlotPB::Role>              parsed_roles;
    std::unordered_set<std::string>              lease_ids;
    uint64_t                                     embedding_rows     = 0;
    uint64_t                                     position_rows      = 0;
    uint64_t                                     total_payload_bytes = 0;
    uint64_t                                     total_tensor_bytes  = 0;
    size_t                                       position_count     = 0;
    size_t                                       extra_count        = 0;
    descriptors.reserve(static_cast<size_t>(receipt.output_rdma_slots_size()));
    for (const auto& slot : receipt.output_rdma_slots()) {
        if (slot.roles_size() != slot.rdma_descriptor().tensors_size()) {
            RTP_LLM_LOG_WARNING("rdma slot has %d roles for %d manifest entries",
                                slot.roles_size(),
                                slot.rdma_descriptor().tensors_size());
            return false;
        }
        if (static_cast<size_t>(slot.roles_size())
            > kMaxManifestEntriesPerReceipt - parsed_roles.size()) {
            RTP_LLM_LOG_WARNING("rdma receipt has too many tensor manifest entries");
            return false;
        }
        const auto& descriptor_pb = slot.rdma_descriptor();
        if (descriptor_pb.nic_keys_size() <= 0
            || static_cast<size_t>(descriptor_pb.nic_keys_size()) > rdma_transport::kMaxRdmaNicKeys
            || descriptor_pb.tensors_size() <= 0
            || static_cast<size_t>(descriptor_pb.tensors_size()) > rdma_transport::kMaxRdmaTensorsPerSlot) {
            RTP_LLM_LOG_WARNING("rdma descriptor exceeds protocol collection limits");
            return false;
        }
        for (const auto& tensor_pb : descriptor_pb.tensors()) {
            if (static_cast<size_t>(tensor_pb.shape_size()) > rdma_transport::kMaxRdmaTensorDimensions) {
                RTP_LLM_LOG_WARNING("rdma tensor shape exceeds protocol dimension limit");
                return false;
            }
        }
        rdma_transport::RdmaDescriptor descriptor;
        if (!rdma_transport::fromProto(descriptor_pb, &descriptor)) {
            RTP_LLM_LOG_WARNING("rdma descriptor contains an unsupported tensor dtype");
            return false;
        }
        if (validate_descriptors_) {
            const auto validation =
                rdma_transport::validateRdmaDescriptor(descriptor, rdma_config_->max_slot_bytes);
            if (!validation.ok()) {
                RTP_LLM_LOG_WARNING("invalid rdma descriptor: %s", validation.ToString().c_str());
                return false;
            }
            const uint64_t max_receipt = rdma_config_->max_receipt_bytes > 0 ?
                                             static_cast<uint64_t>(rdma_config_->max_receipt_bytes) :
                                             std::numeric_limits<uint64_t>::max();
            if (descriptor.payload_bytes > max_receipt - total_payload_bytes) {
                RTP_LLM_LOG_WARNING("rdma receipt payload exceeds configured total output limit");
                return false;
            }
            total_payload_bytes += descriptor.payload_bytes;
            for (const auto& tensor : descriptor.tensors) {
                if (tensor.nbytes > max_receipt - total_tensor_bytes) {
                    RTP_LLM_LOG_WARNING("rdma receipt tensor bytes exceed configured total output limit");
                    return false;
                }
                total_tensor_bytes += tensor.nbytes;
            }
        }
        if (!lease_ids.insert(descriptor.lease_id).second) {
            RTP_LLM_LOG_WARNING("rdma receipt contains duplicate lease id");
            return false;
        }
        for (int i = 0; i < slot.roles_size(); ++i) {
            const auto& tensor = descriptor.tensors[static_cast<size_t>(i)];
            switch (slot.roles(i)) {
                case MMRdmaSlotPB::EMBEDDING:
                    if (tensor.shape.empty() || tensor.shape[0] <= 0
                        || embedding_rows
                               > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - tensor.shape[0])) {
                        return false;
                    }
                    embedding_rows += static_cast<uint64_t>(tensor.shape[0]);
                    break;
                case MMRdmaSlotPB::POS_ID:
                    if (++position_count > 1 || tensor.shape.empty() || tensor.shape[0] <= 0
                        || position_rows
                               > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - tensor.shape[0])) {
                        return false;
                    }
                    position_rows += static_cast<uint64_t>(tensor.shape[0]);
                    break;
                case MMRdmaSlotPB::EXTRA_INPUT:
                    ++extra_count;
                    break;
                case MMRdmaSlotPB::ROLE_UNSPECIFIED:
                default:
                    RTP_LLM_LOG_WARNING("rdma receipt contains invalid tensor role %d", slot.roles(i));
                    return false;
            }
            parsed_roles.push_back(slot.roles(i));
        }
        descriptors.push_back(std::move(descriptor));
    }
    if (embedding_rows != split_total || (position_rows != 0 && position_rows != split_total)
        || (extra_count != 0 && extra_count != static_cast<size_t>(receipt.split_size_size()))) {
        RTP_LLM_LOG_WARNING("rdma receipt roles do not match split_size");
        return false;
    }
    if (validate_descriptors_ && total_tensor_bytes > total_payload_bytes) {
        RTP_LLM_LOG_WARNING("rdma receipt tensor bytes exceed descriptor payload bytes");
        return false;
    }
    int64_t remaining = context.budget.remainingMs();
    if (remaining <= 0) {
        *deadline_exhausted = true;
        return false;
    }
    int64_t read_timeout_ms = remaining;
    if (rdma_config_.has_value() && rdma_config_->read_timeout_ms > 0) {
        read_timeout_ms = std::min(read_timeout_ms, rdma_config_->read_timeout_ms);
    }
    const auto read_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(read_timeout_ms);
    std::unique_lock<std::timed_mutex> provider_lock(provider_mutex_, std::defer_lock);
    if (!provider_lock.try_lock_until(read_deadline)) {
        *deadline_exhausted = context.budget.exhausted();
        return false;
    }
    remaining = context.budget.remainingMs();
    if (remaining <= 0) {
        *deadline_exhausted = true;
        return false;
    }
    const int64_t read_remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                          read_deadline - std::chrono::steady_clock::now())
                                          .count();
    if (read_remaining_ms <= 0) {
        *deadline_exhausted = context.budget.exhausted();
        return false;
    }
    read_timeout_ms = std::min(remaining, read_remaining_ms);
    rdma_transport::RdmaReadResult result;
    try {
        result = reader_->read(descriptors, read_timeout_ms);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("tensor rdma read threw an exception: %s", e.what());
        return false;
    } catch (...) {
        RTP_LLM_LOG_WARNING("tensor rdma read threw an unknown exception");
        return false;
    }
    if (!result.status.ok()) {
        RTP_LLM_LOG_WARNING("tensor rdma read failed: %s", result.status.ToString().c_str());
        return false;
    }
    if (result.tensors.size() != parsed_roles.size()) {
        RTP_LLM_LOG_WARNING("rdma batch returned %zu tensors for %zu manifest entries",
                            result.tensors.size(),
                            parsed_roles.size());
        return false;
    }
    *roles      = std::move(parsed_roles);
    *mm_tensors = std::move(result.tensors);
    return true;
}

std::vector<std::string> MMRdmaReader::handlesOf(const MultimodalOutputPB& receipt) {
    std::vector<std::string> handles;
    std::unordered_set<std::string> seen;
    handles.reserve(static_cast<size_t>(receipt.output_rdma_slots_size()));
    for (const auto& slot : receipt.output_rdma_slots()) {
        const auto& handle = slot.rdma_descriptor().lease_id();
        if (!handle.empty() && seen.insert(handle).second) {
            handles.push_back(handle);
        }
    }
    return handles;
}

void MMRdmaReader::discard(const MultimodalOutputPB& receipt, DeliveryContext& context) {
    const auto handles = handlesOf(receipt);
    if (handles.empty()) {
        return;
    }
    RTP_LLM_LOG_WARNING("discarding %zu unusable rdma slot(s) from a receipt we will not consume",
                        handles.size());
    context.control.release(context.endpoint, handles, context.budget);
}

std::unique_ptr<MMReceiptReader> createMMRdmaReader(std::shared_ptr<rdma_transport::RdmaRead> reader) {
    return std::make_unique<MMRdmaReader>(std::move(reader));
}
}  // namespace rtp_llm
