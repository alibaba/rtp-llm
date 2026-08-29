#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaAdapter.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>
#include <torch/python.h>
#include "rtp_llm/cpp/config/MMTransportConfigExtract.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/rdma_transport/RdmaTransport.h"
#include "rtp_llm/cpp/utils/Logger.h"
namespace rtp_llm {




namespace {
inline uint64_t alignUp(uint64_t x, uint64_t a) {
    return (x + a - 1) / a * a;
}
inline uint64_t tensorBytes(const torch::Tensor& t) {
    return static_cast<uint64_t>(t.numel()) * t.element_size();
}
inline uint64_t slotFootprint(const torch::Tensor& t) {
    return alignUp(tensorBytes(t), rdma_transport::kRdmaSlotAlign);
}
RdmaConfig rdmaConfigFromPython(const py::object& rdma_config) {
    return extractRdmaConfig(rdma_config);
}

}  // namespace

MMRdmaOutputExporter::MMRdmaOutputExporter(const py::object& rdma_config):
    MMRdmaOutputExporter(rdmaConfigFromPython(rdma_config)) {}

MMRdmaOutputExporter::MMRdmaOutputExporter(const RdmaConfig& rdma_config) {
    exporter_       = rdma_transport::createRdmaExport(rdma_config);
    max_slot_bytes_ = rdma_config.max_slot_bytes;
}

bool MMRdmaOutputExporter::exportSlots(const torch::Tensor&                embedding,
                                       const std::optional<torch::Tensor>& pos_id,
                                       const std::vector<torch::Tensor>&   extra_inputs,
                                       std::vector<MMRdmaSlotPB>*          slots) {
    if (exporter_ == nullptr) {
        return false;
    }

    const uint64_t max_slot =
        max_slot_bytes_ > 0 ? static_cast<uint64_t>(max_slot_bytes_) : std::numeric_limits<uint64_t>::max();
    // The packer aligns every tensor start, so chunk against the aligned-down slot limit.
    const uint64_t max_slot_aligned =
        max_slot / rdma_transport::kRdmaSlotAlign * rdma_transport::kRdmaSlotAlign;

    // Manifest order is embedding chunks, optional position ids, then extra inputs.
    std::vector<torch::Tensor>      tensors;
    std::vector<MMRdmaSlotPB::Role> roles;

    if (slotFootprint(embedding) <= max_slot) {
        tensors.push_back(embedding);
        roles.push_back(MMRdmaSlotPB::EMBEDDING);
    } else {
        const int64_t rows = embedding.dim() >= 1 ? embedding.size(0) : 0;
        if (rows <= 0) {
            RTP_LLM_LOG_WARNING("mm rdma chunk: embedding not row-splittable (dim=%ld), fall back to bytes",
                                static_cast<long>(embedding.dim()));
            return false;
        }
        const uint64_t row_bytes = tensorBytes(embedding) / static_cast<uint64_t>(rows);
        if (row_bytes == 0 || alignUp(row_bytes, rdma_transport::kRdmaSlotAlign) > max_slot) {
            RTP_LLM_LOG_WARNING("mm rdma chunk: single embedding row (%lu B) exceeds max_slot (%lu), "
                                "fall back to bytes",
                                row_bytes,
                                max_slot);
            return false;
        }
        int64_t rows_per_chunk = static_cast<int64_t>(max_slot_aligned / std::max<uint64_t>(row_bytes, 1));
        if (rows_per_chunk < 1) {
            rows_per_chunk = 1;
        }
        for (int64_t start = 0; start < rows; start += rows_per_chunk) {
            const int64_t len = std::min<int64_t>(rows_per_chunk, rows - start);
            tensors.push_back(embedding.narrow(0, start, len));
            roles.push_back(MMRdmaSlotPB::EMBEDDING);
        }
    }
    if (pos_id.has_value()) {
        if (slotFootprint(*pos_id) > max_slot) {
            RTP_LLM_LOG_WARNING("mm rdma chunk: pos_id (%lu B) exceeds max_slot (%lu), fall back to bytes",
                                tensorBytes(*pos_id),
                                max_slot);
            return false;
        }
        tensors.push_back(*pos_id);
        roles.push_back(MMRdmaSlotPB::POS_ID);
    }
    for (const auto& extra : extra_inputs) {
        if (slotFootprint(extra) > max_slot) {
            RTP_LLM_LOG_WARNING("mm rdma chunk: extra_input (%lu B) exceeds max_slot (%lu), fall back to bytes",
                                tensorBytes(extra),
                                max_slot);
            return false;
        }
        tensors.push_back(extra);
        roles.push_back(MMRdmaSlotPB::EXTRA_INPUT);
    }

    // Greedily pack tensors while preserving manifest order.
    std::vector<std::vector<torch::Tensor>> groups;
    std::vector<std::vector<MMRdmaSlotPB::Role>> group_roles;
    size_t i = 0;
    while (i < tensors.size()) {
        std::vector<torch::Tensor>      grp_tensors;
        std::vector<MMRdmaSlotPB::Role> grp_roles;
        uint64_t                        grp_bytes = 0;
        while (i < tensors.size()) {
            const uint64_t fp = slotFootprint(tensors[i]);
            if (!grp_tensors.empty() && grp_bytes + fp > max_slot) {
                break;
            }
            grp_tensors.push_back(tensors[i]);
            grp_roles.push_back(roles[i]);
            grp_bytes += fp;
            ++i;
        }
        groups.push_back(std::move(grp_tensors));
        group_roles.push_back(std::move(grp_roles));
    }
    const auto descriptors = exporter_->createBatch(groups);
    if (descriptors.size() != groups.size()) {
        return false;
    }
    for (size_t group = 0; group < descriptors.size(); ++group) {
        MMRdmaSlotPB slot;
        rdma_transport::toProto(descriptors[group], slot.mutable_rdma_descriptor());
        for (auto role : group_roles[group]) {
            slot.add_roles(role);
        }
        slots->push_back(std::move(slot));
    }
    return !slots->empty();
}

std::vector<py::bytes> MMRdmaOutputExporter::exportEmbedding(torch::Tensor                embedding,
                                                             std::optional<torch::Tensor> pos_id,
                                                             std::vector<torch::Tensor>   extra_inputs) {
    std::vector<MMRdmaSlotPB> slots;
    bool                      exported = false;
    {
        // D2D copies and memory registration do not need the GIL.
        py::gil_scoped_release release;
        exported = exportSlots(embedding, pos_id, extra_inputs, &slots);
    }

    std::vector<py::bytes> out;
    if (!exported) {
        return out;
    }
    out.reserve(slots.size());
    for (const auto& slot : slots) {
        out.push_back(py::bytes(slot.SerializeAsString()));
    }
    return out;
}

void MMRdmaOutputExporter::release(const std::vector<std::string>& handles) {
    if (exporter_ == nullptr) {
        return;
    }
    py::gil_scoped_release release;
    exporter_->release(handles);
}

void registerMMRdmaOutputExporter(py::module& m) {
    py::class_<MMRdmaOutputExporter, std::shared_ptr<MMRdmaOutputExporter>>(m, "MMRdmaOutputExporter")
        .def(py::init<const py::object&>(), py::arg("rdma_config"))
        .def_static("available", &rdma_transport::hasRdmaImplementation)
        .def("enabled", &MMRdmaOutputExporter::enabled)
        .def("export_embedding",
             &MMRdmaOutputExporter::exportEmbedding,
             py::arg("embedding"),
             py::arg("pos_id"),
             py::arg("extra_inputs"))
        .def("release", &MMRdmaOutputExporter::release, py::arg("handles"));
}








inline constexpr const char* kReasonRdmaReadError     = "rdma_read_error";
inline constexpr const char* kReasonRdmaManifestError = "rdma_manifest_error";

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
                    if (has_pos_id) return false;
                    mm_position_id = mm_tensors[i].to(torch::kCPU);
                    has_pos_id = true;
                    break;
                case MMRdmaSlotPB::EXTRA_INPUT:
                    extra_inputs.emplace_back(mm_tensors[i]);
                    break;
                case MMRdmaSlotPB::ROLE_UNSPECIFIED:
                    RTP_LLM_LOG_WARNING("rdma manifest tensor %zu has unset role; treating as protocol "
                                        "error and falling back to inline bytes",
                                        i);
                    return false;
                default:
                    return false;
            }
        }
        if (embedding_chunks.empty()) return false;
        auto embedding = embedding_chunks.size() == 1 ? embedding_chunks[0] : torch::cat(embedding_chunks, 0);
        std::vector<int64_t> split_sizes(output_pb->split_size().begin(), output_pb->split_size().end());
        const int64_t split_total = std::accumulate(split_sizes.begin(), split_sizes.end(), int64_t{0});
        if (split_sizes.empty() || split_total != embedding.size(0)) return false;
        if (has_pos_id && split_total != mm_position_id.size(0)) return false;
        if (!extra_inputs.empty() && extra_inputs.size() != split_sizes.size()) return false;
        MultimodalOutput assembled;
        assembled.mm_features = embedding.split(split_sizes, 0);
        if (has_pos_id) assembled.mm_position_ids = mm_position_id.split(split_sizes, 0);
        if (!extra_inputs.empty()) assembled.mm_extra_input = std::move(extra_inputs);
        *mm_output = std::move(assembled);
        return true;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("rdma output materialization failed: %s", e.what());
        return false;
    }
}

std::unordered_map<std::string, RdmaCircuitBreaker::State>& RdmaCircuitBreaker::table() {
    static std::unordered_map<std::string, State> states;
    return states;
}

std::mutex& RdmaCircuitBreaker::mutex() {
    static std::mutex m;
    return m;
}

bool RdmaCircuitBreaker::open(const std::string& endpoint) const {
    std::lock_guard<std::mutex> lock(mutex());
    auto&                       states = table();
    auto                        it     = states.find(endpoint);
    return it != states.end() && it->second.open_until > std::chrono::steady_clock::now();
}

void RdmaCircuitBreaker::recordFailure(const std::string& endpoint) {
    std::lock_guard<std::mutex> lock(mutex());
    auto&                       state = table()[endpoint];
    if (++state.failures >= kFailuresToOpen) {
        state.open_until = std::chrono::steady_clock::now() + std::chrono::seconds(kOpenSeconds);
    }
}

void RdmaCircuitBreaker::recordSuccess(const std::string& endpoint) {
    std::lock_guard<std::mutex> lock(mutex());
    table().erase(endpoint);
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

bool RdmaReceiptReader::advertise(const std::string& endpoint, MultimodalInputsPB& request_pb) {
    if (!ensureReader()) {
        return false;
    }
    const bool circuit_open = circuit_.open(endpoint);
    metrics_->reportCircuitState(name(), endpoint, circuit_open);
    if (circuit_open) {
        return false;
    }
    request_pb.set_support_rdma(true);
    return true;
}

void RdmaReceiptReader::withdraw(MultimodalInputsPB& request_pb) {
    request_pb.set_support_rdma(false);
}

bool RdmaReceiptReader::matches(const MultimodalOutputPB& receipt) const {
    return receipt.output_rdma_slots_size() > 0;
}

ConsumeResult RdmaReceiptReader::consume(const MultimodalOutputPB& receipt, DeliveryContext& context) {
    if (!ensureReader()) {
        return ConsumeResult::failure(
            ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "rdma receipt reached an adapter with no RDMA reader"));
    }

    std::vector<std::string>       handles = handlesOf(receipt);
    std::vector<torch::Tensor>      mm_tensors;
    std::vector<MMRdmaSlotPB::Role> roles;
    SlotLease                       lease(context, std::move(handles));
    const bool read_ok = readAllSlots(receipt, context, &mm_tensors, &roles);

    if (!read_ok) {
        RTP_LLM_LOG_WARNING("rdma read of multimodal embedding failed (%zu slot(s)), "
                            "falling back to inline bytes",
                            static_cast<size_t>(receipt.output_rdma_slots_size()));
        noteFailure(context.endpoint);
        return ConsumeResult::retry(kReasonRdmaReadError);
    }

    MultimodalOutput mm_output;
    if (!assembleMMRdmaOutput(mm_tensors, roles, &receipt, &mm_output)) {
        RTP_LLM_LOG_WARNING("rdma manifest of multimodal embedding is inconsistent (%zu slot(s)), "
                            "falling back to inline bytes",
                            static_cast<size_t>(receipt.output_rdma_slots_size()));
        noteFailure(context.endpoint);
        return ConsumeResult::retry(kReasonRdmaManifestError);
    }

    RTP_LLM_LOG_INFO("[MM-RDMA-HIT] multimodal embedding read over rdma, %d slot(s)",
                     receipt.output_rdma_slots_size());
    circuit_.recordSuccess(context.endpoint);
    metrics_->reportCircuitState(name(), context.endpoint, false);
    lease.releaseAsync();
    return ConsumeResult::success(std::move(mm_output));
}

bool RdmaReceiptReader::readAllSlots(const MultimodalOutputPB&        receipt,
                                     DeliveryContext&                 context,
                                     std::vector<torch::Tensor>*      mm_tensors,
                                     std::vector<MMRdmaSlotPB::Role>* roles) {
    std::vector<rdma_transport::RdmaDescriptor> descriptors;
    descriptors.reserve(static_cast<size_t>(receipt.output_rdma_slots_size()));
    for (const auto& slot : receipt.output_rdma_slots()) {
        if (slot.roles_size() != slot.rdma_descriptor().tensors_size()) {
            RTP_LLM_LOG_WARNING("rdma slot has %d roles for %d manifest entries",
                                slot.roles_size(),
                                slot.rdma_descriptor().tensors_size());
            return false;
        }
        rdma_transport::RdmaDescriptor descriptor;
        if (!rdma_transport::fromProto(slot.rdma_descriptor(), &descriptor)) {
            RTP_LLM_LOG_WARNING("rdma descriptor contains an unsupported tensor dtype");
            return false;
        }
        descriptors.push_back(std::move(descriptor));
        for (int i = 0; i < slot.roles_size(); ++i) {
            roles->push_back(slot.roles(i));
        }
    }
    const int64_t remaining = context.budget.remainingMs();
    if (remaining <= 0) {
        return false;
    }
    auto result = reader_->read(descriptors, remaining);
    if (!result.status.ok()) {
        RTP_LLM_LOG_WARNING("tensor rdma read failed: %s", result.status.ToString().c_str());
        return false;
    }
    if (result.tensors.size() != roles->size()) {
        RTP_LLM_LOG_WARNING("rdma batch returned %zu tensors for %zu manifest entries",
                            result.tensors.size(),
                            roles->size());
        return false;
    }
    *mm_tensors = std::move(result.tensors);
    return true;
}

std::vector<std::string> RdmaReceiptReader::handlesOf(const MultimodalOutputPB& receipt) {
    std::vector<std::string> handles;
    handles.reserve(static_cast<size_t>(receipt.output_rdma_slots_size()));
    for (const auto& slot : receipt.output_rdma_slots()) {
        handles.push_back(slot.rdma_descriptor().lease_id());
    }
    return handles;
}

void RdmaReceiptReader::discard(const MultimodalOutputPB& receipt, DeliveryContext& context) {
    const auto handles = handlesOf(receipt);
    if (handles.empty()) {
        return;
    }
    RTP_LLM_LOG_WARNING("discarding %zu unusable rdma slot(s) from a receipt we will not consume",
                        handles.size());
    context.control.release(context.endpoint, handles, context.budget);
}

void RdmaReceiptReader::noteFailure(const std::string& endpoint) {
    circuit_.recordFailure(endpoint);
    metrics_->reportCircuitState(name(), endpoint, circuit_.open(endpoint));
}

bool RdmaReceiptReader::ensureReader() {
    if (reader_ != nullptr) {
        return true;
    }
    if (!rdma_config_.has_value()) {
        return false;
    }
    // This method is called from gRPC worker threads, which do not hold the
    // Python GIL.  Reader construction is entirely C++/RDMA and must not use
    // py::gil_scoped_release here (it requires the GIL to be held first).
    reader_ = rdma_transport::createRdmaRead(*rdma_config_);
    rdma_config_.reset();
    return reader_ != nullptr;
}

std::unique_ptr<MMReceiptReader> createMMRdmaReceiptReader(std::shared_ptr<rdma_transport::RdmaRead> reader,
                                                           MMTransportMetricsPtr                      metrics) {
    return std::make_unique<RdmaReceiptReader>(std::move(reader), std::move(metrics));
}

std::unique_ptr<MMReceiptReader> createLazyMMRdmaReceiptReader(const RdmaConfig& rdma_config,
                                                               MMTransportMetricsPtr metrics) {
    return std::make_unique<RdmaReceiptReader>(rdma_config, std::move(metrics));
}



}  // namespace rtp_llm
