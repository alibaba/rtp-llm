#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaReceiptReader.h"

#include <utility>

#include <torch/python.h>

#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaOutputAssembler.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace py = pybind11;

namespace rtp_llm {

inline constexpr const char* kReasonRdmaReadError     = "rdma_read_error";
inline constexpr const char* kReasonRdmaManifestError = "rdma_manifest_error";

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
        context_.control.release(context_.endpoint, handles_, context_.budget);
    }
    SlotLease(const SlotLease&)            = delete;
    SlotLease& operator=(const SlotLease&) = delete;

private:
    DeliveryContext&         context_;
    std::vector<std::string> handles_;
};

}  // namespace

bool RdmaReceiptReader::advertise(const std::string& endpoint, MultimodalInputsPB& request_pb) {
    if (transport_ == nullptr) {
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
    if (transport_ == nullptr) {
        return ConsumeResult::failure(
            ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "rdma receipt reached a reader with no transport"));
    }

    std::vector<std::string>         handles = handlesOf(receipt);
    std::vector<const MMRdmaDescPB*> descs;
    descs.reserve(static_cast<size_t>(receipt.output_rdma_slots_size()));
    for (const auto& desc : receipt.output_rdma_slots()) {
        descs.push_back(&desc);
    }

    std::vector<torch::Tensor>        mm_tensors;
    std::vector<MMRdmaTensorPB::Role> roles;
    bool                              read_ok = false;
    {
        SlotLease lease(context, handles);
        read_ok = readAllSlots(descs, context, &mm_tensors, &roles);
    }

    if (!read_ok) {
        RTP_LLM_LOG_WARNING("rdma read of multimodal embedding failed (%zu slot(s)), "
                            "falling back to inline bytes",
                            descs.size());
        noteFailure(context.endpoint);
        return ConsumeResult::retry(kReasonRdmaReadError);
    }

    MultimodalOutput mm_output;
    if (!assembleMMRdmaOutput(mm_tensors, roles, &receipt, &mm_output)) {
        RTP_LLM_LOG_WARNING("rdma manifest of multimodal embedding is inconsistent (%zu slot(s)), "
                            "falling back to inline bytes",
                            descs.size());
        noteFailure(context.endpoint);
        return ConsumeResult::retry(kReasonRdmaManifestError);
    }

    RTP_LLM_LOG_INFO("[MM-RDMA-HIT] multimodal embedding read over rdma, %zu slot(s)", descs.size());
    circuit_.recordSuccess(context.endpoint);
    metrics_->reportCircuitState(name(), context.endpoint, false);
    return ConsumeResult::success(std::move(mm_output));
}

bool RdmaReceiptReader::readAllSlots(const std::vector<const MMRdmaDescPB*>& descs,
                                     DeliveryContext&                        context,
                                     std::vector<torch::Tensor>*             mm_tensors,
                                     std::vector<MMRdmaTensorPB::Role>*      roles) {
    for (const auto* desc : descs) {
        std::vector<torch::Tensor> chunk_tensors;
        const int64_t              remaining = context.budget.remainingMs();
        if (remaining <= 0 || !transport_->readEmbedding(*desc, &chunk_tensors, remaining)) {
            return false;
        }
        for (int i = 0; i < desc->tensors_size(); ++i) {
            roles->push_back(desc->tensors(i).role());
        }
        mm_tensors->insert(mm_tensors->end(), chunk_tensors.begin(), chunk_tensors.end());
    }
    return true;
}

std::vector<std::string> RdmaReceiptReader::handlesOf(const MultimodalOutputPB& receipt) {
    std::vector<std::string> handles;
    handles.reserve(static_cast<size_t>(receipt.output_rdma_slots_size()));
    for (const auto& desc : receipt.output_rdma_slots()) {
        handles.push_back(desc.handle());
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

std::shared_ptr<MMRdmaTransport> createMMRdmaClientTransport(const MMRdmaConfig& rdma_config) {
    py::gil_scoped_release gil_release;
    return createMMRdmaTransport(rdma_config, MMRdmaRole::LLM_CLIENT);
}

std::unique_ptr<MMReceiptReader> createMMRdmaReceiptReader(std::shared_ptr<MMRdmaTransport> transport,
                                                           MMTransportMetricsPtr            metrics) {
    return std::make_unique<RdmaReceiptReader>(std::move(transport), std::move(metrics));
}

}  // namespace rtp_llm
