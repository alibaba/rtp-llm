#include "rtp_llm/cpp/normal_engine/pipeline/PPTransport.h"

#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

PPCommTicket::PPCommTicket(std::unique_ptr<P2PWork> work): work_(std::move(work)) {
    RTP_LLM_CHECK_WITH_INFO(work_ != nullptr, "PPCommTicket requires non-null P2P work");
}

PPCommTicket::~PPCommTicket() = default;

void PPCommTicket::wait() {
    if (work_) {
        work_->wait();
        work_.reset();
    }
}

TorchDistributedPPTransport::TorchDistributedPPTransport(int64_t previous_rank, int64_t next_rank):
    previous_rank_(previous_rank), next_rank_(next_rank) {
#if !USING_CUDA
    RTP_LLM_FAIL("TorchDistributedPPTransport requires a CUDA build");
#endif
}

std::unique_ptr<PPCommTicket> TorchDistributedPPTransport::asyncSend(const torch::Tensor& tensor) {
#if USING_CUDA
    const auto backend = tensor.is_cuda() ? P2PBackend::NCCL : P2PBackend::GLOO;
    return std::make_unique<PPCommTicket>(execISend(tensor, next_rank_, backend));
#else
    (void)tensor;
    RTP_LLM_FAIL("TorchDistributedPPTransport requires a CUDA build");
#endif
}

std::unique_ptr<PPCommTicket> TorchDistributedPPTransport::asyncReceive(torch::Tensor& tensor) {
#if USING_CUDA
    const auto backend = tensor.is_cuda() ? P2PBackend::NCCL : P2PBackend::GLOO;
    return std::make_unique<PPCommTicket>(execIRecv(tensor, previous_rank_, backend));
#else
    (void)tensor;
    RTP_LLM_FAIL("TorchDistributedPPTransport requires a CUDA build");
#endif
}

}  // namespace rtp_llm
