#include "rtp_llm/cpp/normal_engine/pipeline/PPTransport.h"

#include <utility>

#include <ATen/record_function.h>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

PPCommTicket::PPCommTicket(std::unique_ptr<P2PWork> work): work_(std::move(work)) {}

PPCommTicket::~PPCommTicket() = default;

void PPCommTicket::wait() {
    work_->wait();
}

bool PPCommTicket::tryComplete() {
    if (!work_->isCompleted()) {
        return false;
    }
    /** A completed work state may include errors; wait() validates success before the ticket can be released. */
    work_->wait();
    return true;
}

TorchDistributedPPTransport::TorchDistributedPPTransport(int64_t previous_rank, int64_t next_rank):
    previous_rank_(previous_rank), next_rank_(next_rank) {
#if !USING_CUDA
    RTP_LLM_FAIL("TorchDistributedPPTransport requires a CUDA build");
#endif
}

std::unique_ptr<PPCommTicket> TorchDistributedPPTransport::asyncSend(const torch::Tensor& tensor,
                                                                     const std::string&   group_name) {
#if USING_CUDA
    if (tensor.is_cpu()) {
        /** PP CPU groups use Gloo; deferred wait() callbacks must not retain per-step profiler storage. */
        at::DisableRecordFunctionGuard record_guard;
        return std::make_unique<PPCommTicket>(execISend(tensor, next_rank_, group_name));
    }
    return std::make_unique<PPCommTicket>(execISend(tensor, next_rank_, group_name));
#else
    (void)tensor;
    (void)group_name;
    RTP_LLM_FAIL("TorchDistributedPPTransport requires a CUDA build");
#endif
}

std::unique_ptr<PPCommTicket> TorchDistributedPPTransport::asyncReceive(torch::Tensor&     tensor,
                                                                        const std::string& group_name) {
#if USING_CUDA
    if (tensor.is_cpu()) {
        /** PP CPU groups use Gloo; deferred wait() callbacks must not retain per-step profiler storage. */
        at::DisableRecordFunctionGuard record_guard;
        return std::make_unique<PPCommTicket>(execIRecv(tensor, previous_rank_, group_name));
    }
    return std::make_unique<PPCommTicket>(execIRecv(tensor, previous_rank_, group_name));
#else
    (void)tensor;
    (void)group_name;
    RTP_LLM_FAIL("TorchDistributedPPTransport requires a CUDA build");
#endif
}

}  // namespace rtp_llm
