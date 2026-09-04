#include "rtp_llm/cpp/normal_engine/pipeline/PPTransport.h"

#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

#if USING_CUDA
namespace {

class NcclPPCommTicket final: public PPCommTicket {
public:
    NcclPPCommTicket(const torch::Tensor& tensor, std::unique_ptr<P2PWork> work):
        PPCommTicket(tensor), work_(std::move(work)) {}

    void wait() override {
        if (work_) {
            work_->wait();
            work_.reset();
        }
    }

private:
    std::unique_ptr<P2PWork> work_;
};

}  // namespace
#endif

NcclPPTransport::NcclPPTransport(int64_t previous_rank, int64_t next_rank):
    previous_rank_(previous_rank), next_rank_(next_rank) {
#if !USING_CUDA
    RTP_LLM_FAIL("NcclPPTransport requires a CUDA build");
#endif
}

std::unique_ptr<PPCommTicket> NcclPPTransport::asyncSend(const torch::Tensor& tensor) {
#if USING_CUDA
    return std::make_unique<NcclPPCommTicket>(tensor, execISend(tensor, next_rank_));
#else
    (void)tensor;
    RTP_LLM_FAIL("NcclPPTransport requires a CUDA build");
#endif
}

std::unique_ptr<PPCommTicket> NcclPPTransport::asyncReceive(torch::Tensor& tensor) {
#if USING_CUDA
    return std::make_unique<NcclPPCommTicket>(tensor, execIRecv(tensor, previous_rank_));
#else
    (void)tensor;
    RTP_LLM_FAIL("NcclPPTransport requires a CUDA build");
#endif
}

}  // namespace rtp_llm
