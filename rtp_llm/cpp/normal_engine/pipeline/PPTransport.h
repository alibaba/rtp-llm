#pragma once

#include <cstdint>
#include <memory>

#include <torch/torch.h>

namespace rtp_llm {

class P2PWork;

class PPCommTicket {
public:
    explicit PPCommTicket(std::unique_ptr<P2PWork> work);

    ~PPCommTicket();

    /**
     * Different behaviors for CPU or CUDA：
     * 1. CPU waits for local completion
     * 2. CUDA enqueues a wait on the tensor device's current stream and will not block the CPU.
     * P2PWork is released after wait() returns; subsequent calls are no-ops.
     */
    void wait();

private:
    std::unique_ptr<P2PWork> work_;
};

class PPTransport {
public:
    virtual ~PPTransport() = default;

    virtual std::unique_ptr<PPCommTicket> asyncSend(const torch::Tensor& tensor) = 0;
    virtual std::unique_ptr<PPCommTicket> asyncReceive(torch::Tensor& tensor)    = 0;
};

/** Routes by tensor device: CUDA to the NCCL lane group, CPU to its gloo twin. */
class TorchDistributedPPTransport final: public PPTransport {
public:
    TorchDistributedPPTransport(int64_t previous_rank, int64_t next_rank);

    TorchDistributedPPTransport(const TorchDistributedPPTransport&)            = delete;
    TorchDistributedPPTransport& operator=(const TorchDistributedPPTransport&) = delete;

    std::unique_ptr<PPCommTicket> asyncSend(const torch::Tensor& tensor) override;
    std::unique_ptr<PPCommTicket> asyncReceive(torch::Tensor& tensor) override;

private:
    int64_t previous_rank_;
    int64_t next_rank_;
};

}  // namespace rtp_llm
