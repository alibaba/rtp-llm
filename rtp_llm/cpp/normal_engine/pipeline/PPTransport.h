#pragma once

#include <cstdint>
#include <memory>

#include <torch/torch.h>

namespace rtp_llm {

class PPCommTicket {
public:
    explicit PPCommTicket(const torch::Tensor& tensor): tensor_(tensor) {}

    virtual ~PPCommTicket() = default;

    // CUDA tensors: make the caller's current stream wait for completion.
    // CPU tensors: block the calling thread until the bytes arrive.
    virtual void wait() = 0;

protected:
    torch::Tensor tensor_;
};

class PPTransport {
public:
    virtual ~PPTransport() = default;

    // CUDA tensors observe the caller's current stream: a send depends on prior work on it.
    virtual std::unique_ptr<PPCommTicket> asyncSend(const torch::Tensor& tensor) = 0;
    virtual std::unique_ptr<PPCommTicket> asyncReceive(torch::Tensor& tensor)    = 0;
};

// Routes by tensor device: CUDA to the NCCL lane group, CPU to its gloo twin.
class NcclPPTransport final: public PPTransport {
public:
    NcclPPTransport(int64_t previous_rank, int64_t next_rank);

    NcclPPTransport(const NcclPPTransport&)            = delete;
    NcclPPTransport& operator=(const NcclPPTransport&) = delete;

    std::unique_ptr<PPCommTicket> asyncSend(const torch::Tensor& tensor) override;
    std::unique_ptr<PPCommTicket> asyncReceive(torch::Tensor& tensor) override;

private:
    int64_t previous_rank_;
    int64_t next_rank_;
};

}  // namespace rtp_llm
