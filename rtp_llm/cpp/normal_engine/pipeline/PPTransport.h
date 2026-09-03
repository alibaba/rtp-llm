#pragma once

#include <memory>
#include <optional>

#include <torch/torch.h>

namespace rtp_llm {

class PPCommTicket {
public:
    explicit PPCommTicket(const torch::Tensor& tensor): tensor_(tensor) {}

    virtual ~PPCommTicket() = default;

    // CUDA transports make the caller's current stream wait for completion; the CPU thread need not block.
    virtual void wait() = 0;

protected:
    torch::Tensor tensor_;
};

class PPTransport {
public:
    virtual ~PPTransport() = default;

    // CUDA transports observe the caller's current stream: a send depends on prior work on it.
    virtual std::unique_ptr<PPCommTicket> asyncSend(const torch::Tensor& tensor) = 0;
    virtual std::unique_ptr<PPCommTicket> asyncReceive(torch::Tensor& tensor)    = 0;
};

class NcclPPTransport final: public PPTransport {
public:
    NcclPPTransport(std::optional<int> previous_rank, std::optional<int> next_rank);

    NcclPPTransport(const NcclPPTransport&)            = delete;
    NcclPPTransport& operator=(const NcclPPTransport&) = delete;

    std::unique_ptr<PPCommTicket> asyncSend(const torch::Tensor& tensor) override;
    std::unique_ptr<PPCommTicket> asyncReceive(torch::Tensor& tensor) override;

private:
    std::optional<int> previous_rank_;
    std::optional<int> next_rank_;
};

}  // namespace rtp_llm
