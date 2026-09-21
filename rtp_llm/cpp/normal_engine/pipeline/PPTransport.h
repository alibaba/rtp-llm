#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

namespace rtp_llm {

class P2PWork;

/** Thrown by the PP communication watchdog when a peer stops delivering data after shutdown began. */
class PPCommWatchdogTimeout: public std::runtime_error {
public:
    explicit PPCommWatchdogTimeout(const std::string& message): std::runtime_error(message) {}
};

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

    /**
     * Bounded wait; returns false if the timeout expires before completion. The work is
     * kept for a later retry and only released once it completes.
     */
    bool wait(std::chrono::milliseconds timeout);

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
