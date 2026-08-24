#pragma once

#include <memory>
#include <torch/torch.h>

#include "rtp_llm/cpp/normal_engine/pipeline/PPTypes.h"

namespace rtp_llm {

class PPCommTicket {
public:
    virtual ~PPCommTicket() = default;

    // When wait() returns, the transport no longer references the payload
    // borrowed by the asynchronous operation.
    virtual void wait() = 0;
};

// Moves each stage's ordered plan and intermediate tensors to the same TP lane
// in the next stage. TP rank 0 carries the execution payload; other TP lanes send
// an empty plan. Receiving a plan advances that lane by one batch. Model inputs
// are distributed inside a stage by tpSyncModelInputs. The last-stage TP root
// returns compact sample results.
// Blocking receive methods must be interruptible by abort().
// asyncSendTensors/Result must enqueue their communication-stream wait on
// forward_done before returning and must not access the Event afterwards.
class PPTransport {
public:
    virtual ~PPTransport() = default;

    virtual PPExecutionPlan               receivePlan()                              = 0;
    virtual std::unique_ptr<PPCommTicket> asyncSendPlan(const PPExecutionPlan& plan) = 0;

    // The receive starts before TP input synchronization. wait() guarantees
    // tensors have been populated and the transport no longer accesses them.
    virtual std::unique_ptr<PPCommTicket> asyncReceiveTensors(PPIntermediateTensors& tensors)                       = 0;
    virtual std::unique_ptr<PPCommTicket> asyncSendTensors(const PPIntermediateTensors& tensors,
                                                           torch::Event&                forward_done)                              = 0;
    virtual std::unique_ptr<PPCommTicket> asyncSendResult(const PPSampleResult& result, torch::Event& forward_done) = 0;

    // Thread-safe and idempotent. When it returns, blocked receives are awake
    // and no pending operation may access executor-owned payload storage.
    virtual void abort() = 0;
};

}  // namespace rtp_llm
