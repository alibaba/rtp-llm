#pragma once

#include <cstdint>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/engine_base/executor_base/HandlerArgs.h"

namespace rtp_llm {

// Post-layers extension point for the generate path: hands the lm_output-row
// hidden states of each step to a deployment-registered CustomHandler
// (rtp_llm.models.downstream_modules.custom_module). The handler runs on the
// prefill step of each request, over either the last-token hidden states or
// explicitly selected prompt-token hidden states, and its batched output is
// returned to the caller as GptModelOutputs.custom_output.
//
// Argument names come from the same HandlerArgs registry as
// EmbeddingExecutor::postProcess; this path accepts only its hidden-state
// subset.
class PostLayersProcessor {
public:
    ~PostLayersProcessor();

    // Parses the handler protocol (extend_forward_args + trigger_mode).
    // Throws on unsupported protocol so a misconfigured deployment fails at
    // startup rather than degrading silently.
    void setHandler(pybind11::object handler);

    bool hasHandler() const {
        return has_handler_;
    }

    // True when the handler should run on this step (CONTEXT trigger).
    bool shouldRunOnContext(bool has_context_request) const {
        return has_handler_ && wants_context_ && has_context_request;
    }

    // lm_rows: [total_batch, hidden] lm_output rows of this step, decode rows
    // first, context rows at the tail. Runs the handler over the context rows
    // (one GIL acquire, one batched extend_forward call; the slice is a view).
    // Returns [context_batch, ...] or an undefined tensor. Handler failures
    // propagate to PyWrappedModel, which records them in the model output for
    // per-stream error dispatch without terminating the engine loop.
    torch::Tensor runOnContext(const torch::Tensor& lm_rows, int64_t decode_batch_size) const;

    // Runs the handler over dummy [batch, hidden_size] inputs so lazy
    // initialization (cuBLAS handles, imports, allocator sizing) happens
    // before serving traffic. Exceptions propagate: a handler that cannot
    // run must fail startup.
    void warmup(int64_t hidden_size, c10::ScalarType dtype) const;

private:
    // Invokes extend_forward with kwargs per handler_args_; caller holds no
    // GIL. Throws on handler error or output/batch mismatch.
    torch::Tensor invokeHandler(const torch::Tensor& context_rows) const;

    bool              has_handler_   = false;
    bool              wants_context_ = false;
    pybind11::object  handler_;
    HandlerArgs::Flag handler_args_{};
};

}  // namespace rtp_llm
