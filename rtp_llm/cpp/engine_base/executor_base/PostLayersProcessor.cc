#include "rtp_llm/cpp/engine_base/executor_base/PostLayersProcessor.h"

#include <pybind11/stl.h>
#include <stdexcept>
// torch/extension.h registers the pybind11 casters for at::Tensor; without it
// handler_.attr(...)(**kwargs).cast<torch::Tensor>() throws "Unregistered type".
#include <torch/extension.h>
#include <string>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"

namespace py = pybind11;

namespace rtp_llm {

PostLayersProcessor::~PostLayersProcessor() {
    if (!handler_) {
        return;
    }
    try {
        py::gil_scoped_acquire gil;
        py::object             tmp = std::move(handler_);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("error releasing post-layers handler: %s", e.what());
    }
}

void PostLayersProcessor::setHandler(py::object handler) {
    py::gil_scoped_acquire gil;
    if (!handler || handler.is_none()) {
        handler_       = py::object();
        handler_args_  = HandlerArgs::Flag{};
        has_handler_   = false;
        wants_context_ = false;
        return;
    }

    handler_ = handler;

    std::vector<std::string> unknown;
    handler_args_ = HandlerArgs::parse(py::cast<std::vector<std::string>>(handler_.attr("extend_forward_args")()),
                                       &unknown);
    for (const auto& name : unknown) {
        RTP_LLM_LOG_WARNING("unknown handler arg: \"%s\", ignored", name.c_str());
    }

    // v1 assembles last_hidden_states only; a handler declaring args this
    // path cannot provide must fail startup, not fail every step.
    for (size_t i = 0; i < HandlerArgs::NUM_ARG_TYPES; ++i) {
        const auto arg = static_cast<HandlerArgs::Arg>(i);
        if (HandlerArgs::has_arg(handler_args_, arg) && arg != HandlerArgs::Arg::LAST_HIDDEN_STATES) {
            throw std::runtime_error(std::string("post-layers handler arg \"") + HandlerArgs::get_name(arg)
                                     + "\" is not available on the generate path");
        }
    }

    const auto trigger = py::cast<std::string>(handler_.attr("trigger_mode")());
    if (trigger != "context") {
        throw std::runtime_error("post-layers handler trigger_mode \"" + trigger
                                 + "\" is not implemented; only Trigger.CONTEXT is supported");
    }
    wants_context_ = true;
    has_handler_   = true;
    RTP_LLM_LOG_INFO("post-layers handler registered, trigger=%s", trigger.c_str());
}

torch::Tensor PostLayersProcessor::invokeHandler(const torch::Tensor& context_rows) const {
    py::gil_scoped_acquire gil;
    py::dict               kwargs;
    if (HandlerArgs::has_arg(handler_args_, HandlerArgs::Arg::LAST_HIDDEN_STATES)) {
        kwargs[HandlerArgs::get_name(HandlerArgs::Arg::LAST_HIDDEN_STATES)] = context_rows;
    }
    auto output = handler_.attr("extend_forward")(**kwargs).cast<torch::Tensor>();
    if (output.defined() && output.size(0) != context_rows.size(0)) {
        throw std::runtime_error("post-layers handler returned " + std::to_string(output.size(0)) + " rows for "
                                 + std::to_string(context_rows.size(0)) + " context requests");
    }
    return output;
}

torch::Tensor PostLayersProcessor::runOnContext(const torch::Tensor& lm_rows, int64_t decode_batch_size) const {
    const int64_t context_batch_size = lm_rows.size(0) - decode_batch_size;
    if (context_batch_size <= 0) {
        return {};
    }
    try {
        return invokeHandler(lm_rows.narrow(0, decode_batch_size, context_batch_size));
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("post-layers handler failed, custom_output dropped for this step: %s", e.what());
        return {};
    }
}

void PostLayersProcessor::warmup(int64_t hidden_size, c10::ScalarType dtype) const {
    if (!has_handler_) {
        return;
    }
    for (const int64_t batch_size : {int64_t(1), int64_t(8)}) {
        auto dummy = torch::zeros({batch_size, hidden_size},
                                  torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
        invokeHandler(dummy);
    }
    RTP_LLM_LOG_INFO("post-layers handler warmup done, hidden_size=%ld", hidden_size);
}

}  // namespace rtp_llm
