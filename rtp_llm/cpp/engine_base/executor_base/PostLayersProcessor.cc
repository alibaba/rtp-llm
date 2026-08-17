#include "rtp_llm/cpp/engine_base/executor_base/PostLayersProcessor.h"

#include <pybind11/stl.h>
#include <cerrno>
#include <climits>
#include <stdexcept>
// torch/extension.h registers the pybind11 casters for at::Tensor; without it
// handler_.attr(...)(**kwargs).cast<torch::Tensor>() throws "Unregistered type".
#include <torch/extension.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <string>
#include <vector>
#include <cstdlib>

#include "rtp_llm/cpp/utils/Logger.h"

namespace py = pybind11;

namespace rtp_llm {

namespace {

int parseSelectorEnv(const char* name, bool require_non_negative) {
    const char* value = std::getenv(name);
    errno             = 0;
    char* end         = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < INT_MIN || parsed > INT_MAX) {
        throw std::runtime_error(std::string(name) + " must be a valid int32 integer");
    }
    if (require_non_negative && parsed < 0) {
        throw std::runtime_error(std::string(name) + " must be non-negative");
    }
    return static_cast<int>(parsed);
}

}  // namespace

PostLayersProcessor::PostLayersProcessor() = default;

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
        if (HandlerArgs::has_arg(handler_args_, arg) && arg != HandlerArgs::Arg::LAST_HIDDEN_STATES
            && arg != HandlerArgs::Arg::SELECTED_HIDDEN_STATES) {
            throw std::runtime_error(std::string("post-layers handler arg \"") + HandlerArgs::get_name(arg)
                                     + "\" is not available on the generate path");
        }
    }
    if (HandlerArgs::has_arg(handler_args_, HandlerArgs::Arg::LAST_HIDDEN_STATES)
        && HandlerArgs::has_arg(handler_args_, HandlerArgs::Arg::SELECTED_HIDDEN_STATES)) {
        throw std::runtime_error("post-layers handler must request exactly one hidden-state argument");
    }
    if (HandlerArgs::has_arg(handler_args_, HandlerArgs::Arg::SELECTED_HIDDEN_STATES)) {
        const bool has_position = std::getenv("CUSTOM_OUTPUT_TOKEN_POSITION") != nullptr;
        const bool has_token_id = std::getenv("CUSTOM_OUTPUT_TRACKED_TOKEN_ID") != nullptr;
        if (has_position == has_token_id) {
            throw std::runtime_error(
                "selected_hidden_states requires exactly one of CUSTOM_OUTPUT_TOKEN_POSITION or "
                "CUSTOM_OUTPUT_TRACKED_TOKEN_ID");
        }
        if (has_position) {
            parseSelectorEnv("CUSTOM_OUTPUT_TOKEN_POSITION", false);
        } else {
            parseSelectorEnv("CUSTOM_OUTPUT_TRACKED_TOKEN_ID", true);
        }
        const bool has_expected_token_id = std::getenv("CUSTOM_OUTPUT_EXPECTED_TOKEN_ID") != nullptr;
        if (has_expected_token_id && !has_position) {
            throw std::runtime_error("CUSTOM_OUTPUT_EXPECTED_TOKEN_ID requires CUSTOM_OUTPUT_TOKEN_POSITION");
        }
        if (has_expected_token_id) {
            parseSelectorEnv("CUSTOM_OUTPUT_EXPECTED_TOKEN_ID", true);
        }
    }

    const auto trigger = py::cast<std::string>(handler_.attr("trigger_mode")());
    if (trigger != "context") {
        throw std::runtime_error("post-layers handler trigger_mode \"" + trigger
                                 + "\" is not implemented; only Trigger.CONTEXT is supported");
    }

    // compiled mode: ensure_aoti_package compiles on first startup (weights
    // are loaded by now) or returns the hash-cached package. Failures
    // propagate — a deployment declaring compiled mode must not come up
    // degraded to eager.
    if (py::hasattr(handler_, "ensure_aoti_package")) {
        auto package = handler_.attr("ensure_aoti_package")();
        if (!package.is_none()) {
            const auto path = py::cast<std::string>(package);
            aoti_loader_    = std::make_unique<torch::inductor::AOTIModelPackageLoader>(path);
            RTP_LLM_LOG_INFO("post-layers AOTI package loaded: %s", path.c_str());
        }
    }

    wants_context_ = true;
    has_handler_   = true;
    RTP_LLM_LOG_INFO("post-layers handler registered, trigger=%s, mode=%s",
                     trigger.c_str(),
                     aoti_loader_ ? "compiled" : "eager");
}

torch::Tensor PostLayersProcessor::invokeHandler(const torch::Tensor& context_rows) const {
    torch::Tensor output;
    if (aoti_loader_) {
        // compiled tier: runs the AOTI package on the current CUDA stream,
        // no GIL and no python objects on the hot path
        auto outputs = aoti_loader_->run({context_rows.contiguous()});
        if (outputs.empty()) {
            throw std::runtime_error("post-layers AOTI package returned no outputs");
        }
        output = outputs[0];
    } else {
        py::gil_scoped_acquire gil;
        py::dict               kwargs;
        if (HandlerArgs::has_arg(handler_args_, HandlerArgs::Arg::LAST_HIDDEN_STATES)) {
            kwargs[HandlerArgs::get_name(HandlerArgs::Arg::LAST_HIDDEN_STATES)] = context_rows;
        }
        if (HandlerArgs::has_arg(handler_args_, HandlerArgs::Arg::SELECTED_HIDDEN_STATES)) {
            kwargs[HandlerArgs::get_name(HandlerArgs::Arg::SELECTED_HIDDEN_STATES)] = context_rows;
        }
        output = handler_.attr("extend_forward")(**kwargs).cast<torch::Tensor>();
    }
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
    // A configured custom output is part of the request contract. Propagate
    // handler failures so callers never receive a successful response with a
    // silently missing custom_output.
    return invokeHandler(lm_rows.narrow(0, decode_batch_size, context_batch_size));
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
