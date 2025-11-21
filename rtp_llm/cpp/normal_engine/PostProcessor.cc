#include "rtp_llm/cpp/normal_engine/PostProcessor.h"

#include <torch/torch.h>

#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

void PostProcessor::setHandler(pybind11::object handler) {
    if (!handler || handler.is_none()) {
        handler_      = pybind11::object();
        handler_args_ = HandlerArgs::Flag{};
        has_handler_  = false;
    } else {
        pybind11::gil_scoped_acquire gil;
        handler_ = handler;
        handler_args_.reset();

        std::vector<std::string> handler_arg_names =
            pybind11::cast<std::vector<std::string>>(handler_.attr("extend_forward_args")());
        for (const auto& name : handler_arg_names) {
            if (!HandlerArgs::set_by_str(handler_args_, name.c_str())) {
                RTP_LLM_LOG_WARNING("unknown handler arg: \"%s\", ignored", name.c_str());
            }
        }
        has_handler_ = true;
    }
}

StreamUpdateInfo::PostprocessCallback PostProcessor::buildCallback(const rtp_llm::BufferPtr&              hidden_states,
                                                                   const std::vector<rtp_llm::BufferPtr>& gating_buffers,
                                                                   size_t                                 token_offset,
                                                                   size_t                                 token_size,
                                                                   size_t                                 cur_batch_size,
                                                                   const GenerateStreamPtr&               stream) const {
    if (!has_handler_) {
        return {};
    }

    pybind11::object handler;
    {
        pybind11::gil_scoped_acquire gil;
        handler = handler_;
    }
    HandlerArgs::Flag handler_args = handler_args_;
    GenerateStreamPtr stream_capture;
    if (HandlerArgs::has_arg(handler_args, HandlerArgs::Arg::TOKEN_LENGTHS)) {
        stream_capture = stream;
    }

    return [handler,
            handler_args,
            hidden_states,
            gating_buffers,
            token_offset,
            token_size,
            cur_batch_size,
            stream_capture](const StreamUpdateInfo& info) {
        pybind11::gil_scoped_acquire gil;
        pybind11::dict kwargs;

        if (HandlerArgs::has_arg(handler_args, HandlerArgs::Arg::LAST_HIDDEN_STATES)) {
            kwargs[pybind11::str(HandlerArgs::get_name(HandlerArgs::Arg::LAST_HIDDEN_STATES))] =
                Buffer2torchTensor(hidden_states, false);
        }

        if (HandlerArgs::has_arg(handler_args, HandlerArgs::Arg::LAST_MOE_GATING)) {
            pybind11::list gating_list;
            for (const auto& gating : gating_buffers) {
                if (gating) {
                    auto gating_view = gating->slice(token_offset, token_size, false);
                    gating_list.append(Buffer2torchTensor(gating_view, false));
                } else {
                    gating_list.append(pybind11::none());
                }
            }
            kwargs[pybind11::str(HandlerArgs::get_name(HandlerArgs::Arg::LAST_MOE_GATING))] = gating_list;
        }

        if (HandlerArgs::has_arg(handler_args, HandlerArgs::Arg::TOKEN_LENGTHS) && stream_capture) {
            pybind11::list lengths;
            for (size_t i = 0; i < cur_batch_size; ++i) {
                lengths.append(static_cast<int64_t>(stream_capture->currentExecuteTokens(static_cast<int>(i)).size()));
            }
            kwargs[pybind11::str(HandlerArgs::get_name(HandlerArgs::Arg::TOKEN_LENGTHS))] = lengths;
        }

        return handler.attr("extend_forward")(**kwargs).cast<torch::Tensor>();
    };
}

bool PostProcessor::hasArg(HandlerArgs::Arg arg) const {
    return has_handler_ && HandlerArgs::has_arg(handler_args_, arg);
}

}  // namespace rtp_llm
