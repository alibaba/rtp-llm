#pragma once

#include <functional>
#include <pybind11/pybind11.h>

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/engine_base/executor_base/HandlerArgs.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"

namespace rtp_llm {

class PostProcessor {
public:
    void setHandler(pybind11::object handler);

    bool hasArg(HandlerArgs::Arg arg) const;

    StreamUpdateInfo::PostprocessCallback buildCallback(const rtp_llm::BufferPtr&              hidden_states,
                                                        const std::vector<rtp_llm::BufferPtr>& gating_buffers,
                                                        size_t                                 token_offset,
                                                        size_t                                 token_size,
                                                        size_t                                 cur_batch_size,
                                                        const GenerateStreamPtr&               stream) const;

private:
    bool              has_handler_ = false;
    pybind11::object  handler_;
    HandlerArgs::Flag handler_args_{};
};

}  // namespace rtp_llm
