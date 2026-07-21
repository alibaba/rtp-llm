#include <gtest/gtest.h>
#include <pybind11/eval.h>

#include "rtp_llm/cpp/api_server/openai/ChatRender.h"

namespace py = pybind11;
namespace th = torch;

namespace rtp_llm {
namespace {

class ChatRenderTest: public ::testing::Test {
protected:
    void SetUp() override {
        py::gil_scoped_acquire acquire;
        py::module_::import("torch");
        py::dict locals;
        py::exec(R"(
class RenderSpy:
    def __init__(self):
        self.calls = []

    def _create_status_list_sync(self, n, body):
        return [None] * n

    def _record(self, name, args, result):
        self.calls.append((name, len(args)))
        return result

    def render_stream_response_refactor(self, *args):
        return self._record("stream", args, "{}")

    def render_stream_response_flush(self, *args):
        return self._record("stream_flush", args, "{}")

    def render_stream_response_blocking(self, *args):
        return self._record("blocking", args, None)

    def render_stream_response_flush_blocking(self, *args):
        return self._record("blocking_flush", args, None)

    def render_stream_response_final(self, *args):
        return self._record("final", args, "{}")

    def render_stream_response_final_blocking(self, *args):
        return self._record("final_blocking", args, None)
)",
                 py::globals(),
                 locals);
        spy_         = locals["RenderSpy"]();
        chat_render_ = std::make_shared<ChatRender>(spy_);
        context_     = chat_render_->getRenderContext();
        context_->init(1, "{}", chat_render_);
    }

    GenerateOutputs makeOutputs() {
        GenerateOutput output;
        output.output_ids            = th::tensor({1}, th::kInt32);
        output.token_logprobs        = th::tensor({-0.1F}, th::kFloat32);
        output.top_logprob_token_ids = th::tensor({2}, th::kInt32).reshape({1, 1});
        output.top_logprobs          = th::tensor({-0.2F}, th::kFloat32).reshape({1, 1});
        output.aux_info.input_len    = 3;
        output.aux_info.output_len   = 1;
        output.aux_info.reuse_len    = 0;
        GenerateOutputs outputs;
        outputs.generate_outputs.push_back(std::move(output));
        return outputs;
    }

    std::vector<std::pair<std::string, int>> calls() const {
        py::gil_scoped_acquire                   acquire;
        std::vector<std::pair<std::string, int>> result;
        for (const auto& item : spy_.attr("calls")) {
            auto call = py::cast<py::tuple>(item);
            result.emplace_back(py::cast<std::string>(call[0]), py::cast<int>(call[1]));
        }
        return result;
    }

protected:
    py::object                     spy_;
    std::shared_ptr<ChatRender>    chat_render_;
    std::shared_ptr<RenderContext> context_;
};

TEST_F(ChatRenderTest, DisabledLogprobsUsesCompactPythonAbi) {
    auto outputs            = makeOutputs();
    auto config             = std::make_shared<GenerateConfig>();
    config->return_logprobs = false;

    EXPECT_EQ(context_->render_stream_response(outputs, config, true), "{}");
    EXPECT_EQ(context_->render_stream_response_flush(outputs, config, true), "{}");
    context_->render_stream_response_blocking(outputs, config, false);
    context_->render_stream_response_flush_blocking(outputs, config, false);
    EXPECT_EQ(context_->render_stream_response_final(outputs), "{}");
    context_->render_stream_response_final_blocking(outputs);

    EXPECT_EQ(calls(),
              (std::vector<std::pair<std::string, int>>{{"stream", 9},
                                                        {"stream_flush", 6},
                                                        {"blocking", 9},
                                                        {"blocking_flush", 6},
                                                        {"final", 4},
                                                        {"final_blocking", 4}}));
}

TEST_F(ChatRenderTest, EnabledLogprobsKeepsExtendedPythonAbi) {
    auto outputs            = makeOutputs();
    auto config             = std::make_shared<GenerateConfig>();
    config->return_logprobs = true;

    EXPECT_EQ(context_->render_stream_response(outputs, config, true), "{}");
    EXPECT_EQ(context_->render_stream_response_flush(outputs, config, true), "{}");
    context_->render_stream_response_blocking(outputs, config, false);
    context_->render_stream_response_flush_blocking(outputs, config, false);

    EXPECT_EQ(calls(),
              (std::vector<std::pair<std::string, int>>{
                  {"stream", 12}, {"stream_flush", 10}, {"blocking", 12}, {"blocking_flush", 10}}));
}

}  // namespace
}  // namespace rtp_llm
