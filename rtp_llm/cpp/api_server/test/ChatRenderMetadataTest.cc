#include <memory>
#include <string>

#include <gtest/gtest.h>
#include <pybind11/eval.h>

#include "rtp_llm/cpp/api_server/openai/ChatRender.h"

namespace py = pybind11;

namespace rtp_llm {

TEST(ChatRenderMetadataTest, RenderContextsUseDistinctStableMetadata) {
    py::gil_scoped_acquire acquire;
    py::dict               scope;
    scope["__builtins__"] = py::module_::import("builtins");
    py::exec(R"PY(
import json

class FakeRender:
    def __init__(self):
        self.next_id = 0

    def _create_status_list_sync(self, n, body):
        return []

    def create_response_metadata(self):
        self.next_id += 1
        return (f"chatcmpl-sync-request-{self.next_id}", 123456 + self.next_id)

    def _response(self, response_id, created):
        return json.dumps({"id": response_id, "created": created})

    def render_stream_response_first_blocking(self, n):
        return {}

    def collect_complete_response(self, responses, response_id, created):
        return self._response(response_id, created)

    def render_stream_response_first(self, n, debug_info, response_id, created):
        return self._response(response_id, created)

    def render_stream_response_refactor(self, *args):
        return self._response(*args[-2:])

    def render_stream_response_flush(self, *args):
        return self._response(*args[-2:])

    def render_stream_response_final(self, *args):
        return self._response(*args[-2:])
)PY",
             scope);

    auto chat_render = std::make_shared<ChatRender>(scope["FakeRender"]());
    RenderContext first_context;
    RenderContext second_context;
    first_context.init(1, "{}", chat_render);
    second_context.init(1, "{}", chat_render);

    auto assert_metadata = [](const std::string& response, const std::string& response_id, int64_t created) {
        const std::string expected_id      = "\"id\": \"" + response_id + "\"";
        const std::string expected_created = "\"created\": " + std::to_string(created);
        EXPECT_NE(response.find(expected_id), std::string::npos);
        EXPECT_NE(response.find(expected_created), std::string::npos);
    };

    first_context.render_stream_response_first_blocking(1);
    assert_metadata(first_context.collect_complete_response(), "chatcmpl-sync-request-1", 123457);

    second_context.render_stream_response_first_blocking(1);
    assert_metadata(second_context.collect_complete_response(), "chatcmpl-sync-request-2", 123458);

    GenerateOutputs outputs;
    auto            config = std::make_shared<GenerateConfig>();
    assert_metadata(first_context.render_stream_response_first(1, "{}"), "chatcmpl-sync-request-1", 123457);
    assert_metadata(first_context.render_stream_response(outputs, config, true), "chatcmpl-sync-request-1", 123457);
    assert_metadata(
        first_context.render_stream_response_flush(outputs, config, true), "chatcmpl-sync-request-1", 123457);
    assert_metadata(first_context.render_stream_response_final(outputs), "chatcmpl-sync-request-1", 123457);
}

}  // namespace rtp_llm
