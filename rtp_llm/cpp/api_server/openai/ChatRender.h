#pragma once

#include <string>
#include <map>

#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/api_server/openai/ApiDataType.h"

namespace py = pybind11;
namespace th = torch;

namespace rtp_llm {

class EngineInputs;
class EngineOutputs;
class RenderContext;

class RenderedInputs {
public:
    std::vector<int>             input_ids;
    std::vector<MultimodalInput> multimodal_inputs;
    std::string                  rendered_prompt;
    RenderedInputs(std::vector<int> ids, std::vector<MultimodalInput> mm_inputs, std::string rendered_prompt):
        input_ids(ids), multimodal_inputs(mm_inputs), rendered_prompt(rendered_prompt) {}
};

class ChatRender {
public:
    ChatRender(py::object render): render_(render) {}
    virtual ~ChatRender() {}

public:
    // `virtual` for test
    virtual std::vector<std::vector<int>>  tokenize_words(const std::vector<std::string>& words);
    virtual std::vector<std::vector<int>>  get_all_extra_stop_word_ids_list();
    virtual RenderedInputs                 render_chat_request(const std::string& req);
    py::object                             getRender();
    virtual std::shared_ptr<RenderContext> getRenderContext();
    std::string                            toString();

private:
    py::object render_;
};

class RenderContext {
public:
    RenderContext() = default;
    virtual ~RenderContext();

    virtual void init(int n, std::string body, std::shared_ptr<ChatRender> chat_render);

    // non-streaming
    void        render_stream_response_first_blocking(int n);
    void        render_stream_response_blocking(const GenerateOutputs&                 outputs,
                                                const std::shared_ptr<GenerateConfig>& config,
                                                bool                                   is_streaming);
    void        render_stream_response_flush_blocking(const GenerateOutputs&                 outputs,
                                                      const std::shared_ptr<GenerateConfig>& config,
                                                      bool                                   is_streaming);
    void        render_stream_response_final_blocking(const GenerateOutputs& outputs);
    std::string collect_complete_response();

    // streaming
    virtual std::string render_stream_response_first(int n, std::string debug_info);
    virtual std::string render_stream_response(const GenerateOutputs&                 outputs,
                                               const std::shared_ptr<GenerateConfig>& config,
                                               bool                                   is_streaming);
    virtual std::string render_stream_response_flush(const GenerateOutputs&                 outputs,
                                                     const std::shared_ptr<GenerateConfig>& config,
                                                     bool                                   is_streaming);
    virtual std::string render_stream_response_final(const GenerateOutputs& outputs);

private:
    std::string render_common_response(const GenerateOutputs&                 outputs,
                                       const std::shared_ptr<GenerateConfig>& config,
                                       const char*                            function_name,
                                       bool                                   is_streaming);

private:
    std::shared_ptr<py::object> status_list_;
    std::shared_ptr<py::object> render_;
    std::shared_ptr<py::list>   complete_responses_;
};

}  // namespace rtp_llm
