#include "rtp_llm/cpp/api_server/openai/ChatRender.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {

RenderContext::~RenderContext() {
    py::gil_scoped_acquire acquire;
    complete_responses_.reset();
    render_.reset();
    status_list_.reset();
}

void RenderContext::init(int n, std::string body, std::shared_ptr<ChatRender> chat_render) {
    py::gil_scoped_acquire acquire;
    auto                   render = chat_render->getRender();
    render_                       = std::make_shared<py::object>(render);
    auto status_list              = render.attr("_create_status_list_sync")(n, body);
    status_list_                  = std::make_shared<py::object>(status_list);
    complete_responses_           = std::make_shared<py::list>();
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<th::Tensor>, std::vector<th::Tensor>>
getArgs(const GenerateOutputs& outputs) {
    std::vector<int>        input_len_list, output_len_list, reuse_len_list;
    std::vector<th::Tensor> all_probs_list, output_ids_list;
    for (const auto& output : outputs.generate_outputs) {
        const auto& aux_info = output.aux_info;
        input_len_list.push_back(aux_info.input_len);
        output_len_list.push_back(aux_info.output_len);
        reuse_len_list.push_back(aux_info.reuse_len);
        if (aux_info.all_probs.has_value()) {
            all_probs_list.push_back(rtp_llm::Buffer2torchTensor(aux_info.all_probs.value(), true));
        } else {
            all_probs_list.push_back(th::empty({0}));
        }
        output_ids_list.push_back(rtp_llm::Buffer2torchTensor(output.output_ids, true));
    }
    return std::make_tuple(input_len_list, output_len_list, reuse_len_list, all_probs_list, output_ids_list);
}

void RenderContext::render_stream_response_first_blocking(int n) {
    py::gil_scoped_acquire acquire;
    auto                   response = render_->attr("render_stream_response_first_blocking")(n);
    complete_responses_->append(response);
}

void RenderContext::render_stream_response_blocking(const GenerateOutputs&                 outputs,
                                                    const std::shared_ptr<GenerateConfig>& config,
                                                    bool                                   is_streaming) {
    py::gil_scoped_acquire acquire;
    auto [input_len_list, output_len_list, reuse_len_list, all_probs_list, output_ids_list] = getArgs(outputs);
    auto response = render_->attr("render_stream_response_blocking")(*status_list_,
                                                                     input_len_list,
                                                                     output_len_list,
                                                                     reuse_len_list,
                                                                     all_probs_list,
                                                                     output_ids_list,
                                                                     config->max_new_tokens,
                                                                     config->stop_words_str,
                                                                     is_streaming);
    complete_responses_->append(response);
}

void RenderContext::render_stream_response_flush_blocking(const GenerateOutputs&                 outputs,
                                                          const std::shared_ptr<GenerateConfig>& config,
                                                          bool                                   is_streaming) {
    py::gil_scoped_acquire acquire;
    auto [input_len_list, output_len_list, reuse_len_list, all_probs_list, output_ids_list] = getArgs(outputs);
    auto response = render_->attr("render_stream_response_flush_blocking")(*status_list_,
                                                                           input_len_list,
                                                                           output_len_list,
                                                                           reuse_len_list,
                                                                           all_probs_list,
                                                                           output_ids_list,
                                                                           config->stop_words_str,
                                                                           is_streaming);
    complete_responses_->append(response);
}

void RenderContext::render_stream_response_final_blocking(const GenerateOutputs& outputs) {
    py::gil_scoped_acquire acquire;
    auto [input_len_list, output_len_list, reuse_len_list, all_probs_list, output_ids_list] = getArgs(outputs);
    auto response = render_->attr("render_stream_response_final_blocking")(
        *status_list_, input_len_list, output_len_list, reuse_len_list);
    complete_responses_->append(response);
}

std::string RenderContext::collect_complete_response() {
    py::gil_scoped_acquire acquire;
    auto                   json_response = render_->attr("collect_complete_response")(complete_responses_.get());
    auto                   res           = py::cast<std::string>(json_response);
    return res;
}

std::string RenderContext::render_common_response(const GenerateOutputs&                 outputs,
                                                  const std::shared_ptr<GenerateConfig>& config,
                                                  const char*                            function_name,
                                                  bool                                   is_streaming) {
    py::gil_scoped_acquire acquire;
    auto [input_len_list, output_len_list, reuse_len_list, all_probs_list, output_ids_list] = getArgs(outputs);
    auto json_response = render_->attr(function_name)(*status_list_,
                                                      input_len_list,
                                                      output_len_list,
                                                      reuse_len_list,
                                                      all_probs_list,
                                                      output_ids_list,
                                                      config->max_new_tokens,
                                                      config->stop_words_str,
                                                      is_streaming);
    auto res           = py::cast<std::string>(json_response);
    return res;
}

std::string RenderContext::render_stream_response_first(int n, std::string debug_info) {
    py::gil_scoped_acquire acquire;
    auto                   json_response = render_->attr("render_stream_response_first")(n, debug_info);
    auto                   res           = py::cast<std::string>(json_response);
    return res;
}

std::string RenderContext::render_stream_response(const GenerateOutputs&                 outputs,
                                                  const std::shared_ptr<GenerateConfig>& config,
                                                  bool                                   is_streaming) {
    return render_common_response(outputs, config, "render_stream_response_refactor", is_streaming);
}

std::string RenderContext::render_stream_response_flush(const GenerateOutputs&                 outputs,
                                                        const std::shared_ptr<GenerateConfig>& config,
                                                        bool                                   is_streaming) {
    py::gil_scoped_acquire acquire;
    auto [input_len_list, output_len_list, reuse_len_list, all_probs_list, output_ids_list] = getArgs(outputs);
    auto json_response = render_->attr("render_stream_response_flush")(*status_list_,
                                                                       input_len_list,
                                                                       output_len_list,
                                                                       reuse_len_list,
                                                                       all_probs_list,
                                                                       output_ids_list,
                                                                       config->stop_words_str,
                                                                       is_streaming);
    auto res           = py::cast<std::string>(json_response);
    return res;
}

std::string RenderContext::render_stream_response_final(const GenerateOutputs& outputs) {
    py::gil_scoped_acquire acquire;
    auto [input_len_list, output_len_list, reuse_len_list, all_probs_list, output_ids_list] = getArgs(outputs);
    auto json_response =
        render_->attr("render_stream_response_final")(*status_list_, input_len_list, output_len_list, reuse_len_list);
    auto res = py::cast<std::string>(json_response);
    return res;
}

py::object ChatRender::getRender() {
    return render_;
}

std::shared_ptr<RenderContext> ChatRender::getRenderContext() {
    return std::make_shared<RenderContext>();
}

std::vector<std::vector<int>> ChatRender::tokenize_words(const std::vector<std::string>& words) {
    py::gil_scoped_acquire acquire;
    return py::cast<std::vector<std::vector<int>>>(render_.attr("tokenize_words")(words));
}

std::vector<std::vector<int>> ChatRender::get_all_extra_stop_word_ids_list() {
    py::gil_scoped_acquire acquire;
    return py::cast<std::vector<std::vector<int>>>(render_.attr("get_all_extra_stop_word_ids_list")());
}

RenderedInputs ChatRender::render_chat_request(const std::string& reqBody) {
    py::gil_scoped_acquire acquire;

    auto chat_request   = render_.attr("getRequest")(reqBody);
    auto rendered_input = render_.attr("render_chat")(chat_request);

    auto                         input_ids    = py::cast<std::vector<int>>(rendered_input.attr("input_ids"));
    py::list                     mm_inputs_py = rendered_input.attr("multimodal_inputs");
    std::vector<MultimodalInput> mm_inputs;
    for (const auto& item : mm_inputs_py) {
        mm_inputs.emplace_back(py::cast<std::string>(item.attr("url")),
                               std::vector<torch::Tensor>{},
                               py::cast<int>(item.attr("mm_type")),
                               py::cast<int>(item.attr("config").attr("width")),
                               py::cast<int>(item.attr("config").attr("height")),
                               py::cast<int>(item.attr("config").attr("min_pixels")),
                               py::cast<int>(item.attr("config").attr("max_pixels")),
                               py::cast<int>(item.attr("config").attr("fps")),
                               py::cast<int>(item.attr("config").attr("min_frames")),
                               py::cast<int>(item.attr("config").attr("max_frames")));
    }
    auto rendered_prompt = py::cast<std::string>(rendered_input.attr("rendered_prompt"));

    return RenderedInputs(input_ids, mm_inputs, rendered_prompt);
}

std::string ChatRender::toString() {
    py::gil_scoped_acquire acquire;
    py::str                py_str  = py::str(render_);
    std::string            cpp_str = py_str;
    return cpp_str;
}

}  // namespace rtp_llm
