#include "rtp_llm/cpp/api_server/TokenProcessor.h"

#include <algorithm>

#include <pybind11/numpy.h>

#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/api_server/Exception.h"

namespace rtp_llm {

TokenProcessor::TokenProcessor(py::object token_processor): token_processor_(token_processor) {}

TokenProcessor::~TokenProcessor() {
    py::gil_scoped_acquire acquire;
    // make sure all py::object are deconstructed while holding GIL
    py::object tmp;
    tmp = std::move(token_processor_);
}

py::object TokenProcessor::getPyObject() const {
    return token_processor_;
}

std::string TokenProcessor::decode(const std::vector<int>& token_ids) {
    py::gil_scoped_acquire acquire;
    std::string            res = py::cast<std::string>(token_processor_.attr("decode")(token_ids));
    return res;
}

std::vector<int> TokenProcessor::encode(const std::string& prompt) {
    py::gil_scoped_acquire acquire;
    py::bytes              py_prompt_bytes(prompt);
    auto                   res = token_processor_.attr("encode")(py_prompt_bytes);
    std::vector<int>       vecInt;
    if (!py::isinstance<py::list>(res)) {
        throw HttpApiServerException(HttpApiServerException::TOKENIZER_ERROR,
                                     "Expected a list, but get " + py::cast<std::string>(py::str(res)));
    }
    py::list py_list = py::reinterpret_borrow<py::list>(res);
    for (auto item : py_list) {
        vecInt.push_back(py::cast<int>(item));
    }
    return vecInt;
}

std::shared_ptr<TokenizerEncodeResponse> TokenProcessor::tokenizer(const std::string& prompt) {
    auto                   response = std::make_shared<TokenizerEncodeResponse>();
    py::gil_scoped_acquire acquire;
    auto                   res    = token_processor_(prompt);
    auto                   py_res = py::cast<py::dict>(res);

    // offset_mapping
    if (py_res.contains("offset_mapping")) {
        auto py_offset_mapping = py_res["offset_mapping"];
        if (!py::isinstance<py::list>(py_offset_mapping)) {
            RTP_LLM_LOG_WARNING("tokenizer failed, offset mapping expected list but type is %s, offset mapping: %s",
                                py::cast<std::string>(py::str(py::type::of(py_offset_mapping))).c_str(),
                                py::cast<std::string>(py::str(py_offset_mapping)).c_str());
            return nullptr;
        }
        std::vector<std::vector<int>> offset_mapping;
        auto                          py_offset_mapping_list = py::cast<py::list>(py_offset_mapping);
        for (auto& py_offset : py_offset_mapping_list) {
            offset_mapping.push_back({});
            auto py_offset_list = py::cast<py::list>(py_offset);
            for (auto py_num : py_offset_list) {
                offset_mapping.back().push_back(py::cast<int>(py_num));
            }
        }
        response->offset_mapping = offset_mapping;
    } else {
        RTP_LLM_LOG_WARNING("tokenizer result has no offset_mapping");
    }

    // input_ids
    if (py_res.contains("input_ids")) {
        auto py_input_ids = py_res["input_ids"];
        if (!py::isinstance<py::list>(py_input_ids)) {
            RTP_LLM_LOG_WARNING("tokenizer failed, input ids expected list but type is: %s, input ids: %s",
                                py::cast<std::string>(py::str(py::type::of(py_input_ids))).c_str(),
                                py::cast<std::string>(py::str(py_input_ids)).c_str());
            return nullptr;
        }
        std::vector<int> input_ids;
        auto             py_input_ids_list = py::cast<py::list>(py_input_ids);
        for (auto& py_id : py_input_ids_list) {
            input_ids.push_back(py::cast<int>(py_id));
        }
        response->token_ids = input_ids;
    } else {
        RTP_LLM_LOG_WARNING("tokenizer result has no input_ids");
    }

    return response;
}

std::vector<std::string> TokenProcessor::decodeTokens(std::shared_ptr<TokenProcessorPerStream> ctx,
                                                      GenerateOutputs&                         responses,
                                                      std::vector<int>&                        output_lens,
                                                      std::shared_ptr<GenerateConfig>          config) {
    return ctx->decodeTokens(responses, output_lens, config);
}

std::shared_ptr<TokenProcessorPerStream> TokenProcessor::getTokenProcessorCtx(
    bool use_beam_search, int size, const std::shared_ptr<TokenProcessor>& token_processor_cpp) {
    auto ctx = std::make_shared<TokenProcessorPerStream>();
    ctx->init(use_beam_search, size, token_processor_cpp);
    return ctx;
}

void TokenProcessorPerStream::init(bool                                   use_beam_search,
                                   int                                    size,
                                   const std::shared_ptr<TokenProcessor>& token_processor_cpp) {
    py::gil_scoped_acquire acquire;
    py::module             token_processor = py::module::import("rtp_llm.frontend.token_processor");
    py::object             cls             = token_processor.attr("TokenProcessorPerStream");
    token_processor_stream_                = cls(use_beam_search, size, token_processor_cpp->getPyObject());
}

std::vector<std::string> TokenProcessorPerStream::decodeTokens(GenerateOutputs&                responses,
                                                               std::vector<int>&               output_lens,
                                                               std::shared_ptr<GenerateConfig> config) {
    py::gil_scoped_acquire   acquire;
    std::vector<std::string> texts;
    for (size_t i = 0; i < responses.generate_outputs.size(); i++) {
        auto&            response = responses.generate_outputs[i];
        py::array_t<int> token_ids(response.output_ids->size(), response.output_ids->data<int>());
        py::tuple        result = token_processor_stream_.attr("decode_tokens")(i,
                                                                         token_ids,
                                                                         response.finished,
                                                                         config->print_stop_words,
                                                                         config->stop_words_str,
                                                                         config->stop_words_list,
                                                                         config->return_incremental);
        if (result.size() != 2) {
            RTP_LLM_LOG_WARNING("token_processor_per_stream.decodeTokens() failed.");
            continue;
        }
        int         len  = result[0].cast<int>();
        std::string text = result[1].cast<std::string>();
        output_lens.push_back(len);
        texts.push_back(text);
    }
    return texts;
}

TokenProcessorPerStream::~TokenProcessorPerStream() {
    py::gil_scoped_acquire acquire;
    // make sure all py::object are deconstructed while holding GIL
    py::object tmp;
    tmp = std::move(token_processor_stream_);
}

}  // namespace rtp_llm
