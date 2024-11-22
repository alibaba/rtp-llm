#include "maga_transformer/cpp/api_server/TokenProcessor.h"

namespace rtp_llm {

using namespace autil::legacy;
using namespace autil::legacy::json;

class AuxInfoAdapter: public Jsonizable, public AuxInfo {
public:
    void Jsonize(Jsonizable::JsonWrapper& json) override {
        json.Jsonize("cost_time_ms", cost_time_ms, cost_time_ms);
        json.Jsonize("iter_count", iter_count, iter_count);
        json.Jsonize("input_len", input_len, input_len);
        json.Jsonize("prefix_len", prefix_len, prefix_len);
        json.Jsonize("reuse_len", reuse_len, reuse_len);
        json.Jsonize("output_len", output_len, output_len);
        json.Jsonize("fallback_tokens", fallback_tokens, fallback_tokens);
        json.Jsonize("fallback_times", fallback_times, fallback_times);
    }
    AuxInfoAdapter() {
        AuxInfo();
    }
    AuxInfoAdapter(const AuxInfo& base) {
        cost_time_us    = base.cost_time_us;
        iter_count      = base.iter_count;
        input_len       = base.input_len;
        prefix_len      = base.prefix_len;
        reuse_len       = base.reuse_len;
        output_len      = base.output_len;
        fallback_tokens = base.fallback_tokens;
        fallback_times  = base.fallback_times;

        cost_time_ms = cost_time_us / 1000.0;
    }
    float cost_time_ms;
};

struct TokenProcessorResponse: public Jsonizable {
    void Jsonize(Jsonizable::JsonWrapper& json) override {
        json.Jsonize("response", response, response);
        json.Jsonize("finished", finished, finished);
        json.Jsonize("aux_info", aux_info, aux_info);
    }
    std::string    response;
    bool           finished;
    AuxInfoAdapter aux_info;
};

TokenProcessor::TokenProcessor(py::object token_processor): token_processor_(token_processor) {}

std::string TokenProcessor::decode(const std::vector<int>& token_ids) {
    py::gil_scoped_acquire acquire;
    std::string            res = py::cast<std::string>(token_processor_.attr("decode")(token_ids));
    return res;
}

std::vector<int> TokenProcessor::encode(const std::string& prompt) {
    py::gil_scoped_acquire acquire;
    auto                   res = token_processor_.attr("encode")(prompt);
    std::vector<int>       vecInt;
    if (!py::isinstance<py::list>(res)) {
        throw std::runtime_error("Expected a list, but get " + py::cast<std::string>(py::str(res)));
    }
    py::list py_list = py::reinterpret_borrow<py::list>(res);
    for (auto item : py_list) {
        vecInt.push_back(py::cast<int>(item));
    }
    return vecInt;
}

std::string TokenProcessor::formatResponse(const std::string& generate_texts, const GenerateOutputs* generate_outputs) {
    TokenProcessorResponse res;
    res.response = generate_texts;
    res.finished = generate_outputs->generate_outputs[0].finished;
    res.aux_info = AuxInfoAdapter(generate_outputs->generate_outputs[0].aux_info);
    return ToJsonString(res, /*isCompact=*/true);
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
            FT_LOG_WARNING("tokenizer failed, offset mapping expected list but type is %s, offset mapping: %s",
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
        FT_LOG_WARNING("tokenizer result has no offset_mapping");
    }

    // input_ids
    if (py_res.contains("input_ids")) {
        auto py_input_ids = py_res["input_ids"];
        if (!py::isinstance<py::list>(py_input_ids)) {
            FT_LOG_WARNING("tokenizer failed, input ids expected list but type is: %s, input ids: %s",
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
        FT_LOG_WARNING("tokenizer result has no input_ids");
    }

    return response;
}

}  // namespace rtp_llm