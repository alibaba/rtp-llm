#include "rtp_llm/cpp/pybind/multi_gpu_gpt/XGrammarBootstrap.h"

#include <pybind11/stl.h>
#include <xgrammar/tokenizer_info.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"

namespace py = pybind11;

namespace rtp_llm {

namespace {

std::vector<int32_t> collectStopTokenIds(py::object model) {
    py::object           mc = model.attr("model_config");
    py::object           st = mc.attr("special_tokens");
    std::vector<int32_t> ids;
    auto                 add_id = [&](int32_t id) {
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
            ids.push_back(id);
        }
    };

    py::object eos = st.attr("eos_token_id");
    if (py::isinstance<py::int_>(eos)) {
        add_id(eos.cast<int32_t>());
    } else if (py::isinstance<py::list>(eos) || py::isinstance<py::tuple>(eos)) {
        for (py::handle item : eos) {
            add_id(py::reinterpret_borrow<py::object>(item).cast<int32_t>());
        }
    }

    py::object stop_list = st.attr("stop_words_id_list");
    if (!stop_list.is_none()) {
        try {
            auto stop_seqs = stop_list.cast<std::vector<std::vector<int64_t>>>();
            for (const auto& seq : stop_seqs) {
                if (seq.size() == 1) {
                    add_id(static_cast<int32_t>(seq[0]));
                }
            }
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("collectStopTokenIds: stop_words_id_list cast failed: %s", e.what());
        }
    }

    std::sort(ids.begin(), ids.end());
    return ids;
}

}  // namespace

std::string buildXGrammarTokenizerInfoJson(const std::unordered_map<std::string, int32_t>& vocab,
                                           const std::string&                              backend_tokenizer_str,
                                           const std::vector<int32_t>&                     stop_token_ids) {
    static constexpr int64_t kMaxVocabSize = 1'000'000;
    if (vocab.empty()) {
        throw std::invalid_argument("buildXGrammarTokenizerInfoJson: vocab is empty");
    }
    int64_t max_id = -1;
    for (const auto& [_, tid] : vocab) {
        if (tid < 0) {
            throw std::invalid_argument("buildXGrammarTokenizerInfoJson: negative token id " + std::to_string(tid));
        }
        max_id = std::max(max_id, static_cast<int64_t>(tid));
    }
    const int64_t vocab_size = max_id + 1;
    if (vocab_size > kMaxVocabSize) {
        throw std::invalid_argument("vocab_size must be in (0, " + std::to_string(kMaxVocabSize) + "], got "
                                    + std::to_string(vocab_size));
    }
    std::vector<std::string> encoded_vocab(static_cast<size_t>(vocab_size));
    for (const auto& [tok, tid] : vocab) {
        encoded_vocab[static_cast<size_t>(tid)] = tok;
    }

    std::string meta = xgrammar::TokenizerInfo::DetectMetadataFromHF(backend_tokenizer_str);

    std::string stops = "[";
    for (size_t i = 0; i < stop_token_ids.size(); ++i) {
        if (i)
            stops += ",";
        stops += std::to_string(stop_token_ids[i]);
    }
    stops += "]";

    const auto close_pos = meta.rfind('}');
    if (close_pos == std::string::npos) {
        throw std::runtime_error(
            "buildXGrammarTokenizerInfoJson: DetectMetadataFromHF returned a non-object metadata string: '" + meta
            + "'");
    }
    const auto  first_content = meta.find_first_not_of(" \t\r\n", meta.find('{') + 1);
    const bool  has_members   = first_content != std::string::npos && first_content < close_pos;
    std::string injected      = std::string(has_members ? "," : "")
                           + "\"vocab_size\":" + std::to_string(vocab_size) + ",\"stop_token_ids\":" + stops;
    meta.insert(close_pos, injected);
    return xgrammar::TokenizerInfo::FromVocabAndMetadata(encoded_vocab, meta).SerializeJSON();
}

void bootstrapGrammarConfigFromModel(py::object model, GrammarConfig& gc) {
    std::vector<int32_t> stop_token_ids;
    try {
        py::object tokenizer = model.attr("tokenizer").attr("tokenizer");
        auto       vocab     = tokenizer.attr("get_vocab")().cast<std::unordered_map<std::string, int32_t>>();
        stop_token_ids       = collectStopTokenIds(model);
        const std::string backend_str = tokenizer.attr("backend_tokenizer").attr("to_str")().cast<std::string>();
        gc.tokenizer_info_json        = buildXGrammarTokenizerInfoJson(vocab, backend_str, stop_token_ids);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("xgrammar bootstrap failed (%s); grammar disabled", e.what());
        gc.tokenizer_info_json = "";
        return;
    }

    RTP_LLM_LOG_INFO("grammar bootstrap: json=%zuB, stop_tokens=%zu",
                     gc.tokenizer_info_json.size(),
                     stop_token_ids.size());
}

}  // namespace rtp_llm
