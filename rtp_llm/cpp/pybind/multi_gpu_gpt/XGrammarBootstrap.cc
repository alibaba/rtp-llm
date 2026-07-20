#include "rtp_llm/cpp/pybind/multi_gpu_gpt/XGrammarBootstrap.h"

#include <pybind11/stl.h>
#include <xgrammar/tokenizer_info.h>

namespace py = pybind11;

namespace rtp_llm {

std::string buildXGrammarTokenizerInfoJson(const std::unordered_map<std::string, int32_t>& vocab,
                                           const std::string&                              backend_tokenizer_str,
                                           int64_t                                         vocab_size,
                                           const std::vector<int32_t>&                     stop_token_ids) {
    std::vector<std::string> encoded_vocab(vocab_size);
    for (const auto& [token, token_id] : vocab) {
        if (token_id >= 0 && token_id < static_cast<int32_t>(vocab_size)) {
            encoded_vocab[token_id] = token;
        }
    }

    std::string metadata = xgrammar::TokenizerInfo::DetectMetadataFromHF(backend_tokenizer_str);
    std::string stops    = "[";
    for (size_t i = 0; i < stop_token_ids.size(); ++i) {
        if (i != 0) {
            stops += ",";
        }
        stops += std::to_string(stop_token_ids[i]);
    }
    stops += "]";
    metadata.insert(metadata.size() - 1,
                    ",\"vocab_size\":" + std::to_string(vocab_size) + ",\"stop_token_ids\":" + stops);
    return xgrammar::TokenizerInfo::FromVocabAndMetadata(encoded_vocab, metadata).SerializeJSON();
}

void registerXGrammarBootstrap(py::module& m) {
    m.def("build_xgrammar_tokenizer_info_json",
          &buildXGrammarTokenizerInfoJson,
          py::arg("vocab"),
          py::arg("backend_tokenizer_str"),
          py::arg("vocab_size"),
          py::arg("stop_token_ids"));
}

}  // namespace rtp_llm
