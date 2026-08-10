#include "rtp_llm/cpp/pybind/multi_gpu_gpt/XGrammarBootstrap.h"

#include <pybind11/stl.h>
#include <xgrammar/tokenizer_info.h>

#include <limits>
#include <stdexcept>

namespace py = pybind11;

namespace rtp_llm {

namespace {

std::vector<std::string> makeEncodedVocab(const std::unordered_map<std::string, int32_t>& vocab, int64_t vocab_size) {
    if (vocab_size < 0 || vocab_size > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("xgrammar vocab_size must fit in a non-negative int");
    }
    std::vector<std::string> encoded_vocab(vocab_size);
    for (const auto& [token, token_id] : vocab) {
        if (token_id >= 0 && token_id < static_cast<int32_t>(vocab_size)) {
            encoded_vocab[token_id] = token;
        }
    }
    return encoded_vocab;
}

}  // namespace

std::string buildXGrammarTokenizerInfoJson(const std::unordered_map<std::string, int32_t>& vocab,
                                           const std::string&                              backend_tokenizer_str,
                                           int64_t                                         vocab_size,
                                           const std::vector<int32_t>&                     stop_token_ids) {
    auto encoded_vocab = makeEncodedVocab(vocab, vocab_size);

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

std::string buildXGrammarTokenizerInfoJsonFromVocab(const std::unordered_map<std::string, int32_t>& vocab,
                                                    int32_t                                         vocab_type,
                                                    int64_t                                         vocab_size,
                                                    const std::vector<int32_t>&                     stop_token_ids,
                                                    bool                                            add_prefix_space) {
    if (vocab_type < static_cast<int32_t>(xgrammar::VocabType::RAW)
        || vocab_type > static_cast<int32_t>(xgrammar::VocabType::BYTE_LEVEL)) {
        throw std::invalid_argument("xgrammar vocab_type must be 0 (raw), 1 (byte fallback), or 2 (byte level)");
    }
    auto encoded_vocab = makeEncodedVocab(vocab, vocab_size);
    return xgrammar::TokenizerInfo(encoded_vocab,
                                   static_cast<xgrammar::VocabType>(vocab_type),
                                   static_cast<int>(vocab_size),
                                   stop_token_ids,
                                   add_prefix_space)
        .SerializeJSON();
}

void registerXGrammarBootstrap(py::module& m) {
    m.def("build_xgrammar_tokenizer_info_json",
          &buildXGrammarTokenizerInfoJson,
          py::arg("vocab"),
          py::arg("backend_tokenizer_str"),
          py::arg("vocab_size"),
          py::arg("stop_token_ids"));
    m.def("build_xgrammar_tokenizer_info_json_from_vocab",
          &buildXGrammarTokenizerInfoJsonFromVocab,
          py::arg("vocab"),
          py::arg("vocab_type"),
          py::arg("vocab_size"),
          py::arg("stop_token_ids"),
          py::arg("add_prefix_space") = false);
}

}  // namespace rtp_llm
