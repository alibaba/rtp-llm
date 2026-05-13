#include "rtp_llm/cpp/pybind/multi_gpu_gpt/XGrammarBootstrap.h"

#include <pybind11/stl.h>
#include <xgrammar/tokenizer_info.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace rtp_llm {

// Take the vocab as a dict<token_str, token_id> straight from
// tokenizer.get_vocab() and do the "id → token" reverse fill in C++. Saves
// the Python-side `[""] * vocab_size` allocation + per-id assignment loop;
// xgrammar's FromVocabAndMetadata still needs the indexed vector internally.
std::string buildXGrammarTokenizerInfoJson(
    const std::unordered_map<std::string, int32_t>& vocab,
    const std::string&                              backend_tokenizer_str,
    int64_t                                         vocab_size,
    const std::vector<int32_t>&                     stop_token_ids) {
    std::vector<std::string> encoded_vocab(vocab_size);
    for (const auto& [tok, tid] : vocab) {
        if (tid >= 0 && tid < static_cast<int32_t>(vocab_size)) {
            encoded_vocab[tid] = tok;
        }
    }

    // DetectMetadataFromHF returns {"vocab_type":N,"add_prefix_space":bool};
    // splice in vocab_size + stop_token_ids before the closing '}' rather
    // than pulling in a JSON lib just to rebuild 4 keys.
    std::string meta  = xgrammar::TokenizerInfo::DetectMetadataFromHF(backend_tokenizer_str);
    std::string stops = "[";
    for (size_t i = 0; i < stop_token_ids.size(); ++i) {
        if (i) stops += ",";
        stops += std::to_string(stop_token_ids[i]);
    }
    stops += "]";
    meta.insert(meta.size() - 1,
                ",\"vocab_size\":" + std::to_string(vocab_size)
                    + ",\"stop_token_ids\":" + stops);
    return xgrammar::TokenizerInfo::FromVocabAndMetadata(encoded_vocab, meta).SerializeJSON();
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
