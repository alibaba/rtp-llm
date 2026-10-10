
#include "rtp_llm/cpp/api_server/tokenizer/Tokenizer.h"

#include <map>
#include <algorithm>

namespace rtp_llm {

using namespace std;

std::string Tokenizer::sidMappingJson() {
    py::gil_scoped_acquire acquire;
    // RTP BaseTokenizer (including QWenV2Tokenizer) wraps the HF tokenizer and
    // does not expose get_vocab/__len__. Unwrap only for vocabulary inspection;
    // getEosTokenId() below must retain the wrapper's effective EOS override.
    py::object vocab_tokenizer = tokenizer_;
    if (py::hasattr(vocab_tokenizer, "get_real_tokenizer")) {
        vocab_tokenizer = vocab_tokenizer.attr("get_real_tokenizer")();
    }
    if (vocab_tokenizer.is_none() || !py::hasattr(vocab_tokenizer, "get_vocab")
        || !py::hasattr(vocab_tokenizer, "__len__")) {
        return "{}";
    }
    const auto vocab      = vocab_tokenizer.attr("get_vocab")().cast<py::dict>();
    const int  vocab_size = py::len(vocab_tokenizer);
    const auto eos        = getEosTokenId();
    if (!eos || vocab_size <= 1699 || *eos < 0 || *eos >= vocab_size) {
        return "{}";
    }
    std::map<std::string, int> tokens;
    for (auto entry : vocab) {
        const auto symbol = py::cast<std::string>(entry.first);
        if (symbol.size() < 2 || symbol[0] != 'C'
            || !std::all_of(symbol.begin() + 1, symbol.end(), [](char c) { return c >= '0' && c <= '9'; })) {
            continue;
        }
        const auto id = py::cast<int>(entry.second);
        if (id < 0 || id >= vocab_size || id == 1699 || id == *eos) {
            throw std::runtime_error("invalid C-token vocabulary entry");
        }
        tokens.emplace(symbol, id);
    }
    if (tokens.empty()) {
        return "{}";
    }
    std::string canonical =
        "rtp-sid-mapping-v1\n" + std::to_string(vocab_size) + "\n1699\n" + std::to_string(*eos) + "\n";
    py::dict mapping;
    for (const auto& [symbol, id] : tokens) {
        canonical += symbol + "\t" + std::to_string(id) + "\n";
        mapping[py::str(symbol)] = id;
    }
    const auto fingerprint = py::module::import("hashlib").attr("sha256")(py::bytes(canonical)).attr("hexdigest")();
    py::dict   manifest;
    manifest["mapping_fingerprint"] = fingerprint;
    manifest["vocab_size"]          = vocab_size;
    manifest["start_token_id"]      = 1699;
    manifest["end_token_id"]        = *eos;
    manifest["tokens"]              = mapping;
    return py::module::import("json").attr("dumps")(manifest).cast<std::string>();
}

std::optional<int> Tokenizer::getEosTokenId() {
    py::gil_scoped_acquire acquire;
    auto                   res = tokenizer_.attr("eos_token_id");
    if (res.is_none()) {
        return std::nullopt;
    }
    return res.cast<int>();
}

bool Tokenizer::isPreTrainedTokenizer() {
    py::gil_scoped_acquire acquire;
    py::module             transformers            = py::module::import("transformers");
    py::object             PreTrainedTokenizerBase = transformers.attr("PreTrainedTokenizerBase");
    return py::isinstance(tokenizer_, PreTrainedTokenizerBase);
}

std::vector<int> Tokenizer::encode(const std::string& tokens_str) {
    py::gil_scoped_acquire acquire;
    auto                   res = tokenizer_.attr("encode")(py::str(tokens_str));
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

std::string Tokenizer::decode(const std::vector<int>& token_ids) {
    py::gil_scoped_acquire acquire;
    py::list               py_token_ids;
    for (auto token_id : token_ids) {
        py_token_ids.append(token_id);
    }
    std::string res = py::cast<std::string>(tokenizer_.attr("decode")(py_token_ids));
    return res;
}

std::string Tokenizer::toString() {
    py::gil_scoped_acquire acquire;
    py::str                py_str  = py::str(tokenizer_);
    std::string            cpp_str = py_str;
    return cpp_str;
}

vector<int> Tokenizer::convertSelectTokens(const std::vector<std::string>& select_tokens_str, int vocab_size) {
    std::vector<int> select_tokens_id;

    for (const auto& token_str : select_tokens_str) {
        auto vec = encode(token_str);
        select_tokens_id.insert(select_tokens_id.begin(), vec.begin(), vec.end());
    }

    auto areTokensValid = [](const std::vector<int>& select_tokens_id, int vocab_size) {
        return std::all_of(select_tokens_id.begin(), select_tokens_id.end(), [vocab_size](int token_id) {
            return token_id < vocab_size && token_id >= 0;
        });
    };
    if (!areTokensValid(select_tokens_id, vocab_size)) {
        throw std::runtime_error("token_id should be less than vocab_size");
    }

    return select_tokens_id;
}

}  // namespace rtp_llm
