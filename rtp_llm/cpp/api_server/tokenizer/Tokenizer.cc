
#include "rtp_llm/cpp/api_server/tokenizer/Tokenizer.h"

namespace rtp_llm {

using namespace std;

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
