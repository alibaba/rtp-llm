
#include "maga_transformer/cpp/tokenizer/Tokenizer.h"

namespace rtp_llm {

std::optional<int> Tokenizer::getEosTokenId() {
    py::gil_scoped_acquire acquire;
    auto res = tokenizer_.attr("eos_token_id");
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

}  // namespace rtp_llm
