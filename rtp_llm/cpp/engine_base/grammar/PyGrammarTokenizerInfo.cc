#include <Python.h>

#include "rtp_llm/cpp/engine_base/grammar/XGrammarTokenizerInfo.h"

#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

PyObject* serializeGrammarTokenizerInfo(PyObject*, PyObject* args) {
    PyObject*   encoded_vocab_obj       = nullptr;
    const char* tokenizer_metadata_json = nullptr;
    if (!PyArg_ParseTuple(args, "Os:serialize_grammar_tokenizer_info", &encoded_vocab_obj, &tokenizer_metadata_json)) {
        return nullptr;
    }

    PyObject* encoded_vocab_seq =
        PySequence_Fast(encoded_vocab_obj, "encoded_vocab must be a sequence of str or bytes");
    if (encoded_vocab_seq == nullptr) {
        return nullptr;
    }

    std::vector<std::string> encoded_vocab;
    const auto               vocab_size = PySequence_Fast_GET_SIZE(encoded_vocab_seq);
    encoded_vocab.reserve(static_cast<size_t>(vocab_size));

    PyObject** items = PySequence_Fast_ITEMS(encoded_vocab_seq);
    for (Py_ssize_t i = 0; i < vocab_size; ++i) {
        Py_ssize_t  token_size = 0;
        const char* token      = nullptr;
        if (PyUnicode_Check(items[i])) {
            token = PyUnicode_AsUTF8AndSize(items[i], &token_size);
        } else if (PyBytes_Check(items[i])) {
            char* bytes = nullptr;
            if (PyBytes_AsStringAndSize(items[i], &bytes, &token_size) == 0) {
                token = bytes;
            } else {
                token = nullptr;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "encoded_vocab must be a sequence of str or bytes");
        }
        if (token == nullptr) {
            Py_DECREF(encoded_vocab_seq);
            return nullptr;
        }
        encoded_vocab.emplace_back(token, static_cast<size_t>(token_size));
    }
    Py_DECREF(encoded_vocab_seq);

    try {
        const auto result = rtp_llm::xgrammar_impl::serializeTokenizerInfo(encoded_vocab, tokenizer_metadata_json);
        return PyUnicode_FromStringAndSize(result.data(), static_cast<Py_ssize_t>(result.size()));
    } catch (const std::invalid_argument& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    }
    return nullptr;
}

PyMethodDef methods[] = {
    {"serialize_grammar_tokenizer_info",
     serializeGrammarTokenizerInfo,
     METH_VARARGS,
     "Serialize tokenizer metadata for the xgrammar backend."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "libth_grammar_tokenizer_info",
    nullptr,
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit_libth_grammar_tokenizer_info() {
    return PyModule_Create(&module);
}
