#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>

namespace rtp_llm {

// Render a py::object's type as "<module>.<class>" — e.g.
// "rtp_llm.async_decoder_engine.xgrammar_backend.XGrammarGrammar".
// Intended for debug logging; never throws (any pybind error is swallowed
// and "<unknown_py_type>" returned), so it is safe inside log call sites
// that must not raise. Caller must hold the GIL.
inline std::string pyObjTypeName(const pybind11::object& obj) {
    try {
        if (obj.is_none()) {
            return "None";
        }
        pybind11::object cls   = obj.attr("__class__");
        std::string      mod   = pybind11::str(cls.attr("__module__")).cast<std::string>();
        std::string      cname = pybind11::str(cls.attr("__name__")).cast<std::string>();
        return mod + "." + cname;
    } catch (...) {
        return "<unknown_py_type>";
    }
}

}  // namespace rtp_llm
