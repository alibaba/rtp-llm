#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace rtp_llm {

namespace py = pybind11;

const std::string capturePythonStackTrace(bool trace_memory) {
    if (trace_memory) {
        py::gil_scoped_acquire gil;
        try {
            py::object traceback_module = py::module::import("traceback");
            py::object format_stack     = traceback_module.attr("format_stack");
            py::list   stack_list       = format_stack(py::none(), py::int_(10));

            std::string result;
            result.reserve(1024);

            for (auto item : stack_list) {
                result += py::str(item).cast<std::string>();
            }
            return result;
        } catch (...) {
            return "torch_allocated";
        }
    } else {
        return "torch_allocated";
    }
}

}  // namespace rtp_llm
