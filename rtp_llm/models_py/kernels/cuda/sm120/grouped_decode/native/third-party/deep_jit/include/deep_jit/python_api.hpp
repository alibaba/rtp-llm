#pragma once

#include <memory>

#include <pybind11/pybind11.h>

#include <deep_jit/runtime/runtime.hpp>
#include <deep_jit/utils/lazy.hpp>

namespace deep_jit {

template <typename Backend>
inline void register_python_api(pybind11::module_& module, LazyInit<Runtime<Backend>>& jit) {
    pybind11::class_<Runtime<Backend>, std::shared_ptr<Runtime<Backend>>>(module, "Runtime", pybind11::module_local());

    // Help the user library to export some functions to do warmup
    module.def("get_jit", [&jit]() {
        return jit.get();
    });
}

}  // namespace deep_jit
