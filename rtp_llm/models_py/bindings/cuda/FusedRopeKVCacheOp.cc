#include "rtp_llm/models_py/bindings/cuda/FusedRopeKVCacheOp.h"

namespace rtp_llm {

void registerTRTAttn(const py::module& m) {
    pybind11::class_<TRTAttn, std::shared_ptr<TRTAttn>, rtp_llm::ParamsBase>(m, "TRTAttn")
        .def(pybind11::init<>())
        .def_readwrite("kv_cache_offset", &TRTAttn::kv_cache_offset)
        .def(
            "__cpp_ptr__",
            [](TRTAttn& self) { return reinterpret_cast<uintptr_t>(&self); },
            "Get C++ object pointer address");
}

}  // namespace rtp_llm
