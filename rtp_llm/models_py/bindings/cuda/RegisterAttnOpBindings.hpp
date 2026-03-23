#pragma once

#include "rtp_llm/models_py/bindings/cuda/FlashInferOp.h"
#include "rtp_llm/models_py/bindings/cuda/TRTAttnOp.h"
#include "rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.h"
#include "rtp_llm/models_py/bindings/cuda/SparseMlaParams.h"

#ifdef USING_CUDA12
#include "rtp_llm/models_py/bindings/cuda/XQAAttnOp.h"
#endif

namespace torch_ext {

void registerAttnOpBindings(py::module& rtp_ops_m) {
    pybind11::class_<rtp_llm::TRTAttn, std::shared_ptr<rtp_llm::TRTAttn>, rtp_llm::ParamsBase>(rtp_ops_m, "TRTAttn")
        .def(pybind11::init<>())
        .def_readwrite("kv_cache_offset", &rtp_llm::TRTAttn::kv_cache_offset)
        .def(
            "__cpp_ptr__",
            [](rtp_llm::TRTAttn& self) { return reinterpret_cast<uintptr_t>(&self); },
            "Get C++ object pointer address");
    rtp_llm::registerFlashInferOp(rtp_ops_m);
    rtp_llm::registerTRTAttnOp(rtp_ops_m);
    rtp_llm::registerPyFlashInferMlaParams(rtp_ops_m);
    rtp_llm::registerPySparseMlaParams(rtp_ops_m);
#ifdef USING_CUDA12
    rtp_llm::registerXQAAttnOp(rtp_ops_m);
#endif
}

}  // namespace torch_ext
