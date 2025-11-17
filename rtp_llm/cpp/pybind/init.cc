#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

namespace rtp_llm {

void registerMultimodalInput(const py::module& m) {
    pybind11::class_<MultimodalInput>(m, "MultimodalInput")
        .def(pybind11::init<std::string, torch::Tensor, int32_t>(),
             py::arg("url"),
             py::arg("tensor"),
             py::arg("mm_type"))
        .def_readwrite("url", &MultimodalInput::url)
        .def_readwrite("mm_type", &MultimodalInput::mm_type)
        .def_readwrite("tensor", &MultimodalInput::tensor);
}

PYBIND11_MODULE(libth_transformer, m) {
    registerRtpLLMOp(m);
    registerMultimodalInput(m);
    registerRtpEmbeddingOp(m);
}

}  // namespace rtp_llm
