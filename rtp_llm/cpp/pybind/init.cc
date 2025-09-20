#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/dataclass/KvCacheInfo.h"
#include "rtp_llm/cpp/dataclass/WorkerStatusInfo.h"
#include "rtp_llm/cpp/dataclass/EngineScheduleInfo.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "rtp_llm/models_py/bindings/RegisterOps.h"
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
    registerKvCacheInfo(m);
    registerWorkerStatusInfo(m);
    registerEngineScheduleInfo(m);

    registerRtpLLMOp(m);
    registerRtpEmbeddingOp(m);

    registerDeviceOps(m);
    registerPyOpDefs(m);

    registerMultimodalInput(m);

    py::module rtp_ops_m = m.def_submodule("rtp_llm_ops", "rtp llm custom ops");
    registerPyModuleOps(rtp_ops_m);
}

}  // namespace rtp_llm
