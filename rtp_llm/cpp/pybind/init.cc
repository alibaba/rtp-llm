#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/cache/types.h"
#include "rtp_llm/cpp/engine_base/WorkerStatusInfo.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingQuery.h"
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

namespace rtp_llm {

void registerKvCacheInfo(const pybind11::module& m) {
    pybind11::class_<KVCacheInfo>(m, "KVCacheInfo")
        .def(pybind11::init<>())
        .def_readwrite("cached_keys", &KVCacheInfo::cached_keys)
        .def_readwrite("available_kv_cache", &KVCacheInfo::available_kv_cache)
        .def_readwrite("total_kv_cache", &KVCacheInfo::total_kv_cache)
        .def_readwrite("block_size", &KVCacheInfo::block_size)
        .def_readwrite("version", &KVCacheInfo::version);
}

void registerEngineScheduleInfo(const pybind11::module& m) {
    pybind11::class_<EngineScheduleInfo::TaskInfo>(m, "EngineTaskInfo")
        .def(pybind11::init<>())
        .def_readwrite("request_id", &EngineScheduleInfo::TaskInfo::request_id)
        .def_readwrite("inter_request_id", &EngineScheduleInfo::TaskInfo::inter_request_id)
        .def_readwrite("prefix_length", &EngineScheduleInfo::TaskInfo::prefix_length)
        .def_readwrite("input_length", &EngineScheduleInfo::TaskInfo::input_length)
        .def_readwrite("waiting_time_ms", &EngineScheduleInfo::TaskInfo::waiting_time_ms)
        .def_readwrite("iterate_count", &EngineScheduleInfo::TaskInfo::iterate_count)
        .def_readwrite("end_time_ms", &EngineScheduleInfo::TaskInfo::end_time_ms);

    pybind11::class_<EngineScheduleInfo>(m, "EngineScheduleInfo")
        .def(pybind11::init<>())
        .def_readwrite("last_schedule_delta", &EngineScheduleInfo::last_schedule_delta)
        .def_readwrite("finished_task_info_list", &EngineScheduleInfo::finished_task_info_list)
        .def_readwrite("running_task_info_list", &EngineScheduleInfo::running_task_info_list);
}

void registerWorkerStatusInfo(const pybind11::module& m) {
    pybind11::class_<WorkerStatusInfo>(m, "WorkerStatusInfo")
        .def(pybind11::init<>())
        .def_readwrite("role", &WorkerStatusInfo::role)
        .def_readwrite("engine_schedule_info", &WorkerStatusInfo::engine_schedule_info)
        .def_readwrite("status_version", &WorkerStatusInfo::status_version)
        .def_readwrite("alive", &WorkerStatusInfo::alive)
        .def_readwrite("dp_size", &WorkerStatusInfo::dp_size)
        .def_readwrite("tp_size", &WorkerStatusInfo::tp_size)
        .def_readwrite("dp_rank", &WorkerStatusInfo::dp_rank)
        .def_readwrite("precision", &WorkerStatusInfo::precision);
}

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

void registerEmbeddingOutput(const py::module& m) {
    py::class_<TypedOutput>(m, "TypedOutput")
        .def(py::init<>())
        .def_readwrite("isTensor", &TypedOutput::isTensor)
        .def_property(
            "t",
            [](const TypedOutput& self) -> py::object {
                return self.t.has_value() ? py::cast(self.t.value()) : py::none();
            },
            [](TypedOutput& self, const at::Tensor& tensor) { self.setTensorOuput(tensor); })
        .def_property(
            "map",
            [](const TypedOutput& self) -> py::object {
                return self.map.has_value() ? py::cast(self.map.value()) : py::none();
            },
            // FIX: Take by value instead of const reference
            [](TypedOutput& self, std::vector<std::map<std::string, at::Tensor>> map_val) {
                self.setMapOutput(map_val);
            });

    py::class_<EmbeddingOutput>(m, "EmbeddingCppOutput")
        .def(py::init<>())
        .def_readwrite("output", &EmbeddingOutput::output)
        .def_readwrite("error_info", &EmbeddingOutput::error_info)
        .def("setTensorOutput", &EmbeddingOutput::setTensorOutput)
        .def("setMapOutput", &EmbeddingOutput::setMapOutput)
        .def("setError", &EmbeddingOutput::setError);
}

PYBIND11_MODULE(libth_transformer, m) {
    registerKvCacheInfo(m);
    registerEngineScheduleInfo(m);
    registerWorkerStatusInfo(m);
    registerRtpLLMOp(m);
    registerMultimodalInput(m);
    registerRtpEmbeddingOp(m);
    registerEmbeddingOutput(m);
}

}  // namespace rtp_llm
