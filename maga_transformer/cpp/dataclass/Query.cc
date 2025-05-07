#include <atomic>
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/core/Buffer.h"
#include "maga_transformer/cpp/core/BufferHelper.h"
#include "maga_transformer/cpp/devices/DeviceFactory.h"

using namespace std;

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

} // namespace rtp_llm
