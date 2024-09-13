#include <atomic>
#include "maga_transformer/cpp/dataclass/Query.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/devices/DeviceFactory.h"

using namespace std;

namespace rtp_llm {

void registerMultimodalInput(const py::module& m) {
    pybind11::class_<MultimodalInput>(m, "MultimodalInput")
        .def(pybind11::init<std::string, int32_t>())
        .def_readwrite("url", &MultimodalInput::url)
        .def_readwrite("mm_type", &MultimodalInput::mm_type);
}

} // namespace rtp_llm
