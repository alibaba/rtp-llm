#include "rtp_llm/cpp/multimodal_processor/MMRdmaEncoderOp.h"

#include "rtp_llm/cpp/multimodal_processor/MMRdmaVitConfig.h"

namespace py = pybind11;

namespace rtp_llm {

MMRdmaEncoderOp::MMRdmaEncoderOp(const py::object& vit_config) {
    VitConfig config;
    extractMMRdmaVitConfig(vit_config, config);
    transport_ = createMMRdmaTransport(config, MMRdmaRole::ENCODER_SERVER);
}

py::bytes MMRdmaEncoderOp::exportEmbedding(const torch::Tensor& embedding) {
    if (!transport_) {
        return py::bytes();
    }
    MMRdmaDescPB descriptor;
    bool         exported = false;
    {
        py::gil_scoped_release release;
        exported = transport_->exportEmbedding({embedding}, {MMRdmaTensorPB::EMBEDDING}, &descriptor);
    }
    if (!exported) {
        return py::bytes();
    }
    try {
        return py::bytes(descriptor.SerializeAsString());
    } catch (...) {
        release({descriptor.handle()});
        throw;
    }
}

void MMRdmaEncoderOp::release(const std::vector<std::string>& handles) {
    if (transport_) {
        py::gil_scoped_release release;
        transport_->releaseEmbedding(handles);
    }
}

void registerMMRdmaEncoderOp(py::module& m) {
    py::class_<MMRdmaEncoderOp>(m, "MMRdmaEncoderOp")
        .def(py::init<const py::object&>())
        .def("enabled", &MMRdmaEncoderOp::enabled)
        .def("export_embedding", &MMRdmaEncoderOp::exportEmbedding)
        .def("release", &MMRdmaEncoderOp::release);
}

}  // namespace rtp_llm
