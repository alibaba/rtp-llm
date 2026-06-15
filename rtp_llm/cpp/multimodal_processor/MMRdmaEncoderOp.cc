#include "rtp_llm/cpp/multimodal_processor/MMRdmaEncoderOp.h"

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/multimodal_processor/MMRdmaVitConfig.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

MMRdmaEncoderOp::MMRdmaEncoderOp(const py::object& vit_config) {
    VitConfig cfg;
    extractMMRdmaVitConfig(vit_config, cfg);
    transport_ = createMMRdmaTransport(cfg, MMRdmaRole::ENCODER_SERVER);
}

py::bytes MMRdmaEncoderOp::exportEmbedding(torch::Tensor embedding) {
    if (transport_ == nullptr) {
        return py::bytes(std::string());
    }
    MMRdmaDescPB desc;
    bool         ok = false;
    {
        py::gil_scoped_release release;  // D2D copy + MR lookup, no Python involved
        ok = transport_->exportEmbedding(embedding, &desc);
    }
    if (!ok) {
        return py::bytes(std::string());
    }
    return py::bytes(desc.SerializeAsString());
}

void MMRdmaEncoderOp::release(const std::vector<std::string>& handles) {
    if (transport_ == nullptr) {
        return;
    }
    py::gil_scoped_release release;
    transport_->releaseEmbedding(handles);
}

void registerMMRdmaEncoderOp(py::module& m) {
    py::class_<MMRdmaEncoderOp, std::shared_ptr<MMRdmaEncoderOp>>(m, "MMRdmaEncoderOp")
        .def(py::init<const py::object&>(), py::arg("vit_config"))
        .def("enabled", &MMRdmaEncoderOp::enabled)
        .def("export_embedding", &MMRdmaEncoderOp::exportEmbedding, py::arg("embedding"))
        .def("release", &MMRdmaEncoderOp::release, py::arg("handles"));
}

}  // namespace rtp_llm
