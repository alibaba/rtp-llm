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

py::bytes MMRdmaEncoderOp::exportEmbedding(torch::Tensor                embedding,
                                           std::optional<torch::Tensor> pos_id,
                                           std::vector<torch::Tensor>   extra_inputs) {
    if (transport_ == nullptr) {
        return py::bytes(std::string());
    }
    // Pack order must stay in lockstep with the manifest the LLM slices by: embedding first,
    // then the optional pos_id, then each extra_input in its original (per-image) order.
    std::vector<torch::Tensor>        tensors;
    std::vector<MMRdmaTensorPB::Role> roles;
    tensors.reserve(2 + extra_inputs.size());
    roles.reserve(2 + extra_inputs.size());
    tensors.push_back(embedding);
    roles.push_back(MMRdmaTensorPB::EMBEDDING);
    if (pos_id.has_value()) {
        tensors.push_back(*pos_id);
        roles.push_back(MMRdmaTensorPB::POS_ID);
    }
    for (auto& extra : extra_inputs) {
        tensors.push_back(extra);
        roles.push_back(MMRdmaTensorPB::EXTRA_INPUT);
    }

    MMRdmaDescPB desc;
    bool         ok = false;
    {
        py::gil_scoped_release release;  // D2D copy + MR lookup, no Python involved
        ok = transport_->exportEmbedding(tensors, roles, &desc);
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
        .def("export_embedding",
             &MMRdmaEncoderOp::exportEmbedding,
             py::arg("embedding"),
             py::arg("pos_id"),
             py::arg("extra_inputs"))
        .def("release", &MMRdmaEncoderOp::release, py::arg("handles"));
}

}  // namespace rtp_llm
