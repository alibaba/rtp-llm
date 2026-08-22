#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaEncoderOp.h"

#include <algorithm>
#include <limits>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/config/MMTransportConfigExtract.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {
inline uint64_t alignUp(uint64_t x, uint64_t a) {
    return (x + a - 1) / a * a;
}
inline uint64_t tensorBytes(const torch::Tensor& t) {
    return static_cast<uint64_t>(t.numel()) * t.element_size();
}
inline uint64_t slotFootprint(const torch::Tensor& t) {
    return alignUp(tensorBytes(t), kMMRdmaSlotAlign);
}
MMRdmaConfig rdmaConfigFromPython(const py::object& rdma_config) {
    return extractMMRdmaConfig(rdma_config);
}
}  // namespace

MMRdmaEncoderOp::MMRdmaEncoderOp(const py::object& rdma_config):
    MMRdmaEncoderOp(rdmaConfigFromPython(rdma_config)) {}

MMRdmaEncoderOp::MMRdmaEncoderOp(const MMRdmaConfig& rdma_config) {
    transport_      = createMMRdmaTransport(rdma_config, MMRdmaRole::ENCODER_SERVER);
    max_slot_bytes_ = rdma_config.max_slot_bytes;
}

bool MMRdmaEncoderOp::exportSlots(const torch::Tensor&                embedding,
                                  const std::optional<torch::Tensor>& pos_id,
                                  const std::vector<torch::Tensor>&   extra_inputs,
                                  std::vector<MMRdmaDescPB>*          descs) {
    if (transport_ == nullptr) {
        return false;
    }

    const uint64_t max_slot =
        max_slot_bytes_ > 0 ? static_cast<uint64_t>(max_slot_bytes_) : std::numeric_limits<uint64_t>::max();
    // The packer aligns every tensor start, so chunk against the aligned-down slot limit.
    const uint64_t max_slot_aligned = max_slot / kMMRdmaSlotAlign * kMMRdmaSlotAlign;

    // Manifest order is embedding chunks, optional position ids, then extra inputs.
    std::vector<torch::Tensor>        tensors;
    std::vector<MMRdmaTensorPB::Role> roles;

    if (slotFootprint(embedding) <= max_slot) {
        tensors.push_back(embedding);
        roles.push_back(MMRdmaTensorPB::EMBEDDING);
    } else {
        const int64_t rows = embedding.dim() >= 1 ? embedding.size(0) : 0;
        if (rows <= 0) {
            RTP_LLM_LOG_WARNING("mm rdma chunk: embedding not row-splittable (dim=%ld), fall back to bytes",
                                static_cast<long>(embedding.dim()));
            return false;
        }
        const uint64_t row_bytes = tensorBytes(embedding) / static_cast<uint64_t>(rows);
        if (row_bytes == 0 || alignUp(row_bytes, kMMRdmaSlotAlign) > max_slot) {
            RTP_LLM_LOG_WARNING("mm rdma chunk: single embedding row (%lu B) exceeds max_slot (%lu), "
                                "fall back to bytes",
                                row_bytes,
                                max_slot);
            return false;
        }
        int64_t rows_per_chunk = static_cast<int64_t>(max_slot_aligned / std::max<uint64_t>(row_bytes, 1));
        if (rows_per_chunk < 1) {
            rows_per_chunk = 1;
        }
        for (int64_t start = 0; start < rows; start += rows_per_chunk) {
            const int64_t len = std::min<int64_t>(rows_per_chunk, rows - start);
            tensors.push_back(embedding.narrow(0, start, len));
            roles.push_back(MMRdmaTensorPB::EMBEDDING);
        }
    }
    if (pos_id.has_value()) {
        if (slotFootprint(*pos_id) > max_slot) {
            RTP_LLM_LOG_WARNING("mm rdma chunk: pos_id (%lu B) exceeds max_slot (%lu), fall back to bytes",
                                tensorBytes(*pos_id),
                                max_slot);
            return false;
        }
        tensors.push_back(*pos_id);
        roles.push_back(MMRdmaTensorPB::POS_ID);
    }
    for (const auto& extra : extra_inputs) {
        if (slotFootprint(extra) > max_slot) {
            RTP_LLM_LOG_WARNING("mm rdma chunk: extra_input (%lu B) exceeds max_slot (%lu), fall back to bytes",
                                tensorBytes(extra),
                                max_slot);
            return false;
        }
        tensors.push_back(extra);
        roles.push_back(MMRdmaTensorPB::EXTRA_INPUT);
    }

    // Greedily pack tensors while preserving manifest order.
    size_t i = 0;
    while (i < tensors.size()) {
        std::vector<torch::Tensor>        grp_tensors;
        std::vector<MMRdmaTensorPB::Role> grp_roles;
        uint64_t                          grp_bytes = 0;
        while (i < tensors.size()) {
            const uint64_t fp = slotFootprint(tensors[i]);
            if (!grp_tensors.empty() && grp_bytes + fp > max_slot) {
                break;
            }
            grp_tensors.push_back(tensors[i]);
            grp_roles.push_back(roles[i]);
            grp_bytes += fp;
            ++i;
        }
        MMRdmaDescPB desc;
        if (!transport_->exportEmbedding(grp_tensors, grp_roles, &desc)) {
            // Roll back slots exported before this failure.
            std::vector<std::string> handles;
            handles.reserve(descs->size());
            for (const auto& d : *descs) {
                handles.push_back(d.handle());
            }
            if (!handles.empty()) {
                transport_->releaseEmbedding(handles);
            }
            descs->clear();
            return false;
        }
        descs->push_back(std::move(desc));
    }
    return !descs->empty();
}

std::vector<py::bytes> MMRdmaEncoderOp::exportEmbedding(torch::Tensor                embedding,
                                                        std::optional<torch::Tensor> pos_id,
                                                        std::vector<torch::Tensor>   extra_inputs) {
    std::vector<MMRdmaDescPB> descs;
    bool                      exported = false;
    {
        // D2D copies and memory registration do not need the GIL.
        py::gil_scoped_release release;
        exported = exportSlots(embedding, pos_id, extra_inputs, &descs);
    }

    std::vector<py::bytes> out;
    if (!exported) {
        return out;
    }
    out.reserve(descs.size());
    for (const auto& d : descs) {
        out.push_back(py::bytes(d.SerializeAsString()));
    }
    return out;
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
        .def(py::init<const py::object&>(), py::arg("rdma_config"))
        .def("enabled", &MMRdmaEncoderOp::enabled)
        .def("export_embedding",
             &MMRdmaEncoderOp::exportEmbedding,
             py::arg("embedding"),
             py::arg("pos_id"),
             py::arg("extra_inputs"))
        .def("release", &MMRdmaEncoderOp::release, py::arg("handles"));
}

}  // namespace rtp_llm
