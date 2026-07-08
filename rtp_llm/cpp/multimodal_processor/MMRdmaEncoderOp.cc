#include "rtp_llm/cpp/multimodal_processor/MMRdmaEncoderOp.h"

#include <algorithm>
#include <limits>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/multimodal_processor/MMRdmaVitConfig.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {
// Each packed tensor starts on a kSlotAlign boundary inside its slot (mirrors the transport's
// packer), so account for that padding when estimating whether a tensor group fits one slot.
constexpr uint64_t kSlotAlign = 256;
inline uint64_t alignUp(uint64_t x, uint64_t a) {
    return (x + a - 1) / a * a;
}
inline uint64_t tensorBytes(const torch::Tensor& t) {
    return static_cast<uint64_t>(t.numel()) * t.element_size();
}
// A tensor's footprint inside a slot, including alignment padding.
inline uint64_t slotFootprint(const torch::Tensor& t) {
    return alignUp(tensorBytes(t), kSlotAlign);
}
}  // namespace

MMRdmaEncoderOp::MMRdmaEncoderOp(const py::object& vit_config) {
    VitConfig cfg;
    extractMMRdmaVitConfig(vit_config, cfg);
    transport_      = createMMRdmaTransport(cfg, MMRdmaRole::ENCODER_SERVER);
    max_slot_bytes_ = cfg.mm_rdma_max_slot_bytes;
}

std::vector<py::bytes> MMRdmaEncoderOp::exportEmbedding(torch::Tensor                embedding,
                                                       std::optional<torch::Tensor> pos_id,
                                                       std::vector<torch::Tensor>   extra_inputs) {
    std::vector<py::bytes> out;
    if (transport_ == nullptr) {
        return out;  // empty => caller falls back to inline bytes
    }

    const uint64_t max_slot =
        max_slot_bytes_ > 0 ? static_cast<uint64_t>(max_slot_bytes_) : std::numeric_limits<uint64_t>::max();

    // Build the ordered (tensor, role) list. Pack order must stay in lockstep with the manifest
    // the LLM slices by: embedding first, then the optional pos_id, then each extra_input in its
    // original (per-image) order. The embedding may exceed one slot, so row-split it (along dim 0)
    // into pieces that each fit; the LLM concatenates the EMBEDDING pieces back in order. pos_id
    // and each extra_input are per-image and expected to fit in one slot.
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
            return {};
        }
        const uint64_t row_bytes = tensorBytes(embedding) / static_cast<uint64_t>(rows);
        if (row_bytes == 0 || alignUp(row_bytes, kSlotAlign) > max_slot) {
            RTP_LLM_LOG_WARNING("mm rdma chunk: single embedding row (%lu B) exceeds max_slot (%lu), "
                                "fall back to bytes",
                                row_bytes,
                                max_slot);
            return {};
        }
        int64_t rows_per_chunk = static_cast<int64_t>(max_slot / std::max<uint64_t>(row_bytes, 1));
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
            return {};
        }
        tensors.push_back(*pos_id);
        roles.push_back(MMRdmaTensorPB::POS_ID);
    }
    for (auto& extra : extra_inputs) {
        if (slotFootprint(extra) > max_slot) {
            RTP_LLM_LOG_WARNING("mm rdma chunk: extra_input (%lu B) exceeds max_slot (%lu), fall back to bytes",
                                tensorBytes(extra),
                                max_slot);
            return {};
        }
        tensors.push_back(extra);
        roles.push_back(MMRdmaTensorPB::EXTRA_INPUT);
    }

    // Greedy-pack the ordered list into groups whose aligned footprint fits one slot; each group
    // becomes one RDMA slot + descriptor. Order is preserved so EMBEDDING/EXTRA_INPUT stay in
    // sequence across descriptors.
    std::vector<MMRdmaDescPB> descs;
    {
        py::gil_scoped_release release;  // narrow views are cheap; D2D copy + MR lookup need no GIL
        size_t                 i = 0;
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
                // A slot failed mid-way: release everything already registered and signal the
                // caller to fall back to the inline-bytes path.
                std::vector<std::string> handles;
                handles.reserve(descs.size());
                for (const auto& d : descs) {
                    handles.push_back(d.handle());
                }
                if (!handles.empty()) {
                    transport_->releaseEmbedding(handles);
                }
                descs.clear();
                break;
            }
            descs.push_back(std::move(desc));
        }
    }
    if (descs.empty()) {
        return {};
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
