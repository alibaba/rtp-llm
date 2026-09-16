#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaExporter.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "rtp_llm/cpp/pybind/ConfigExtract.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

uint64_t alignUp(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

uint64_t tensorBytes(const torch::Tensor& tensor) {
    return static_cast<uint64_t>(tensor.numel()) * tensor.element_size();
}

uint64_t slotFootprint(const torch::Tensor& tensor) {
    return alignUp(tensorBytes(tensor), rdma_transport::kRdmaSlotAlign);
}

}  // namespace

MMRdmaExporter::MMRdmaExporter(const py::object& rdma_config):
    MMRdmaExporter(extractRdmaConfig(rdma_config), -1) {}

MMRdmaExporter::MMRdmaExporter(const py::object& rdma_config, int device_id):
    MMRdmaExporter(extractRdmaConfig(rdma_config), device_id) {}

MMRdmaExporter::MMRdmaExporter(const RdmaConfig& rdma_config) {
    exporter_       = rdma_transport::createRdmaExport(rdma_config, -1);
    max_slot_bytes_ = rdma_config.max_slot_bytes;
}

MMRdmaExporter::MMRdmaExporter(const RdmaConfig& rdma_config, int device_id) {
    exporter_       = rdma_transport::createRdmaExport(rdma_config, device_id);
    max_slot_bytes_ = rdma_config.max_slot_bytes;
}

bool MMRdmaExporter::exportSlots(const torch::Tensor&                  embedding,
                                 const std::optional<torch::Tensor>&   pos_id,
                                 const std::vector<torch::Tensor>&     extra_inputs,
                                 std::vector<MMRdmaSlotPB>*            slots) {
    if (exporter_ == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> provider_lock(provider_mutex_);

    const uint64_t max_slot = max_slot_bytes_ > 0 ? static_cast<uint64_t>(max_slot_bytes_)
                                                   : std::numeric_limits<uint64_t>::max();
    const uint64_t max_slot_aligned =
        max_slot / rdma_transport::kRdmaSlotAlign * rdma_transport::kRdmaSlotAlign;

    std::vector<torch::Tensor>      tensors;
    std::vector<MMRdmaSlotPB::Role> roles;
    if (slotFootprint(embedding) <= max_slot) {
        tensors.push_back(embedding);
        roles.push_back(MMRdmaSlotPB::EMBEDDING);
    } else {
        const int64_t rows = embedding.dim() >= 1 ? embedding.size(0) : 0;
        if (rows <= 0) {
            RTP_LLM_LOG_WARNING("mm rdma chunk: embedding not row-splittable (dim=%ld)",
                                static_cast<long>(embedding.dim()));
            return false;
        }
        const uint64_t row_bytes = tensorBytes(embedding) / static_cast<uint64_t>(rows);
        if (row_bytes == 0 || alignUp(row_bytes, rdma_transport::kRdmaSlotAlign) > max_slot) {
            RTP_LLM_LOG_WARNING("mm rdma chunk: single embedding row (%lu B) exceeds max_slot (%lu)",
                                row_bytes,
                                max_slot);
            return false;
        }
        int64_t rows_per_chunk =
            static_cast<int64_t>(max_slot_aligned / std::max<uint64_t>(row_bytes, 1));
        rows_per_chunk         = std::max<int64_t>(rows_per_chunk, 1);
        for (int64_t start = 0; start < rows; start += rows_per_chunk) {
            const int64_t len = std::min<int64_t>(rows_per_chunk, rows - start);
            tensors.push_back(embedding.narrow(0, start, len));
            roles.push_back(MMRdmaSlotPB::EMBEDDING);
        }
    }

    if (pos_id.has_value()) {
        if (slotFootprint(*pos_id) > max_slot) {
            RTP_LLM_LOG_WARNING("mm rdma chunk: pos_id (%lu B) exceeds max_slot (%lu)",
                                tensorBytes(*pos_id),
                                max_slot);
            return false;
        }
        tensors.push_back(*pos_id);
        roles.push_back(MMRdmaSlotPB::POS_ID);
    }
    for (const auto& extra : extra_inputs) {
        if (slotFootprint(extra) > max_slot) {
            RTP_LLM_LOG_WARNING("mm rdma chunk: extra_input (%lu B) exceeds max_slot (%lu)",
                                tensorBytes(extra),
                                max_slot);
            return false;
        }
        tensors.push_back(extra);
        roles.push_back(MMRdmaSlotPB::EXTRA_INPUT);
    }

    std::vector<std::vector<torch::Tensor>>      groups;
    std::vector<std::vector<MMRdmaSlotPB::Role>> group_roles;
    size_t                                       index = 0;
    while (index < tensors.size()) {
        std::vector<torch::Tensor>      group;
        std::vector<MMRdmaSlotPB::Role> group_role;
        uint64_t                        group_bytes = 0;
        while (index < tensors.size()) {
            const uint64_t footprint = slotFootprint(tensors[index]);
            if (!group.empty() && group_bytes + footprint > max_slot) {
                break;
            }
            group.push_back(tensors[index]);
            group_role.push_back(roles[index]);
            group_bytes += footprint;
            ++index;
        }
        groups.push_back(std::move(group));
        group_roles.push_back(std::move(group_role));
    }

    const auto descriptors = exporter_->createBatch(groups);
    if (descriptors.size() != groups.size()) {
        return false;
    }
    for (size_t group = 0; group < descriptors.size(); ++group) {
        MMRdmaSlotPB slot;
        rdma_transport::toProto(descriptors[group], slot.mutable_rdma_descriptor());
        for (const auto role : group_roles[group]) {
            slot.add_roles(role);
        }
        slots->push_back(std::move(slot));
    }
    return !slots->empty();
}

std::vector<py::bytes> MMRdmaExporter::exportEmbedding(torch::Tensor                  embedding,
                                                        std::optional<torch::Tensor>   pos_id,
                                                        std::vector<torch::Tensor>     extra_inputs) {
    std::vector<MMRdmaSlotPB> slots;
    bool                      exported = false;
    {
        py::gil_scoped_release release;
        exported = exportSlots(embedding, pos_id, extra_inputs, &slots);
    }

    std::vector<py::bytes> output;
    if (!exported) {
        return output;
    }
    output.reserve(slots.size());
    for (const auto& slot : slots) {
        output.push_back(py::bytes(slot.SerializeAsString()));
    }
    return output;
}

void MMRdmaExporter::release(const std::vector<std::string>& handles) {
    if (exporter_ == nullptr) {
        return;
    }
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> provider_lock(provider_mutex_);
    exporter_->release(handles);
}

void registerMMRdmaExporter(py::module& module) {
    py::class_<MMRdmaExporter, std::shared_ptr<MMRdmaExporter>>(module, "MMRdmaExporter")
        .def(py::init<const py::object&>(), py::arg("rdma_config"))
        .def(py::init<const py::object&, int>(), py::arg("rdma_config"), py::arg("device_id"))
        .def_static("available", &rdma_transport::hasRdmaImplementation)
        .def("enabled", &MMRdmaExporter::enabled)
        .def("export_embedding",
             &MMRdmaExporter::exportEmbedding,
             py::arg("embedding"),
             py::arg("pos_id"),
             py::arg("extra_inputs"))
        .def("release", &MMRdmaExporter::release, py::arg("handles"));
}

}  // namespace rtp_llm
