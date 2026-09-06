#include "rtp_llm/cpp/multimodal_processor/transport/kvcm/MMKvcmWriter.h"

#include <limits>
#include <stdexcept>
#include <unordered_set>

#include <pybind11/stl.h>

#include "rtp_llm/cpp/pybind/ConfigExtract.h"

namespace rtp_llm {
namespace {

std::vector<MMKvcmBuffer> makeBuffers(const std::vector<std::string>&   keys,
                                      const std::vector<torch::Tensor>& tensors,
                                      uint64_t                          max_object_bytes,
                                      uint64_t                          max_receipt_bytes) {
    if (keys.empty() || keys.size() > kMMKvcmMaxObjectsPerReceipt || keys.size() != tensors.size()) {
        throw std::invalid_argument("KVCM keys and tensors must be non-empty and aligned");
    }
    std::unordered_set<std::string> unique_keys;
    unique_keys.reserve(keys.size());
    std::vector<MMKvcmBuffer> objects;
    objects.reserve(keys.size());
    uint64_t total_bytes = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        const auto& tensor = tensors[i];
        if (keys[i].empty() || keys[i].size() > kMMKvcmMaxKeyBytes || !unique_keys.insert(keys[i]).second) {
            throw std::invalid_argument("KVCM keys must contain 1 to 512 bytes and be unique");
        }
        if (!tensor.defined() || !tensor.is_contiguous() || tensor.numel() <= 0) {
            throw std::invalid_argument("KVCM tensors must be defined, non-empty and contiguous");
        }
        if (!tensor.device().is_cpu() && !tensor.device().is_cuda()) {
            throw std::invalid_argument("KVCM tensors must use CPU or CUDA storage");
        }
        const auto dtype = tensor.scalar_type();
        if (dtype != torch::kFloat32 && dtype != torch::kInt32 && dtype != torch::kFloat16
            && dtype != torch::kBFloat16) {
            throw std::invalid_argument("KVCM tensor dtype is not representable in the multimodal receipt");
        }
        const uint64_t numel = static_cast<uint64_t>(tensor.numel());
        const uint64_t width = static_cast<uint64_t>(tensor.element_size());
        if (width == 0 || numel > std::numeric_limits<uint64_t>::max() / width) {
            throw std::invalid_argument("KVCM tensor byte size overflows");
        }
        const uint64_t nbytes = numel * width;
        if (nbytes == 0 || nbytes > max_object_bytes
            || nbytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::invalid_argument("KVCM tensor exceeds max_object_bytes");
        }
        if (nbytes > max_receipt_bytes - total_bytes) {
            throw std::invalid_argument("KVCM tensors exceed max_receipt_bytes");
        }
        total_bytes += nbytes;
        objects.push_back({keys[i], tensor.data_ptr(), nbytes, tensor.is_cuda()});
    }
    return objects;
}

}  // namespace

MMKvcmWriter::MMKvcmWriter(const py::object& kvcm_config): MMKvcmWriter(extractMMKvcmConfig(kvcm_config)) {}

MMKvcmWriter::MMKvcmWriter(const MMKvcmConfig& kvcm_config):
    client_(createMMKvcmClient(kvcm_config)),
    max_object_bytes_(kvcm_config.max_object_bytes > 0 ? static_cast<uint64_t>(kvcm_config.max_object_bytes) : 0),
    max_receipt_bytes_(kvcm_config.max_receipt_bytes > 0 ? static_cast<uint64_t>(kvcm_config.max_receipt_bytes) : 0) {
    const auto config_error = validateMMKvcmConfig(kvcm_config);
    if (!config_error.empty()) {
        throw std::invalid_argument(config_error);
    }
    if (client_ == nullptr) {
        if (!hasMMKvcmImplementation()) {
            throw std::runtime_error("KVCM EMB storage is not linked; rebuild with --define=use_kvcm_emb_storage=true");
        }
        throw std::runtime_error("failed to initialize KVCM EMB object client");
    }
}

void MMKvcmWriter::save(const std::vector<std::string>& keys, const std::vector<torch::Tensor>& tensors) {
    auto        objects = makeBuffers(keys, tensors, max_object_bytes_, max_receipt_bytes_);
    std::string error;
    {
        py::gil_scoped_release release;
        error = client_->save("rtp-mm-kvcm-save", objects);
    }
    if (!error.empty()) {
        throw std::runtime_error(error);
    }
}

void MMKvcmWriter::remove(const std::vector<std::string>& keys) {
    if (keys.empty()) {
        return;
    }
    std::string error;
    {
        py::gil_scoped_release release;
        error = client_->remove("rtp-mm-kvcm-remove", keys);
    }
    if (!error.empty()) {
        throw std::runtime_error(error);
    }
}

void registerMMKvcmWriter(py::module& module) {
    py::class_<MMKvcmWriter, std::shared_ptr<MMKvcmWriter>>(module, "MMKvcmWriter")
        .def(py::init<const py::object&>(), py::arg("kvcm_config"))
        .def_static("available", &hasMMKvcmImplementation)
        .def("enabled", &MMKvcmWriter::enabled)
        .def("save", &MMKvcmWriter::save, py::arg("keys"), py::arg("tensors"))
        .def("remove", &MMKvcmWriter::remove, py::arg("keys"));
}

}  // namespace rtp_llm
