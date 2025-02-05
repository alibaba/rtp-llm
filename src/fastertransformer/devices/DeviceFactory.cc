#include "src/fastertransformer/devices/DeviceFactory.h"
#include "autil/EnvUtil.h"
#include <cassert>

using namespace std;
using namespace torch_ext;

namespace fastertransformer {

DeviceType getDeviceType(const std::string& device_name) {
    if (device_name == "CPU") {
        return DeviceType::Cpu;
    } else if (device_name == "CUDA") {
        return DeviceType::Cuda;
    } else if (device_name == "ROCM") {
        return DeviceType::ROCm;
    } else if (device_name == "YITIAN") {
        return DeviceType::Yitian;
    } else if (device_name == "ARM") {
        return DeviceType::ArmCpu;
    } else {
        FT_LOG_ERROR("Unknown device type: %s", device_name.c_str());
        abort();
    }
}

GlobalDeviceParams DeviceFactory::getDefaultGlobalDeviceParams() {
    GlobalDeviceParams params;
    const std::vector<DeviceType> types_to_try = {
        DeviceType::Cuda, DeviceType::Yitian, DeviceType::ArmCpu, DeviceType::Cpu, DeviceType::ROCm, DeviceType::Ppu};
    for (const auto type : types_to_try) {
        if (getRegistrationMap().find(type) != getRegistrationMap().end()) {
            FT_LOG_INFO("found device type %d, use as default.", static_cast<int>(type));
            params.device_params.push_back({type, DeviceInitParams{type}});
        } else {
            FT_LOG_INFO("Device type %d is not registered, skip.", static_cast<int>(type));
        }
    }
    if (!params.device_params.size()) {
        FT_LOG_ERROR("FATAL: No device is registered !");
        abort();
    }
    return params;
}

int64_t getDefaultDeviceReserveMemoryBytes(const GptInitParameter& params) {
    auto reserve_bytes = -256L * 1024 * 1024;
    FT_LOG_INFO("Default device reserve memory bytes: %ld", reserve_bytes);
    return reserve_bytes;
}

void DeviceFactory::initDevices(const GptInitParameter& params) {
    if (getCurrentDevices().size()) {
        FT_LOG_WARNING("Devices are already initialized! will do nothing.");
        return;
    }
    auto global_params = getDefaultGlobalDeviceParams();
    auto& device_params       = global_params.device_params[0].second;
    device_params.tp_size     = params.tp_size_;
    device_params.tp_rank     = params.tp_rank_;
    device_params.device_id   = params.local_rank_;
    device_params.master_ip   = params.nccl_ip_;
    device_params.master_port = params.nccl_port_;
    device_params.tokens_per_block = params.seq_size_per_block_;
    size_t max_batch_size =
        params.max_context_batch_size_ + params.max_generate_batch_size_ + std::max((long)0, params.gen_num_per_circle_) * 32;

    device_params.max_batch_size =
        std::max((size_t)autil::EnvUtil::getEnv("MAX_BATCH_SIZE", 0L), std::max((size_t)1024, max_batch_size * 2));  // set static max batch size to avoid sampler reset memory

    const auto device_mem_reserve_env = autil::EnvUtil::getEnv("DEVICE_RESERVE_MEMORY_BYTES", 0L);
    FT_LOG_INFO("Device reserve memory bytes from env: %ld", device_mem_reserve_env);
    device_params.device_reserve_memory_bytes = device_mem_reserve_env
                                                ? device_mem_reserve_env
                                                : getDefaultDeviceReserveMemoryBytes(params);
    FT_LOG_INFO("Device reserve memory bytes: %ld", device_params.device_reserve_memory_bytes);

    device_params.host_reserve_memory_bytes = autil::EnvUtil::getEnv("HOST_RESERVE_MEMORY_BYTES", (int64_t)(4L * 1024 * 1024 * 1024)); // 4GB
    FT_LOG_INFO("Host reserve memory bytes: %ld", device_params.host_reserve_memory_bytes);

    if (!global_params.device_params.size()) {
        FT_LOG_ERROR("No device is specified to init !");
        abort();
    }
    for (const auto& [type, device_params] : global_params.device_params) {
        auto& registrationMap = getRegistrationMap();
        auto it = registrationMap.find(type);
        if (it == registrationMap.end()) {
            FT_LOG_ERROR("Device type %d is not registered !", static_cast<int>(type));
            abort();
        }
        auto device = it->second.create(device_params);
        getCurrentDevices().push_back(device);
    }
    FT_LOG_INFO("init devices done");
}

unordered_map<DeviceType, DeviceCreatorType>& DeviceFactory::getRegistrationMap() {
    static unordered_map<DeviceType, DeviceCreatorType> registrationMap;
    return registrationMap;
}

vector<DeviceBase*>& DeviceFactory::getCurrentDevices() {
    static vector<DeviceBase *> devices;
    return devices;
}

DeviceBase* DeviceFactory::getDevice(DeviceType type, int device_id) {
    if (!getCurrentDevices().size()) {
        FT_LOG_ERROR("You must explicitly initialize devices before getting device !");
        abort();
    }
    for (const auto device: getCurrentDevices()) {
        const auto& props = device->getDeviceProperties();
        if (props.type == type && int(props.id) == device_id) {
            return device;
        }
    }
    FT_LOG_ERROR("Device type %d with id %d is not found !", static_cast<int>(type), device_id);
    abort();
}

DeviceBase* DeviceFactory::getDefaultDevice() {
    if (!getCurrentDevices().size()) {
        FT_LOG_ERROR("You must explicitly initialize devices before getting device !");
        abort();
    }
    return getCurrentDevices()[0];
}

std::shared_ptr<DeviceExporter> DeviceFactory::getDeviceExporter() {
    const auto params = getDefaultGlobalDeviceParams();
    const auto registration = getRegistrationMap()[params.device_params[0].first];
    const auto exporter = registration.createExporter(params.device_params[0].second);
    return std::shared_ptr<DeviceExporter>(exporter);
}

void DeviceFactory::registerDevice(DeviceType type, DeviceCreatorType creator) {
    auto& registrationMap = getRegistrationMap();
    FT_CHECK_WITH_INFO((registrationMap.find(type) == registrationMap.end()),
        "Can not find device: %d", type);
    registrationMap[type] = creator;
}

void registerDeviceOps(py::module& m) {
    pybind11::class_<DeviceExporter, std::shared_ptr<DeviceExporter>>(m, "DeviceExporter")
        .def("get_device_type", &DeviceExporter::getDeviceType)
        .def("get_device_id", &DeviceExporter::getDeviceId)
        .def("preprocess_gemm_weight_by_key", &DeviceExporter::preprocessGemmWeightByKey)
        .def("pack_int8_tensor_to_packed_int4", &DeviceExporter::packInt8TensorToPackedInt4)
        .def("preprocess_weights_for_mixed_gemm", &DeviceExporter::preprocessWeightsForMixedGemm)
        .def("symmetric_quantize_last_axis_of_batched_matrix", &DeviceExporter::symmetricQuantizeLastAxisOfBatchedMatrix);

    pybind11::enum_<DeviceType>(m, "DeviceType")
        .value("Cpu", DeviceType::Cpu)
        .value("Cuda", DeviceType::Cuda)
        .value("Yitian", DeviceType::Yitian)
        .value("ArmCpu", DeviceType::ArmCpu)
        .value("ROCm", DeviceType::ROCm)
        .value("Ppu", DeviceType::Ppu);

    m.def("get_device", &DeviceFactory::getDeviceExporter);
}

} // namespace fastertransformer

