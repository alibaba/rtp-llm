#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "autil/EnvUtil.h"
#include <cassert>

using namespace std;
using namespace torch_ext;

namespace rtp_llm {

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
        RTP_LLM_LOG_ERROR("Unknown device type: %s", device_name.c_str());
        abort();
    }
}

GlobalDeviceParams DeviceFactory::getDefaultGlobalDeviceParams() {
    GlobalDeviceParams params;
    const std::vector<DeviceType> types_to_try = {
        DeviceType::Cuda, DeviceType::Yitian, DeviceType::ArmCpu, DeviceType::Cpu, DeviceType::ROCm, DeviceType::Ppu};
    for (const auto type : types_to_try) {
        if (getRegistrationMap().find(type) != getRegistrationMap().end()) {
            RTP_LLM_LOG_INFO("found device type %d, use as default.", static_cast<int>(type));
            params.device_params.push_back({type, DeviceInitParams{type}});
        } else {
            RTP_LLM_LOG_INFO("Device type %d is not registered, skip.", static_cast<int>(type));
        }
    }
    if (!params.device_params.size()) {
        RTP_LLM_LOG_ERROR("FATAL: No device is registered !");
        abort();
    }
    return params;
}

int64_t getDefaultDeviceReserveMemoryBytes(const GptInitParameter& params) {
    auto reserve_bytes = -512L * 1024 * 1024;
    RTP_LLM_LOG_INFO("Default device reserve memory bytes: %ld", reserve_bytes);
    return reserve_bytes;
}

void DeviceFactory::initDevices(const GptInitParameter& params) {
    if (getCurrentDevices().size()) {
        RTP_LLM_LOG_WARNING("Devices are already initialized! will do nothing.");
        return;
    }
    auto  global_params             = getDefaultGlobalDeviceParams();
    auto& device_params             = global_params.device_params[0].second;
    device_params.tp_size           = params.tp_size_;
    device_params.dp_size           = params.dp_size_;
    device_params.ep_size           = params.ep_size_;
    device_params.ep_rank           = params.ep_rank_;
    device_params.tp_rank           = params.tp_rank_;
    device_params.dp_rank           = params.dp_rank_;
    device_params.ffn_tp_size       = params.ffn_tp_size_;
    device_params.ffn_tp_rank       = params.ffn_tp_rank_;
    device_params.enable_sp         = params.enable_sp_;
    device_params.device_id         = params.local_rank_;
    device_params.master_ip         = params.nccl_ip_;
    device_params.tp_master_port    = params.tp_nccl_port_;
    device_params.dp_tp_master_port = params.dp_tp_nccl_port_;
    device_params.ffn_tp_master_port = params.ffn_tp_nccl_port_;
    device_params.tokens_per_block  = params.seq_size_per_block_;
    device_params.mla_ops_type      = params.mla_ops_type_;
    device_params.max_seq_len       = params.max_seq_len_;
    device_params.hidden_size       = params.hidden_size_;
    device_params.num_experts       = params.expert_num_;
    device_params.extra_experts     = params.phy_exp_num_ - params.expert_num_;

    size_t max_batch_size           = params.max_context_batch_size_ + params.max_generate_batch_size_
                            + std::max((long)0, params.gen_num_per_circle_) * 32;

    device_params.overlap_math_sm_count = autil::EnvUtil::getEnv("OVERLAP_MATH_SM_COUNT", 0UL);
    device_params.overlap_comm_type = autil::EnvUtil::getEnv("OVERLAP_COMM_TYPE", 0UL);
    device_params.max_seq_len       = params.max_seq_len_;
    RTP_LLM_LOG_INFO("set overlap type to be %d", device_params.overlap_comm_type);
    device_params.m_split = autil::EnvUtil::getEnv("M_SPLIT", 0UL);
    device_params.max_generate_batch_size = params.max_generate_batch_size_;
    device_params.max_batch_size =
        std::max((size_t)autil::EnvUtil::getEnv("MAX_BATCH_SIZE", 0L), std::max((size_t)1024, max_batch_size * 2));  // set static max batch size to avoid sampler reset memory

    const auto device_mem_reserve_env = autil::EnvUtil::getEnv("DEVICE_RESERVE_MEMORY_BYTES", 0L);
    RTP_LLM_LOG_INFO("Device reserve memory bytes from env: %ld", device_mem_reserve_env);
    device_params.device_reserve_memory_bytes = device_mem_reserve_env
                                                ? device_mem_reserve_env
                                                : getDefaultDeviceReserveMemoryBytes(params);
    RTP_LLM_LOG_INFO("Device reserve memory bytes: %ld", device_params.device_reserve_memory_bytes);

    device_params.host_reserve_memory_bytes = autil::EnvUtil::getEnv("HOST_RESERVE_MEMORY_BYTES", (int64_t)(4L * 1024 * 1024 * 1024)); // 4GB
    RTP_LLM_LOG_INFO("Host reserve memory bytes: %ld", device_params.host_reserve_memory_bytes);

    device_params.enable_comm_overlap = autil::EnvUtil::getEnv("ENABLE_COMM_OVERLAP", 1L);
    device_params.enable_layer_micro_batch = static_cast<MicroBatchType>(autil::EnvUtil::getEnv("ENABLE_LAYER_MICRO_BATCH", 0));

    RTP_LLM_LOG_INFO("enable comm overlap: %d, enable layer micro batch: %d",
                device_params.enable_comm_overlap, device_params.enable_layer_micro_batch);

    device_params.use_deepep_moe = autil::EnvUtil::getEnv("USE_DEEPEP_MOE", 0L);
    device_params.use_deepep_internode = autil::EnvUtil::getEnv("USE_DEEPEP_INTERNODE", 0L);
    device_params.use_deepep_low_latency = autil::EnvUtil::getEnv("USE_DEEPEP_LOW_LATENCY", 1L);
    auto sp_type = autil::EnvUtil::getEnv("SP_TYPE", "");
    auto sp_model_type = autil::EnvUtil::getEnv("SP_MODEL_TYPE", "");
    RTP_LLM_LOG_INFO("device_params sp_type is %s", sp_type.c_str());
    RTP_LLM_LOG_INFO("device_params sp_model_type is %s", sp_model_type.c_str());
    if (((sp_type == "vanilla") && (sp_model_type == "mixtbstars-mtp"))      ||
        ((sp_type == "vanilla") && (sp_model_type == "deepseek-v3-mtp"))    ||
        (sp_type == "mtp"))
    {
        device_params.is_mtp = true;
        RTP_LLM_LOG_INFO("device_params.is_mtp true");
    }


    RTP_LLM_LOG_INFO("use deepep moe: %d, use deepep low latency: %d",
                device_params.use_deepep_moe, device_params.use_deepep_low_latency);

    if (!global_params.device_params.size()) {
        RTP_LLM_LOG_ERROR("No device is specified to init !");
        abort();
    }
    for (const auto& [type, device_params] : global_params.device_params) {
        auto& registrationMap = getRegistrationMap();
        auto it = registrationMap.find(type);
        if (it == registrationMap.end()) {
            RTP_LLM_LOG_ERROR("Device type %d is not registered !", static_cast<int>(type));
            abort();
        }
        auto device = it->second.create(device_params);
        getCurrentDevices().push_back(device);
    }
    RTP_LLM_LOG_INFO("init devices done");
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
        RTP_LLM_LOG_ERROR("You must explicitly initialize devices before getting device !");
        abort();
    }
    for (const auto device: getCurrentDevices()) {
        const auto& props = device->getDeviceProperties();
        if (props.type == type && int(props.id) == device_id) {
            return device;
        }
    }
    RTP_LLM_LOG_ERROR("Device type %d with id %d is not found !", static_cast<int>(type), device_id);
    abort();
}

DeviceBase* DeviceFactory::getDefaultDevice() {
    if (!getCurrentDevices().size()) {
        RTP_LLM_LOG_ERROR("You must explicitly initialize devices before getting device !");
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
    RTP_LLM_CHECK_WITH_INFO((registrationMap.find(type) == registrationMap.end()),
        "Can not find device: %d", type);
    registrationMap[type] = creator;
}

void registerDeviceOps(py::module& m) {
    pybind11::enum_<DeviceType>(m, "DeviceType")
        .value("Cpu", DeviceType::Cpu)
        .value("Cuda", DeviceType::Cuda)
        .value("Yitian", DeviceType::Yitian)
        .value("ArmCpu", DeviceType::ArmCpu)
        .value("ROCm", DeviceType::ROCm)
        .value("Ppu", DeviceType::Ppu);

    pybind11::class_<DeviceExporter, std::shared_ptr<DeviceExporter>>(m, "DeviceExporter")
        .def("get_device_type", &DeviceExporter::getDeviceType)
        .def("get_device_id", &DeviceExporter::getDeviceId)
        .def("preprocess_gemm_weight_by_key",
             &DeviceExporter::preprocessGemmWeightByKey,
             py::arg("key"),
             py::arg("weight"))
        .def("pack_int8_tensor_to_packed_int4", &DeviceExporter::packInt8TensorToPackedInt4, py::arg("weight"))
        .def("preprocess_weights_for_mixed_gemm",
             &DeviceExporter::preprocessWeightsForMixedGemm,
             py::arg("weight"),
             py::arg("quant_type"),
             py::arg("arch"))
        .def("symmetric_quantize_last_axis_of_batched_matrix",
             &DeviceExporter::symmetricQuantizeLastAxisOfBatchedMatrix,
             py::arg("weight"),
             py::arg("quant_type"),
             py::arg("arch"))
        .def("preprocess_weight_scale", &DeviceExporter::preprocessWeightScale, py::arg("weight"), py::arg("scale"));

    m.def("get_device", &DeviceFactory::getDeviceExporter);
}

} // namespace rtp_llm
