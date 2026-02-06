#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "autil/EnvUtil.h"
#include <cassert>
#include <cstdlib>

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
    GlobalDeviceParams            params;
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

int64_t getDefaultDeviceReserveMemoryBytes() {
    auto reserve_bytes = -512L * 1024 * 1024;
    RTP_LLM_LOG_INFO("Default device reserve memory bytes: %ld", reserve_bytes);
    return reserve_bytes;
}

bool DeviceFactory::isAlreadyInit() {
    if (getCurrentDevices().size()) {
        return true;
    } else {
        return false;
    }
}

void DeviceFactory::initDevices(const ParallelismConfig&           parallelism_config,
                                const ModelConfig&                 model_config,
                                const EPLBConfig&                  eplb_config,
                                const FMHAConfig&                  fmha_config,
                                const DeviceResourceConfig&        device_resource_config,
                                const MoeConfig&                   moe_config,
                                const SpeculativeExecutionConfig&  sp_config,
                                const MiscellaneousConfig&         misc_config,
                                const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                                const HWKernelConfig&              hw_kernel_config,
                                const ConcurrencyConfig&           concurrency_config,
                                const FfnDisAggregateConfig&       ffn_disaggregate_config,
                                const RuntimeConfig&               runtime_config,
                                const ModelSpecificConfig&         model_specific_config,
                                const NcclCommConfig&              nccl_comm_config) {
    if (getCurrentDevices().size()) {
        RTP_LLM_LOG_WARNING("Devices are already initialized! will do nothing.");
        return;
    }
    auto  global_params = getDefaultGlobalDeviceParams();
    auto& device_params = global_params.device_params[0].second;

    device_params.tp_size     = parallelism_config.tp_size;
    device_params.dp_size     = parallelism_config.dp_size;
    device_params.ep_size     = parallelism_config.ep_size;
    device_params.ep_rank     = parallelism_config.ep_rank;
    device_params.tp_rank     = parallelism_config.tp_rank;
    device_params.dp_rank     = parallelism_config.dp_rank;
    device_params.ffn_tp_size = parallelism_config.ffn_tp_size;
    device_params.ffn_tp_rank = parallelism_config.ffn_tp_rank;
    device_params.enable_sp   = parallelism_config.enable_sp;
    // use_all_gather is now in moe_config, but we need to ensure it's not used
    // when use_deepep_low_latency is True
    device_params.use_all_gather = moe_config.use_all_gather && !moe_config.use_deepep_low_latency;
    // local_rank is calculated from parallelism_config
    device_params.device_id              = parallelism_config.world_rank % parallelism_config.local_world_size;
    device_params.master_ip              = nccl_comm_config.master_ip;
    device_params.tp_master_port         = nccl_comm_config.tp_port;
    device_params.dp_tp_master_port      = nccl_comm_config.dp_tp_port;
    device_params.ffn_tp_master_port     = nccl_comm_config.ffn_tp_port;
    device_params.tokens_per_block       = model_config.attn_config.tokens_per_block;
    device_params.mla_ops_type           = model_config.mla_ops_type;
    device_params.max_seq_len            = model_config.max_seq_len;
    device_params.hidden_size            = model_config.hidden_size;
    device_params.num_experts            = model_config.expert_num;
    device_params.extra_experts          = eplb_config.phy_exp_num(model_config.expert_num) - model_config.expert_num;
    device_params.fmha_config            = fmha_config;
    device_params.device_resource_config = device_resource_config;
    device_params.moe_config             = moe_config;
    device_params.sp_config              = sp_config;
    // FIFOSchedulerConfig fields are now in RuntimeConfig
    device_params.runtime_config                      = runtime_config;
    device_params.misc_config                         = misc_config;
    device_params.parallelism_config.tp_size          = parallelism_config.tp_size;
    device_params.parallelism_config.ep_size          = parallelism_config.ep_size;
    device_params.parallelism_config.dp_size          = parallelism_config.dp_size;
    device_params.parallelism_config.pp_size          = parallelism_config.pp_size;
    device_params.parallelism_config.world_size       = parallelism_config.world_size;
    device_params.parallelism_config.world_rank       = parallelism_config.world_rank;
    device_params.parallelism_config.local_world_size = parallelism_config.local_world_size;
    device_params.parallelism_config.ffn_sp_size      = parallelism_config.ffn_sp_size;
    device_params.profile_debug_logging_config        = profiling_debug_logging_config;
    device_params.hw_kernel_config                    = hw_kernel_config;
    device_params.concurrency_config                  = concurrency_config;
    device_params.ffn_as_service                      = ffn_disaggregate_config.is_ffn_service();
    device_params.max_seq_len                         = model_config.max_seq_len;
    RTP_LLM_LOG_INFO("set overlap type to be %d", device_params.device_resource_config.overlap_comm_type);
    device_params.m_split                 = device_resource_config.m_split;
    device_params.max_generate_batch_size = runtime_config.max_generate_batch_size;

    const auto device_mem_reserve_env = device_resource_config.device_reserve_memory_bytes;
    RTP_LLM_LOG_INFO("Device reserve memory bytes from env: %ld", device_mem_reserve_env);
    device_params.device_reserve_memory_bytes =
        device_mem_reserve_env ? device_mem_reserve_env : getDefaultDeviceReserveMemoryBytes();
    RTP_LLM_LOG_INFO("Device reserve memory bytes: %ld", device_params.device_reserve_memory_bytes);

    device_params.host_reserve_memory_bytes = device_resource_config.host_reserve_memory_bytes;  // 4GB
    RTP_LLM_LOG_INFO("Host reserve memory bytes: %ld", device_params.host_reserve_memory_bytes);

    device_params.enable_comm_overlap = device_resource_config.enable_comm_overlap;
    device_params.enable_layer_micro_batch =
        static_cast<MicroBatchType>(device_resource_config.enable_layer_micro_batch);
    RTP_LLM_LOG_INFO("enable comm overlap: %d, enable layer micro batch: %d",
                     device_params.enable_comm_overlap,
                     device_params.enable_layer_micro_batch);
    device_params.user_deep_gemm_num_sm  = hw_kernel_config.deep_gemm_num_sm;
    device_params.use_aiter_pa           = fmha_config.use_aiter_pa;
    device_params.use_asm_pa             = fmha_config.use_asm_pa;
    device_params.use_deepep_moe         = moe_config.use_deepep_moe;
    device_params.use_deepep_internode   = moe_config.use_deepep_internode;
    device_params.use_deepep_low_latency = moe_config.use_deepep_low_latency;
    auto sp_type_str                     = SpeculativeExecutionConfig::to_string(sp_config.type);
    auto sp_model_type                   = sp_config.model_type;
    RTP_LLM_LOG_INFO("device_params sp_type is %s", sp_type_str.c_str());
    RTP_LLM_LOG_INFO("device_params sp_model_type is %s", sp_model_type.c_str());
    if (((sp_config.type == SP_TYPE_VANILLA) && (sp_model_type == "mixtbstars-mtp"))
        || ((sp_config.type == SP_TYPE_VANILLA) && (sp_model_type == "deepseek-v3-mtp"))
        || (sp_config.type == SP_TYPE_MTP) || (sp_config.type == SP_TYPE_EAGLE)) {
        device_params.is_mtp = true;
        RTP_LLM_LOG_INFO("device_params.is_mtp true");
    }

    if (((sp_config.type == SP_TYPE_VANILLA) && (sp_model_type == "qwen_3_moe_eagle"))
        || (sp_config.type == SP_TYPE_EAGLE3)) {
        device_params.is_eagle3 = true;
        RTP_LLM_LOG_INFO("device_params.eagle3 true");
    }

    RTP_LLM_LOG_INFO("use deepep moe: %d, use deepep low latency: %d",
                     device_params.use_deepep_moe,
                     device_params.use_deepep_low_latency);

    device_params.model_specific_config = model_specific_config;

    if (!global_params.device_params.size()) {
        RTP_LLM_LOG_ERROR("No device is specified to init !");
        abort();
    }
    for (const auto& [type, device_params] : global_params.device_params) {
        auto& registrationMap = getRegistrationMap();
        auto  it              = registrationMap.find(type);
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
    return devices;
}

DeviceBase* DeviceFactory::getDevice(DeviceType type, int device_id) {
    if (!getCurrentDevices().size()) {
        RTP_LLM_LOG_ERROR("You must explicitly initialize devices before getting device !");
        abort();
    }
    for (const auto device : getCurrentDevices()) {
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
    static std::shared_ptr<DeviceExporter> exporter = nullptr;
    if (!exporter) {
        const auto params       = getDefaultGlobalDeviceParams();
        const auto registration = getRegistrationMap()[params.device_params[0].first];
        exporter.reset(registration.createExporter(params.device_params[0].second));
    }
    return exporter;
}

void DeviceFactory::registerDevice(DeviceType type, DeviceCreatorType creator) {
    auto& registrationMap = getRegistrationMap();
    RTP_LLM_CHECK_WITH_INFO((registrationMap.find(type) == registrationMap.end()), "Can not find device: %d", type);
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
        .def("update_current_torch_stream", &DeviceExporter::updateCurrentTorchStream)
        .def("preprocess_gemm_weight_by_key",
             &DeviceExporter::preprocessGemmWeightByKey,
             py::arg("key"),
             py::arg("weight"),
             py::arg("user_arm_gemm_use_kai"))
        .def("preprocess_weight_scale", &DeviceExporter::preprocessWeightScale, py::arg("weight"), py::arg("scale"));

    m.def("get_device", &DeviceFactory::getDeviceExporter);
    m.def("init_device",
          &DeviceFactory::initDevices,
          py::arg("parallelism_config"),
          py::arg("model_config"),
          py::arg("eplb_config"),
          py::arg("fmha_config"),
          py::arg("device_resource_config"),
          py::arg("moe_config"),
          py::arg("sp_config"),
          py::arg("misc_config"),
          py::arg("profiling_debug_logging_config"),
          py::arg("hw_kernel_config"),
          py::arg("concurrency_config"),
          py::arg("ffn_disaggregate_config"),
          py::arg("runtime_config"),
          py::arg("model_specific_config"),
          py::arg("nccl_comm_config") = NcclCommConfig{});
}

}  // namespace rtp_llm
