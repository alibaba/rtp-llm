#pragma once

#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/DeviceExport.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <optional>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

struct GlobalDeviceParams {
    std::vector<std::pair<DeviceType, DeviceInitParams>> device_params;
};

DeviceType getDeviceType(const std::string& device_name);

class DeviceCreatorType {
public:
    std::function<DeviceBase*(const DeviceInitParams&)>                create;
    std::function<torch_ext::DeviceExporter*(const DeviceInitParams&)> createExporter;
};

class DeviceFactory {
public:
    static void        initDevices(const ParallelismConfig&           parallelism_config,
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
                                   const NcclCommConfig&              nccl_comm_config);
    static bool        isAlreadyInit();
    static DeviceBase* getDefaultDevice();
    static void        registerDevice(DeviceType type, DeviceCreatorType creator);
    static void        releaseDevices();
    // This function exports default device to python world.
    static std::shared_ptr<torch_ext::DeviceExporter> getDeviceExporter();
    static inline std::vector<DeviceBase*>            devices;

private:
    static DeviceBase*                                        getDevice(DeviceType type, int device_id = 0);
    static GlobalDeviceParams                                 getDefaultGlobalDeviceParams();
    static std::unordered_map<DeviceType, DeviceCreatorType>& getRegistrationMap();
    static std::vector<DeviceBase*>&                          getCurrentDevices();
};

void registerDeviceOps(py::module& m);

#define RTP_LLM_REGISTER_DEVICE(type)                                                                                                \
    static DeviceBase* type##_device __attribute__((used)) = nullptr;                                                                \
    static auto        type##_device_reg_creator           = []() {                                                                  \
        DeviceFactory::registerDevice(DeviceType::type,                                                             \
                                                       {[](const DeviceInitParams& params) {                                         \
                                           auto device = new type##Device(params);                                  \
                                           device->init();                                                          \
                                           return device;                                                           \
                                       },                                                                           \
                                       [](const DeviceInitParams& params) {                                         \
                                           auto exporter = new torch_ext::DeviceExporterImpl<type##Device>(params); \
                                           return exporter;                                                         \
                                       }});                                                                         \
        return true;                                                                                                \
    }();

}  // namespace rtp_llm
