#pragma once

#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/DeviceExport.h"
#include "rtp_llm/cpp/th_op/GptInitParameter.h"
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
    std::function<std::shared_ptr<GraphBase>(const DeviceInitParams& params,
                                             py::object              py_instance,
                                             int                     kv_cache_block_offset,
                                             DeviceBase*             device,
                                             bool                    in_test)>
        graph_creator;
};

class DeviceFactory {
public:
    static void        initDevices(const GptInitParameter& params);
    static DeviceBase* getDefaultDevice();
    static void        registerDevice(DeviceType type, DeviceCreatorType creator);

    // This function exports default device to python world.
    static std::shared_ptr<torch_ext::DeviceExporter> getDeviceExporter();
    static std::shared_ptr<GraphBase>                 getDeviceGraphRunner(const DeviceInitParams& params,
                                                                           py::object              py_instance,
                                                                           int                     kv_cache_block_offset,
                                                                           DeviceBase*             device,
                                                                           bool                    in_test);
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
                                       },                                                                           \
                                       [](const DeviceInitParams& params,                                           \
                                          py::object              py_instance,                                      \
                                          int                     kv_cache_block_offset,                            \
                                          DeviceBase*             device,                                           \
                                          bool                    in_test) {                                                           \
                                           auto runner = std::make_shared<type##GraphRunner>(                       \
                                               params, py_instance, kv_cache_block_offset, device, in_test);        \
                                           return runner;                                                           \
                                       }});                                                                         \
        return true;                                                                                                \
    }();

}  // namespace rtp_llm
