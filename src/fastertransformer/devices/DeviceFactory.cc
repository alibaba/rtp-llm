#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/cpu_impl/CpuDevice.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

namespace fastertransformer {

#define REGISTER_DEVICE(type) \
    static DeviceBase* get##type##Device() { \
        static DeviceBase* type##_device = nullptr; \
        if (type##_device == nullptr) { \
            type##_device = new type##Device(); \
            type##_device->init(); \
        } \
        return type##_device; \
    }

REGISTER_DEVICE(Cpu);
REGISTER_DEVICE(Cuda);

DeviceBase* DeviceFactory::getDevice(DeviceType type) {
    switch (type) {
        case DeviceType::Cpu:
            return getCpuDevice();
        case DeviceType::Cuda:
            return getCudaDevice();
        default:
            return nullptr;
    }
}

} // namespace fastertransformer

