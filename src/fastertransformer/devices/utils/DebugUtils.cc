#include <torch/torch.h>

#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/CommonDefines.h"

namespace fastertransformer {

void printBufferData(const Buffer& buffer, const std::string& hint, DeviceBase* device, bool force_print) {
    if (!force_print) {
        if (!enableDebugPrint()) {
            return;
        }

        if (buffer.isQBuffer()) {
            printf("skip QBuffer [%s]: %s \n", hint.c_str(), buffer.debugString().c_str());
            return;
        }
    }

    if (!device) {
        device = DeviceFactory::getDefaultDevice();
    }

    auto host_buffer = device->allocateBuffer(
        {buffer.type(), buffer.shape(), AllocationType::HOST}
    );
    device->copy({*host_buffer, buffer});
    device->syncAndCheck();

    auto tensor = torch::from_blob(
        host_buffer->data(),
        bufferShapeToTorchShape(buffer),
        c10::TensorOptions().device(torch::Device(torch::kCPU))
                            .dtype(dataTypeToTorchType(buffer.type()))
    );

    std::cout << "Buffer " << hint << " : " << tensor << std::endl;
}

void saveBufferDataToTorch(const Buffer& buffer, DeviceBase* device, const std::string& fileName) {
    if (!device) {
        device = DeviceFactory::getDefaultDevice();
    }

    auto host_buffer = device->allocateBuffer(
        {buffer.type(), buffer.shape(), AllocationType::HOST}
    );
    device->copy({*host_buffer, buffer});
    device->syncAndCheck();
    auto tensor = torch::from_blob(
        host_buffer->data(),
        bufferShapeToTorchShape(buffer),
        c10::TensorOptions().device(torch::Device(torch::kCPU))
                            .dtype(dataTypeToTorchType(buffer.type()))
    );
    auto pickled = torch::pickle_save(tensor);
    std::ofstream fout(fileName, std::ios::out | std::ios::binary);
    fout.write(pickled.data(), pickled.size());
    fout.close();
}

} // namespace fastertransformer


