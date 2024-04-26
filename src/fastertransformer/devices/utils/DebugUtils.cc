#include <torch/torch.h>

#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/devices/utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/CommonDefines.h"

namespace fastertransformer {

void printBufferData(const Buffer& buffer, const std::string& hint, DeviceBase* device) {
    if (!isDebugMode()) {
        return;
    }

    if (!device) {
        device = DeviceFactory::getDefaultDevice();
    }
    device->syncAndCheck();

    auto host_buffer = device->allocateBuffer(
        {buffer.type(), buffer.shape(), AllocationType::HOST}
    );
    device->copy(CopyParams(*host_buffer, buffer));

    auto tensor = torch::from_blob(
        host_buffer->data(),
        bufferShapeToTorchShape(buffer),
        c10::TensorOptions().device(torch::Device(torch::kCPU))
                            .dtype(dataTypeToTorchType(buffer.type()))
    );

    std::cout << "Buffer " << hint << " : " << tensor << std::endl;
}

} // namespace fastertransformer


