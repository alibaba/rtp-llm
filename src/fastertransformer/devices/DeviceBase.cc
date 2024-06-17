#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/core/TrackerAllocator.h"

#include <numeric>

using namespace std;

namespace fastertransformer {

DeviceBase::DeviceBase(const DeviceInitParams& params)
    : device_id_(params.device_id)
    , init_params_(params)
    {}

void DeviceBase::init() {
    buffer_manager_.reset(new BufferManager(getAllocator(), getHostAllocator()));
}

DeviceStatus DeviceBase::getDeviceStatus() {
    return DeviceStatus();
}

void DeviceBase::traceMemoryUsage() {
    FT_LOG_INFO("Device Memory: %s", buffer_manager_->printAllocationRecords(getAllocator()).c_str());
    FT_LOG_INFO("Host Memory: %s", buffer_manager_->printAllocationRecords(getHostAllocator()).c_str());
    return;
}

AllocationType DeviceBase::getMemAllocationType(const MemoryType type) {
    return (type == getAllocator()->memoryType()) ? AllocationType::DEVICE : AllocationType::HOST;
}

BufferStatus DeviceBase::queryBufferStatus() {
    return buffer_manager_->queryStatus();
}

BufferPtr DeviceBase::allocateBuffer(const BufferParams& params, const BufferHints& hints) {
    return buffer_manager_->allocate(params, hints);
}

BufferPtr DeviceBase::allocateBufferLike(const Buffer& buffer,
                                         const AllocationType atype,
                                         const BufferHints& hints) {
    if (buffer.isQuantify()) {
        auto kernel = allocateBufferLike((reinterpret_cast<const QBuffer*>(&buffer)->kernel()),
                                         atype,
                                         hints);
        auto scales = allocateBufferLike((reinterpret_cast<const QBuffer*>(&buffer)->scales()),
                                         atype,
                                         hints);
        auto zeros = allocateBufferLike((reinterpret_cast<const QBuffer*>(&buffer)->zeros()),
                                        atype,
                                        hints);
        return BufferPtr(new QBuffer(std::move(kernel),
                                     std::move(scales),
                                     std::move(zeros)));
    }
    return allocateBuffer({buffer.type(), buffer.shape(), atype}, hints);
}

void DeviceBase::syncAndCheck() {
    return;
}

void DeviceBase::syncCommunication(bool timeout) {
    return;
}

CloneOutput DeviceBase::clone(const CloneParams& params) {
    const auto& src = params.input;
    auto dst = allocateBufferLike(src, params.alloc_type, params.hints);
    copy({*dst, src});
    return move(dst);
}

ConcatOutput DeviceBase::concat(const ConcatParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.dim == 0, "Concat only support dim 0, but got %d.", params.dim);
    RUNTIME_ASSERT_OP_ARG(params.inputs.size() > 0, "Concat requires at least 1 input.");
    if (params.inputs.size() == 1) {
        return params.inputs[0];
    }

    const auto concated_length = std::accumulate(
        params.inputs.begin(), params.inputs.end(), 0UL,
        [](size_t sum, const BufferPtr& buffer) {
            return sum + buffer->shape()[0];
        });
    auto concated_shape = params.inputs[0]->shape();
    concated_shape[0] = concated_length;
    const auto type = params.inputs[0]->type();
    auto concated = allocateBuffer({
        type, concated_shape, getMemAllocationType(params.inputs[0]->where())});

    size_t offset = 0;
    for (auto i = 0; i < params.inputs.size(); i++) {
        const auto& input = params.inputs[i];
        const auto& shape = input->shape();
        RUNTIME_ASSERT_OP_ARG(
            shape.size() == concated_shape.size(),
            "Concat input [%d] shape size %d does not match concated shape size %d.",
            i, shape.size(), concated_shape.size());
        for (auto j = 1; j < concated_shape.size(); j++) {
            RUNTIME_ASSERT_OP_ARG(
                shape[j] == concated_shape[j],
                "Concat input [%d] shape[%d] %d does not match concated shape[%d] %d.",
                i, j, shape[j], j, concated_shape[j]);
        }
        RUNTIME_ASSERT_OP_ARG(
            input->type() == type,
            "Concat input [%d] type %d does not match concated type %d.",
            i, input->type(), type);

        copy({*concated, *input, offset, 0, shape[0]});
        offset += shape[0];
    }
    return move(concated);
}

}; // namespace fastertransformer

