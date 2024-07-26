#include "src/fastertransformer/devices/DeviceBase.h"
#include "ATen/ops/cross_entropy_loss.h"
#include "c10/util/Optional.h"
#include "src/fastertransformer/core/TrackerAllocator.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "torch/extension.h"
#include "torch/types.h"
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
    if (buffer.isQBuffer()) {
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
    return dst;
}

SelectOutput DeviceBase::select(const SelectParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.dim < params.input.shape().size(),
                          "Select dim %ld out of range with input shape %s.",
                          params.dim, params.input.debugString().c_str());
    RUNTIME_ASSERT_OP_ARG(params.index.shape().size() == 1, "Select index must be 1D.");
    RUNTIME_ASSERT_OP_ARG(params.index.type() == DataType::TYPE_INT32, "Select index must be int32.");
    RUNTIME_ASSERT_OP_ARG(params.index.where() != MemoryType::MEMORY_GPU, "Select index must on CPU.");

    const auto& src = params.input;
    const auto& idx_buf = params.index;
    const auto dim = params.dim;
    auto selected_shape = src.shape();
    selected_shape[dim] = idx_buf.shape()[0];
    auto selected = allocateBuffer({src.type(), selected_shape, getMemAllocationType(src.where())});

    const int pre_select_size = std::accumulate(
        selected_shape.begin(), selected_shape.begin() + dim, 1UL, std::multiplies<size_t>());
    const auto post_select_stride = (int32_t)std::accumulate(
        selected_shape.begin() + dim + 1, selected_shape.end(), 1UL, std::multiplies<size_t>());

    // both src and dst needs to be viewed into 1-d buffer.
    auto src_view = src.reshape({src.size()});
    auto dst_view = selected->reshape({selected->size()});

    for (auto i = 0; i < int(idx_buf.shape()[0]); i++) {
        const auto idx = idx_buf.data<int32_t>()[i];
        for (auto j = 0; j < pre_select_size; j++) {
            const auto src_offset = j * src.shape()[dim] * post_select_stride + idx * post_select_stride;
            const auto dst_offset = j * idx_buf.size() * post_select_stride + i * post_select_stride;
            copy({dst_view.view(dst_offset, post_select_stride), src_view.view(src_offset, post_select_stride)});
        }
    }

    return selected;
}

ConcatOutput DeviceBase::concat(const ConcatParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.dim == 0, "Concat only support dim 0, but got %lu.", params.dim);
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
    for (int i = 0; i < int(params.inputs.size()); i++) {
        const auto& input = params.inputs[i];
        const auto& shape = input->shape();
        RUNTIME_ASSERT_OP_ARG(
            shape.size() == concated_shape.size(),
            "Concat input [%d] shape size %ld does not match concated shape size %lu.",
            i, shape.size(), concated_shape.size());
        for (int j = 1; j < int(concated_shape.size()); j++) {
            RUNTIME_ASSERT_OP_ARG(
                shape[j] == concated_shape[j],
                "Concat input [%d] shape[%d] %ld does not match concated shape[%d] %ld.",
                i, j, shape[j], j, concated_shape[j]);
        }
        RUNTIME_ASSERT_OP_ARG(
            input->type() == type,
            "Concat input [%d] type %d does not match concated type %d.",
            i, input->type(), type);

        copy({concated->view(offset, (int64_t)shape[0]), *input});
        offset += shape[0];
    }
    return concated;
}

LossOutput DeviceBase::loss(const LossParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.logits.where() == params.labels.where(), "logits and labels must be same device, but got %d and %d.", (int)params.logits.where(), (int)params.labels.where());
    RUNTIME_ASSERT_OP_ARG(params.logits.shape()[0] == params.labels.shape()[0], "logits and labels must be same dim0, but got %d and %d.", (int)params.logits.shape()[0], (int)params.labels.shape()[0]);
    torch::Tensor logits = Buffer2torchTensor(params.logits, false);
    torch::Tensor labels = Buffer2torchTensor(params.labels, false).toType(torch::kInt64);
    torch::Tensor output;
    switch (params.calculate_loss) {
    case 1:
        output = torch::cross_entropy_loss(logits, labels, torch::nullopt, at::Reduction::Mean);
        output = output.exp();
        break;
    case 2:
        output = torch::cross_entropy_loss(logits, labels, torch::nullopt, at::Reduction::None);
        break;
    default:
        RUNTIME_ASSERT_OP_ARG(false, "calculate_loss not support %d.", params.calculate_loss);
    }
    return clone({*torchTensor2Buffer(output)});
}

} // namespace fastertransformer
