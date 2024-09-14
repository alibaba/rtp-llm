#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/rocm/quantizePreprocessors.h"
#include "src/fastertransformer/kernels/rocm/quantization_rocm.h"

using namespace std;
namespace fastertransformer {
using namespace rocm;

inline rocm::QuantType quantTypeConvert(DataType dtype) {
    switch (dtype) {
        case DataType::TYPE_QINT8: {
            return rocm::QuantType::INT8_WEIGHT_ONLY;
        }
        default: {
            ROCM_FAIL("Invalid quant type");
        }
    }
}

BufferPtr ROCmDevice::quantize(const QuantizeParams& params) {
    ROCM_CHECK_VALUE((params.input.type() == DataType::TYPE_FP16 || params.input.type() == DataType::TYPE_FP32
                        || params.input.type() == DataType::TYPE_BF16),
                       "quantize only support half or float quantize. but get %d.",
                       params.input.type());

    ROCM_CHECK_VALUE((params.qtype == DataType::TYPE_QINT8 || params.qtype == DataType::TYPE_QINT4X2),
                       "cuda quantize only support qint8 or qint4x2 quantize. but get %d.",
                       params.qtype);

    ROCM_CHECK_VALUE((params.input.dim() == 2 || params.input.dim() == 3), "quantize only support 2D or 3D input.");

    ROCM_CHECK_VALUE((params.axis == (params.input.dim() - 1)), "quantize only support last axis.");

    if (params.input.where() == MemoryType::MEMORY_GPU) {
        ROCM_CHECK_VALUE((params.input.dim() == 2), "quantize only support 2D input.");
        size_t groupSize   = params.groupSize;
        size_t scales_dim0 = params.input.shape()[0] / groupSize;

        auto kernel =
            allocateBuffer({params.qtype == DataType::TYPE_QINT8 ? DataType::TYPE_INT8 : DataType::TYPE_INT4X2,
                            params.input.shape(),
                            getMemAllocationType(params.input.where())},
                           {"kernel"});
        auto scales = allocateBuffer(
            {DataType::TYPE_FP16, {scales_dim0, params.input.shape()[1]}, getMemAllocationType(params.input.where())},
            {"scales"});
        auto zeros = allocateBuffer(
            {DataType::TYPE_FP16, {scales_dim0, params.input.shape()[1]}, getMemAllocationType(params.input.where())},
            {"zeros"});

        if (params.qtype == DataType::TYPE_QINT4X2) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(params.input.type(),
                                             invokePerColQuantizationInt4x2,
                                             params.input.data(),
                                             params.input.shape()[0],
                                             params.input.shape()[1],
                                             groupSize,
                                             (uint8_t*)(kernel->data()),
                                             scales->data<half>(),
                                             zeros->data<half>(),
                                             stream_);
        } else {
            ROCM_FAIL("other quantize not implemented");
        }

        ROCM_SYNC_AND_CHECK();
        return BufferPtr(new QBuffer(std::move(kernel), std::move(scales), std::move(zeros)));

    } else {
        ROCM_FAIL("cpu quantize not implemented");
    }
}

BufferPtr ROCmDevice::dequantize(const QuantizeParams& params) {
    if (params.input.where() == MemoryType::MEMORY_GPU) {
        if (params.qtype == DataType::TYPE_QINT4X2) {
            const QBuffer& QB          = reinterpret_cast<const QBuffer&>(params.input);
            size_t         kernel_dim0 = QB.kernel().shape()[0];
            size_t         scales_dim0 = QB.scales().shape()[0];
            size_t         group_size  = (kernel_dim0 / scales_dim0);

            BufferPtr fpB =
                allocateBuffer({QB.scales().type(), {QB.kernel().shape()}, AllocationType::DEVICE}, {"fpB"});

            DISPATCH_CUDA_FUNCTION_DATA_TYPE(fpB.get()->type(),
                                             invokePerColDequantizationInt4x2,
                                             fpB.get()->data(),
                                             (size_t)(QB.kernel().shape()[0]),
                                             (size_t)(QB.kernel().shape()[1]),
                                             group_size,
                                             (int8_t*)(QB.kernel().data()),
                                             QB.scales().data<half>(),
                                             QB.zeros().data<half>(),
                                             stream_);

            ROCM_SYNC_AND_CHECK();
            return fpB;
        } else {
            ROCM_FAIL("other dequantize not implemented");
        }
    } else {
        ROCM_FAIL("cpu dequantize not implemented");
    }
}

}  // namespace fastertransformer
