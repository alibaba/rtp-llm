#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/rocm/quantizePreprocessors.h"
#include "src/fastertransformer/kernels/quantization_tensor.h"

using namespace std;
namespace fastertransformer {
using namespace rocm;

inline rocm::QuantType quantTypeConvert(DataType dtype) {
    switch (dtype) {
        case DataType::TYPE_QINT8: {
            return rocm::QuantType::INT8_WEIGHT_ONLY;
        }
        default: {
            FT_CHECK_WITH_INFO(false, "Invalid quant type");
        }
    }
}
/**
 *  Symmetric Per Channle
 *  Inputs: kernel、 scales（optional）、quantType
 *  Limits：kernel（2D、3D）scales（1D）、quantType（int8、int4x2）
 *  Outputs: QBuffer(kernel, scales, zeros(empty))
 *  note：if scales is null, compute scales
 *
 * **/
BufferPtr ROCmDevice::quantize(const QuantizeParams& params) {
    FT_CHECK_WITH_INFO((params.input.type() == DataType::TYPE_FP16 || params.input.type() == DataType::TYPE_FP32
                        || params.input.type() == DataType::TYPE_BF16),
                       "quantize only support half or float quantize. but get %d.",
                       params.input.type());

    FT_CHECK_WITH_INFO((params.qtype == DataType::TYPE_QINT8 || params.qtype == DataType::TYPE_QINT4X2),
                       "cuda quantize only support qint8 or qint4x2 quantize. but get %d.",
                       params.qtype);

    FT_CHECK_WITH_INFO((params.input.dim() == 2 || params.input.dim() == 3), "quantize only support 2D or 3D input.");

    FT_CHECK_WITH_INFO((params.axis == (params.input.dim() - 1)), "quantize only support last axis.");

    if (params.input.where() == MemoryType::MEMORY_CPU) {
        FT_LOG_INFO("cpu quantize");
        size_t axis   = params.input.dim() - 1;
        auto   scales = allocateBuffer(
            {DataType::TYPE_FP16, {params.input.shape()[axis]}, getMemAllocationType(params.input.where())},
            {"scales"});

        auto kernel = allocateBuffer(
            {DataType::TYPE_INT8, params.input.shape(), getMemAllocationType(params.input.where())}, {"kernel"});
        // TODO(lidongjin) The dispatch maro only support multi template type but without data cast,
        // or one template type with data cast, here need multi template type with data cast.
        if (params.input.type() == DataType::TYPE_FP16) {
            rocm::symmetric_quantize(kernel->data<int8_t>(),
                                     nullptr,
                                     scales->data<half>(),
                                     params.input.data<half>(),
                                     params.input.shape(),
                                     quantTypeConvert(params.qtype));
        } else if (params.input.type() == DataType::TYPE_FP32) {
            rocm::symmetric_quantize(kernel->data<int8_t>(),
                                     nullptr,
                                     scales->data<half>(),
                                     params.input.data<float>(),
                                     params.input.shape(),
                                     quantTypeConvert(params.qtype));
        } else {
            FT_CHECK_WITH_INFO(false, "ERROR data type [%d] for  quantize input.", params.input.type());
        }

        return BufferPtr(new QBuffer(
            std::move(kernel),
            std::move(scales),
            std::move(BufferPtr(new Buffer(MemoryType::MEMORY_CPU_PINNED, DataType::TYPE_INVALID, {0}, nullptr)))));
    } else if (params.input.where() == MemoryType::MEMORY_GPU) {
        FT_CHECK_WITH_INFO((params.input.dim() == 2), "quantize only support 2D input.");
        auto scales = allocateBuffer(
            {DataType::TYPE_FP16, {params.input.shape()[1]}, getMemAllocationType(params.input.where())}, {"scales"});

        auto kernel =
            allocateBuffer({params.qtype == DataType::TYPE_QINT8 ? DataType::TYPE_INT8 : DataType::TYPE_INT4X2,
                            params.input.shape(),
                            getMemAllocationType(params.input.where())},
                           {"kernel"});

        if (params.qtype == DataType::TYPE_QINT8) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(
                params.input.type(),
                invokePerColQuantizationInt8,
                kernel->data<int8_t>(),
                params.input.data(),
                params.input.shape()[0],
                params.input.shape()[1],
                scales->data<half>(),
                params.smoother.has_value() ? params.smoother.value().get().data<float>() : nullptr,
                params.shift.has_value() ? params.shift.value().get().data<float>() : nullptr,
                stream_);
        } else if (params.qtype == DataType::TYPE_QINT4X2) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(
                params.input.type(),
                invokePerColQuantizationInt4x2,
                (int8_t*)(kernel->data()),
                params.input.data(),
                params.input.shape()[0],
                params.input.shape()[1],
                scales->data<half>(),
                params.smoother.has_value() ? params.smoother.value().get().data<float>() : nullptr,
                params.shift.has_value() ? params.shift.value().get().data<float>() : nullptr,
                stream_);
        }

        sync_check_cuda_error();
        return BufferPtr(new QBuffer(
            std::move(kernel),
            std::move(scales),
            std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));

    } else {
        FT_FAIL("other device not implemented");
    }
}

}  // namespace fastertransformer
