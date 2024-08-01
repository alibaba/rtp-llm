#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/kernels/quantization_tensor.h"

using namespace std;

namespace trt = tensorrt_llm::kernels::cutlass_kernels;

namespace fastertransformer {

inline trt::QuantType trtQuantTypeConvert(DataType dtype) {
    switch (dtype) {
        case DataType::TYPE_QINT8: {
            return trt::QuantType::INT8_WEIGHT_ONLY;
        }
        default: {
            FT_FAIL("Invalid quant type");
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

BufferPtr CudaDevice::quantize(const QuantizeParams& params) {
    FT_CHECK_WITH_INFO((params.input.type() == DataType::TYPE_FP16 ||
                        params.input.type() == DataType::TYPE_FP32 ||
                        params.input.type() == DataType::TYPE_BF16),
        "cuda quantize only support half or float quantize. but get %d.", params.input.type());

    FT_CHECK_WITH_INFO((params.input.dim() == 2 || params.input.dim() == 3),
        "cuda quantize only support 2D or 3D input.");

    FT_CHECK_WITH_INFO((params.axis == (params.input.dim() - 1)),
        "cuda quantize only support last axis.");

    if (params.input.where() == MemoryType::MEMORY_CPU) {
        FT_LOG_INFO("cpu quantize");
        size_t axis = params.input.dim() - 1;
        auto scales = allocateBuffer({DataType::TYPE_FP16,
                                    {params.input.shape()[axis]},
                                    getMemAllocationType(params.input.where())},
                                    {"scales"});

        auto kernel = allocateBuffer({DataType::TYPE_INT8,
                                    params.input.shape(),
                                    getMemAllocationType(params.input.where())},
                                    {"kernel"});
        // TODO(lidongjin) The dispatch maro only support multi template type but without data cast,
        // or one template type with data cast, here need multi template type with data cast.
        if (params.input.type() == DataType::TYPE_FP16) {
            trt::symmetric_quantize(kernel->data<int8_t>(),
                                    nullptr,
                                    scales->data<half>(),
                                    params.input.data<half>(),
                                    params.input.shape(),
                                    trtQuantTypeConvert(params.qtype));
        } else if (params.input.type() == DataType::TYPE_BF16) {
            trt::symmetric_quantize(kernel->data<int8_t>(),
                                    nullptr,
                                    scales->data<half>(),
                                    params.input.data<__nv_bfloat16>(),
                                    params.input.shape(),
                                    trtQuantTypeConvert(params.qtype));
        } else if (params.input.type() == DataType::TYPE_FP32) {
            trt::symmetric_quantize(kernel->data<int8_t>(),
                                    nullptr,
                                    scales->data<half>(),
                                    params.input.data<float>(),
                                    params.input.shape(),
                                    trtQuantTypeConvert(params.qtype));
        } else {
            FT_CHECK_WITH_INFO(false,
                "ERROR data type [%d] for cuda quantize input.", params.input.type());
        }


        return BufferPtr(new QBuffer(std::move(kernel),
                                    std::move(scales),
                                    std::move(BufferPtr(new Buffer(MemoryType::MEMORY_CPU_PINNED,
                                                                    DataType::TYPE_INVALID,
                                                                    {0},
                                                                    nullptr)))));
    } else if (params.input.where() == MemoryType::MEMORY_GPU) {
        FT_CHECK_WITH_INFO((params.input.dim() == 2),
            "cuda quantize only support 2D input.");
        // FT_LOG_INFO("invoke invokePerTokenQuantization");
        BufferPtr scales;
        if (params.qscheme == QScheme::Qint8PerToken) {
            scales = allocateBuffer(
                {DataType::TYPE_FP32, {params.input.shape()[0]}, getMemAllocationType(params.input.where())},
                {"scales"});
        } else if (params.qscheme == QScheme::Qint8PerTensor) {
            FT_CHECK_WITH_INFO(params.static_scale_reciprocal.has_value(), "static_scale_reciprocal should not be nullptr in Qint8PerTensor");
            scales = BufferPtr(new Buffer(params.static_scale_reciprocal.value().get().where(),
                                      params.static_scale_reciprocal.value().get().type(),
                                      params.static_scale_reciprocal.value().get().shape(),
                                      params.static_scale_reciprocal.value().get().data()));
        } else {
            FT_CHECK_WITH_INFO(false, "unknown qscheme %d", int(params.qscheme));
        }

        auto kernel = allocateBuffer({DataType::TYPE_INT8,
                                    params.input.shape(),
                                    getMemAllocationType(params.input.where())},
                                    {"kernel"});

        if (params.qscheme == QScheme::Qint8PerToken) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(params.input.type(), invokePerTokenQuantization,
                                            kernel->data<int8_t>(),
                                            params.input.data(),
                                            params.input.shape()[0],
                                            params.input.shape()[1],
                                            scales->data<float>(),
                                            params.smoother.has_value() ? params.smoother.value().get().data<float>() : nullptr,
                                            params.shift.has_value() ? params.shift.value().get().data<float>() : nullptr,
                                            stream_);
        } else if (params.qscheme == QScheme::Qint8PerTensor) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(params.input.type(), invokeQuantization,
                                            kernel->data<int8_t>(),
                                            params.input.data(),
                                            params.input.size(),
                                            params.static_scale.value().get().data<float>(),
                                            stream_,
                                            -1);
        } else {
            FT_CHECK_WITH_INFO(false, "params qscheme type unknown: %d", int(params.qscheme));
        }
        sync_check_cuda_error();
        print_bsd(-1, "kernel", reinterpret_cast<int8_t*>(kernel->data()), 1, params.input.shape()[0], params.input.shape()[1], 0, params.input.shape()[1]);
        print_bsd(-1, "scale", reinterpret_cast<float*>(scales->data()), 1, params.input.shape()[0], 1, 0, 1);

        return BufferPtr(new QBuffer(
            std::move(kernel),
            std::move(scales),
            std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));

    } else {
        FT_FAIL("other device not implemented");
    }
}

}   // namespace fastertransformer
