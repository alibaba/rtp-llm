#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/cuda/cuda_fp8_utils.h"
#include "src/fastertransformer/kernels/quantization_tensor.h"


using namespace std;

namespace trt = tensorrt_llm::kernels::cutlass_kernels;
namespace trt_common = tensorrt_llm::common;

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

    FT_CHECK_WITH_INFO((params.input.dim() == 2),
        "cuda quantize only support 2D or 3D input.");

    FT_CHECK_WITH_INFO((params.axis == (params.input.dim() - 1)),
        "cuda quantize only support last axis.");

    BufferPtr scales = nullptr;
    DataType out_data_type = params.qscheme == QScheme::Qfp8PerTensor ? DataType::TYPE_FP8_E4M3 : DataType::TYPE_INT8;
    auto kernel = allocateBuffer({out_data_type,
                                 params.input.shape(),
                                 getMemAllocationType(params.input.where())},
                                 {"kernel"});

    if (params.qscheme == QScheme::Qint8WeightOnly) {
        FT_CHECK_WITH_INFO(params.input.where() == MemoryType::MEMORY_CPU, "cpu quantize");
        size_t axis = params.input.dim() - 1;
        scales = allocateBuffer({DataType::TYPE_FP16,
                                {params.input.shape()[axis]},
                                getMemAllocationType(params.input.where())},
                                {"scales"});
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
    } else if (params.qscheme == QScheme::Qint8PerToken) {
        scales = allocateBuffer({DataType::TYPE_FP32,
                                {params.input.shape()[0]},
                                getMemAllocationType(params.input.where())},
                                {"scales"});
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
        FT_CHECK_WITH_INFO(params.static_scale_reciprocal.has_value(),
            "static_scale_reciprocal should not be nullptr in Qint8PerTensor");
        scales = BufferPtr(new Buffer(params.static_scale_reciprocal.value().get().where(),
                                      params.static_scale_reciprocal.value().get().type(),
                                      params.static_scale_reciprocal.value().get().shape(),
                                      params.static_scale_reciprocal.value().get().data()));

        DISPATCH_CUDA_FUNCTION_DATA_TYPE(params.input.type(), invokeQuantization,
                                         kernel->data<int8_t>(),
                                         params.input.data(),
                                         params.input.size(),
                                         params.static_scale.value().get().data<float>(),
                                         stream_,
                                         -1);
#ifdef ENABLE_FP8
        } else if (params.qscheme == QScheme::Qfp8PerTensor) {
            FT_CHECK_WITH_INFO(params.static_scale_reciprocal.has_value(),
                "static_scale_reciprocal should not be nullptr in Qint8PerTensor");
            scales = BufferPtr(new Buffer(params.static_scale_reciprocal.value().get().where(),
                                        params.static_scale_reciprocal.value().get().type(),
                                        params.static_scale_reciprocal.value().get().shape(),
                                        params.static_scale_reciprocal.value().get().data()));
            switch (params.input.type()) {
                case DataType::TYPE_FP32:
                    trt_common::invokeQuantizeMatrix( kernel->data<__nv_fp8_e4m3>(),
                                params.static_scale.value().get().data<float>(),
                                params.input.data<float>(),
                                params.input.size(),
                                params.input.shape()[0],
                                trt_common::QuantizeMode::PER_TENSOR,
                                stream_);
                    break;
                case DataType::TYPE_FP16:
                    trt_common::invokeQuantizeMatrix(kernel->data<__nv_fp8_e4m3>(),
                                params.static_scale.value().get().data<float>(),
                                params.input.data<half>(),
                                params.input.size(),
                                params.input.shape()[0],
                                trt_common::QuantizeMode::PER_TENSOR,
                                stream_);
                    break;
#ifdef ENABLE_BF16
                case DataType::TYPE_BF16:
                    trt_common::invokeQuantizeMatrix(kernel->data<__nv_fp8_e4m3>(),
					     params.static_scale.value().get().data<float>(),
					     params.input.data<__nv_bfloat16>(),
					     params.input.size(),
					     params.input.shape()[0],
					     trt_common::QuantizeMode::PER_TENSOR,
					     stream_);
                    break;
                default:
                    FT_CHECK_WITH_INFO(false, "unsupport data type");
            }
#endif
#endif
    } else {
        FT_CHECK_WITH_INFO(false, "params qscheme type unknown: %d", int(params.qscheme));
    }

    sync_check_cuda_error();

    auto zeros_type = scales->where();
    return BufferPtr(new QBuffer(std::move(kernel),
                                 std::move(scales),
                                 std::move(BufferPtr(new Buffer(zeros_type,
                                                                DataType::TYPE_INVALID,
                                                                {0},
                                                                nullptr)))));
}

}   // namespace fastertransformer
