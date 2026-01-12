#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmAllocator.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/devices/ShapeCheck.h"
#include "autil/StringUtil.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/rocm/quantizePreprocessors.h"
#include "rtp_llm/cpp/kernels/rocm/quantization_rocm.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"

#include <numeric>
#include <utility>

// aiter kenels
#include "gemm_a8w8_blockscale.h"
#include "gemm_a8w8_bpreshuffle.h"
#include "gemm_a8w8.h"

// #include "aiter_meta/csrc/ck_gemm_a8w8_blockscale/include/gemm_a8w8_blockscale.h"
// #include "aiter_meta/csrc/ck_gemm_a8w8_bpreshuffle/include/gemm_a8w8_bpreshuffle.h"

using namespace std;

namespace rtp_llm {
using namespace rocm;

template<typename T>
T getRocmValue(const T* ptr, int index) {
    T tmp;
    ROCM_CHECK(hipMemcpy(&tmp, ptr + index, sizeof(T), hipMemcpyDeviceToHost));
    return tmp;
}

hipblasOperation_t opConvert(TransposeOperation op) {
    switch (op) {
        case TransposeOperation::NONE:
            return hipblasOperation_t::HIPBLAS_OP_N;
        case TransposeOperation::TRANSPOSE:
            return hipblasOperation_t::HIPBLAS_OP_T;
        default:
            ROCM_FAIL("[GEMM]: Other TransposeOperation not implemented");
    }
};

hipblasLtEpilogue_t epilogueConvert(ActivationType activationType, bool hasBias) {
    switch (activationType) {
        case ActivationType::Identity:
            return hasBias ? HIPBLASLT_EPILOGUE_BIAS : HIPBLASLT_EPILOGUE_DEFAULT;
        default:
            ROCM_FAIL("[GEMM]: Other ActivationType not implemented");
    }
}

static hipDataType dtypeConvert(DataType dtype) {
    switch (dtype) {
        case DataType::TYPE_BF16:
            return hipDataType::HIP_R_16BF;
        case DataType::TYPE_FP16:
            return hipDataType::HIP_R_16F;
        case DataType::TYPE_FP32:
            return hipDataType::HIP_R_32F;
        case DataType::TYPE_FP8_E4M3:
            return hipDataType::HIP_R_8F_E4M3_FNUZ;
        default:
            ROCM_FAIL("[GEMM]: Other DataType not implemented");
    }
};

struct ROCmGemmDispatch {

    enum GemmImplementType {
        hipblas_basic_gemm,
        hipblas_batch_gemm,
        WeightOnlyQuantMatmulPlugin,
        invalid,
    };

    static GemmImplementType dispatch(const GemmParams& params) {
        size_t dim = params.A.dim();
        if (params.C != std::nullopt) {
            return GemmImplementType::invalid;
        }
        if (dim == 2 && params.A.isFloat() && params.B.isFloat()) {
            return GemmImplementType::hipblas_basic_gemm;
        } else if (dim == 2 && params.A.type() == DataType::TYPE_FP8_E4M3
                   && params.B.type() == DataType::TYPE_FP8_E4M3) {
            return GemmImplementType::hipblas_basic_gemm;
        } else if (dim > 2 && params.A.isFloat() && params.B.isFloat()) {
            return GemmImplementType::hipblas_batch_gemm;
        } else if (dim == 2 && (params.A.type() == DataType::TYPE_FP16 || params.A.type() == DataType::TYPE_BF16)
                   && params.B.type() == DataType::TYPE_QINT8) {
            return GemmImplementType::WeightOnlyQuantMatmulPlugin;
        }
        return GemmImplementType::invalid;
    }
};

struct ROCmGemmArguments {
    std::vector<size_t> Ashape;
    std::vector<size_t> Bshape;
    std::vector<size_t> Cshape;
    std::vector<size_t> Dshape;

    DataType ADtype;
    DataType BDtype;
    DataType CDtype;
    DataType DDtype;
    DataType compute_type;

    size_t dim;
    size_t batch_size;
    size_t m;
    size_t k;
    size_t n;

    float alpha = 1.0f;
    float beta  = 0.0f;

    size_t lda;
    size_t stride_a;
    size_t ldb;
    size_t stride_b;
    size_t ldc;
    size_t stride_c;

    ROCmGemmArguments(const GemmParams& params) {

        Ashape = params.A.shape();
        Bshape = params.B.shape();

        if (params.transA == TransposeOperation::TRANSPOSE) {
            std::iter_swap(Ashape.end() - 1, Ashape.end() - 2);
        }

        if (params.transB == TransposeOperation::TRANSPOSE) {
            std::iter_swap(Bshape.end() - 1, Bshape.end() - 2);
        }

        if (params.C != std::nullopt) {
            Cshape = params.C.value().get().shape();
        }

        ADtype = params.A.type();
        BDtype = params.A.type();
        if (params.C != std::nullopt) {
            CDtype = params.C.value().get().type();
        }
        compute_type = (params.compute_type == DataType::TYPE_INVALID) ? DataType::TYPE_FP32 : params.compute_type;
        if (params.D) {
            DDtype = params.D->type();
        } else if (params.A.type() == DataType::TYPE_INT8 || params.A.type() == DataType::TYPE_QINT8){
            DDtype = DataType::TYPE_FP16;
        } else if (params.D_type){
            DDtype = params.D_type;
        } else {
            DDtype = params.compute_type == DataType::TYPE_INVALID ? ADtype : compute_type;
        }

        dim        = params.A.dim();
        batch_size = std::accumulate(Ashape.begin(), Ashape.end() - 2, (size_t)1, std::multiplies<size_t>());

        m = Ashape[dim - 2];
        k = Ashape[dim - 1];
        n = Bshape[dim - 1];

        Dshape = std::vector<size_t>(Ashape.begin(), Ashape.end() - 2);
        Dshape.insert(Dshape.end(), {m, n});

        lda      = params.A.shape()[dim - 1];
        stride_a = m * k;
        ldb      = params.B.shape()[dim - 1];
        stride_b = k * n;
        ldc      = n;
        stride_c = m * n;
        
        alpha = params.alpha;
        beta = params.beta;
    }

    void dump() {
        std::cout << "Ashape is : " << ShapeStringView(Ashape) << "\n"
                  << "Bshape is : " << ShapeStringView(Bshape) << "\n"
                  << "Cshape is : " << ShapeStringView(Cshape) << "\n"
                  << "Dshape is : " << ShapeStringView(Dshape) << "\n"
                  << "dim is : " << dim << "\n"
                  << "batch size is : " << batch_size << "\n"
                  << "m is : " << m << "\n"
                  << "n is : " << n << "\n"
                  << "k is : " << k << "\n"
                  << "lda is : " << lda << "\n"
                  << "ldb is : " << ldb << "\n"
                  << "ldc is : " << ldc << "\n"
                  << "stride_a is : " << stride_a << "\n"
                  << "stride_b is : " << stride_b << "\n"
                  << "stride_c is : " << stride_c << "\n"
                  << std::endl;
    }
};

void ROCmDevice::InvokeROCmDeepGemm(const GemmParams& params, BufferPtr output) {
    RTP_LLM_LOG_DEBUG("use rocm deep gemm.");
    RTP_LLM_CHECK_WITH_INFO(params.activationType == ActivationType::Identity,
                            "rocm deep gemm activation type should be identity");
    RTP_LLM_CHECK_WITH_INFO(params.C == std::nullopt, "rocm deep gemm bias should be nullopt");
    BufferPtr W_kernel = reinterpret_cast<const QBuffer&>(params.B).kernelPtr();
    BufferPtr W_scales = reinterpret_cast<const QBuffer&>(params.B).scalesPtr();

    const size_t m = params.A.shape()[0];
    const size_t k = params.A.shape()[1];
    const size_t n = W_kernel->shape()[1];

    QBufferPtr q_hidden = std::dynamic_pointer_cast<QBuffer>(
        quantize(QuantizeParams(params.A, DataType::TYPE_QFP8_E4M3, 1, QScheme::Qfp8PerTokenBlock, 128, 0)));

    torch::Tensor A_quant_tensor       = Buffer2torchTensor(q_hidden->kernelPtr(), false);
    torch::Tensor A_quant_scale_tensor = Buffer2torchTensor(q_hidden->scalesPtr(), false);

    torch::Tensor W_kernel_tensor = Buffer2torchTensor(W_kernel, false);
    torch::Tensor W_scale_tensor  = Buffer2torchTensor(W_scales, false);

    W_kernel_tensor = W_kernel_tensor.view({(int)W_kernel->shape()[1], (int)W_kernel->shape()[0]});
    W_scale_tensor  = W_scale_tensor.view({(int)W_scales->shape()[1], (int)W_scales->shape()[0]});

    torch::Tensor output_tensor = Buffer2torchTensor(output, false);

    gemm_a8w8_blockscale(A_quant_tensor, W_kernel_tensor, A_quant_scale_tensor, W_scale_tensor, output_tensor);
}

void ROCmDevice::InvokeROCmPTPCGemm(const GemmParams& params, BufferPtr output) {
    RTP_LLM_LOG_DEBUG("use rocm per-token per-channel gemm.");
    RTP_LLM_CHECK_WITH_INFO(params.activationType == ActivationType::Identity,
                            "rocm per-token per-channel gemm activation type should be identity");
    RTP_LLM_CHECK_WITH_INFO(params.C == std::nullopt, "rocm per-token per-channel gemm bias should be nullopt");
    BufferPtr W_kernel = reinterpret_cast<const QBuffer&>(params.B).kernelPtr();
    BufferPtr W_scales = reinterpret_cast<const QBuffer&>(params.B).scalesPtr();

    const size_t m = params.A.shape()[0];
    const size_t k = params.A.shape()[1];
    const size_t n = W_kernel->shape()[1];

    BufferPtr  A_quant_buffer;
    BufferPtr  A_scales;
    if (params.A.isQBuffer()) {
        RTP_LLM_LOG_DEBUG("aiter ptpc gemm: A is already QBuffer, skip quantize.");
        A_quant_buffer = reinterpret_cast<const QBuffer&>(params.A).kernelPtr();
        A_scales       = reinterpret_cast<const QBuffer&>(params.A).scalesPtr();
    } else {
        QBufferPtr q_hidden = std::dynamic_pointer_cast<QBuffer>(
            quantize(QuantizeParams(params.A, DataType::TYPE_QFP8_E4M3, 1, QScheme::Qfp8PerToken, 0, 0)));
        A_quant_buffer = q_hidden->kernelPtr();
        A_scales       = q_hidden->scalesPtr();
    }


    torch::Tensor A_quant_tensor       = Buffer2torchTensor(A_quant_buffer, false);
    torch::Tensor A_quant_scale_tensor = Buffer2torchTensor(A_scales, false);

    torch::Tensor W_kernel_tensor = Buffer2torchTensor(W_kernel, false);
    torch::Tensor W_scale_tensor  = Buffer2torchTensor(W_scales, false);

    // view from [k,n] to [n,k]
    W_kernel_tensor = W_kernel_tensor.view({(int)W_kernel->shape()[1], (int)W_kernel->shape()[0]});
    W_scale_tensor  = W_scale_tensor.view({(int)W_scales->shape()[1], (int)W_scales->shape()[0]});

    torch::Tensor output_tensor = Buffer2torchTensor(output, false);

    gemm_a8w8_bpreshuffle(A_quant_tensor, W_kernel_tensor, A_quant_scale_tensor, W_scale_tensor, output_tensor);
}

void ROCmDevice::HipblasltPTPCGemm(const GemmParams& params, BufferPtr output) {
    RTP_LLM_LOG_DEBUG("use hipBLASLt ptpc gemm.");
    ROCmGemmArguments arguments(params);
    BufferPtr  A_quant_buffer;
    BufferPtr  A_scales;
    if (params.A.isQBuffer()) {
        RTP_LLM_LOG_DEBUG("hipBLASLt ptpc gemm: A is already QBuffer, skip quantize.");
        A_quant_buffer = reinterpret_cast<const QBuffer&>(params.A).kernelPtr();
        A_scales       = reinterpret_cast<const QBuffer&>(params.A).scalesPtr();
    } else {
        QBufferPtr q_hidden = std::dynamic_pointer_cast<QBuffer>(
            quantize(QuantizeParams(params.A, DataType::TYPE_QFP8_E4M3, 1, QScheme::Qfp8PerToken, 0, 0)));
        A_quant_buffer = q_hidden->kernelPtr();
        A_scales       = q_hidden->scalesPtr();
    }

    BufferPtr W_kernel = reinterpret_cast<const QBuffer&>(params.B).kernelPtr();
    BufferPtr W_scales = reinterpret_cast<const QBuffer&>(params.B).scalesPtr();

    auto A    = A_quant_buffer->data();
    auto B    = W_kernel->data();
    auto D    = output->data();
    auto a_op = opConvert(params.transA);
    auto b_op = opConvert(params.transB);
    bool has_bias = (params.C != std::nullopt && params.C->get().data() != nullptr);
    auto epilogue = epilogueConvert(params.activationType, has_bias);
    RTP_LLM_CHECK_WITH_INFO(epilogue == HIPBLASLT_EPILOGUE_BIAS || epilogue == HIPBLASLT_EPILOGUE_DEFAULT, 
        "epilogue should be HIPBLASLT_EPILOGUE_BIAS or HIPBLASLT_EPILOGUE_DEFAULT in HipblasltPTPCGemm, but got %d", epilogue);
    
    hipblas_mm_wrapper_->setStream(current_stream_);
    hipblas_mm_wrapper_->setGemmConfig(dtypeConvert(A_quant_buffer->type()), dtypeConvert(W_kernel->type()), 
                                       dtypeConvert(output->type()), dtypeConvert(arguments.compute_type));

    hipblas_mm_wrapper_->FP8_Gemm(b_op, a_op, arguments.n, arguments.m, arguments.k, B, arguments.ldb, 
                            A, arguments.lda, D, arguments.ldc, reinterpret_cast<const float*>(W_scales->data()),
                            reinterpret_cast<const float*>(A_scales->data()), has_bias ? params.C->get().data() : nullptr, 
                            epilogue, arguments.alpha, arguments.beta);
}

void ROCmDevice::InvokeROCmDeepGemmWi8Ai8(const GemmParams& params,
                                          BufferPtr         output){
    RTP_LLM_LOG_DEBUG("use rocm deep gemm.");
    RTP_LLM_CHECK_WITH_INFO(params.activationType == ActivationType::Identity, "rocm deep gemm activation type should be identity");
    BufferPtr A_kernel = reinterpret_cast<const QBuffer&>(params.A).kernelPtr();
    BufferPtr A_scales = reinterpret_cast<const QBuffer&>(params.A).scalesPtr();
    BufferPtr W_kernel = reinterpret_cast<const QBuffer&>(params.B).kernelPtr();
    BufferPtr W_scales = reinterpret_cast<const QBuffer&>(params.B).scalesPtr();

    const int m = A_kernel->shape()[0];
    const int k = A_kernel->shape()[1];
    const int n = W_kernel->shape()[1];

    torch::Tensor A_kernel_tensor = Buffer2torchTensor(A_kernel, false);
    torch::Tensor A_scale_tensor = Buffer2torchTensor(A_scales, false);
    torch::Tensor W_kernel_tensor = Buffer2torchTensor(W_kernel, false);
    torch::Tensor W_scale_tensor = Buffer2torchTensor(W_scales, false);

    // view from [k,n] to [n,k]
    W_kernel_tensor = W_kernel_tensor.view({(int)W_kernel->shape()[1],(int)W_kernel->shape()[0]}); 
    W_scale_tensor = W_scale_tensor.view({n, 1});

    torch::Tensor output_tensor;

    output_tensor = Buffer2torchTensor(output, false).contiguous();

    // broadcast single value to [m,1]
    A_scale_tensor = A_scale_tensor.view({-1, 1}).expand({m, 1}).contiguous();

    // bias process
    std::optional<torch::Tensor> bias_opt = std::nullopt;
    if (params.C != std::nullopt) {
        const Buffer& bias_buf = params.C->get();
        torch::Tensor bias_tensor = Buffer2torchTensor(const_cast<Buffer&>(bias_buf), false).contiguous();
        bias_tensor = bias_tensor.view({n});
        bias_opt = bias_tensor;
    }
    int splitK = 0;
    gemm_a8w8(A_kernel_tensor, W_kernel_tensor, A_scale_tensor, W_scale_tensor, output_tensor, bias_opt, splitK);
}

/// @brief   basic gemm ops
/// @details D = alpha * op(A) * op(B) + beta * C
///          A [b, ..., m, k]
///          B [b, ..., k, n]
///          C [b, ..., m, n]
BufferPtr ROCmDevice::gemm(const GemmParams& params) {
    params.check();

    using GemmImplementType = ROCmGemmDispatch::GemmImplementType;
    ROCmGemmArguments arguments(params);
    BufferPtr output;
    if (params.D) {
        output = params.D;
        RUNTIME_ASSERT_OP_ARG((arguments.DDtype == params.D->type()) && (arguments.Dshape == params.D->shape()),
                              "Gemm output D shape and dtype mismatch: expected [%d][%s] but got [%s]",
                              arguments.DDtype,
                              autil::StringUtil::toString(arguments.Dshape).c_str(),
                              params.D->debugString().c_str());                         
    } else {
        output = allocateBuffer({arguments.DDtype, arguments.Dshape, AllocationType::DEVICE}, {"gemm_output"});
    }

    if (params.dispatch() == GemmType::BufferA_QBufferB_BufferC_2DGemm) {
        if (reinterpret_cast<const QBuffer&>(params.B).zerosData() != nullptr) {
            ROCM_CHECK_VALUE(reinterpret_cast<const QBuffer&>(params.B).scales().dim() == 2,
                             "scales().dim() = %d",
                             reinterpret_cast<const QBuffer&>(params.B).scales().dim());
            size_t kernel_dim0 = params.B.shape()[0];
            size_t scales_dim0 = reinterpret_cast<const QBuffer&>(params.B).scales().shape()[0];
            ROCM_CHECK_VALUE((kernel_dim0 % scales_dim0 == 0), "kernel_dim0 % scales_dim0 != 0");
            size_t group_size = (kernel_dim0 / scales_dim0);
            ROCM_CHECK_VALUE((group_size == 64 || group_size == 128), "group_size != 64 and group_size != 128");
            size_t type_bits = getTypeBits(params.B.type());
            ROCM_CHECK_VALUE((type_bits == 4 || type_bits == 8), "type_bits != 4 and type_bits != 8");

            BUFFER_DTYPE_CHECK(params.A, {DataType::TYPE_FP16, DataType::TYPE_BF16});
            BUFFER_DTYPE_CHECK(params.B, {DataType::TYPE_QINT4X2});

            const QBuffer& QB  = reinterpret_cast<const QBuffer&>(params.B);
            auto           fpB = allocateBuffer({params.A.type(), {params.B.shape()}, AllocationType::DEVICE}, {"fpB"});

#if USING_CK_INT4
            // Using CK int4-dequant fusion Gemm kernel
            auto ck_gemm_params = ckGemmParam({params.A.data(),
                                               QB.kernel().data(),
                                               QB.scales().data(),
                                               QB.zeros().data(),
                                               output->data(),
                                               arguments.m,
                                               arguments.n,
                                               arguments.k,
                                               group_size,
                                               arguments.k,  // arguments.lda,
                                               arguments.k,  // arguments.ldb,
                                               arguments.n,  // arguments.ldc,
                                               stream_});
            ck_gemm_runner_->runCKGemm(ck_gemm_params, params.A.type(), params.B.type());

#else
            // dequant B
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(params.A.type(),
                                             invokePerColDequantizationInt4x2,
                                             fpB.get()->data(),
                                             arguments.k,
                                             arguments.n,
                                             group_size,
                                             (int8_t*)(QB.kernel().data()),
                                             QB.scales().data<half>(),
                                             QB.zeros().data<half>(),
                                             stream_);

            const auto A = params.A.data();
            const auto B = fpB.get()->data();
            auto       D = output->data();

            auto a_op = opConvert(params.transA);
            auto b_op = opConvert(params.transB);

            auto A_data_type = dtypeConvert(arguments.ADtype);
            auto B_data_type = dtypeConvert(fpB.get()->type());
            auto D_data_type = dtypeConvert(arguments.DDtype);
            auto computeType = dtypeConvert(arguments.compute_type);

            hipblas_mm_wrapper_->stridedBatchedGemm(b_op,
                                                    a_op,
                                                    arguments.n,
                                                    arguments.m,
                                                    arguments.k,
                                                    arguments.alpha,
                                                    B,
                                                    B_data_type,
                                                    arguments.ldb,
                                                    arguments.stride_b,
                                                    A,
                                                    A_data_type,
                                                    arguments.lda,
                                                    arguments.stride_a,
                                                    arguments.beta,
                                                    D,
                                                    D_data_type,
                                                    arguments.ldc,
                                                    arguments.stride_c,
                                                    arguments.batch_size,
                                                    computeType);
#endif
            return std::move(output);
        } else if (params.A.type() != DataType::TYPE_QFP8_E4M3 && params.B.type() == DataType::TYPE_QFP8_E4M3) {
            const QBuffer& qB        = reinterpret_cast<const QBuffer&>(params.B);
            Buffer         qB_kernel = qB.kernel();
            Buffer         qB_scales = qB.scales();
            ROCM_CHECK_VALUE((qB_kernel.dim() == 2), "quant Gemm only support 2D");
            size_t kernel_K = qB_kernel.shape()[0], kernel_N = qB_kernel.shape()[1];
            size_t scale_K = qB_scales.shape()[0], scale_N = qB_scales.shape()[1];
            if (kernel_K == scale_K * 128) {
                InvokeROCmDeepGemm(params, output);
            } else if ((1 == scale_K && scale_N == kernel_N) || (1 == scale_N && scale_K == kernel_K)) {
                if (hipblas_mm_wrapper_->use_swizzleA()){
                    HipblasltPTPCGemm(params, output);
                }
                else {
                    InvokeROCmPTPCGemm(params, output);
                }
            } else {
                ROCM_FAIL(
                    "[GEMM]: Other FP8 weight quantization not implemented, with weight kernel [%d, %d], weight scales [%d, %d]",
                    kernel_K,
                    kernel_N,
                    scale_K,
                    scale_N);
            }
            return std::move(output);
        } else {
            ROCM_FAIL("[GEMM]: Other weight quantization not implemented");
        }
    }

    if (params.dispatch() == GemmType::QBufferA_QBufferB_BufferC_2DGemm) {
        if ((params.A.type() == DataType::TYPE_INT8 || params.A.type() == DataType::TYPE_QINT8)){
            BUFFER_DTYPE_CHECK(params.A, {DataType::TYPE_INT8, DataType::TYPE_QINT8});
            BUFFER_DTYPE_CHECK(params.B, {DataType::TYPE_INT8, DataType::TYPE_QINT8});
            InvokeROCmDeepGemmWi8Ai8(params, output);
            return std::move(output);
        } else if (params.A.type() == DataType::TYPE_QFP8_E4M3 && params.B.type() == DataType::TYPE_QFP8_E4M3) {
            const QBuffer& qB        = reinterpret_cast<const QBuffer&>(params.B);
            Buffer         qB_kernel = qB.kernel();
            Buffer         qB_scales = qB.scales();
            ROCM_CHECK_VALUE((qB_kernel.dim() == 2), "quant Gemm only support 2D");
            size_t kernel_K = qB_kernel.shape()[0], kernel_N = qB_kernel.shape()[1];
            size_t scale_K = qB_scales.shape()[0], scale_N = qB_scales.shape()[1];
            if (1 == scale_K && scale_N == kernel_N) {
                if (hipblas_mm_wrapper_->use_swizzleA()){
                    HipblasltPTPCGemm(params, output);
                }
                else {
                    InvokeROCmPTPCGemm(params, output);
                }
            } else {
                ROCM_FAIL(
                    "[GEMM]: Other FP8 weight quantization not implemented, with weight kernel [%d, %d], weight scales [%d, %d]",
                    kernel_K,
                    kernel_N,
                    scale_K,
                    scale_N);
            }
            return std::move(output);            

        } else {
            ROCM_FAIL("[GEMM]: Other weight quantization not implemented");
        }
    }

    auto A_data_type = dtypeConvert(arguments.ADtype);
    auto B_data_type = dtypeConvert(arguments.BDtype);
    auto D_data_type = dtypeConvert(arguments.DDtype);
    auto computeType = dtypeConvert(arguments.compute_type);

    const auto A    = params.A.data();
    const auto B    = params.B.data();
    auto       D    = output->data();
    auto       a_op = opConvert(params.transA);
    auto       b_op = opConvert(params.transB);

    hipblas_mm_wrapper_->setGemmConfig(A_data_type, B_data_type, D_data_type, computeType);
    hipblas_mm_wrapper_->setStream(current_stream_);

    bool enable_swizzle = !params.shared_gate_gemm
                          && (hipblas_mm_wrapper_->use_swizzleA());

    if (ROCmGemmDispatch::dispatch(params) == GemmImplementType::hipblas_basic_gemm) {
        hipblas_mm_wrapper_->Gemm(b_op,
                                  a_op,
                                  arguments.n,
                                  arguments.m,
                                  arguments.k,
                                  B,
                                  arguments.ldb,
                                  A,
                                  arguments.lda,
                                  D,
                                  arguments.ldc,
                                  arguments.alpha,
                                  arguments.beta,
                                  enable_swizzle);

        return std::move(output);
    } else if (ROCmGemmDispatch::dispatch(params) == GemmImplementType::hipblas_batch_gemm) {
        hipblas_mm_wrapper_->stridedBatchedGemm(b_op,
                                                a_op,
                                                arguments.n,
                                                arguments.m,
                                                arguments.k,
                                                arguments.alpha,
                                                B,
                                                B_data_type,
                                                arguments.ldb,
                                                arguments.stride_b,
                                                A,
                                                A_data_type,
                                                arguments.lda,
                                                arguments.stride_a,
                                                arguments.beta,
                                                D,
                                                D_data_type,
                                                arguments.ldc,
                                                arguments.stride_c,
                                                arguments.batch_size,
                                                computeType);
        return std::move(output);
    } else {
        ROCM_FAIL("[GEMM]:other dispatch not implemented");
    }
    return std::move(output);
}

}  // namespace rtp_llm
