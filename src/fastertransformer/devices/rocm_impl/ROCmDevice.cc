#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
#include "src/fastertransformer/devices/rocm_impl/ROCmAllocator.h"
#include "src/fastertransformer/core/TrackerAllocator.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/utils/ShapeCheck.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include <cstring>

#include "src/fastertransformer/kernels/hello_world.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/rmsnormKernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"

//layerNorm
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/alpha_layernorm_kernels.h"
#include "src/fastertransformer/kernels/rmsnormKernels.h"

#include "src/fastertransformer/cuda/nccl/nccl_utils_torch.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils.h"

// TODO(rocm): Idealy we just link compiler_rt for this symbol.
extern "C" half __truncdfhf2(double a) {
    return (half)(float)a;
}

namespace fastertransformer {
using namespace rocm;

ROCmDevice::ROCmDevice(const DeviceInitParams& params): DeviceBase(params) {
    RUNTIME_ASSERT_OP_ARG(params.tp_rank == 0, "rocm device doesn't support nccl");
    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipSetDevice(params.device_id));  // TODO(rocm): ensure this is setup every op
    HIP_CHECK(hipStreamCreate(&stream_));
    check_hip_error(hipGetDeviceProperties(&rocmDevProp, device_id_));

    if (params.tp_size > 1) {
        const auto rank = params.tp_rank;
        const auto world_size = params.tp_size;

        nccl_param_.rank_ = rank;
        nccl_param_.world_size_ = world_size;
        auto tcpStore = createTcpStore(
            params.master_ip, params.master_port, world_size, rank);
        const auto nccl_id = &(nccl_param_.nccl_uid_);

        const std::string tp_group_name = "RTP_LLM_TP_GROUP_";
        if (rank == 0) {
            FT_LOG_INFO("rank %d creates nccl uid in group %s.", rank, tp_group_name.c_str());
            NCCLCHECK(ncclGetUniqueId(nccl_id));
            setUniqueId(nccl_id, tp_group_name, tcpStore);
        } else {
            FT_LOG_INFO("rank %d get nccl uid in group %s.", rank, tp_group_name.c_str());
            getUniqueId(nccl_id, tp_group_name, tcpStore);
        }

        FT_LOG_INFO("Initialize NCCL communicators rank %d of %d.", rank, world_size);
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclCommInitRank(&nccl_param_.nccl_comm_, world_size, *nccl_id, rank));
        NCCLCHECK(ncclGroupEnd());
    }    
    
    // Initialize custom all reduce communicator
    // Note: custom all reduce communicator will allocate cuda mem through cudaMalloc, it must be called before allocator init
    if (nccl_param_.world_size_ > 1) {
        FT_LOG_INFO("Initialize custom all reduce communicator rank %d of %d", nccl_param_.rank_, nccl_param_.world_size_);
        std::vector<int> tp_ranks = fcNcclGatherRanks(nccl_param_, stream_);
        custom_allreduce_comm_ = initCustomAllReduceComm(nccl_param_, tp_ranks, stream_);
    }

    allocator_.reset(new Allocator<AllocatorType::ROCM>());
    hostAllocator_.reset(new Allocator<AllocatorType::ROCM_HOST>());
    if (params.device_reserve_memory_bytes) {
        size_t free_bytes, total_bytes;
        HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
        TrackerAllocatorParams tracker_params;
        tracker_params.real_allocator     = allocator_.get();  // TODO(rocm): leak?
        tracker_params.target_track_bytes = params.device_reserve_memory_bytes > 0 ?
                                                params.device_reserve_memory_bytes :
                                                free_bytes + params.device_reserve_memory_bytes;
        tracker_params.align_size         = 16;
        FT_LOG_INFO("rocm device %d has %lu bytes free memory, trying to reserve %lu bytes.",
                    device_id_,
                    free_bytes,
                    tracker_params.target_track_bytes);
        allocator_.reset(new TrackerAllocator(tracker_params));
    }

    if (params.host_reserve_memory_bytes) {
        RUNTIME_ASSERT_OP_ARG(params.host_reserve_memory_bytes > 0,
                              "rocm host memory can not reserve as much as possible (%lu), must specify concrete size.",
                              params.host_reserve_memory_bytes);
        TrackerAllocatorParams tracker_params;
        tracker_params.real_allocator     = hostAllocator_.release();
        tracker_params.target_track_bytes = params.host_reserve_memory_bytes;
        tracker_params.align_size         = 32;
        hostAllocator_.reset(new TrackerAllocator(tracker_params));
    }

    check_hip_error(hipGetDeviceProperties(&device_prop_, device_id_));

    hipblasCreate(&hipblas_handle_);
    hipblasLtCreate(&hipblaslt_handle_);
    
    hipblas_algo_map_.reset(new hipblasAlgoMap(GEMM_CONFIG));
    hipblas_mm_wrapper_.reset(new hipblasMMWrapper(hipblas_handle_,
                                                   hipblaslt_handle_,
                                                   stream_,
                                                   hipblas_algo_map_.get(),
                                                   &hipblas_wrapper_mutex_,
                                                   allocator_.get()));
    hipblas_mm_wrapper_->setGemmConfig(hipblasDatatype_t::HIPBLAS_R_16F,
                                       hipblasDatatype_t::HIPBLAS_R_16F,
                                       hipblasDatatype_t::HIPBLAS_R_16F,
                                       hipblasDatatype_t::HIPBLAS_R_32F);

    fmha_runner_.reset(new rocmFmhaWrapper());
    fmha_runner_->init(stream_);
    moe_runner_.reset(new rocmMoeWrapper());
}

ROCmDevice::~ROCmDevice() {
    hipblas_mm_wrapper_.reset();
    hipStreamDestroy(stream_);
    hipblasDestroy(hipblas_handle_);
    hipblasLtDestroy(hipblaslt_handle_);
    curandstate_buf_.reset();

    if (stream_ != nullptr) {
        HIP_CHECK(hipStreamDestroy(stream_));
    }

    if (nccl_param_.nccl_comm_) {
        ncclCommDestroy(nccl_param_.nccl_comm_);
    }
}

void ROCmDevice::init() {
    DeviceBase::init();
    FT_LOG_INFO("max batch size: %d\n", init_params_.max_batch_size);
    curandstate_buf_ = allocateBuffer(
        {init_params_.max_batch_size * sizeof(curandState_t)}, {"curandstate"});
}

DeviceProperties ROCmDevice::getDeviceProperties() {
    DeviceProperties props;
    props.type = DeviceType::ROCm;
    props.id   = device_id_;
    props.tp_rank = nccl_param_.rank_;
    props.tp_size = nccl_param_.world_size_;
    return props;
}

void ROCmDevice::copy(const CopyParams& params) {
    FT_CHECK_WITH_INFO(params.src.type() == params.dst.type(),
                       "copy dst[%d] and src[%d] need has same type.",
                       params.src.type(), params.dst.type());

    if (params.dst.isQBuffer() && params.src.isQBuffer()) {
        auto dst_ptr = reinterpret_cast<const QBuffer*>(&params.dst);
        auto src_ptr = reinterpret_cast<const QBuffer*>(&params.src);
        copy({dst_ptr->kernel(), src_ptr->kernel()});
        copy({dst_ptr->scales(), src_ptr->scales()});
        copy({dst_ptr->zeros(), src_ptr->zeros()});
        return;
    }

    const auto& src = params.src;
    const auto& dst = params.dst;

    RUNTIME_ASSERT_OP_ARG(src.sizeBytes() == dst.sizeBytes(),
                          "src and dst copy size mismatch: [%s] vs [%s]",
                          src.debugString().c_str(),
                          dst.debugString().c_str());

    if (src.data() == dst.data()) {
        return;
    }

    hipMemcpyKind copyType;
    if (src.where() == MemoryType::MEMORY_GPU && dst.where() != MemoryType::MEMORY_GPU) {
        copyType = hipMemcpyDeviceToHost;
    } else if (src.where() != MemoryType::MEMORY_GPU && dst.where() == MemoryType::MEMORY_GPU) {
        copyType = hipMemcpyHostToDevice;
    } else if (src.where() == MemoryType::MEMORY_GPU && dst.where() == MemoryType::MEMORY_GPU) {
        copyType = hipMemcpyDeviceToDevice;
    } else {
        copyType = hipMemcpyHostToHost;
    }

    (void)hipMemcpyWithStream(dst.data(), src.data(), src.sizeBytes(), copyType, stream_);
    (void)hipStreamSynchronize(stream_);
}

TransposeOutput ROCmDevice::transpose(const TransposeParams& params) {
    const auto& input = params.input;
    const auto data_type = input.type();
    const auto shape = input.shape();

    RUNTIME_ASSERT_OP_ARG(shape.size() == 2 || shape.size() == 3,
        "You can only transpose a 2D buffer, but got [%s]", input.debugString().c_str());
    if (shape.size() == 2) {
        auto output = allocateBuffer({data_type, {shape[1], shape[0]}});
        DISPATCH_CUDA_FUNCTION_GENERAL_TYPE(data_type, invokeTransposeAxis01,
                                            output->data(), input.data(), shape[0], shape[1], stream_
                                            );
        return std::move(output);
    } else {
        auto output = allocateBuffer({data_type, {shape[1], shape[0], shape[2]}});
        DISPATCH_CUDA_FUNCTION_GENERAL_TYPE(data_type, invokeTransposeAxis012,
                                            output->data(), input.data(), shape[0], shape[1], shape[2], stream_
                                            );
        return std::move(output);
    }
}

void ROCmDevice::syncAndCheck() {
    (void)hipDeviceSynchronize();
}

void ROCmDevice::syncCommunication(bool timeout) {
    if (nccl_param_.world_size_ > 1) {
        FT_LOG_DEBUG("Synchronize NCCL communicators rank %d of %d.", nccl_param_.rank_, nccl_param_.world_size_);
        ftNcclStreamSynchronize(nccl_param_, stream_, timeout);
    }
}

SelectOutput ROCmDevice::select(const SelectParams& params) {
    if ((params.input.where() != MemoryType::MEMORY_GPU) || (params.dim > 0)) {
        return DeviceBase::select(params);
    }

    RUNTIME_ASSERT_OP_ARG(params.index.type() == DataType::TYPE_INT32, "Select index must be int32.");
    RUNTIME_ASSERT_OP_ARG(params.dim == 0, "select op tmp only support dim == 0");

    const auto& input         = params.input;
    auto        output_shape  = input.shape();
    output_shape[0]           = params.index.size();
    auto num_selected_element = input.size() / input.shape()[0];
    auto output               = allocateBuffer({input.type(), output_shape});
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(input.type(),
                                     invokeLookupHiddenStateOfLastToken,
                                     output->data(),
                                     input.data(),
                                     (int*)params.index.data(),
                                     (int)params.index.size(),
                                     num_selected_element,
                                     0,
                                     stream_);
    return std::move(output);
}

BufferPtr ROCmDevice::embeddingLookup(const EmbeddingLookupParams& params) {
    const auto& tokens           = params.combo_tokens;
    const auto& embedding_table  = params.embedding_table;
    const auto& position_ids     = params.position_ids;
    const auto& postition_table  = params.position_table;
    const auto& token_types      = params.token_types;
    const auto& token_type_table = params.token_type_table;

    const auto token_num   = tokens.size();
    const auto hidden_size = embedding_table.shape()[1];
    const auto data_type   = embedding_table.type();

    auto embeddings = allocateBuffer({data_type, {token_num, hidden_size}}, {"embedding"});

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                     invokeEmebeddingLookup,
                                     embeddings->data(),
                                     embedding_table.data(),
                                     params.input_embedding_scalar,
                                     postition_table.has_value() ? postition_table.value().get().data() : nullptr,
                                     token_type_table.has_value() ? token_type_table.value().get().data() : nullptr,
                                     tokens.data<int>(),
                                     position_ids.has_value() ? position_ids.value().get().data<int>() : nullptr,
                                     token_types.has_value() ? token_types.value().get().data<int>() : nullptr,
                                     nullptr, //mask
                                     token_num,
                                     hidden_size,
                                     stream_);

    return std::move(embeddings);
}

LayernormOutput ROCmDevice::layernorm(const LayernormParams& params) {
    BufferPtr input = params.input;
    BufferPtr norm_output = input;
    BufferPtr output = params.before_norm_output;
    float* scales_ptr = nullptr;
    int8_t* quant_output = nullptr;
    const auto data_type = input->type();
    const auto m = input->shape()[0];
    const auto n = input->shape()[1];
    auto norm_weight = params.norm_weight;
    const auto& gamma = norm_weight ? norm_weight->get().gamma.get()->data() : nullptr;
    const auto& beta = (norm_weight && norm_weight->get().beta) ? norm_weight->get().beta.get()->data() : nullptr;
    const auto norm_type = params.norm_type;
    const auto eps       = params.eps;
    const auto& weights  = params.norm_weight;

    if (!params.is_inplace && params.qscheme == QScheme::NoQuantize) {
        norm_output = allocateBufferLike(*params.input);
    } else if (params.qscheme == Qint8PerChannelLastAxis) {
        auto kernel = allocateBuffer({DataType::TYPE_INT8,
                                            {input->shape()},
                                            AllocationType::DEVICE},
                                            {"kernel"});
        auto scales = allocateBuffer({DataType::TYPE_FP32,
                                        {input->shape()[1]},
                                        AllocationType::DEVICE},
                                        {"scales"});
        norm_output = BufferPtr(new QBuffer(std::move(kernel),
                                            std::move(scales),
                                            std::move(BufferPtr(
                                                new Buffer(MemoryType::MEMORY_GPU,
                                                DataType::TYPE_INVALID,
                                                {0},
                                                nullptr)))));
        quant_output = std::dynamic_pointer_cast<QBuffer>(norm_output)->kernel().data<int8_t>();
        scales_ptr = std::dynamic_pointer_cast<QBuffer>(norm_output)->scalesData<float>();
    }

    if (!weights.has_value()) {
        if (params.alpha != 0 || (norm_type == NormType::alphanorm)) {
            const auto alpha = params.alpha;
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                             invokeAlphaAddBiasResidual,
                                             norm_output->data(),
                                             input->data(),
                                             params.residual1 ? params.residual1.value().get().data() : nullptr,
                                             params.bias ? params.bias.value().get().data() : nullptr,
                                             params.alpha,
                                             m,
                                             n,
                                             stream_);
            sync_check_cuda_error();
            return LayernormOutput({std::move(norm_output), nullptr});
        } else if (params.bias.has_value() || params.residual1.has_value() || params.residual2.has_value()) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                             invokeAddBiasResidual,
                                             output->data(),
                                             input->data(),
                                             params.residual1 ? params.residual1.value().get().data() : nullptr,
                                             params.residual2 ? params.residual2.value().get().data() : nullptr,
                                             params.bias.has_value() ? params.bias.value().get().data() : nullptr,
                                             nullptr,  // scale_inter
                                             nullptr,  // scale_out
                                             m,
                                             n,
                                             stream_);
            sync_check_cuda_error();
            return LayernormOutput({std::move(norm_output), nullptr});
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    }

    if (!(norm_type == NormType::layernorm || norm_type == NormType::rmsnorm)) {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }

    if (params.residual1.has_value() || params.bias.has_value()) {
        //const auto& add_bias_output = params.bias ? params.bias.value() : output;
        if (params.norm_type == NormType::layernorm) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                             invokeGeneralAddBiasResidualLayerNorm,
                                             //add_bias_output.data(),
                                             (params.before_norm_output == nullptr) ? input->data() : params.before_norm_output->data(),
                                             norm_output->data(),
                                             input->data(),
                                             params.bias ? params.bias.value().get().data() : nullptr,
                                             params.residual1 ? params.residual1.value().get().data() : nullptr,
                                             gamma,
                                             beta,
                                             eps,
                                             m,
                                             n,
                                             stream_,
                                             true,     // use_diff_of_squares
                                             nullptr,  // scale
                                             scales_ptr,  // dynamic_scale
                                             quant_output,   // out_quant
                                             params.return_normed_output
            );
            sync_check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        } else if (params.norm_type == NormType::rmsnorm) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                             invokeAddBiasResidualRmsNorm,
                                             //add_bias_output.data(),
                                             (params.before_norm_output == nullptr) ? input->data() : params.before_norm_output->data(), // or null
                                             norm_output->data(),
                                             input->data(),
                                             params.bias ? params.bias.value().get().data() : nullptr,
                                             params.residual1 ? params.residual1.value().get().data() : nullptr,
                                             gamma,
                                             beta,
                                             eps,
                                             m,
                                             n,
                                             stream_,
                                             nullptr,  // scale
                                             scales_ptr,  // dynamic_scale
                                             quant_output   // out_quant
            );
            sync_check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    } else {
        if (params.norm_type == NormType::layernorm) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                             invokeGeneralLayerNorm,
                                             nullptr,
                                             norm_output->data(),
                                             input->data(),
                                             gamma,
                                             beta,
                                             eps,
                                             m,
                                             n,
                                             stream_,
                                             true,     // use_diff_of_squares
                                             nullptr,  // scale
                                             scales_ptr,  // dynamic_scale
                                             quant_output,   // out_quant
                                             params.return_normed_output
            );
            sync_check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        } else if (params.norm_type == NormType::rmsnorm) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                             invokeGeneralRmsNorm,
                                             norm_output->data(),
                                             input->data(),
                                             gamma,
                                             beta,
                                             eps,
                                             m,
                                             n,
                                             stream_,
                                             nullptr,  // scale
                                             scales_ptr,  // dynamic_scale
                                             quant_output   // out_quant
            );
            sync_check_cuda_error();
            return LayernormOutput({norm_output, params.before_norm_output});
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    }
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

#define ARGS_DISPATCH(Atype, Dtype, out, bias, gate, gate_bias, m, n, stream)                                          \
    do {                                                                                                               \
        invokeGenericActivation<Atype>((Dtype*)out,                                                                    \
                                       (const Dtype*)bias,                                                             \
                                       (const Dtype*)gate,                                                             \
                                       (const Dtype*)gate_bias,                                                        \
                                       (const int*)nullptr,                                                            \
                                       (const Dtype*)nullptr,                                                          \
                                       (int)m,                                                                         \
                                       (int)n,                                                                         \
                                       0,                                                                              \
                                       (const float*)nullptr,                                                          \
                                       (const float*)nullptr,                                                          \
                                       (const Dtype*)nullptr,                                                          \
                                       stream);                                                                        \
    } while (0)

#define ATYPE_DISPATCH(Dtype, Atype, KERNEL, ...)                                                                      \
    do {                                                                                                               \
        if (Atype == ActivationType::Silu) {                                                                           \
            KERNEL(SiluActivation, Dtype, __VA_ARGS__);                                                                \
        } else if (Atype == ActivationType::Gelu) {                                                                    \
            KERNEL(GeluActivation, Dtype, __VA_ARGS__);                                                                \
        } else if (Atype == ActivationType::Swiglu) {                                                                  \
            KERNEL(SiluActivation, Dtype, __VA_ARGS__);                                                                \
        } else {                                                                                                       \
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);                                                       \
        }                                                                                                              \
    } while (0)

#define DTYPE_DISPATCH(Dtype, ...)                                                                                     \
    do {                                                                                                               \
        if (Dtype == DataType::TYPE_FP16) {                                                                            \
            ATYPE_DISPATCH(half, __VA_ARGS__);                                                                         \
        } else {                                                                                                       \
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);                                                       \
        }                                                                                                              \
    } while (0)

#define DISPATCH(Dtype, Atype, ...)                                                                                    \
    do {                                                                                                               \
        DTYPE_DISPATCH(Dtype, Atype, ARGS_DISPATCH, __VA_ARGS__);                                                      \
    } while (0)


// AttentionModuleOutput ROCmDevice::contextAttention(const AttentionModuleParams& params) {
//     std::cerr << "contextAttention\n";
// }

BufferPtr ROCmDevice::testVecAdd(const BufferPtr a, const BufferPtr b) {
    BufferPtr           output;
    DataType            dtype  = a.get()->type();
    std::vector<size_t> dshape = a.get()->shape();

    output = allocateBuffer({dtype, dshape, AllocationType::DEVICE}, {"vec_add_rslt"});
    invokeHelloWorld<float>((const float*)(a.get()->data()),
                            ((const float*)b.get()->data()),
                            ((float*)output.get()->data()),
                            output.get()->size(),
                            stream_);

    return output;
}

DeviceStatus ROCmDevice::getDeviceStatus() {
    DeviceStatus status;

    size_t total_bytes;
    auto error = hipMemGetInfo(&status.device_memory_status.free_bytes, &total_bytes);
    status.device_memory_status.used_bytes = total_bytes - status.device_memory_status.free_bytes;

    const auto buffer_status = queryBufferStatus();
    status.device_memory_status.allocated_bytes = buffer_status.device_allocated_bytes;
    status.device_memory_status.preserved_bytes = buffer_status.device_preserved_bytes;
    status.host_memory_status.allocated_bytes = buffer_status.host_allocated_bytes;
    status.device_memory_status.available_bytes = status.device_memory_status.free_bytes + status.device_memory_status.preserved_bytes;

    return status;
}

RTP_LLM_REGISTER_DEVICE(ROCm);

}  // namespace fastertransformer
