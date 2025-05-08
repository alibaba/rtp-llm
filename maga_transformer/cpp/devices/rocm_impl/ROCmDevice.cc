#include "maga_transformer/cpp/devices/rocm_impl/ROCmDevice.h"
#include "maga_transformer/cpp/devices/rocm_impl/ROCmAllocator.h"
#include "maga_transformer/cpp/core/TrackerAllocator.h"
#include "maga_transformer/cpp/devices/DeviceFactory.h"
#include "maga_transformer/cpp/kernels/gpt_kernels.h"
#include "maga_transformer/cpp/kernels/add_residual_kernels.h"
#include "maga_transformer/cpp/utils/ShapeCheck.h"
#include "maga_transformer/cpp/cuda/Dispatch.h"
#include <cstring>

#include "maga_transformer/cpp/kernels/hello_world.h"
#include "maga_transformer/cpp/kernels/rmsnormKernels.h"
#include "maga_transformer/cpp/kernels/activation_kernels.h"
#include "maga_transformer/cpp/cuda/nccl/nccl_utils_torch.h"
#include "maga_transformer/cpp/cuda/nccl/nccl_utils.h"

// TODO(rocm): Idealy we just link compiler_rt for this symbol.
extern "C" half __truncdfhf2(double a) {
    return (half)(float)a;
}

namespace rtp_llm {
using namespace rocm;

ROCmDevice::ROCmDevice(const DeviceInitParams& params): DeviceBase(params) {
    ROCM_CHECK(hipSetDevice(params.device_id));
    stream_ = at::hip::getCurrentHIPStream().stream();
    ROCM_CHECK(hipStreamCreate(&assist_stream_));
    current_stream_ = stream_;
    ROCM_CHECK(hipGetDeviceProperties(&rocmDevProp, device_id_));

    if (params.tp_size > 1) {
        const auto rank       = params.tp_rank;
        const auto world_size = params.tp_size;

        nccl_param_.rank_       = rank;
        nccl_param_.world_size_ = world_size;
        auto       tcpStore     = createTcpStore(params.master_ip, params.tp_master_port, world_size, rank);
        const auto nccl_id      = &(nccl_param_.nccl_uid_);

        const std::string tp_group_name = "RTP_LLM_TP_GROUP_";
        if (rank == 0) {
            RTP_LLM_LOG_INFO("rank %d creates nccl uid in group %s.", rank, tp_group_name.c_str());
            NCCLCHECK(ncclGetUniqueId(nccl_id));
            setUniqueId(nccl_id, tp_group_name, tcpStore);
        } else {
            RTP_LLM_LOG_INFO("rank %d get nccl uid in group %s.", rank, tp_group_name.c_str());
            getUniqueId(nccl_id, tp_group_name, tcpStore);
        }

        RTP_LLM_LOG_INFO("Initialize NCCL communicators rank %d of %d.", rank, world_size);
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclCommInitRank(&nccl_param_.nccl_comm_, world_size, *nccl_id, rank));
        NCCLCHECK(ncclGroupEnd());
    }

    // Initialize custom all reduce communicator
    // Note: custom all reduce communicator will allocate cuda mem through cudaMalloc, it must be called before allocator init
    if (nccl_param_.world_size_ > 1) {
        RTP_LLM_LOG_INFO("Initialize custom all reduce communicator rank %d of %d", nccl_param_.rank_, nccl_param_.world_size_);
        std::vector<size_t> tp_ranks = fcNcclGatherRanks(nccl_param_, stream_);
        // custom_allreduce_comm_ = initCustomAllReduceComm(nccl_param_, tp_ranks, stream_);
    }

    auto allocator_ptr     = new Allocator<AllocatorType::ROCM>();
    auto hostAllocator_ptr = new Allocator<AllocatorType::ROCM_HOST>();

    if (params.device_reserve_memory_bytes) {
        syncAndCheck();
        size_t free_bytes, total_bytes;
        ROCM_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
        TrackerAllocatorParams tracker_params;
        tracker_params.real_allocator     = allocator_ptr;
        tracker_params.target_track_bytes = params.device_reserve_memory_bytes > 0 ?
                                                params.device_reserve_memory_bytes :
                                                free_bytes + params.device_reserve_memory_bytes - ROCM_RUNTIME_MEM_SIZE;
        tracker_params.align_size         = 16;
        RTP_LLM_LOG_INFO("[ROCM] total = %.2f(GB), free = %.2f(GB), reserve = %.2f(GB), track = %.2f(GB)\n",
                    total_bytes / 1024.0 / 1024.0 / 1024.0,
                    free_bytes / 1024.0 / 1024.0 / 1024.0,
                    params.device_reserve_memory_bytes / 1024.0 / 1024.0 / 1024.0,
                    tracker_params.target_track_bytes / 1024.0 / 1024.0 / 1024.0);
        assert(tracker_params.target_track_bytes <= free_bytes && tracker_params.target_track_bytes > 0);
        allocator_.reset(new TrackerAllocator(tracker_params));
        syncAndCheck();
    } else {
        allocator_.reset(allocator_ptr);
    }

    origin_torch_hip_allocator_ = at::hip::HIPCachingAllocator::allocator;
    initTorchHIPAllocator(allocator_.get(), device_id_);
    // change torch hip gpu allocate
    at::hip::HIPCachingAllocator::allocator.store(getTorchHIPAllocator());

    if (params.host_reserve_memory_bytes) {
        RUNTIME_ASSERT_OP_ARG(params.host_reserve_memory_bytes > 0,
                              "rocm host memory can not reserve as much as possible (%lu), must specify concrete size.",
                              params.host_reserve_memory_bytes);
        TrackerAllocatorParams tracker_params;
        tracker_params.real_allocator     = hostAllocator_ptr;
        tracker_params.target_track_bytes = params.host_reserve_memory_bytes;
        tracker_params.align_size         = 32;
        hostAllocator_.reset(new TrackerAllocator(tracker_params));
    } else {
        hostAllocator_.reset(hostAllocator_ptr);
    }

    ROCM_CHECK(hipGetDeviceProperties(&device_prop_, device_id_));

    ROCM_CHECK(hipblasCreate(&hipblas_handle_));
    ROCM_CHECK(hipblasLtCreate(&hipblaslt_handle_));

    hipblas_mm_wrapper_.reset(new hipblasMMWrapper(hipblas_handle_, hipblaslt_handle_, stream_, allocator_ptr));
    hipblas_mm_wrapper_->setGemmConfig(hipDataType::HIP_R_16F,
                                       hipDataType::HIP_R_16F,
                                       hipDataType::HIP_R_16F,
                                       hipDataType::HIP_R_32F);

    hipblas_mm_wrapper_->setStream(stream_);
    fmha_runner_.reset(new rocmFmhaWrapper());
    fmha_runner_->init(stream_);
    moe_runner_.reset(new rocmMoeWrapper());
    ck_gemm_runner_.reset(new rocmCKGemmWrapper());

    // select mla type
    if (params.mla_ops_type != MlaOpsType::AUTO) {
        mla_ops_type = params.mla_ops_type;
    } else {
        mla_ops_type = device_prop_.major >= 9 ? MlaOpsType::FLASH_MLA : MlaOpsType::FLASH_INFER;
    }
}

ROCmDevice::~ROCmDevice() {
    // change torch hip gpu allocate
    if (origin_torch_hip_allocator_) {
        at::hip::HIPCachingAllocator::allocator.store(origin_torch_hip_allocator_);
    }
    hipblas_mm_wrapper_.reset();
    ROCM_CHECK(hipStreamDestroy(stream_));
    ROCM_CHECK(hipStreamDestroy(assist_stream_));
    ROCM_CHECK(hipblasDestroy(hipblas_handle_));
    ROCM_CHECK(hipblasLtDestroy(hipblaslt_handle_));
    curandstate_buf_.reset();

    if (stream_ != nullptr) {
        ROCM_CHECK(hipStreamDestroy(stream_));
    }

    if (nccl_param_.nccl_comm_) {
        ncclCommDestroy(nccl_param_.nccl_comm_);
    }
}

void ROCmDevice::init() {
    DeviceBase::init();
    RTP_LLM_LOG_INFO("max batch size: %d", init_params_.max_batch_size);
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

DevicePrepOutput ROCmDevice::prepareModelRun(const DevicePrepParams& params) {
    DevicePrepOutput output;
    output.need_mask = false;
    output.decode_flash_infer_attn_params = FlashInferAttnParams::prepareDecodeFlashInferAttnParams(
             this,
             params.configs,
             params.sequence_lengths,
             params.input_lengths,
             params.kv_cache_block_id,
             params.attn_dtype);
     output.prefill_flash_infer_attn_params = FlashInferAttnParams::preparePrefillFlashInferAttnParams(
             this,
             params.configs,
             params.prefix_lengths,
             params.sequence_lengths,
             params.input_lengths,
             params.kv_cache_block_id,
             params.attn_dtype
     );
    return std::move(output);
}

void ROCmDevice::copy(const CopyParams& params) {
    ROCM_CHECK_VALUE(params.src.type() == params.dst.type(),
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

    // ROCM_CHECK(hipMemcpy(dst.data(), src.data(), src.sizeBytes(), copyType));
    if (copyType == hipMemcpyHostToHost) {
        std::memcpy(dst.data(), src.data(), src.sizeBytes());
    } else {
        ROCM_CHECK(hipMemcpyAsync(dst.data(), src.data(), src.sizeBytes(), copyType, stream_));
    }

    if (copyType == hipMemcpyDeviceToHost) {
        ROCM_CHECK(hipStreamSynchronize(stream_));
    }
}

void ROCmDevice::noBlockCopy(const CopyParams& params) {
    params.check();
    const auto& src = params.src;
    const auto& dst = params.dst;
    ROCM_CHECK(hipMemcpyAsync(dst.data(), src.data(), src.sizeBytes(), hipMemcpyDefault, no_block_copy_stream_));
    ROCM_CHECK(hipStreamSynchronize(no_block_copy_stream_));
}

void ROCmDevice::bufMemset(Buffer& buf, int val, DeviceStream stream) {
    if (buf.where() == MemoryType::MEMORY_CPU || buf.where() == MemoryType::MEMORY_CPU_PINNED) {
        std::memset(buf.data(), val, buf.sizeBytes());
    } else {
        ROCM_CHECK(hipMemsetAsync(buf.data(), val, buf.sizeBytes(), stream_));
    }
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
    syncCommunication();
    ROCM_CHECK(hipDeviceSynchronize());
    ROCM_SYNC_AND_CHECK();
}

void ROCmDevice::syncCommunication(bool timeout) {
    if (nccl_param_.world_size_ > 1) {
        RTP_LLM_LOG_DEBUG("Synchronize NCCL communicators rank %d of %d.", nccl_param_.rank_, nccl_param_.world_size_);
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
    const auto& tokens = params.combo_tokens;
    const auto& embedding_table = params.embedding_table;
    const auto& mask = params.text_tokens_mask;
    const auto& position_ids = params.position_ids;
    const auto& postition_table = params.position_table;
    const auto& token_types = params.token_types;
    const auto& token_type_table = params.token_type_table;

    const auto token_num = tokens.size();
    const auto hidden_size = embedding_table.shape()[1];
    const auto data_type = embedding_table.type();

    auto embeddings = allocateBuffer({data_type, {token_num, hidden_size}}, {"embedding"});

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeEmebeddingLookup,
        embeddings->data(),
        embedding_table.data(),
        params.input_embedding_scalar,
        postition_table.has_value() ? postition_table.value().get().data() : nullptr,
        token_type_table.has_value() ? token_type_table.value().get().data() : nullptr,
        tokens.data<int>(),
        position_ids.has_value() ? position_ids.value().get().data<int>() : nullptr,
        token_types.has_value() ? token_types.value().get().data<int>() : nullptr,
        mask.has_value() ? mask.value().get().data<int>() : nullptr,
        token_num,
        hidden_size,
        stream_
    );

    return embeddings;
}

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

MemoryStatus ROCmDevice::getDeviceMemoryStatus() {
    MemoryStatus status;
    size_t total_bytes;
    auto   error = hipMemGetInfo(&status.free_bytes, &total_bytes);
    RTP_LLM_CHECK(error == hipSuccess);
    status.used_bytes = total_bytes - status.free_bytes;
    return status;
}

static float cpu_half2float(uint16_t h) {
    unsigned sign     = ((((uint16_t)h) >> 15) & 1);
    unsigned exponent = ((((uint16_t)h) >> 10) & 0x1f);
    unsigned mantissa = ((((uint16_t)h) & 0x3ff) << 13);

    if (exponent == 0x1f) { /* NaN or Inf */
        mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
        exponent = 0xff;
    } else if (!exponent) { /* Denorm or Zero */
        if (mantissa) {
            unsigned int msb;
            exponent = 0x71;
            do {
                msb = (mantissa & 0x400000);
                mantissa <<= 1; /* normalize */
                --exponent;
            } while (!msb);
            mantissa &= 0x7fffff; /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70;
    }

    int temp = ((sign << 31) | (exponent << 23) | mantissa);
    return *((float*)((void*)&temp));
}
void ROCmDevice::printBuffer(const BufferPtr b) {
    BufferPtr hb = b;
    if (b.get()->where() == MemoryType::MEMORY_GPU) {
        hb = clone({*b, AllocationType::HOST});
    }

    if (b.get()->type() == DataType::TYPE_FP16) {
        printf("%s", hb.get()->debugString().c_str());
        uint16_t* phb = (uint16_t*)(hb.get()->data());
        for (uint32_t i = 0; i < hb.get()->size(); i++) {
            if (i % hb.get()->shape()[0] == 0)
                printf("\n");
            uint16_t val = phb[i];
            printf("%.2e, ", cpu_half2float(val));
        }
        printf("\n");
    } else if (b.get()->type() == DataType::TYPE_INT4X2) {
        printf("%s", hb.get()->debugString().c_str());
        uint16_t* phb = (uint16_t*)(hb.get()->data());
        for (uint32_t i = 0; i < hb.get()->size(); i++) {
            if (i % (hb.get()->shape()[0] / 2) == 0)
                printf("\n");
            uint8_t val = phb[i];
            printf("%d, %d, ", (val & 0x0F), ((val & 0xF0) >> 4));
        }
        printf("\n");
    } else if (b.get()->type() == DataType::TYPE_QINT4X2) {
        const QBuffer& qb = reinterpret_cast<const QBuffer&>(*hb);

        size_t kernel_dim0 = qb.kernel().shape()[0];
        size_t scales_dim0 = qb.scales().shape()[0];
        size_t group_size  = (kernel_dim0 / scales_dim0);

        printf("QBuffer( group_size = %zu\n [\n", group_size);
        printf("   kernel: ");
        printBuffer(BufferPtr(new Buffer(qb.kernel())));
        printf("   scales: ");
        printBuffer(BufferPtr(new Buffer(qb.scales())));
        printf("   zeros: ");
        printBuffer(BufferPtr(new Buffer(qb.zeros())));
        printf("])\n");
    }
}

RTP_LLM_REGISTER_DEVICE(ROCm);

DeviceEventPtr ROCmDevice::createEvent() {
    return std::make_unique<ROCmEvent>(stream_);
}

ROCmEvent::ROCmEvent(hipStream_t stream) : stream_(stream) {
    ROCM_CHECK(hipEventCreate(&event_));
    ROCM_CHECK(hipEventRecord(event_, stream));
}

ROCmEvent::~ROCmEvent() {
    ROCM_CHECK(hipEventDestroy(event_));
}

void ROCmEvent::synchronize() const {
    ROCM_CHECK(hipEventSynchronize(event_));
}

}  // namespace rtp_llm
