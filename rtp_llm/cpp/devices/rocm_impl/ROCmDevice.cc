#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmAllocator.h"
#include "rtp_llm/cpp/core/TrackerAllocator.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/kernels/eplb/experts_stats_kernels.h"
#include "rtp_llm/cpp/kernels/add_residual_kernels.h"
#include "rtp_llm/cpp/devices/ShapeCheck.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include <cstring>

#include "rtp_llm/cpp/kernels/rmsnormKernels.h"
#include "rtp_llm/cpp/kernels/activation_kernels.h"
#include "rtp_llm/cpp/kernels/tensor_ops_kernels.h"
#include "rtp_llm/cpp/kernels/embedding_kernels.h"
#include "rtp_llm/cpp/kernels/mask_logits.h"
#include "rtp_llm/cpp/cuda/nccl/nccl_utils_torch.h"
#include "rtp_llm/cpp/cuda/nccl/nccl_utils.h"
#include "rtp_llm/cpp/rocm/speculative_sampling/sampling.cuh"

#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"

// TODO(rocm): Idealy we just link compiler_rt for this symbol.
extern "C" half __truncdfhf2(double a) {
    return (half)(float)a;
}

namespace rtp_llm {
using namespace rocm;

ROCmDevice::ROCmDevice(const DeviceInitParams& params): DeviceBase(params) {
    ROCM_CHECK(hipSetDevice(params.device_id));
    torch_default_stream_ =
        std::make_unique<at::hip::HIPStreamMasqueradingAsCUDA>(at::hip::getDefaultHIPStreamMasqueradingAsCUDA());
    stream_ = torch_default_stream_->stream();
    ROCM_CHECK(hipStreamCreate(&assist_stream_));
    current_stream_ = stream_;
    ROCM_CHECK(hipGetDeviceProperties(&rocmDevProp, device_id_));

    if (params.tp_size > 1) {
        auto master_ip = params.master_ip;
        if (params.dp_size > 1) {
            master_ip = "127.0.0.1";
        }
        initNcclParam(
            params.tp_rank, params.tp_size, master_ip, params.tp_master_port, "RTP_LLM_TP_GROUP_", tp_nccl_param_);
    }
    if (params.ffn_tp_size > 1) {
        if (params.ffn_tp_size != params.tp_size) {
            initNcclParam(params.ffn_tp_rank,
                          params.ffn_tp_size,
                          params.master_ip,
                          params.ffn_tp_master_port - params.tp_rank / params.ffn_tp_size,
                          "RTP_LLM_FFN_TP_GROUP_",
                          ffn_tp_nccl_param_);
        } else {
            ffn_tp_nccl_param_ = tp_nccl_param_;
        }
    }
    if (params.ep_size > 1) {
        initNcclParam(params.dp_rank * params.tp_size + params.tp_rank,
                      params.dp_size * params.tp_size,
                      params.master_ip,
                      params.dp_tp_master_port,
                      "RTP_LLM_DP_TP_GROUP_",
                      dp_tp_nccl_param_);
    }

    // Initialize custom/quick all reduce communicator
    // Note: custom all reduce communicator will allocate cuda mem through cudaMalloc, it must be called before
    // allocator init
    if (tp_nccl_param_.world_size_ > 1) {
        auto&               nccl_param = tp_nccl_param_;
        std::vector<size_t> tp_ranks   = fcNcclGatherRanks(nccl_param, stream_);
        // Initialization may fail, and the variable will still be nullptr. When allreduce is called, it will fall back to the normal allreduce.
        custom_allreduce_comm_         = initCustomAllReduceComm(nccl_param, tp_ranks, stream_, params.hw_kernel_config);
        quick_allreduce_comm_          = initQuickAllReduceComm(nccl_param, tp_ranks, stream_);
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
    initTorchHIPAllocator(allocator_.get(), device_id_, this);
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

    hipblas_mm_wrapper_.reset(new hipblasMMWrapper(
        hipblas_handle_, hipblaslt_handle_, stream_, allocator_ptr, init_params_.hw_kernel_config));
    hipblas_mm_wrapper_->setGemmConfig(
        hipDataType::HIP_R_16F, hipDataType::HIP_R_16F, hipDataType::HIP_R_16F, hipDataType::HIP_R_32F);

    hipblas_mm_wrapper_->setStream(stream_);
    aiter_wrapper_.reset(new AiterWrapper(params));
    fmha_runner_.reset(new rocmFmhaWrapper());
    fmha_runner_->init(stream_);
    // moe_runner_.reset(new rocmMoeWrapper());
    ck_gemm_runner_.reset(new rocmCKGemmWrapper());
    ck_w8a8_gelu_gemm_runner_.reset(new rocmCKW8A8GeluGemmWrapper());

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

    if (stream_ != nullptr) {
        ROCM_CHECK(hipStreamDestroy(stream_));
    }

    if (ffn_tp_nccl_param_ != tp_nccl_param_ && ffn_tp_nccl_param_.nccl_comm_) {
        ncclCommDestroy(ffn_tp_nccl_param_.nccl_comm_);
    }
    if (tp_nccl_param_.nccl_comm_) {
        ncclCommDestroy(tp_nccl_param_.nccl_comm_);
    }
    if (dp_nccl_param_.nccl_comm_) {
        ncclCommDestroy(dp_nccl_param_.nccl_comm_);
    }
    if (dp_tp_nccl_param_.nccl_comm_) {
        ncclCommDestroy(dp_tp_nccl_param_.nccl_comm_);
    }
}

void ROCmDevice::init() {
    DeviceBase::init();
#ifdef ENABLE_DEEP_EP
    if (init_params_.use_deepep_moe) {
        if (!initDeepEPBuffer()) {
            RTP_LLM_CHECK_WITH_INFO(false, "init deepep buffer failed");
        } else {
            RTP_LLM_LOG_INFO("init deepep buffer success");
        }
    }
#else
    RTP_LLM_LOG_INFO("deep_ep is not enabled");
#endif
}

DeviceProperties ROCmDevice::getDeviceProperties() {
    static DeviceProperties* prop = nullptr;
    if (prop == nullptr) {
        prop                           = new DeviceProperties();
        prop->type                     = DeviceType::ROCm;
        prop->id                       = device_id_;
        prop->tp_rank                  = init_params_.tp_rank;
        prop->tp_size                  = init_params_.tp_size;
        prop->dp_rank                  = init_params_.dp_rank;
        prop->dp_size                  = init_params_.dp_size;
        prop->enable_comm_overlap      = init_params_.enable_comm_overlap;
        prop->enable_layer_micro_batch = init_params_.enable_layer_micro_batch;
        prop->enable_sp                = init_params_.enable_sp;
        prop->overlap_math_sm_count    = init_params_.device_resource_config.overlap_math_sm_count;
        prop->overlap_comm_type        = init_params_.device_resource_config.overlap_comm_type;
        prop->ffn_tp_size              = init_params_.ffn_tp_size;
        prop->ffn_tp_rank              = init_params_.ffn_tp_rank;
        prop->m_split                  = init_params_.m_split;
        prop->use_all_gather           = init_params_.use_all_gather;
    }
    return *prop;
}

bool ROCmDevice::checkSpecDecode(const DevicePrepParams& params, bool skip_no_prefix) {
    bool has_prefix = params.prefix_lengths != nullptr && params.prefix_lengths->size();
    if (!params.configs.use_mla && has_prefix) {
        auto input_lengths_host = params.input_lengths->slice(params.decoder_batch_size, params.context_batch_size);
        const int batch_size    = input_lengths_host->shape()[0];
        size_t    sp_seq_len    = init_params_.sp_config.gen_num_per_cycle;
        size_t    max_context_input_seq_len =
            *std::max_element(input_lengths_host->data<int>(), input_lengths_host->data<int>() + batch_size);
        size_t min_prefix_len =
            *std::min_element(params.prefix_lengths->data<int>(), params.prefix_lengths->data<int>() + batch_size);

        RTP_LLM_LOG_DEBUG("max_context_input_seq_len %d min_prefix_len %d sp_seq_len %d.",
                          max_context_input_seq_len,
                          min_prefix_len,
                          sp_seq_len);

        if (skip_no_prefix && (min_prefix_len == 0 || max_context_input_seq_len > sp_seq_len + 1)) {
            return false;
        }
    }
    return has_prefix;
}

DevicePrepOutput ROCmDevice::prepareModelRun(const DevicePrepParams& params) {
    DevicePrepOutput output;
    output.need_mask                = false;
    output.decode_flash_infer_attn  = FlashInferAttnParams::prepareDecodeFlashInferAttnParams(this,
                                                                                             params.configs,
                                                                                             params.sequence_lengths,
                                                                                             params.input_lengths,
                                                                                             params.kv_cache_block_id,
                                                                                             params.attn_dtype);
    output.prefill_flash_infer_attn = FlashInferAttnParams::preparePrefillFlashInferAttnParams(this,
                                                                                               params.configs,
                                                                                               params.prefix_lengths,
                                                                                               params.sequence_lengths,
                                                                                               params.input_lengths,
                                                                                               params.kv_cache_block_id,
                                                                                               params.attn_dtype);
    const int kv_cache_offset       = params.kv_cache ? params.kv_cache->shape()[0] * params.kv_cache->shape()[1] : 0;
    auto      decode_kv_cache_block_id_d =
        params.kv_cache_block_id_d ? params.kv_cache_block_id_d->slice(0, params.decoder_batch_size) : nullptr;
    output.decode_aiter_attn = AiterAttnParams::prepareDecodeAiterAttnParams(
        this, params.sequence_lengths, params.configs, kv_cache_offset, decode_kv_cache_block_id_d);
    use_mtp_pa_ = checkSpecDecode(params);
    return std::move(output);
}

void ROCmDevice::copy(const CopyParams& params) {
    ROCM_CHECK_VALUE(params.src.type() == params.dst.type(),
                     "copy src[%d] and dst[%d] need has same type.",
                     params.src.type(),
                     params.dst.type());

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
        if (params.async) {
            ROCM_CHECK(hipMemcpyAsync(dst.data(), src.data(), src.sizeBytes(), copyType, stream_));
        } else {
            ROCM_CHECK(hipMemcpyWithStream(dst.data(), src.data(), src.sizeBytes(), copyType, stream_));
        }
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

hipStream_t ROCmDevice::getStream(DeviceStream stream) {
    switch (stream) {
        default:
            return stream_;
    }
}

void ROCmDevice::bufMemset(Buffer& buf, int val, DeviceStream stream) {
    if (buf.where() == MemoryType::MEMORY_CPU || buf.where() == MemoryType::MEMORY_CPU_PINNED) {
        std::memset(buf.data(), val, buf.sizeBytes());
    } else {
        hipStream_t cur_stream = getStream(stream);
        ROCM_CHECK(hipMemsetAsync(buf.data(), val, buf.sizeBytes(), cur_stream));
    }
}

TransposeOutput ROCmDevice::transpose(const TransposeParams& params) {
    const auto& input     = params.input;
    const auto  data_type = input.type();
    const auto  shape     = input.shape();

    RUNTIME_ASSERT_OP_ARG(shape.size() == 2 || shape.size() == 3,
                          "You can only transpose a 2D buffer, but got [%s]",
                          input.debugString().c_str());
    if (shape.size() == 2) {
        auto output = allocateBuffer({data_type, {shape[1], shape[0]}});
        DISPATCH_CUDA_FUNCTION_GENERAL_TYPE(
            data_type, invokeTransposeAxis01, output->data(), input.data(), shape[0], shape[1], stream_);
        return std::move(output);
    } else {
        auto output = allocateBuffer({data_type, {shape[1], shape[0], shape[2]}});
        DISPATCH_CUDA_FUNCTION_GENERAL_TYPE(
            data_type, invokeTransposeAxis012, output->data(), input.data(), shape[0], shape[1], shape[2], stream_);
        return std::move(output);
    }
}

void ROCmDevice::checkError() {
    ROCM_CHECK_ERROR();
}

void ROCmDevice::initNcclParam(size_t             rank,
                               size_t             world_size,
                               const std::string& ip,
                               size_t             port,
                               const std::string& group_name,
                               NcclParam&         nccl_param) {
    nccl_param.rank_       = rank;
    nccl_param.world_size_ = world_size;
    auto       tcpStore    = createTcpStore(ip, port, world_size, rank);
    const auto nccl_id     = &(nccl_param.nccl_uid_);

    if (rank == 0) {
        RTP_LLM_LOG_INFO("rank %d creates nccl uid in group %s.", rank, group_name.c_str());
        NCCLCHECK(ncclGetUniqueId(nccl_id));
        setUniqueId(nccl_id, group_name, tcpStore);
    } else {
        RTP_LLM_LOG_INFO("rank %d get nccl uid in group %s.", rank, group_name.c_str());
        getUniqueId(nccl_id, group_name, tcpStore);
    }

    RTP_LLM_LOG_INFO("Initialize NCCL communicators [%s] rank %d of %d.", group_name.c_str(), rank, world_size);
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclCommInitRank(&nccl_param.nccl_comm_, world_size, *nccl_id, rank));
    NCCLCHECK(ncclGroupEnd());
}

void ROCmDevice::syncAndCheck() {
    syncCommunication();
    ROCM_CHECK(hipDeviceSynchronize());
    ROCM_CHECK_ERROR();
}

void ROCmDevice::syncCommunication(bool timeout) {
    if (tp_nccl_param_.world_size_ > 1) {
        RTP_LLM_LOG_DEBUG(
            "Synchronize tp NCCL communicators rank %d of %d.", tp_nccl_param_.rank_, tp_nccl_param_.world_size_);
        ftNcclStreamSynchronize(tp_nccl_param_, stream_, timeout);
    }
    if (dp_nccl_param_.world_size_ > 1) {
        RTP_LLM_LOG_DEBUG(
            "Synchronize dp NCCL communicators rank %d of %d.", dp_nccl_param_.rank_, dp_nccl_param_.world_size_);
        ftNcclStreamSynchronize(dp_nccl_param_, stream_, timeout);
    }
    if (dp_tp_nccl_param_.world_size_ > 1) {
        RTP_LLM_LOG_DEBUG("Synchronize dp_tp NCCL communicators rank %d of %d.",
                          dp_tp_nccl_param_.rank_,
                          dp_tp_nccl_param_.world_size_);
        ftNcclStreamSynchronize(dp_tp_nccl_param_, stream_, timeout);
    }
    if (ffn_tp_nccl_param_.world_size_ > 1 && ffn_tp_nccl_param_ != tp_nccl_param_) {
        RTP_LLM_LOG_DEBUG("Synchronize ffn_tp NCCL communicators rank %d of %d.",
                          ffn_tp_nccl_param_.rank_,
                          ffn_tp_nccl_param_.world_size_);
        ftNcclStreamSynchronize(ffn_tp_nccl_param_, stream_, timeout);
    }
}

void ROCmDevice::overlappedCommBarrier() {
    // NOTE: when all the overlapped communication and computation done,
    // we need to ensure the communication has been finished before starting the next computation.
    if (tp_nccl_param_.world_size_ * dp_nccl_param_.world_size_ * ffn_tp_nccl_param_.world_size_ > 1) {
        hipEvent_t event;
        ROCM_CHECK(hipEventCreate(&event));
        ROCM_CHECK(hipEventRecord(event, communication_stream_));
        ROCM_CHECK(hipStreamWaitEvent(stream_, event, 0));
        ROCM_CHECK(hipEventDestroy(event));
    }
}

DeviceHookPtr ROCmDevice::createCommHook() {
    return std::make_unique<ROCmCommHook>(stream_, communication_stream_);
}

void ROCmDevice::overlappedComputeBarrier() {
    // NOTE: when all the overlapped communication and computation done,
    // we need to ensure the communication has been finished before starting the next computation.
    if (tp_nccl_param_.world_size_ * dp_nccl_param_.world_size_ * ffn_tp_nccl_param_.world_size_ > 1) {
        hipEvent_t event;
        ROCM_CHECK(hipEventCreate(&event));
        ROCM_CHECK(hipEventRecord(event, stream_));
        ROCM_CHECK(hipStreamWaitEvent(communication_stream_, event, 0));
        ROCM_CHECK(hipEventDestroy(event));
    }
}

SelectOutput ROCmDevice::select(const SelectParams& params) {
    if ((params.input.where() != MemoryType::MEMORY_GPU) || (params.dim > 0)) {
        return DeviceBase::select(params);
    }

    RUNTIME_ASSERT_OP_ARG(params.index.type() == DataType::TYPE_INT32, "Select index must be int32.");
    RUNTIME_ASSERT_OP_ARG(params.dim == 0, "select op tmp only support dim == 0");
    const auto& input        = params.input;
    auto        output_shape = input.shape();
    output_shape[0]          = params.index.size();
    auto output              = allocateBuffer({input.type(), output_shape});
    if (output_shape[0] == 0 || input.shape()[0] == 0) {
        return output;
    }
    auto num_selected_element = input.size() / input.shape()[0];
    DISPATCH_CUDA_FUNCTION_GENERAL_TYPE(input.type(),
                                        invokeLookupHiddenStateOfLastToken,
                                        output->data(),
                                        input.data(),
                                        (int*)params.index.data(),
                                        params.index.size(),
                                        num_selected_element,
                                        0,
                                        stream_);
    return output;
}

BufferPtr ROCmDevice::embeddingLookup(const EmbeddingLookupParams& params) {
    const auto& tokens           = params.combo_tokens;
    const auto& embedding_table  = params.embedding_table;
    const auto& mask             = params.text_tokens_mask;
    const auto& position_ids     = params.position_ids;
    const auto& postition_table  = params.position_table;
    const auto& token_types      = params.token_types;
    const auto& token_type_table = params.token_type_table;

    const auto token_num   = tokens.size();
    const auto hidden_size = embedding_table.shape()[1];
    const auto data_type   = embedding_table.type();

    auto embeddings = allocateBuffer({data_type, {token_num, hidden_size}}, {"embedding"});

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                     invokeEmbeddingLookup,
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
                                     stream_);

    return embeddings;
}

void ROCmDevice::updateExpertGpuLoads(const MoeConfigs&          moe_conf,
                                      const OptionalExpertStats& expert_stats,
                                      BufferPtr                  expert_ids) {
    if (expert_stats.has_value() && expert_ids->size()) {
        auto& stats = expert_stats.value();
        launch_update_gpu_loads(expert_ids->data<int>(),
                                stats.getLayerGpuLoads(),
                                expert_ids->size(),
                                stats.phy_exp_num,
                                moe_conf.ep_rank,
                                moe_conf.ep_size,
                                stream_);
    }
}

void ROCmDevice::balanceExperts(BufferPtr                  expert_ids,
                                const OptionalExpertStats& expert_stats,
                                const MoeConfigs&          moe_conf,
                                const FfnLayerWeights&     weights) {
    if (expert_stats.has_value() && weights.log2phy) {
        const auto& expert_stats_v = expert_stats.value();

        int* log2phy          = weights.log2phy->data<int>();
        int* logic_expert_cnt = weights.logic_expert_cnt->data<int>();

        switch (moe_conf.balance_method) {
            case EplbBalanceMethod::EQUAL:
                if (expert_ids->type() == DataType::TYPE_INT64) {
                    launch_equal_expert_balance(expert_ids->data<int64_t>(),
                                                expert_stats_v.getLayerLogStats(),
                                                log2phy,
                                                logic_expert_cnt,
                                                expert_stats_v.log_exp_num,
                                                expert_stats_v.phy_exp_num,
                                                expert_ids->size(),
                                                moe_conf.use_all_gather ? 0 : moe_conf.ep_rank,
                                                stream_);
                } else {
                    launch_equal_expert_balance(expert_ids->data<int>(),
                                                expert_stats_v.getLayerLogStats(),
                                                log2phy,
                                                logic_expert_cnt,
                                                expert_stats_v.log_exp_num,
                                                expert_stats_v.phy_exp_num,
                                                expert_ids->size(),
                                                moe_conf.use_all_gather ? 0 : moe_conf.ep_rank,
                                                stream_);
                }
                break;
            default:
                throw std::runtime_error("Unsupported balance method");
                break;
        }
        ROCM_CHECK_ERROR();
    }
}

MemoryStatus ROCmDevice::getDeviceMemoryStatus() {
    MemoryStatus status;
    size_t       total_bytes;
    auto         error = hipMemGetInfo(&status.free_bytes, &total_bytes);
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

ROCmEvent::ROCmEvent(hipStream_t stream): stream_(stream) {
    ROCM_CHECK(hipEventCreate(&event_));
    ROCM_CHECK(hipEventRecord(event_, stream));
}

ROCmEvent::~ROCmEvent() {
    ROCM_CHECK(hipEventDestroy(event_));
}

void ROCmEvent::synchronize() const {
    ROCM_CHECK(hipEventSynchronize(event_));
    ROCM_CHECK(hipStreamSynchronize(stream_));
    ROCM_CHECK_ERROR();
    hipDeviceSynchronize();
}

bool ROCmEvent::checkReadiness() const {
    auto status = hipEventQuery(event_);
    if (status == hipSuccess) {
        return true;
    } else if (status == hipErrorNotReady) {
        return false;
    } else {
        RTP_LLM_LOG_ERROR("ROCmEvent checkReadiness failed with status: %d", status);
        ROCM_CHECK_ERROR();
        return false;
    }
}

ROCmCommHook::ROCmCommHook(hipStream_t main_stream, hipStream_t comm_stream):
    main_stream_(main_stream), comm_stream_(comm_stream) {
    ROCM_CHECK(hipEventCreate(&hook_event_));
    ROCM_CHECK(hipEventRecord(hook_event_, comm_stream_));
}

ROCmCommHook::~ROCmCommHook() {
    ROCM_CHECK(hipEventDestroy(hook_event_));
}

void ROCmCommHook::hook_sync() const {
    ROCM_CHECK(hipStreamWaitEvent(main_stream_, hook_event_, 0));
}

void ROCmDevice::chainSpeculativeSampling(const SpeculativeSamplingParams& params) {
    chain_speculative_sampling(params.draft_probs_d,
                               params.draft_token_ids_d,
                               params.uniform_samples_d,
                               params.target_probs_d,
                               params.output_token_ids_d,
                               params.output_accepted_token_num_d,
                               params.output_emitted_token_num_d,
                               false,
                               int64_t(stream_));
}

// void ROCmDevice::prepareCommBuffer(const PrepareCommBufferParams& params) {
//     if (attn_rs_comm_buffer_) {
//         return;
//     }

//     RTP_LLM_LOG_INFO("[PrepareCommBuffer] max_batch_seq_len %d, attn_rs_hidden %d, ffn_rs_hidden %d, attn_ag_hidden
//     %d, ffn_ag_hidden %d, rs_output_type %d, ag_input_type %d, enable_per_token_scale %d, enable_ffn_tp %d",
//             params.max_batch_seq_len, params.attn_rs_hidden, params.ffn_rs_hidden, params.attn_ag_hidden,
//             params.ffn_ag_hidden, params.rs_output_type, params.ag_input_type, params.enable_per_token_scale,
//             params.enable_ffn_tp);

//     size_t m = params.max_batch_seq_len * 1.1;
//     std::vector<size_t> tp_ranks = fcNcclGatherRanks(tp_nccl_param_, stream_);

//     RTP_LLM_LOG_INFO("[PrepareCommBuffer] prepare attn_rs_comm_buffer_");
//     std::vector<size_t> attn_rs_buffer_shape = {m, params.attn_rs_hidden};
//     attn_rs_comm_buffer_ = initCommBuffer(attn_rs_buffer_shape, params.rs_output_type, tp_nccl_param_, tp_ranks,
//     false, stream_);

//     RTP_LLM_LOG_INFO("[PrepareCommBuffer] prepare attn_ag_comm_buffer_");
//     std::vector<size_t> attn_ag_buffer_shape = {m, params.attn_ag_hidden};
//     attn_ag_comm_buffer_ = initCommBuffer(attn_ag_buffer_shape, params.ag_input_type, tp_nccl_param_, tp_ranks, true,
//     stream_);

//     if (params.enable_per_token_scale) {
//         RTP_LLM_LOG_INFO("[PrepareCommBuffer] prepare attn_ag_scale_comm_buffer_");
//         std::vector<size_t> attn_ag_scale_shape = {m, 1};
//         attn_ag_scale_comm_buffer_ = initCommBuffer(attn_ag_scale_shape, DataType::TYPE_FP32, tp_nccl_param_,
//         tp_ranks, true, stream_);
//     }

//     if (params.enable_ffn_tp) {
//         std::vector<size_t> ffn_tp_ranks = fcNcclGatherRanks(ffn_tp_nccl_param_, stream_);

//         RTP_LLM_LOG_INFO("[PrepareCommBuffer] prepare ffn_rs_comm_buffer_");
//         std::vector<size_t> ffn_rs_buffer_shape = {m, params.ffn_rs_hidden};
//         ffn_rs_comm_buffer_ = initCommBuffer(ffn_rs_buffer_shape, params.rs_output_type, ffn_tp_nccl_param_,
//         ffn_tp_ranks, false, stream_);

//         RTP_LLM_LOG_INFO("[PrepareCommBuffer] prepare ffn_ag_comm_buffer_");
//         std::vector<size_t> ffn_ag_buffer_shape = {m, params.ffn_ag_hidden};
//         ffn_ag_comm_buffer_ = initCommBuffer(ffn_ag_buffer_shape, params.ag_input_type, ffn_tp_nccl_param_,
//         ffn_tp_ranks, true, stream_);

//         RTP_LLM_LOG_INFO("[PrepareCommBuffer] prepare ffn_ag_scale_comm_buffer_");
//         if (params.enable_per_token_scale) {
//             std::vector<size_t> ffn_ag_scale_shape = {m, 1};
//             ffn_ag_scale_comm_buffer_ = initCommBuffer(ffn_ag_scale_shape, DataType::TYPE_FP32, ffn_tp_nccl_param_,
//             ffn_tp_ranks, true, stream_);
//         }
//     }
// }

BufferPtr ROCmDevice::mhaQKVGemm(const AttentionLayerParams& params) {
    const auto& input      = params.input;
    const auto& qkv_weight = params.weights.qkv_weight;

    // typically local_head_num * size_per_head + 2 * local_head_num_kv * size_per_head
    const auto qkv_merged_size = qkv_weight->kernel->shape()[1];

    BufferPtr qkv;
    if (!params.configs.fuse_qkv_add_bias && params.weights.qkv_weight && params.qscheme == QScheme::Qint8PerTensor) {
        BufferPtr D = allocateBuffer({DataType::TYPE_FP16, {input.shape()[0], qkv_weight->kernel->shape()[1]}});
        OptionalConstBufferRef bias = std::nullopt;
        if (qkv_weight->bias) {
            bias = *(qkv_weight->bias);
        }
        GemmParams qkv_gemm_params{input,
                                   *(qkv_weight->kernel),
                                   bias,
                                   D,
                                   DataType::TYPE_FP16,
                                   DataType::TYPE_FP16,
                                   TransposeOperation::NONE,
                                   TransposeOperation::NONE};
        qkv = loraLinear(LoraLinearParams(qkv_gemm_params, params.common.lora_input.qkv_lora_input)).output;
    } else if (!params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias) {
        ActivationParams act_params(ActivationType::Identity,
                                    nullptr,
                                    mayGetRef(params.weights.qkv_weight->bias),
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    nullptr,
                                    false,
                                    params.qscheme);
        auto             qkv_gemm_params = GemmParams(input, *(qkv_weight->kernel));
        auto lora_linear_params          = LoraLinearParams(qkv_gemm_params, params.common.lora_input.qkv_lora_input);
        qkv = loraLinearWithActivation(LoraLinearWithActivationParams(lora_linear_params, act_params));
    } else {
        auto qkv_gemm_params = GemmParams(
            input, *(qkv_weight->kernel), std::nullopt, nullptr, DataType::TYPE_INVALID, params.output->type());
        qkv = loraLinear(LoraLinearParams(qkv_gemm_params, params.common.lora_input.qkv_lora_input)).output;
    }
    printBufferData(*qkv, "qkv");
    if (params.weights.q_norm_weight) {
        RTP_LLM_CHECK_WITH_INFO(params.weights.k_norm_weight != nullptr,
                                "q_norm_weight and k_norm_weight should both be provided");
        RTP_LLM_CHECK_WITH_INFO(params.ln_params.norm_type == NormType::rmsnorm, "qkRmsNorm only support rmsnorm");
        auto qk_rmsnorm_output = qkRmsNorm(QkRmsNormParams({qkv,
                                                            *params.weights.q_norm_weight,
                                                            *params.weights.k_norm_weight,
                                                            params.ln_params.eps,
                                                            params.configs.head_num,
                                                            params.configs.kv_head_num,
                                                            params.configs.size_per_head}));
        printBufferData(*qkv, "qkv_after_qk_norm");
    }
    return qkv;
}

void ROCmDevice::maskLogits(Buffer& logits, const Buffer& mask) {
    size_t batch_size = logits.shape()[0];
    size_t vocab_size = logits.shape()[1];
    if (logits.type() == DataType::TYPE_FP32) {
        invokeMaskLogits<float>((float*)(logits.data()), (const uint8_t*)mask.data(), batch_size, vocab_size, stream_);
    } else if (logits.type() == DataType::TYPE_FP16) {
        invokeMaskLogits<half>((half*)(logits.data()), (const uint8_t*)mask.data(), batch_size, vocab_size, stream_);
    } else if (logits.type() == DataType::TYPE_BF16) {
        invokeMaskLogits<__nv_bfloat16>(
            (__nv_bfloat16*)(logits.data()), (const uint8_t*)mask.data(), batch_size, vocab_size, stream_);
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}

}  // namespace rtp_llm
