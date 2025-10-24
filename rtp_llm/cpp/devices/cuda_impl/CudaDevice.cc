#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/kernels/eplb/experts_stats_kernels.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/cuda/custom_ar/custom_ar_comm.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/cuda/allocator_cuda.h"
#include "rtp_llm/cpp/cuda/nccl/nccl_utils_torch.h"
#include "rtp_llm/cpp/cuda/nccl/nccl_utils.h"
#include "rtp_llm/cpp/core/TrackerAllocator.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/core/torch_utils/torch_cuda_allocator.h"
#include "rtp_llm/cpp/core/torch_utils/TorchEvent.h"
#include "rtp_llm/cpp/kernels/mask_logits.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <unistd.h>
#include "3rdparty/flashinfer/flashinfer.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

#ifdef USING_CUDA12
#include "rtp_llm/cpp/devices/cuda_impl/CudaXqa.h"
#endif

using namespace std;
using namespace rtp_llm;
using namespace tensorrt_llm;
using namespace tensorrt_llm::kernels;
using namespace rtp_llm;

namespace rtp_llm {

CudaDevice::CudaDevice(const DeviceInitParams& params): DeviceBase(params) {
    RTP_LLM_LOG_INFO("Initialize CudaDevice. %d", device_id_);
    check_cuda_value(cudaSetDevice(device_id_));
    printDeviceMemoryUsage("before init");
    if (init_params_.device_resource_config.not_use_default_stream) {
        torch_default_stream_ = std::make_unique<at::cuda::CUDAStream>(at::cuda::getStreamFromPool(true));
    } else {
        torch_default_stream_ = std::make_unique<at::cuda::CUDAStream>(at::cuda::getDefaultCUDAStream());
    }
    torch_comm_stream_ = std::make_unique<at::cuda::CUDAStream>(at::cuda::getStreamFromPool(true));
    at::cuda::setCurrentCUDAStream(*torch_default_stream_);
    stream_               = torch_default_stream_->stream();
    communication_stream_ = torch_comm_stream_->stream();
    check_cuda_value(cudaStreamCreateWithFlags(&no_block_copy_stream_, cudaStreamNonBlocking));
    check_cuda_value(cublasCreate(&cublas_handle_));
    check_cuda_value(cublasLtCreate(&cublaslt_handle_));
    check_cuda_value(cublasSetStream(cublas_handle_, stream_));
    check_cuda_value(cudaGetDeviceProperties(&device_prop_, device_id_));

    weight_only_matmul_plugin_ = std::make_unique<trt_plugins::WeightOnlyQuantMatmulPlugin>();

    smooth_quant_plugin_ = std::make_unique<trt_plugins::SmoothQuantGemmPlugin>();

    weight_only_groupwise_matmul_plugin_ = std::make_unique<trt_plugins::WeightOnlyGroupwiseQuantMatmulPlugin>();

    moe_plugin_ = std::make_unique<trt_plugins::MixtureOfExpertsPlugin>();

    if (init_params_.moe_config.hack_moe_expert) {
        hack_moe_expert_ = true;
    }

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

    cuggemm_runner_.reset(new cuggemm());
    cuggemm_runner_->init(stream_);
    bool fmha_env = params.fmha_config.enable_fmha;
    if (!fmha_env) {
        RTP_LLM_LOG_WARNING("FMHA is not enbaled");
    } else {
        checkUseTrtV1FMHA();
        checkUseTrtV2FMHA();
        checkUseOpenSourceFMHA();
        checkUseXQA();
        checkSupportTrtFp8FMHA();
    }
    checkUseMultiBlockMode();
    checkUseGroupGemm();
    checkUseFlashinferSampleKernel();

    // Initialize custom all reduce communicator
    // Note: custom all reduce communicator will allocate cuda mem through cudaMalloc, it must be called before
    // allocator init
    if (tp_nccl_param_.world_size_ > 1) {
        auto&               nccl_param = tp_nccl_param_;
        std::vector<size_t> tp_ranks   = fcNcclGatherRanks(nccl_param, stream_);
        custom_allreduce_comm_ = initCustomAllReduceComm(nccl_param, tp_ranks, stream_, params.hw_kernel_config);
    }
    printDeviceMemoryUsage("after init communicator");
    // cudaHostMalloc needs page table on GPU memory, retain this part first.
    auto host_allocator_ptr = new Allocator<AllocatorType::CUDA_HOST>(device_id_);
    host_allocator_ptr->setStream(stream_);
    if (params.host_reserve_memory_bytes) {
        RUNTIME_ASSERT_OP_ARG(params.host_reserve_memory_bytes > 0,
                              "cuda host memory can not reserve as much as possible (%lu), must specify concrete size.",
                              params.host_reserve_memory_bytes);
        TrackerAllocatorParams tracker_params;
        tracker_params.real_allocator     = host_allocator_ptr;
        tracker_params.target_track_bytes = params.host_reserve_memory_bytes;
        tracker_params.align_size         = 32;  // required by avx512
        host_allocator_.reset(new TrackerAllocator(tracker_params));
    } else {
        host_allocator_.reset(host_allocator_ptr);
    }

    auto allocator_ptr = new Allocator<AllocatorType::CUDA>(device_id_);
    allocator_ptr->setStream(stream_);

    if (init_params_.use_deepep_moe) {
        // init deepep buffer before buffer manager init to avoid out of mem
        buffer_manager_.reset(
            new BufferManager(allocator_ptr, host_allocator_ptr, init_params_.profile_debug_logging_config));
        if (!initDeepEPBuffer()) {
            RTP_LLM_CHECK_WITH_INFO(false, "init deepep buffer failed");
        } else {
            RTP_LLM_LOG_INFO("init deepep buffer success");
        }
        printDeviceMemoryUsage("after init deepep buffer");
        buffer_manager_.reset();
    }

    if (params.device_reserve_memory_bytes) {
        size_t free_bytes, total_bytes;
        check_cuda_value(cudaMemGetInfo(&free_bytes, &total_bytes));
        TrackerAllocatorParams tracker_params;
        tracker_params.real_allocator     = allocator_ptr;
        tracker_params.target_track_bytes = params.device_reserve_memory_bytes > 0 ?
                                                params.device_reserve_memory_bytes :
                                                (int64_t)free_bytes + params.device_reserve_memory_bytes;
        tracker_params.align_size         = 128;
        RTP_LLM_LOG_INFO("cuda device %d has %lu bytes free memory, trying to reserve %lu bytes.",
                         device_id_,
                         free_bytes,
                         tracker_params.target_track_bytes);
        allocator_.reset(new TrackerAllocator(tracker_params));
        syncAndCheck();  // sync check tracker malloc cuda mem
    } else {
        allocator_.reset(allocator_ptr);
    }

    // hijack torch cuda allocator
    origin_torch_cuda_allocator_  = at::cuda::CUDACachingAllocator::allocator;
    managed_torch_cuda_allocator_ = std::make_unique<TorchCudaAllocator>(this);
    at::cuda::CUDACachingAllocator::allocator.store(managed_torch_cuda_allocator_.get());

    cublas_algo_map_.reset(new cublasAlgoMap(GEMM_CONFIG));
    cublas_mm_wrapper_.reset(new cublasMMWrapper(
        cublas_handle_, cublaslt_handle_, stream_, cublas_algo_map_.get(), &cublas_wrapper_mutex_, allocator_.get()));

    // select mla type
    if (params.mla_ops_type != MlaOpsType::AUTO) {
        mla_ops_type = params.mla_ops_type;
    } else {
        mla_ops_type = device_prop_.major >= 9 ? MlaOpsType::FLASH_MLA : MlaOpsType::FLASH_INFER;
    }

    use_stable_scatter_add = init_params_.hw_kernel_config.enable_stable_scatter_add;

    RTP_LLM_LOG_INFO("use_stable_scatter_add: %d", use_stable_scatter_add);
}

void CudaDevice::printDeviceMemoryUsage(std::string stage) {
    size_t free_bytes, total_bytes;
    check_cuda_value(cudaMemGetInfo(&free_bytes, &total_bytes));
    RTP_LLM_LOG_INFO("stage:[%s] cuda device %d has %lu bytes free memory, total %lu bytes.",
                     stage.c_str(),
                     device_id_,
                     free_bytes,
                     total_bytes);
}

CudaDevice::~CudaDevice() {
    // change torch cuda gpu allocate
    if (origin_torch_cuda_allocator_) {
        at::cuda::CUDACachingAllocator::allocator.store(origin_torch_cuda_allocator_);
        origin_torch_cuda_allocator_ = nullptr;
    }
    curandstate_buf_.reset();
    cublas_mm_wrapper_.reset();
    check_cuda_value(cudaStreamDestroy(no_block_copy_stream_));
    check_cuda_value(cublasDestroy(cublas_handle_));
    check_cuda_value(cublasLtDestroy(cublaslt_handle_));
    if (ffn_tp_nccl_param_ != tp_nccl_param_ && ffn_tp_nccl_param_.nccl_comm_) {
        NCCLCHECK(ncclCommDestroy(ffn_tp_nccl_param_.nccl_comm_));
    }
    if (tp_nccl_param_.nccl_comm_) {
        NCCLCHECK(ncclCommDestroy(tp_nccl_param_.nccl_comm_));
    }
    if (dp_tp_nccl_param_.nccl_comm_) {
        NCCLCHECK(ncclCommDestroy(dp_tp_nccl_param_.nccl_comm_));
    }
    cache_store_.reset();
}

void CudaDevice::preRun() {
    check_cuda_value(cudaSetDevice(device_id_));
    at::cuda::setCurrentCUDAStream(*torch_default_stream_);
}

void CudaDevice::printDebugInfo() {
    RTP_LLM_LOG_INFO("default_stream: %d, device_id_: %d, stream_: %d",
                     torch_default_stream_->id(),
                     at::cuda::current_device(),
                     at::cuda::getCurrentCUDAStream(at::cuda::current_device()).id());
}

void CudaDevice::init() {
    // should init cuda device first to avoid set it in device reserve
    DeviceBase::init();

    RTP_LLM_LOG_INFO("cuda device init max batch size: %d\n", init_params_.max_batch_size);
    curandstate_buf_ = allocateBuffer({init_params_.max_batch_size * sizeof(curandState_t)}, {"curandstate"});
}

// pre-allocate buffer before buffer managaer
void CudaDevice::commBarrier(const NcclParam& nccl_param) {
    void* tmpBuffer = nullptr;
    check_cuda_value(cudaMalloc(&tmpBuffer, 32));
    check_cuda_value(cudaMemset(tmpBuffer, 0, 32));
    ftNcclAllReduceSum((float*)tmpBuffer, (float*)tmpBuffer, 32, nccl_param, stream_);
    check_cuda_value(cudaStreamSynchronize(stream_));
    check_cuda_value(cudaFree(tmpBuffer));
}

void CudaDevice::initNcclParam(size_t             rank,
                               size_t             world_size,
                               const std::string& ip,
                               size_t             port,
                               const string&      group_name,
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
    commBarrier(nccl_param);
}

void CudaDevice::checkError() {
    check_cuda_error();
}

void CudaDevice::syncAndCheck() {
    syncCommunication();
    check_cuda_value(cudaStreamSynchronize(stream_));
    check_cuda_value(cudaStreamSynchronize(communication_stream_));
    check_cuda_value(cudaStreamSynchronize(no_block_copy_stream_));
    check_cuda_error();
}

void CudaDevice::syncDeviceStream(DeviceStream stream) {
    auto sync_stream = getStream(stream);
    check_cuda_value(cudaStreamSynchronize(sync_stream));
}

void CudaDevice::syncCommunication(bool timeout) {
    if (tp_nccl_param_.world_size_ > 1) {
        RTP_LLM_LOG_DEBUG(
            "Synchronize tp NCCL communicators rank %d of %d.", tp_nccl_param_.rank_, tp_nccl_param_.world_size_);
        ftNcclStreamSynchronize(tp_nccl_param_, stream_, timeout);
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

void CudaDevice::syncCommunication(ParallelMode mode, bool timeout) {
    auto nccl_param = getNcclParam(mode);
    auto stream     = getCommStream(mode, false);
    if (nccl_param.world_size_ > 1) {
        RTP_LLM_LOG_DEBUG("Synchronize NCCL communicators rank %d of %d.", nccl_param.rank_, nccl_param.world_size_);
        ftNcclStreamSynchronize(nccl_param, stream, timeout);
    }
}

void CudaDevice::overlappedCommBarrier() {
    // NOTE: when all the overlapped communication and computation done,
    // we need to ensure the communication has been finished before starting the next computation.
    if (tp_nccl_param_.world_size_ * init_params_.dp_size * ffn_tp_nccl_param_.world_size_ > 1) {
        cudaEvent_t event;
        check_cuda_value(cudaEventCreate(&event));
        check_cuda_value(cudaEventRecord(event, communication_stream_));
        check_cuda_value(cudaStreamWaitEvent(stream_, event, 0));
        check_cuda_value(cudaEventDestroy(event));
    }
}

DeviceHookPtr CudaDevice::createCommHook() {
    return std::make_unique<CudaCommHook>(stream_, communication_stream_);
}
void CudaDevice::overlappedComputeBarrier() {
    // NOTE: when all the overlapped communication and computation done,
    // we need to ensure the communication has been finished before starting the next computation.
    if (tp_nccl_param_.world_size_ * init_params_.dp_size * ffn_tp_nccl_param_.world_size_ > 1) {
        cudaEvent_t event;
        check_cuda_value(cudaEventCreate(&event));
        check_cuda_value(cudaEventRecord(event, stream_));
        check_cuda_value(cudaStreamWaitEvent(communication_stream_, event, 0));
        check_cuda_value(cudaEventDestroy(event));
    }
}

DeviceProperties CudaDevice::getDeviceProperties() {
    static DeviceProperties* prop = nullptr;
    if (prop == nullptr) {
        prop                           = new DeviceProperties();
        prop->type                     = DeviceType::Cuda;
        prop->id                       = device_id_;
        prop->use_all_gather           = init_params_.use_all_gather;
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
        prop->use_deepep_moe           = init_params_.use_deepep_moe;
        prop->use_deepep_internode     = init_params_.use_deepep_internode;
        prop->use_deepep_low_latency   = init_params_.use_deepep_low_latency;
        prop->is_mtp                   = init_params_.is_mtp;
        prop->is_eagle3                = init_params_.is_eagle3;
        prop->ffn_as_service           = init_params_.ffn_as_service;
    }
    return *prop;
}

// only for cuda graph batch prefill test
void CudaDevice::setIsPadded(bool is_s_padded) {
    RTP_LLM_CHECK_WITH_INFO(!cufmha_runner_pool_.empty(), "cufmha_runner_pool_ is empty, cannot call setIsPadded");
    cufmha_runner_ = cufmha_runner_pool_.back();
    RTP_LLM_LOG_INFO("cufmha runner nums: %d", cufmha_runner_pool_.size());
    RTP_LLM_CHECK_WITH_INFO(cufmha_runner_ != nullptr, "cufmha_runner_ can't be nullptr");
    cufmha_runner_->setIsPadded(is_s_padded);
}

std::shared_ptr<cufmha>
CudaDevice::selectCuFMHARunner(const AttentionConfigs& configs, DataType attn_dtype, bool has_alibi_slopes) {
    bool     found_cufmha_runner = false;
    DataType fmha_datatype       = use_fp8_fmha_ ? DataType::TYPE_FP8_E4M3 : attn_dtype;
    for (auto& runner : cufmha_runner_pool_) {
        if (runner->checkSignature(fmha_datatype,
                                   configs.mask_type,
                                   configs.head_num,
                                   configs.kv_head_num,
                                   configs.size_per_head,
                                   configs.q_scaling / configs.softmax_extra_scale,
                                   has_alibi_slopes)) {
            cufmha_runner_      = runner;
            found_cufmha_runner = true;
            return cufmha_runner_;
        }
    }

    if (!found_cufmha_runner) {
        cufmha_runner_pool_.emplace_back();
        bool is_s_padded = (graph_runner_ != nullptr);
        cufmha_runner_pool_.back().reset(
            new cufmha(fmha_datatype,
                       configs.mask_type,
                       configs.head_num,
                       configs.kv_head_num,
                       configs.size_per_head,
                       configs.tokens_per_block,
                       configs.q_scaling / configs.softmax_extra_scale,  // div scale for DeepSeek V2
                       has_alibi_slopes,
                       use_trtv1_fmha,
                       use_trtv2_fmha,
                       use_trtv2_fmha_paged,
                       use_open_source_fmha,
                       use_open_source_fmha_paged,
                       is_s_padded,
                       stream_));
        cufmha_runner_ = cufmha_runner_pool_.back();
    }
    return cufmha_runner_;
}

bool CudaDevice::checkSpecDecode(const DevicePrepParams& params, bool skip_no_prefix) {
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

DevicePrepOutput CudaDevice::prepareModelRun(const DevicePrepParams& params) {
    if (init_params_.model_specific_config.load_python_model) {
        assert(!(params.context_batch_size && params.decoder_batch_size));
    }
    use_fp8_fmha_           = useFp8Fmha(params);
    DevicePrepOutput output = prepareModelRunCommon(params);

    fmha_type_ = FMHAType::NONE;
    if (params.attn_dtype == DataType::TYPE_FP32) {
        fmha_type_       = FMHAType::NONE;
        output.need_mask = true;
    } else if (params.context_batch_size) {
        selectCuFMHARunner(params.configs, params.attn_dtype, params.has_alibi_slopes);
        bool paged_kv_fmha =
            params.diff_qkv_len && params.k_cache && (params.configs.kv_cache_dtype != KvCacheDataType::INT8);

        if (!params.configs.use_mla && checkSpecDecode(params)) {
#ifdef USING_CUDA12
            if (use_xqa && use_fp8_fmha_
                && supportXqa(DataType::TYPE_BF16,
                              DataType::TYPE_BF16,
                              DataType::TYPE_FP8_E4M3,
                              params.configs.head_num / params.configs.kv_head_num,
                              params.configs.size_per_head,
                              params.configs.tokens_per_block)) {
                fmha_type_ = FMHAType::XQA;
            } else if (output.prefill_flash_infer_attn != nullptr) {
                fmha_type_ = FMHAType::FLASH_INFER;
            }
#else
            if (output.prefill_flash_infer_attn != nullptr) {
                fmha_type_ = FMHAType::FLASH_INFER;
            }
#endif
            else if (paged_kv_fmha) {
                if (use_trtv2_fmha_paged && cufmha_runner_->trtV2FmhaPagedSupport()) {
                    fmha_type_ = FMHAType::PAGED_TRT_V2;
                } else if (use_open_source_fmha_paged && cufmha_runner_->openSourceFmhaSupport()
                           && params.configs.tokens_per_block % 256 == 0) {
                    fmha_type_ = FMHAType::PAGED_OPEN_SOURCE;
                }
            }
        } else if (paged_kv_fmha) {
            if (use_trtv2_fmha_paged && cufmha_runner_->trtV2FmhaPagedSupport()) {
                fmha_type_ = FMHAType::PAGED_TRT_V2;
            } else if (use_open_source_fmha_paged && cufmha_runner_->openSourceFmhaSupport()
                       && params.configs.tokens_per_block % 256 == 0) {
                fmha_type_ = FMHAType::PAGED_OPEN_SOURCE;
            }
        } else if (!params.diff_qkv_len) {
            if (use_trtv2_fmha && cufmha_runner_->trtV2FmhaSupport()) {
                fmha_type_ = FMHAType::TRT_V2;
            } else if (use_open_source_fmha && cufmha_runner_->openSourceFmhaSupport()) {
                fmha_type_ = FMHAType::OPEN_SOURCE;
            } else if (use_trtv1_fmha && cufmha_runner_->trtV1FmhaSupport() && mla_ops_type == MlaOpsType::MHA) {
                fmha_type_ = FMHAType::TRT_V1;
            }
        } else {
            fmha_type_ = FMHAType::NONE;
        }
        output.need_mask = (fmha_type_ == FMHAType::NONE);
    }
    return output;
}

DevicePrepOutput CudaDevice::prepareModelRunCommon(const DevicePrepParams& params) {
    DevicePrepOutput output;
    auto             decode_kv_cache_block_id_d =
        params.kv_cache_block_id_d ? params.kv_cache_block_id_d->slice(0, params.decoder_batch_size) : nullptr;
    auto prefill_kv_cache_block_id_d =
        params.kv_cache_block_id_d ?
            params.kv_cache_block_id_d->slice(params.decoder_batch_size, params.context_batch_size) :
            nullptr;
    output.decode_flash_infer_attn = FlashInferAttnParams::prepare(
        this,
        params.configs,
        nullptr,
        params.sequence_lengths->slice(0, params.decoder_batch_size),
        params.input_lengths->slice(0, params.decoder_batch_size),
        params.kv_cache_block_id ? params.kv_cache_block_id->slice(0, params.decoder_batch_size) : nullptr,
        decode_kv_cache_block_id_d,
        params.attn_dtype);
    output.prefill_flash_infer_attn = FlashInferAttnParams::prepare(
        this,
        params.configs,
        params.prefix_lengths,
        nullptr,
        params.input_lengths->slice(params.decoder_batch_size, params.context_batch_size),
        params.kv_cache_block_id ?
            params.kv_cache_block_id->slice(params.decoder_batch_size, params.context_batch_size) :
            nullptr,
        prefill_kv_cache_block_id_d,
        params.attn_dtype);
    output.decode_trt_attn =
        prepareTrtAttn(params.configs, params.k_cache, decode_kv_cache_block_id_d, params.decoder_batch_size);
    output.prefill_trt_attn =
        prepareTrtAttn(params.configs, params.k_cache, prefill_kv_cache_block_id_d, params.context_batch_size);
    return output;
}

bool CudaDevice::useGroupGemm() const {
    return use_group_gemm;
}

cudaStream_t CudaDevice::getStream(DeviceStream stream) {
    switch (stream) {
        default:
            return stream_;
    }
}

void CudaDevice::bufMemset(Buffer& buf, int val, DeviceStream stream) {
    if (buf.where() == MemoryType::MEMORY_CPU || buf.where() == MemoryType::MEMORY_CPU_PINNED) {
        std::memset(buf.data(), val, buf.sizeBytes());
    } else {
        cudaStream_t cur_stream = getStream(stream);
        check_cuda_value(cudaMemsetAsync(buf.data(), val, buf.sizeBytes(), cur_stream));
    }
}

void CudaDevice::checkUseOpenSourceFMHA() {
    if (!(is_sm8x() || is_sm90())) {
        RTP_LLM_LOG_WARNING("opensource FMHA is disabled for sm %d", get_sm());
        return;
    }

    bool fmha_env = init_params_.fmha_config.enable_open_source_fmha;
    if (!fmha_env) {
        RTP_LLM_LOG_WARNING("opensource FMHA is disabled for by env");
        return;
    }

    RTP_LLM_LOG_INFO("use opensource fmha");
    use_open_source_fmha = true;
    bool paged_fmha_env  = init_params_.fmha_config.enable_paged_open_source_fmha;
    if (!paged_fmha_env) {
        RTP_LLM_LOG_INFO("Paged open source FMHA is disabled for by ENABLE_PAGED_OPEN_SOURCE_TRT_FMHA=OFF env");
        return;
    }
    if (init_params_.tokens_per_block % 256 != 0) {
        RTP_LLM_LOG_INFO("Paged open source FMHA is disabled since tokens_per_block % 256 != 0");
        return;
    }
    RTP_LLM_LOG_INFO("use opensource fmha paged");
    use_open_source_fmha_paged = true;
}

void CudaDevice::checkUseTrtV1FMHA() {
    bool fmha_env = init_params_.fmha_config.enable_trtv1_fmha;
    if (!fmha_env) {
        RTP_LLM_LOG_WARNING("TRTV1 FMHA is not enbaled");
        return;
    }
    RTP_LLM_LOG_INFO("use TRTV1 fmha");
    use_trtv1_fmha = true;
}

void CudaDevice::checkUseTrtV2FMHA() {
    if (!(is_sm8x() || is_sm90() || is_sm70())) {
        RTP_LLM_LOG_WARNING("TRT FMHA is disabled for sm %d", get_sm());
        return;
    }
    bool fmha_env = init_params_.fmha_config.enable_trt_fmha;
    if (!fmha_env) {
        RTP_LLM_LOG_WARNING("TRT FMHA is disabled for by env");
        return;
    }
    RTP_LLM_LOG_INFO("use TRTV2 fmha");
    use_trtv2_fmha = true;
    if (!(is_sm8x() || is_sm90())) {
        RTP_LLM_LOG_INFO("Paged TRT FMHA is disabled for sm %d", get_sm());
        return;
    }
    bool paged_fmha_env = init_params_.fmha_config.enable_paged_trt_fmha;
    if (!paged_fmha_env) {
        RTP_LLM_LOG_INFO("Paged TRT FMHA is disabled for by ENABLE_PAGED_TRT_FMHA=OFF env");
        return;
    }
    RTP_LLM_LOG_INFO("use TRTV2 fmha paged");
    use_trtv2_fmha_paged = true;
}

void CudaDevice::checkUseXQA() {
    if (!is_sm90()) {
        RTP_LLM_LOG_WARNING("not use xqa: unsupported sm %d (90)", get_sm());
        return;
    }
    if (!init_params_.fmha_config.enable_xqa) {
        RTP_LLM_LOG_WARNING("not use xqa: env disabled");
        return;
    }
    RTP_LLM_LOG_INFO("use xqa");
    use_xqa = true;
}

void CudaDevice::checkSupportTrtFp8FMHA() {
    int sm = get_sm();
    if (sm < 90 || !use_trtv2_fmha) {
        RTP_LLM_LOG_WARNING("sm is [%d], use_trtv2_fmha:[%d] not support fp8 fmha", sm, use_trtv2_fmha);
        return;
    }
    RTP_LLM_LOG_INFO("support fp8 fmha");
    support_trt_fp8_fmha = true;
}

bool CudaDevice::useFp8Fmha(const DevicePrepParams& params) const {
#ifdef ENABLE_FP8
    if (support_trt_fp8_fmha && params.configs.kv_cache_dtype == KvCacheDataType::FP8) {
        RTP_LLM_LOG_DEBUG("use fp8 fmha");
        return true;
    }
#endif
    return false;
}

void CudaDevice::checkUseFlashinferSampleKernel() {
    bool flashinfer_sample_env = init_params_.sampler_config.enable_flashinfer_sample_kernel;
    if (!flashinfer_sample_env) {
        RTP_LLM_LOG_WARNING("Flashinfer sample is disabled for by env");
        return;
    }
    RTP_LLM_LOG_INFO("use Flashinfer sample kernel");
    use_flashinfer_sample_kernel = true;
}

void CudaDevice::checkUseMultiBlockMode() {
    if (!init_params_.hw_kernel_config.enable_multi_block_mode) {
        RTP_LLM_LOG_WARNING("MMHA multi_block_mode is disabled");
        use_multi_block_mode = false;
        return;
    }
    if (get_sm() == 80 || get_sm() >= 89) {
        RTP_LLM_LOG_INFO("MMHA multi_block_mode is enabled");
        use_multi_block_mode = true;
        return;
    }
    use_multi_block_mode = true;
}

void CudaDevice::checkUseGroupGemm() {
    if (is_sm8x()) {
        use_group_gemm = true;
    } else {
        use_group_gemm = false;
    }
}

MemoryStatus CudaDevice::getDeviceMemoryStatus() {
    MemoryStatus status;
    size_t       total_bytes;
    auto         error = cudaMemGetInfo(&status.free_bytes, &total_bytes);
    RTP_LLM_CHECK(error == cudaSuccess);
    status.used_bytes = total_bytes - status.free_bytes;
    return status;
}

void CudaDevice::maskLogits(Buffer& logits, const Buffer& mask) {
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

nvinfer1::DataType nvinfer1DtypeConvert(rtp_llm::DataType dtype) {
    switch (dtype) {
        case rtp_llm::DataType::TYPE_FP16:
            return nvinfer1::DataType::kHALF;
        case rtp_llm::DataType::TYPE_BF16:
            return nvinfer1::DataType::kBF16;
        case rtp_llm::DataType::TYPE_FP32:
            return nvinfer1::DataType::kFLOAT;
        case rtp_llm::DataType::TYPE_QINT8:
            return nvinfer1::DataType::kINT8;
        case rtp_llm::DataType::TYPE_QINT4X2:
            return nvinfer1::DataType::kINT4;
        case rtp_llm::DataType::TYPE_QFP8_E4M3:
            return nvinfer1::DataType::kFP8;
        default:
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}

DeviceEventPtr CudaDevice::createEvent() {
    return std::make_unique<CudaEvent>(stream_);
}

DeviceEventPtr CudaDevice::createTorchEvent() {
    return std::make_unique<TorchEvent>(*torch_default_stream_);
}

void CudaDevice::updateCurrentTorchStream() {
    torch_default_stream_ =
        std::make_unique<at::cuda::CUDAStream>(at::cuda::getCurrentCUDAStream(at::cuda::current_device()));
    stream_ = torch_default_stream_->stream();
}

CudaEvent::CudaEvent(cudaStream_t stream): stream_(stream) {
    check_cuda_value(cudaEventCreate(&event_));
    check_cuda_value(cudaEventRecord(event_, stream));
}

CudaEvent::~CudaEvent() {
    check_cuda_value(cudaEventDestroy(event_));
}

void CudaEvent::synchronize() const {
    check_cuda_value(cudaEventSynchronize(event_));
    check_cuda_value(cudaStreamSynchronize(stream_));
    check_cuda_error();
    cudaDeviceSynchronize();
}

bool CudaEvent::checkReadiness() const {
    auto status = cudaEventQuery(event_);
    if (status == cudaSuccess) {
        return true;
    } else if (status == cudaErrorNotReady) {
        return false;
    } else {
        RTP_LLM_LOG_ERROR("CudaEvent checkReadiness failed with status: %d", status);
        check_cuda_error();
        return false;
    }
}

CudaCommHook::CudaCommHook(cudaStream_t main_stream, cudaStream_t comm_stream):
    main_stream_(main_stream), comm_stream_(comm_stream) {
    check_cuda_value(cudaEventCreate(&hook_event_));
    check_cuda_value(cudaEventRecord(hook_event_, comm_stream_));
}

CudaCommHook::~CudaCommHook() {
    check_cuda_value(cudaEventDestroy(hook_event_));
}

void CudaCommHook::hook_sync() const {
    check_cuda_value(cudaStreamWaitEvent(main_stream_, hook_event_, 0));
}

void CudaDevice::prepareCommBuffer(const PrepareCommBufferParams& params) {
    if (attn_rs_comm_buffer_) {
        return;
    }

    RTP_LLM_LOG_INFO(
        "[PrepareCommBuffer] max_batch_seq_len %d, attn_rs_hidden %d, ffn_rs_hidden %d, attn_ag_hidden %d, ffn_ag_hidden %d, rs_output_type %d, ag_input_type %d, enable_per_token_scale %d, enable_ffn_tp %d",
        params.max_batch_seq_len,
        params.attn_rs_hidden,
        params.ffn_rs_hidden,
        params.attn_ag_hidden,
        params.ffn_ag_hidden,
        params.rs_output_type,
        params.ag_input_type,
        params.enable_per_token_scale,
        params.enable_ffn_tp);

    size_t              m        = params.max_batch_seq_len * 1.1;
    std::vector<size_t> tp_ranks = fcNcclGatherRanks(tp_nccl_param_, stream_);

    RTP_LLM_LOG_INFO("[PrepareCommBuffer] prepare attn_rs_comm_buffer_");
    std::vector<size_t> attn_rs_buffer_shape = {m, params.attn_rs_hidden};
    attn_rs_comm_buffer_ =
        initCommBuffer(attn_rs_buffer_shape, params.rs_output_type, tp_nccl_param_, tp_ranks, false, stream_);

    RTP_LLM_LOG_INFO("[PrepareCommBuffer] prepare attn_ag_comm_buffer_");
    std::vector<size_t> attn_ag_buffer_shape = {m, params.attn_ag_hidden};
    attn_ag_comm_buffer_ =
        initCommBuffer(attn_ag_buffer_shape, params.ag_input_type, tp_nccl_param_, tp_ranks, true, stream_);

    if (params.enable_per_token_scale) {
        RTP_LLM_LOG_INFO("[PrepareCommBuffer] prepare attn_ag_scale_comm_buffer_");
        std::vector<size_t> attn_ag_scale_shape = {m, 1};
        attn_ag_scale_comm_buffer_ =
            initCommBuffer(attn_ag_scale_shape, DataType::TYPE_FP32, tp_nccl_param_, tp_ranks, true, stream_);
    }

    if (params.enable_ffn_tp) {
        std::vector<size_t> ffn_tp_ranks = fcNcclGatherRanks(ffn_tp_nccl_param_, stream_);

        RTP_LLM_LOG_INFO("[PrepareCommBuffer] prepare ffn_rs_comm_buffer_");
        std::vector<size_t> ffn_rs_buffer_shape = {m, params.ffn_rs_hidden};
        ffn_rs_comm_buffer_                     = initCommBuffer(
            ffn_rs_buffer_shape, params.rs_output_type, ffn_tp_nccl_param_, ffn_tp_ranks, false, stream_);

        RTP_LLM_LOG_INFO("[PrepareCommBuffer] prepare ffn_ag_comm_buffer_");
        std::vector<size_t> ffn_ag_buffer_shape = {m, params.ffn_ag_hidden};
        ffn_ag_comm_buffer_ =
            initCommBuffer(ffn_ag_buffer_shape, params.ag_input_type, ffn_tp_nccl_param_, ffn_tp_ranks, true, stream_);

        RTP_LLM_LOG_INFO("[PrepareCommBuffer] prepare ffn_ag_scale_comm_buffer_");
        if (params.enable_per_token_scale) {
            std::vector<size_t> ffn_ag_scale_shape = {m, 1};
            ffn_ag_scale_comm_buffer_              = initCommBuffer(
                ffn_ag_scale_shape, DataType::TYPE_FP32, ffn_tp_nccl_param_, ffn_tp_ranks, true, stream_);
        }
    }
}

void CudaDevice::updateExpertGpuLoads(const MoeConfigs&          moe_conf,
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

void CudaDevice::balanceExperts(BufferPtr                  expert_ids,
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
        check_cuda_error();
    }
}

void CudaDevice::chainSpeculativeSampling(const SpeculativeSamplingParams& params) {
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
};  // namespace rtp_llm
