#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/cuda/custom_ar/custom_ar_comm.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/cuda/allocator_cuda.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils_torch.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils.h"
#include "src/fastertransformer/core/TrackerAllocator.h"
#include "src/fastertransformer/devices/OpData.h"
#include "maga_transformer/cpp/utils/Logger.h"
#include "src/fastertransformer/utils/compiler_config.h"
#include "src/fastertransformer/core/torch_utils/torch_cuda_allocator.h"
#include "maga_transformer/cpp/disaggregate/cache_store/NormalCacheStore.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <unistd.h>

using namespace std;
using namespace rtp_llm;
using namespace tensorrt_llm;
using namespace tensorrt_llm::kernels;
using namespace rtp_llm;

namespace fastertransformer {

CudaDevice::CudaDevice(const DeviceInitParams& params) : DeviceBase(params) {
    FT_LOG_INFO("Initialize CudaDevice. %d", device_id_);
    check_cuda_error(cudaSetDevice(device_id_));
    stream_ = at::cuda::getCurrentCUDAStream().stream();
    check_cuda_error(cudaStreamCreateWithFlags(&no_block_copy_stream_, cudaStreamNonBlocking));
    check_cuda_error(cudaStreamCreateWithFlags(&communication_stream_, cudaStreamNonBlocking));
    check_cuda_error(cublasCreate(&cublas_handle_));
    check_cuda_error(cublasLtCreate(&cublaslt_handle_));
    check_cuda_error(cublasSetStream(cublas_handle_, stream_));
    check_cuda_error(cudaGetDeviceProperties(&device_prop_, device_id_));

    weight_only_matmul_plugin_ = std::make_unique<trt_plugins::WeightOnlyQuantMatmulPlugin>();

    smooth_quant_plugin_ = std::make_unique<trt_plugins::SmoothQuantGemmPlugin>();

    weight_only_groupwise_matmul_plugin_ = std::make_unique<trt_plugins::WeightOnlyGroupwiseQuantMatmulPlugin>();

    moe_plugin_ = std::make_unique<trt_plugins::MixtureOfExpertsPlugin>();

    if (params.tp_size > 1) {
        auto master_ip = params.master_ip;
        if (params.dp_size > 1) {
            master_ip = "127.0.0.1";
        }
        initNcclParam(params.tp_rank, params.tp_size, master_ip,
                      params.tp_master_port, "RTP_LLM_TP_GROUP_", tp_nccl_param_);
    }
    if (params.dp_size > 1 && params.tp_rank == 0) {
        initNcclParam(params.dp_rank, params.dp_size, params.master_ip,
                      params.dp_master_port, "RTP_LLM_DP_GROUP_", dp_nccl_param_);
    }
    if (params.ep_size > 1) {
        initNcclParam(params.dp_rank * params.tp_size + params.tp_rank, params.dp_size * params.tp_size, params.master_ip,
                      params.dp_tp_master_port, "RTP_LLM_DP_TP_GROUP_", dp_tp_nccl_param_);
    }
    cuggemm_runner_.reset(new cuggemm());
    cuggemm_runner_->init(stream_);

    auto fmha_env = std::getenv("ENABLE_FMHA");
    if (fmha_env && std::string(fmha_env) == "OFF") {
        FT_LOG_WARNING("FMHA is not enbaled");
    } else {
        checkUseTrtV1FMHA();
        checkUseTrtV2FMHA();
        checkUseOpenSourceFMHA();
        checkSupportTrtFp8FMHA();
    }
    checkUseMultiBlockMode();
    checkUseGroupGemm();
    checkUseFlashinferSampleKernel();

    // Initialize custom all reduce communicator
    // Note: custom all reduce communicator will allocate cuda mem through cudaMalloc, it must be called before allocator init
    if (tp_nccl_param_.world_size_ > 1) {
        auto& nccl_param = tp_nccl_param_;
        FT_LOG_INFO("Initialize tp custom all reduce communicator rank %d of %d", nccl_param.rank_, nccl_param.world_size_);
        std::vector<size_t> tp_ranks = fcNcclGatherRanks(nccl_param, stream_);
        custom_allreduce_comm_ = initCustomAllReduceComm(nccl_param, tp_ranks, stream_);
    }
    if (dp_tp_nccl_param_.world_size_ > 1) {
        auto& nccl_param = dp_tp_nccl_param_;
        FT_LOG_INFO("Initialize dp_tp custom all reduce communicator rank %d of %d", nccl_param.rank_, nccl_param.world_size_);
        std::vector<size_t> dp_tp_ranks = fcNcclGatherRanks(nccl_param, stream_);
        dp_tp_custom_allreduce_comm_ = initCustomAllReduceComm(nccl_param, dp_tp_ranks, stream_);
    }

    // cudaHostMalloc needs page table on GPU memory, retain this part first.
    auto host_allocator_ptr = new Allocator<AllocatorType::CUDA_HOST>(device_id_);
    host_allocator_ptr->setStream(stream_);
    if (params.host_reserve_memory_bytes) {
        RUNTIME_ASSERT_OP_ARG(params.host_reserve_memory_bytes > 0,
            "cuda host memory can not reserve as much as possible (%lu), must specify concrete size.",
            params.host_reserve_memory_bytes);
        TrackerAllocatorParams tracker_params;
        tracker_params.real_allocator = host_allocator_ptr;
        tracker_params.target_track_bytes = params.host_reserve_memory_bytes;
        tracker_params.align_size = 32; // required by avx512
        host_allocator_.reset(new TrackerAllocator(tracker_params));
    } else {
        host_allocator_.reset(host_allocator_ptr);
    }

    auto allocator_ptr = new Allocator<AllocatorType::CUDA>(device_id_);
    allocator_ptr->setStream(stream_);
    if (params.device_reserve_memory_bytes) {
        size_t free_bytes, total_bytes;
        check_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes));
        TrackerAllocatorParams tracker_params;
        tracker_params.real_allocator = allocator_ptr;
        tracker_params.target_track_bytes = params.device_reserve_memory_bytes > 0
            ? params.device_reserve_memory_bytes
            : free_bytes + params.device_reserve_memory_bytes;
        tracker_params.align_size = 16;
        FT_LOG_INFO("cuda device %d has %lu bytes free memory, trying to reserve %lu bytes.",
                    device_id_, free_bytes, tracker_params.target_track_bytes);
        allocator_.reset(new TrackerAllocator(tracker_params));
        syncAndCheck(); // sync check tracker malloc cuda mem
    } else {
        allocator_.reset(allocator_ptr);
    }

    // hijack torch cuda allocator
    origin_torch_cuda_allocator_ = at::cuda::CUDACachingAllocator::allocator;
    managed_torch_cuda_allocator_ = std::make_unique<TorchCudaAllocator>(this);
    at::cuda::CUDACachingAllocator::allocator.store(managed_torch_cuda_allocator_.get());

    cublas_algo_map_.reset(new cublasAlgoMap(GEMM_CONFIG));
    cublas_mm_wrapper_.reset(new cublasMMWrapper(
        cublas_handle_, cublaslt_handle_, stream_, cublas_algo_map_.get(),
        &cublas_wrapper_mutex_, allocator_.get()));

    // select mla type
    if (params.mla_ops_type != MlaOpsType::AUTO) {
        mla_ops_type = params.mla_ops_type;
    } else {
        mla_ops_type = device_prop_.major >= 9 ? MlaOpsType::FLASH_MLA : MlaOpsType::FLASH_INFER;
    }
}

CudaDevice::~CudaDevice() {
    // change torch cuda gpu allocate
    if (origin_torch_cuda_allocator_) {
        at::cuda::CUDACachingAllocator::allocator.store(origin_torch_cuda_allocator_);
        origin_torch_cuda_allocator_ = nullptr;
    }
    curandstate_buf_.reset();
    cublas_mm_wrapper_.reset();
    check_cuda_error(cudaStreamDestroy(no_block_copy_stream_));
    check_cuda_error(cublasDestroy(cublas_handle_));
    check_cuda_error(cublasLtDestroy(cublaslt_handle_));
    if (tp_nccl_param_.nccl_comm_) {
        ncclCommDestroy(tp_nccl_param_.nccl_comm_);
    }
    if (dp_nccl_param_.nccl_comm_) {
        ncclCommDestroy(dp_nccl_param_.nccl_comm_);
    }
    if (dp_tp_nccl_param_.nccl_comm_) {
        ncclCommDestroy(dp_tp_nccl_param_.nccl_comm_);
    }
    cache_store_.reset();
}

void CudaDevice::init() {
    DeviceBase::init();

    FT_LOG_INFO("cuda device init max batch size: %d\n", init_params_.max_batch_size);
    curandstate_buf_ = allocateBuffer(
        {init_params_.max_batch_size * sizeof(curandState_t)}, {"curandstate"});
}

void CudaDevice::initNcclParam(size_t rank, size_t world_size, const std::string& ip, size_t port,
                               const string& group_name, NcclParam& nccl_param) {
    nccl_param.rank_ = rank;
    nccl_param.world_size_ = world_size;
    auto tcpStore = createTcpStore(
            ip, port, world_size, rank);
    const auto nccl_id = &(nccl_param.nccl_uid_);

    if (rank == 0) {
        FT_LOG_INFO("rank %d creates nccl uid in group %s.", rank, group_name.c_str());
        NCCLCHECK(ncclGetUniqueId(nccl_id));
        setUniqueId(nccl_id, group_name, tcpStore);
    } else {
        FT_LOG_INFO("rank %d get nccl uid in group %s.", rank, group_name.c_str());
        getUniqueId(nccl_id, group_name, tcpStore);
    }

    FT_LOG_INFO("Initialize NCCL communicators [%s] rank %d of %d.", group_name.c_str(), rank, world_size);
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclCommInitRank(&nccl_param.nccl_comm_, world_size, *nccl_id, rank));
    NCCLCHECK(ncclGroupEnd());
}

void CudaDevice::syncAndCheck() {
    syncCommunication();
    cudaDeviceSynchronize();
    sync_check_cuda_error();
}

void CudaDevice::syncCommunication(bool timeout) {
    if (tp_nccl_param_.world_size_ > 1) {
        FT_LOG_DEBUG("Synchronize tp NCCL communicators rank %d of %d.", tp_nccl_param_.rank_, tp_nccl_param_.world_size_);
        ftNcclStreamSynchronize(tp_nccl_param_, stream_, timeout);
    }
    if (dp_nccl_param_.world_size_ > 1) {
        FT_LOG_DEBUG("Synchronize dp NCCL communicators rank %d of %d.", dp_nccl_param_.rank_, dp_nccl_param_.world_size_);
        ftNcclStreamSynchronize(dp_nccl_param_, stream_, timeout);
    }
    if (dp_tp_nccl_param_.world_size_ > 1) {
        FT_LOG_DEBUG("Synchronize dp_tp NCCL communicators rank %d of %d.", dp_tp_nccl_param_.rank_, dp_tp_nccl_param_.world_size_);
        ftNcclStreamSynchronize(dp_tp_nccl_param_, stream_, timeout);
    }
}

void CudaDevice::overlappedCommBarrier() {
    // NOTE: when all the overlapped communication and computation done,
    // we need to ensure the communication has been finished before starting the next computation.
    if (tp_nccl_param_.world_size_ * dp_nccl_param_.world_size_ > 1) {
        cudaEvent_t event;
        check_cuda_error(cudaEventCreate(&event));
        check_cuda_error(cudaEventRecord(event, communication_stream_));
        check_cuda_error(cudaStreamWaitEvent(stream_, event, 0));
        check_cuda_error(cudaEventDestroy(event));
    }
}

DeviceHookPtr CudaDevice::createCommHook() {
    return std::make_unique<CudaCommHook>(stream_, communication_stream_);
}

DeviceProperties CudaDevice::getDeviceProperties() {
    static DeviceProperties* prop = nullptr;
    if (prop == nullptr) {
        prop = new DeviceProperties();
        prop->type = DeviceType::Cuda;
        prop->id = device_id_;
        prop->tp_rank = init_params_.tp_rank;
        prop->tp_size = init_params_.tp_size;
        prop->dp_rank = init_params_.dp_rank;
        prop->dp_size = init_params_.dp_size;
        prop->enable_layer_micro_batch = init_params_.enable_layer_micro_batch;
    }
    return *prop;
}

void CudaDevice::selectCuFMHARunner(const DevicePrepParams& params) {
    bool found_cufmha_runner = false;
    use_fp8_fmha_ = useFp8Fmha(params);
    DataType fmha_datatype = use_fp8_fmha_ ? DataType::TYPE_FP8_E4M3 : params.dtype;
    for (auto& runner: cufmha_runner_pool_) {
        if (runner->checkSignature(fmha_datatype,
                                   params.configs.mask_type,
                                   params.configs.head_num,
                                   params.configs.kv_head_num,
                                   params.configs.size_per_head,
                                   params.configs.q_scaling / params.configs.softmax_extra_scale,
                                   params.has_alibi_slopes)) {
            cufmha_runner_ = runner;
            found_cufmha_runner = true;
            return;
        }
    }

    if (!found_cufmha_runner) {
        cufmha_runner_pool_.emplace_back();
        cufmha_runner_pool_.back().reset(
            new cufmha(fmha_datatype,
                       params.configs.mask_type,
                       params.configs.head_num,
                       params.configs.kv_head_num,
                       params.configs.size_per_head,
                       params.configs.q_scaling / params.configs.softmax_extra_scale, // div scale for DeepSeek V2
                       params.has_alibi_slopes,
                       use_trtv1_fmha,
                       use_trtv2_fmha,
                       use_trtv2_fmha_paged,
                       use_open_source_fmha,
                       use_open_source_fmha_paged,
                       stream_));
        cufmha_runner_ = cufmha_runner_pool_.back();
    }
}

DevicePrepOutput CudaDevice::prepareModelRun(const DevicePrepParams& params) {
    DevicePrepOutput output;
    fmha_type_ = FMHAType::NONE;
    if (params.dtype == DataType::TYPE_FP32) {
        fmha_type_ = FMHAType::NONE;
        output.need_mask = true;
    } else if (params.context_batch_size) {
        selectCuFMHARunner(params);
        bool paged_kv_fmha = params.diff_qkv_len && params.has_kv_cache && (params.configs.kv_cache_dtype == KvCacheDataType::BASE);
        if (paged_kv_fmha) {
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

    output.flash_infer_attn_params = FlashInferAttnParams::prepareFlashInferAttnParams(
            this,
            params.configs,
            params.sequence_lengths,
            params.input_lengths,
            params.kv_cache_block_id,
            params.dtype);

    return output;
}

bool CudaDevice::useGroupGemm() const {
    return use_group_gemm;

}

void CudaDevice::bufMemset(Buffer& buf, int val) {
    if (buf.where() == MemoryType::MEMORY_CPU || buf.where() == MemoryType::MEMORY_CPU_PINNED) {
        std::memset(buf.data(), val, buf.sizeBytes());
    } else {
        check_cuda_error(cudaMemset(buf.data(), val, buf.sizeBytes()));
    }
}

void CudaDevice::checkUseOpenSourceFMHA() {
    if (!(is_sm8x() || is_sm90())) {
        FT_LOG_WARNING("opensource FMHA is disabled for sm %d", get_sm());
        return;
    }

    char* fmha_env = std::getenv("ENABLE_OPENSOURCE_FMHA");
    if (fmha_env && std::string(fmha_env) == "OFF") {
        FT_LOG_WARNING("opensource FMHA is disabled for by env");
        return;
    }

    FT_LOG_INFO("use opensource fmha");
    use_open_source_fmha = true;
    char* paged_fmha_env = std::getenv("ENABLE_PAGED_OPEN_SOURCE_FMHA");
    if (paged_fmha_env && std::string(paged_fmha_env) == "OFF") {
        FT_LOG_INFO("Paged open source FMHA is disabled for by ENABLE_PAGED_TRT_FMHA=OFF env");
        return;
    }
    if (init_params_.tokens_per_block % 256 != 0) {
        FT_LOG_INFO("Paged open source FMHA is disabled since tokens_per_block % 256 != 0");
        return;
    }
    FT_LOG_INFO("use opensource fmha paged");
    use_open_source_fmha_paged = true;
}

void CudaDevice::checkUseTrtV1FMHA() {
    if (!CompileConfig::use_old_trt_fmha) {
        return;
    }
    char* fmha_env = std::getenv("ENABLE_TRTV1_FMHA");
    if (fmha_env && std::string(fmha_env) == "OFF") {
        FT_LOG_WARNING("TRTV1 FMHA is not enbaled");
        return;
    }
    FT_LOG_INFO("use TRTV1 fmha");
    use_trtv1_fmha = true;
}

void CudaDevice::checkUseTrtV2FMHA() {
    if (!(is_sm8x() || is_sm90() || is_sm70())) {
        FT_LOG_WARNING("TRT FMHA is disabled for sm %d", get_sm());
        return;
    }
    char* fmha_env = std::getenv("ENABLE_TRT_FMHA");
    if (fmha_env && std::string(fmha_env) == "OFF") {
        FT_LOG_WARNING("TRT FMHA is disabled for by env");
        return;
    }
    if(CompileConfig::cudart_version < 12000) {
        FT_LOG_WARNING("cudart version %d not support need >= 12000!", CompileConfig::cudart_version);
        return;
    }
    FT_LOG_INFO("use TRTV2 fmha");
    use_trtv2_fmha = true;
    if (!(is_sm8x() || is_sm90())) {
        FT_LOG_INFO("Paged TRT FMHA is disabled for sm %d", get_sm());
        return;
    }
    char* paged_fmha_env = std::getenv("ENABLE_PAGED_TRT_FMHA");
    if (paged_fmha_env && std::string(paged_fmha_env) == "OFF") {
        FT_LOG_INFO("Paged TRT FMHA is disabled for by ENABLE_PAGED_TRT_FMHA=OFF env");
        return;
    }
    FT_LOG_INFO("use TRTV2 fmha paged");
    use_trtv2_fmha_paged = true;
}

void CudaDevice::checkSupportTrtFp8FMHA() {
    int sm = get_sm();
    if (sm < 90 || !use_trtv2_fmha) {
      FT_LOG_WARNING("sm is [%d], use_trtv2_fmha:[%d] not support fp8 fmha", sm, use_trtv2_fmha);
        return;
    }
    FT_LOG_INFO("support fp8 fmha");
    support_trt_fp8_fmha = true;
}

bool CudaDevice::useFp8Fmha(const DevicePrepParams& params) const {
#ifdef ENABLE_FP8
    if (support_trt_fp8_fmha && params.configs.kv_cache_dtype == KvCacheDataType::FP8) {
        return true;
    }
#endif
    return false;
}

void CudaDevice::checkUseFlashinferSampleKernel() {
    char* flashinfer_sample_env = std::getenv("ENABLE_FLASHINFER_SAMPLE_KERNEL");
    if (flashinfer_sample_env && std::string(flashinfer_sample_env) == "OFF") {
        FT_LOG_WARNING("Flashinfer sample is disabled for by env");
        return;
    }
    FT_LOG_INFO("use Flashinfer sample kernel");
    use_flashinfer_sample_kernel = true;
}


void CudaDevice::checkUseMultiBlockMode() {
    if constexpr (CompileConfig::cudart_version < 11070) {
        FT_LOG_WARNING("MMHA multi_block_mode for cudart_version %d is disabled",
                        CompileConfig::cudart_version);
        use_multi_block_mode = false;
        return;
    }
    char* multi_block_mode_env = std::getenv("ENABLE_MULTI_BLOCK_MODE");
    if (multi_block_mode_env != nullptr && std::string(multi_block_mode_env) == "OFF") {
        FT_LOG_WARNING("MMHA multi_block_mode is disabled");
        use_multi_block_mode = false;
        return;
    }
    if (get_sm() == 80 || get_sm() >= 89) {
        FT_LOG_INFO("MMHA multi_block_mode is enabled");
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
    size_t total_bytes;
    auto error = cudaMemGetInfo(&status.free_bytes, &total_bytes);
    FT_CHECK(error == cudaSuccess);
    status.used_bytes = total_bytes - status.free_bytes;
    return status;
}

nvinfer1::DataType nvinfer1DtypeConvert(fastertransformer::DataType dtype)
 {
    switch (dtype) {
        case fastertransformer::DataType::TYPE_FP16 : return nvinfer1::DataType::kHALF;
        case fastertransformer::DataType::TYPE_BF16 : return nvinfer1::DataType::kBF16;
        case fastertransformer::DataType::TYPE_FP32 : return nvinfer1::DataType::kFLOAT;
        case fastertransformer::DataType::TYPE_QINT8 : return nvinfer1::DataType::kINT8;
        case fastertransformer::DataType::TYPE_QINT4X2 : return nvinfer1::DataType::kINT4;
        default: throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}

DeviceEventPtr CudaDevice::createEvent() {
    return std::make_unique<CudaEvent>(stream_);
}

CudaEvent::CudaEvent(cudaStream_t stream) : stream_(stream) {
    check_cuda_error(cudaEventCreate(&event_));
    check_cuda_error(cudaEventRecord(event_, stream));
}

CudaEvent::~CudaEvent() {
    check_cuda_error(cudaEventDestroy(event_));
}

void CudaEvent::synchronize() const {
    check_cuda_error(cudaEventSynchronize(event_));
    check_cuda_error(cudaStreamSynchronize(stream_));
    sync_check_cuda_error();
    cudaDeviceSynchronize();
}

CudaCommHook::CudaCommHook(cudaStream_t main_stream, cudaStream_t comm_stream)
    : main_stream_(main_stream), comm_stream_(comm_stream) {
    check_cuda_error(cudaEventCreate(&hook_event_));
    check_cuda_error(cudaEventRecord(hook_event_, comm_stream_));
}

CudaCommHook::~CudaCommHook() {
    check_cuda_error(cudaEventDestroy(hook_event_));
}

void CudaCommHook::hook_sync() const {
    check_cuda_error(cudaStreamWaitEvent(main_stream_, hook_event_, 0));
}

}; // namespace fastertransformer
