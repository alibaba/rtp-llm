#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/cuda/custom_ar/custom_ar_comm.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/cuda/allocator_cuda.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils_torch.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils.h"
#include "src/fastertransformer/core/TrackerAllocator.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/compiler_config.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <unistd.h>

using namespace std;
using namespace tensorrt_llm;
using namespace tensorrt_llm::kernels;

namespace fastertransformer {

static const size_t DEFAULT_MAX_BATCH_SIZE = 256;

CudaDevice::CudaDevice(const DeviceInitParams& params) : DeviceBase(params) {
    FT_LOG_INFO("Initialize CudaDevice. %d", device_id_);
    check_cuda_error(cudaSetDevice(device_id_));
    check_cuda_error(cudaStreamCreate(&stream_));

    check_cuda_error(cublasCreate(&cublas_handle_));
    check_cuda_error(cublasLtCreate(&cublaslt_handle_));
    check_cuda_error(cublasSetStream(cublas_handle_, stream_));
    check_cuda_error(cudaGetDeviceProperties(&device_prop_, device_id_));

    weight_only_matmul_plugin_ = std::make_unique<trt_plugins::WeightOnlyQuantMatmulPlugin>();

    smooth_quant_plugin_ = std::make_unique<trt_plugins::SmoothQuantGemmPlugin>();

    weight_only_groupwise_matmul_plugin_ = std::make_unique<trt_plugins::WeightOnlyGroupwiseQuantMatmulPlugin>();

    auto ret = nvmlInit();
    FT_CHECK(ret == NVML_SUCCESS);
    ret = nvmlDeviceGetHandleByIndex(device_id_, &nvml_device_);
    FT_CHECK(ret == NVML_SUCCESS);

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
    cufmha_runner_.reset(new cufmha());
    cufmha_runner_->init(stream_);
    cuggemm_runner_.reset(new cuggemm());
    cuggemm_runner_->init(stream_);

    auto fmha_env = std::getenv("ENABLE_FMHA");
    if (fmha_env && std::string(fmha_env) == "OFF") {
        FT_LOG_WARNING("FMHA is not enbaled");
    } else {
        checkUseTrtV1FMHA();
        checkUseTrtV2FMHA();
        checkUseOpenSourceFMHA();
    }
    checkUseMultiBlockMode();

    // Initialize custom all reduce communicator
    // Note: custom all reduce communicator will allocate cuda mem through cudaMalloc, it must be called before allocator init
    if (nccl_param_.world_size_ > 1) {
        FT_LOG_INFO("Initialize custom all reduce communicator rank %d of %d", nccl_param_.rank_, nccl_param_.world_size_);
        std::vector<int> tp_ranks = fcNcclGatherRanks(nccl_param_, stream_);
        custom_allreduce_comm_ = initCustomAllReduceComm(nccl_param_, tp_ranks, stream_);
    }

    auto allocator_ptr = new Allocator<AllocatorType::CUDA>(device_id_);
    allocator_ptr->setStream(stream_);
    auto host_allocator_ptr = new Allocator<AllocatorType::CUDA_HOST>(device_id_);
    host_allocator_ptr->setStream(stream_);

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
    cublas_algo_map_.reset(new cublasAlgoMap(GEMM_CONFIG));
    cublas_mm_wrapper_.reset(new cublasMMWrapper(
        cublas_handle_, cublaslt_handle_, stream_, cublas_algo_map_.get(),
        &cublas_wrapper_mutex_, allocator_.get()));
}

CudaDevice::~CudaDevice() {
    curandstate_buf_.reset();
    cublas_mm_wrapper_.reset();
    check_cuda_error(cudaStreamDestroy(stream_));
    check_cuda_error(cublasDestroy(cublas_handle_));
    check_cuda_error(cublasLtDestroy(cublaslt_handle_));
    if (nccl_param_.nccl_comm_) {
        ncclCommDestroy(nccl_param_.nccl_comm_);
    }
}

void CudaDevice::init() {
    DeviceBase::init();
    FT_LOG_INFO("max batch size: %d\n", init_params_.max_batch_size);
    curandstate_buf_ = allocateBuffer(
        {init_params_.max_batch_size * sizeof(curandState_t)}, {"curandstate"});
}

void CudaDevice::syncAndCheck() {
    syncCommunication();
    cudaDeviceSynchronize();
    sync_check_cuda_error();
}

void CudaDevice::syncCommunication(bool timeout) {
    if (nccl_param_.world_size_ > 1) {
        FT_LOG_DEBUG("Synchronize NCCL communicators rank %d of %d.", nccl_param_.rank_, nccl_param_.world_size_);
        ftNcclStreamSynchronize(nccl_param_, stream_, timeout);
    }
}

DeviceProperties CudaDevice::getDeviceProperties() {
    static DeviceProperties* prop = nullptr;
    if (prop == nullptr) {
        prop = new DeviceProperties();
        prop->type = DeviceType::Cuda;
        prop->id = device_id_;
        prop->tp_rank = nccl_param_.rank_;
        prop->tp_size = nccl_param_.world_size_;
        prop->attention_need_mask = !(use_openSource_fmha || use_trtv1_fmha || use_trtv2_fmha);
    }
    return *prop;
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
    use_openSource_fmha = true;
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
        use_multi_block_mode = false;
        return;
    }
    use_multi_block_mode = true;
}

template <typename ComputeT, typename WeightsT, cutlass::WeightOnlyQuantOp QuantOp>
void initMoeRunnerImpl(unique_ptr<CutlassMoeFCRunnerInterface>& moe_runner) {
    if (moe_runner && dynamic_cast<CutlassMoeFCRunner<ComputeT, WeightsT, QuantOp>*>(moe_runner.get())) {
        return;
    } else {
        moe_runner.reset(new CutlassMoeFCRunner<ComputeT, WeightsT, QuantOp>());
    }
}

void CudaDevice::initMoeRunner(const DataType compute_type, const DataType weights_type) {
    if (compute_type == DataType::TYPE_FP16 && weights_type == DataType::TYPE_FP16) {
        initMoeRunnerImpl<half, half, cutlass::WeightOnlyQuantOp::UNDEFINED>(moe_runner_);
    } else if (compute_type == DataType::TYPE_FP32 && weights_type == DataType::TYPE_FP32) {
        initMoeRunnerImpl<float, float, cutlass::WeightOnlyQuantOp::UNDEFINED>(moe_runner_);
    } else if (compute_type == DataType::TYPE_FP16 && weights_type == DataType::TYPE_QINT8) {
        initMoeRunnerImpl<half, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>(moe_runner_);
    } else if (compute_type == DataType::TYPE_BF16 && weights_type == DataType::TYPE_BF16) {
        initMoeRunnerImpl<__nv_bfloat16, __nv_bfloat16, cutlass::WeightOnlyQuantOp::UNDEFINED>(moe_runner_);
    } else if (compute_type == DataType::TYPE_BF16 && weights_type == DataType::TYPE_QINT8) {
        initMoeRunnerImpl<__nv_bfloat16, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>(moe_runner_);
    } else if (compute_type == DataType::TYPE_FP16 && weights_type == DataType::TYPE_QINT4X2) {
        initMoeRunnerImpl<half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>(moe_runner_);
    } else if (compute_type == DataType::TYPE_BF16 && weights_type == DataType::TYPE_QINT4X2) {
        initMoeRunnerImpl<__nv_bfloat16, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>(moe_runner_);
    } else {
        RUNTIME_ASSERT_OP_ARG(
            false,
            "MoE runner can not be initialized via compute type %d and weights bits %d",
            compute_type, weights_type);
    }
}

// TODO(wangyin.yx): fill all memory status.
DeviceStatus CudaDevice::getDeviceStatus() {
    DeviceStatus status;

    size_t total_bytes;
    auto error = cudaMemGetInfo(&status.device_memory_status.free_bytes, &total_bytes);
    FT_CHECK(error == cudaSuccess);
    status.device_memory_status.used_bytes = total_bytes - status.device_memory_status.free_bytes;

    const auto buffer_status = queryBufferStatus();
    status.device_memory_status.allocated_bytes = buffer_status.device_allocated_bytes;
    status.device_memory_status.preserved_bytes = buffer_status.device_preserved_bytes;
    status.host_memory_status.allocated_bytes = buffer_status.host_allocated_bytes;
    status.device_memory_status.available_bytes = status.device_memory_status.free_bytes + status.device_memory_status.preserved_bytes;

    nvmlUtilization_t utilization;
    auto ret = nvmlDeviceGetUtilizationRates(nvml_device_, &utilization);
    FT_CHECK(ret == NVML_SUCCESS);
    status.device_utilization = (float)utilization.gpu;

    return status;
}

RTP_LLM_REGISTER_DEVICE(Cuda);

}; // namespace fastertransformer
