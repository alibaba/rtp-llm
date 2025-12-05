#include <vector>
#include <unordered_set>

#include <hip/hip_runtime.h>

#include "quick_ar_comm.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"

namespace rtp_llm {

QuickAllReduceComm::QuickAllReduceComm(const std::vector<size_t>& tp_ranks, size_t rank, size_t rank_index):
    rank_(rank),
    rank_index_(rank_index),
    world_size_(tp_ranks.size()),
    support_nv_link_(true),
    tp_ranks_(std::move(tp_ranks)) {}

QuickAllReduceComm::~QuickAllReduceComm() {
    aiter::qr_destroy(ptr_);
    ptr_ = 0;
}

void QuickAllReduceComm::init(const NcclParam& nccl_para, hipStream_t stream) {
    // 0. set nccl_para_
    nccl_para_ = nccl_para;

    // 1. set use_fp16_kernels flag
    // FIXME(liyangcheng.lyc): always set this flag to true, for bf16 kernels are slower than fp16 on ROCm
    use_fp16_kernels_ = true;

    // 2. set qr_quant_level flag
    char* qr_quant_level_str = std::getenv("AITER_QUICK_REDUCE_QUANTIZATION");
    std::string qr_quant_level_value(qr_quant_level_str);
    qr_quant_level_ = QuickReduceRegime[qr_quant_level_value];
    RTP_LLM_LOG_INFO("Quick allreduce quantization level set to: %d", qr_quant_level_);

    // 3. set qr_max_size
    char* qr_max_size_str = std::getenv("AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB");
    if (nullptr != qr_max_size_str) {
        qr_max_size_ = std::stoll(std::string(qr_max_size_str));
        qr_max_size_ *= 1024 * 1024;
    } else {
        qr_max_size_ = aiter::qr_max_size();
    }
    RTP_LLM_LOG_INFO("Quick allreduce max size set to %d bytes", qr_max_size_);

    // 4. init ptr_
    ptr_ = aiter::init_custom_qr(rank_index_, world_size_, qr_max_size_);

    // 5. create shared buffer
    torch::Tensor handle = aiter::qr_get_handle(ptr_);

    auto _handles = all_gather(handle.data_ptr(), handle.element_size() * handle.numel(), stream);
    std::vector<torch::Tensor> handles(world_size_);
    for (int i = 0; i < world_size_; ++i) {
        handles[i] = torch::from_blob(_handles[i].data(), handle.sizes(), handle.dtype());
    }

    aiter::qr_open_handles(ptr_, handles);
}

void QuickAllReduceComm::allReduce(torch::Tensor& input_tensor, torch::Tensor& output_tensor) {
    aiter::qr_all_reduce(ptr_, input_tensor, output_tensor, qr_quant_level_, use_fp16_kernels_);
}

bool QuickAllReduceComm::checkAllReduceAvailable(size_t elts_total_num, DataType data_type, size_t world_size) {
    // 0. check data_type, must be fp16 or bf16
    if (data_type != DataType::TYPE_FP16 and data_type != DataType::TYPE_BF16) {
        return false;
    }

    // 1. check size, shoule be multiples of 16
    size_t elts_total_size = elts_total_num * getTypeSize(data_type);

    if (elts_total_size % 16 != 0) {
        return false;
    }

    // 2. TODO(liyangcheng.lyc): check input be weak contiguous

    // 3. check size threshold
    if (use_fp16_kernels_) {
        data_type = DataType::TYPE_FP16;
    }

    int64_t qr_min_thres = QR_SIZE_RANGE[{data_type, world_size_}][qr_quant_level_].first;
    qr_min_thres *= 1024 * 1024;
    int64_t qr_max_thres = QR_SIZE_RANGE[{data_type, world_size_}][qr_quant_level_].second;
    qr_max_thres *= 1024 * 1024;

    if (not (elts_total_size <= qr_max_size_ and elts_total_size >= qr_min_thres and elts_total_size <= qr_max_thres)) {
        return false;
    }

    return true;
}
    
bool QuickAllReduceComm::shouldQuickAR(const std::vector<size_t>& tp_ranks, size_t rank) {
    // 0. check env flag
    // reuse this environment variable because quick allreduce is a special case of custom allreduce with stricter conditions.
    char* disable_quick_ar_str = std::getenv("FT_DISABLE_CUSTOM_AR");
    bool  disable_quick_ar     = disable_quick_ar_str != nullptr && std::string(disable_quick_ar_str) == "1";
    if (disable_quick_ar) {
        RTP_LLM_LOG_INFO("Disable quick ar since FT_DISABLE_CUSTOM_AR is set");
        return false;
    }

    // 1. check rocm arch
    hipDeviceProp_t prop;
    ROCM_CHECK(hipGetDeviceProperties(&prop, 0));
    std::string gcn_arch = std::string(prop.gcnArchName);

    std::unordered_set<std::string> supported_archs = {"gfx94", "gfx50"};
    bool is_supported_arch = false;
    for (const auto& gfx : supported_archs) {
        if (gcn_arch.find(gfx) != std::string::npos) {
            is_supported_arch = true;
            break;
        }
    }
    if (not is_supported_arch) {
        RTP_LLM_LOG_INFO("Disable quick ar since gcn_arch is not supported, supported archs are rocm >= gfx942");
        return false;
    }

    // 2. check aiter quick allreduce library
    // this is guaranteed in compile time

    // 3. check whether group spans across nodes
    size_t world_size       = tp_ranks.size();
    size_t local_world_size = rocm::getDeviceCount();

    if (world_size > local_world_size) {
        RTP_LLM_LOG_INFO(
            "Disable quick ar since TP is performanced on multi nodes, world_size=%d, local_world_size=%d",
            world_size,
            local_world_size);
        return false;
    }

    // 4. check supported world sizes
    std::unordered_set<size_t> available_world_sizes = {2, 4, 8};
    if (available_world_sizes.find(world_size) == available_world_sizes.end()) {
        RTP_LLM_LOG_INFO("Disable quick ar for invalid world size %d", world_size);
        return false;
    }

    // 5. check fully nvlink
    // true

    // 6. check quantization level
    std::unordered_set<std::string> supported_quant_types = {"FP", "FP8", "INT6", "INT4", "NONE"};
    char* qr_quant_level_str = std::getenv("AITER_QUICK_REDUCE_QUANTIZATION");
    if (qr_quant_level_str) {
        std::string qr_quant_level_value(qr_quant_level_str);
        if (supported_quant_types.find(qr_quant_level_value) == supported_quant_types.end()) {
            RTP_LLM_LOG_INFO("Disable quick ar since invalid quantization level");
            return false;
        }

        if (qr_quant_level_value == "NONE") {
            RTP_LLM_LOG_INFO("Disable quick ar since quant level set to NONE");
            return false;
        }
    } else {
        RTP_LLM_LOG_INFO("Disable quick ar since ENV AITER_QUICK_REDUCE_QUANTIZATION not set");
        return false;
    }

    // can use quick ar, return true
    return true;
}

std::vector<std::vector<char>> QuickAllReduceComm::all_gather(void* addr, size_t size, hipStream_t stream) {
    char* device_buffer;
    ROCM_CHECK(hipMalloc(&device_buffer, size * world_size_));
    ROCM_CHECK(hipMemcpyAsync(device_buffer + rank_index_ * size, addr, size, hipMemcpyHostToDevice, stream));
    ftNcclAllGather(device_buffer, device_buffer, size, rank_index_, nccl_para_, stream);
    ROCM_CHECK(hipStreamSynchronize(stream));
    std::vector<std::vector<char>> ret(world_size_);
    for (size_t i = 0; i < world_size_; ++i) {
        std::vector<char> tmp(size);
        ROCM_CHECK(hipMemcpyAsync(tmp.data(), device_buffer + size * i, size, hipMemcpyDeviceToHost, stream));
        ret[i] = tmp;
    }
    ROCM_CHECK(hipFreeAsync(device_buffer, stream));
    return ret;
}

std::unique_ptr<QuickAllReduceComm>
initQuickAllReduceComm(const NcclParam& nccl_para, const std::vector<size_t>& tp_ranks, hipStream_t stream) {
    size_t rank_index = 0;
    for (size_t i = 0; i < tp_ranks.size(); ++i) {
        if (tp_ranks[i] == nccl_para.rank_) {
            rank_index = i;
            break;
        }
    }

    if (!QuickAllReduceComm::shouldQuickAR(tp_ranks, nccl_para.rank_)) {
        return nullptr;
    }

    auto comm = std::make_unique<QuickAllReduceComm>(tp_ranks, nccl_para.rank_, rank_index);
    comm->init(nccl_para, stream);
    RTP_LLM_LOG_INFO("Quick all reduce is enabled on rank %d of %d", nccl_para.rank_, tp_ranks.size());
    return comm;
}

} // namespace rtp_llm
