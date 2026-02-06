#include "rtp_llm/cpp/devices/arm_impl/ArmDevice.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/core/cpu_allocator.h"
#include "rtp_llm/cpp/core/TrackerAllocator.h"
#include <cstring>
#include <sys/sysinfo.h>

namespace rtp_llm {
ConstBufferPtr (*armPrepareWeightFunc)(ConstBufferPtr input, bool isTranspose, bool isForceF32Out);

int getMemoryInfo(unsigned long* free_bytes, unsigned long* total_bytes) {
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        // sysinfo call failed
        return -1;
    }

    *free_bytes  = info.freeram * info.mem_unit;
    *total_bytes = info.totalram * info.mem_unit;

    return 0;
}

ArmCpuDevice::ArmCpuDevice(const DeviceInitParams& params): DeviceBase(params) {

    auto allocator_ptr = new Allocator<AllocatorType::CPU>();
    if (params.device_reserve_memory_bytes) {
        size_t free_bytes, total_bytes;
        RTP_LLM_CHECK(getMemoryInfo(&free_bytes, &total_bytes) == 0);
        TrackerAllocatorParams tracker_params;
        tracker_params.real_allocator     = allocator_ptr;
        tracker_params.target_track_bytes = params.device_reserve_memory_bytes > 0 ?
                                                params.device_reserve_memory_bytes :
                                                free_bytes + params.device_reserve_memory_bytes;
        tracker_params.align_size         = 16;
        tracker_params.metrics_reporter   = DeviceFactory::getMetricsReporter();
        RTP_LLM_LOG_INFO("Arm device %d has %lu bytes free memory, trying to reserve %lu bytes.",
                         device_id_,
                         free_bytes,
                         tracker_params.target_track_bytes);
        allocator_.reset(new TrackerAllocator(tracker_params));
    } else {
        allocator_.reset(allocator_ptr);
    }

    if (!params.hw_kernel_config.arm_gemm_use_kai) {
        isKAIenabled = false;
        gemmFunc     = &ArmCpuDevice::gemm_opt;
    } else {
        isKAIenabled = true;
        gemmFunc     = &ArmCpuDevice::gemm_kai_bf16;
    }
}

ArmCpuDevice::~ArmCpuDevice() {}

DeviceProperties ArmCpuDevice::getDeviceProperties() {
    DeviceProperties props;
    props.type = DeviceType::ArmCpu;
    return props;
}

arm_compute::DataType ArmCpuDevice::getAclDataType(DataType type) {
    using dt = arm_compute::DataType;
    switch (type) {
        case DataType::TYPE_FP32:
            return dt::F32;
        case DataType::TYPE_BF16:
            return dt::BFLOAT16;
        case DataType::TYPE_FP16:
            return dt::F16;
        case DataType::TYPE_UINT8:
            return dt::U8;
        case DataType::TYPE_UINT16:
            return dt::U16;
        case DataType::TYPE_UINT32:
            return dt::U32;
        case DataType::TYPE_INT8:
            return dt::S8;
        case DataType::TYPE_UINT64:
            return dt::U64;
        default:
            return dt::UNKNOWN;
    }
}

void ArmCpuDevice::copy(const CopyParams& params) {
    auto& src  = params.src;
    auto& dst  = params.dst;
    auto  size = params.src.sizeBytes();
    memcpy(dst.data(), src.data(), size);
}

GroupedGemmOutput ArmCpuDevice::groupedGemm(const GroupedGemmParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BeamSearchOutput ArmCpuDevice::sampleBeamSearch(const BeamSearchParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void ArmCpuDevice::broadcast(const BroadcastParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void ArmCpuDevice::allReduceSum(const AllReduceParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

MemoryStatus ArmCpuDevice::getDeviceMemoryStatus() {
    MemoryStatus status;
    size_t       total_bytes;
    auto         error = getMemoryInfo(&status.free_bytes, &total_bytes);
    RTP_LLM_CHECK(error == 0);
    status.used_bytes = total_bytes - status.free_bytes;
    return status;
}

#define MAX_PRE_CALC_SEQ_LEN 1024  // TODO: get it from model config
DevicePrepOutput ArmCpuDevice::prepareModelRun(const DevicePrepParams& params) {
    auto output = DevicePrepOutput();
    /* Prepare cos/sin values used in RoPE. */
    auto base = params.configs.rope_config.base;
    auto dim  = params.configs.rope_config.dim;

    auto it = ropeCosSin.find(base);
    if (it == ropeCosSin.end()) {
        size_t inv_freq_size = (dim + 1) / 2;
        float* inv_freq      = (float*)malloc(inv_freq_size * sizeof(float));

        for (size_t i = 0; i < inv_freq_size; i++) {
            inv_freq[i] = 1.0f / powf(base, (float)(i * 2) / dim);
        }
        float* emb_cos = (float*)malloc(MAX_PRE_CALC_SEQ_LEN * inv_freq_size * sizeof(float));
        float* emb_sin = (float*)malloc(MAX_PRE_CALC_SEQ_LEN * inv_freq_size * sizeof(float));

        ropeCosSin[base] = std::make_tuple(MAX_PRE_CALC_SEQ_LEN, emb_cos, emb_sin);

        for (size_t i = 0; i < MAX_PRE_CALC_SEQ_LEN; i++) {
            float* pcos = emb_cos + i * inv_freq_size;
            float* psin = emb_sin + i * inv_freq_size;

            for (size_t j = 0; j < inv_freq_size; j++) {
                float val     = i * inv_freq[j];
                float cos_tmp = cosf(val);
                float sin_tmp = sinf(val);

                pcos[j] = cos_tmp;
                psin[j] = sin_tmp;
            }
        }
        free(inv_freq);
    }

    return output;
}

RTP_LLM_REGISTER_DEVICE(ArmCpu);

}  // namespace rtp_llm
