#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
#include "src/fastertransformer/devices/rocm_impl/ROCmAllocator.h"
#include "src/fastertransformer/core/TrackerAllocator.h"
#include "src/fastertransformer/devices/DeviceFactory.h"

#include <hip/hip_runtime.h>

namespace fastertransformer {

ROCmDevice::ROCmDevice(const DeviceInitParams& params): DeviceBase(params) {
    RUNTIME_ASSERT_OP_ARG(params.tp_rank == 0, "rocm device doesn't support nccl");
    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipSetDevice(params.device_id));  // TODO(rocm): ensure this is setup every op
    HIP_CHECK(hipStreamCreate(&stream_));

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
}

ROCmDevice::~ROCmDevice() {
    if (stream_ != nullptr) {
        HIP_CHECK(hipStreamDestroy(stream_));
    }
}

DeviceProperties ROCmDevice::getDeviceProperties() {
    DeviceProperties props;
    props.type = DeviceType::ROCm;
    props.id   = device_id_;
    return props;
}

void ROCmDevice::copy(const CopyParams& params) {
    FT_CHECK_WITH_INFO(params.src.type() == params.dst.type(),
                       "dst[%d] and src[%d,] need has same type.",
                       params.src.type(),
                       params.dst.type());

    RUNTIME_ASSERT_OP_ARG(!params.dst.isQuantify() && !params.src.isQuantify(),
                          "rocm device doesn't support qint8 copy");

    const auto src_offset  = params.src_offset;
    const auto dst_offset  = params.dst_offset;
    auto       copy_length = params.copy_length;

    if (copy_length < 0) {
        RUNTIME_ASSERT_OP_ARG(params.src.shape()[0] == params.dst.shape()[0],
                              "src and dst 0-dim size mismatch: [%s] vs [%s]",
                              params.src.debugString().c_str(),
                              params.dst.debugString().c_str());
        copy_length = params.src.shape()[0];
    }

    if (copy_length == 0) {
        return;
    }

    const auto src = params.src.view(src_offset, copy_length);
    const auto dst = params.dst.view(dst_offset, copy_length);

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

    HIP_CHECK(hipMemcpyWithStream(dst.data(), src.data(), src.sizeBytes(), copyType, stream_));

    if (copyType == hipMemcpyDeviceToHost)
        HIP_CHECK(hipStreamSynchronize(stream_));
}

void ROCmDevice::syncAndCheck() {
    HIP_CHECK(hipStreamSynchronize(stream_));
}

RTP_LLM_REGISTER_DEVICE(ROCm);

}  // namespace fastertransformer
