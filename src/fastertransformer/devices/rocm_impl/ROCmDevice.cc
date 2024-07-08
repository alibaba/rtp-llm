#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
#include "src/fastertransformer/devices/rocm_impl/ROCmAllocator.h"
#include "src/fastertransformer/core/TrackerAllocator.h"
#include "src/fastertransformer/devices/DeviceFactory.h"

#include "src/fastertransformer/kernels/hello_world.h"

namespace fastertransformer {
using namespace fastertransformer::rocm;

ROCmDevice::ROCmDevice(const DeviceInitParams& params): DeviceBase(params) {
    RUNTIME_ASSERT_OP_ARG(params.tp_rank == 0, "rocm device doesn't support nccl");
    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipSetDevice(params.device_id));  // TODO(rocm): ensure this is setup every op
    HIP_CHECK(hipStreamCreate(&stream_));
    check_hip_error(hipGetDeviceProperties(&rocmDevProp, device_id_));
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
}

ROCmDevice::~ROCmDevice() {
    hipblas_mm_wrapper_.reset();
    hipStreamDestroy(stream_);
    hipblasDestroy(hipblas_handle_);
    hipblasLtDestroy(hipblaslt_handle_);

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

    HIP_CHECK(hipMemcpyWithStream(dst.data(), src.data(), src.sizeBytes(), copyType, stream_));

    if (copyType == hipMemcpyDeviceToHost)
        HIP_CHECK(hipStreamSynchronize(stream_));
}

void ROCmDevice::syncAndCheck() {
    HIP_CHECK(hipStreamSynchronize(stream_));
}
SelectOutput ROCmDevice::select(const SelectParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}
BufferPtr ROCmDevice::embeddingLookup(const EmbeddingLookupParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}
LayernormOutput ROCmDevice::layernorm(const LayernormParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}
void ROCmDevice::activation(const ActivationParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}
AttentionModuleOutput ROCmDevice::contextAttention(const AttentionModuleParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
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

RTP_LLM_REGISTER_DEVICE(ROCm);

}  // namespace fastertransformer
