#include "rtp_llm/models_py/bindings/core/GpuBlockCopy.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include "rtp_llm/cpp/utils/ProfilingScope.h"

#if USING_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#elif USING_ROCM
#include <ATen/hip/HIPEvent.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPException.h>
#endif

#if USING_CUDA || USING_ROCM
#include "rtp_llm/models_py/bindings/common/kernels/block_id_copy.h"
#endif

namespace rtp_llm {

#if USING_CUDA || USING_ROCM
namespace {
#if USING_ROCM
using DeviceGuard  = c10::hip::HIPGuardMasqueradingAsCUDA;
using StreamHandle = hipStream_t;
auto currentStream(int device_index) {
    return c10::hip::getCurrentHIPStreamMasqueradingAsCUDA(device_index);
}
#else
using DeviceGuard  = c10::cuda::CUDAGuard;
using StreamHandle = cudaStream_t;
auto currentStream(int device_index) {
    return at::cuda::getCurrentCUDAStream(device_index);
}
#endif
}  // namespace

struct GpuBlockCopy::Impl {
    struct Slot {
        torch::Tensor       host;
        torch::Tensor       device;
        at::cuda::CUDAEvent done;
        size_t              capacity  = 0;
        bool                in_flight = false;
    };

    explicit Impl(std::vector<GpuBlockCopyPlane> inputs): owners(std::move(inputs)) {
        TORCH_CHECK(!owners.empty(), "block ID copy needs at least one storage plane");
        TORCH_CHECK(owners.front().blocks.is_cuda(), "block ID copy requires GPU storage");
        device_index = owners.front().blocks.get_device();
        DeviceGuard guard(device_index);
        TORCH_CHECK(owners.size() <= 65535, "too many block copy planes");
        std::vector<kernels::BlockCopyPlane> descriptors;
        for (const auto& plane : owners) {
            const auto& tensor = plane.blocks;
            TORCH_CHECK(tensor.is_cuda() && tensor.get_device() == device_index && tensor.dim() >= 1,
                        "block copy storage planes must be on the same GPU device");
            TORCH_CHECK(tensor.stride(0) > 0, "block copy needs a positive storage stride");
            const auto stride = tensor.stride(0) * tensor.element_size();
            TORCH_CHECK(tensor.size(0) > 0 && plane.copy_bytes > 0 && plane.copy_bytes <= static_cast<size_t>(stride),
                        "invalid block copy stride or size");
            TORCH_CHECK((plane.tokens_per_block == 0) == (plane.token_bytes == 0),
                        "partial block copy requires both token count and token bytes");
            if (plane.tokens_per_block) {
                const uint64_t head_bytes = uint64_t(plane.tokens_per_block) * plane.token_bytes;
                TORCH_CHECK(plane.copy_bytes <= std::numeric_limits<uint32_t>::max() && head_bytes <= plane.copy_bytes
                                && plane.copy_bytes % head_bytes == 0,
                            "invalid HND block copy layout");
                max_valid_tokens = std::min(max_valid_tokens, plane.tokens_per_block);
            }
            block_count = std::min(block_count, tensor.size(0));
            descriptors.push_back({static_cast<uint8_t*>(tensor.data_ptr()),
                                   static_cast<uint64_t>(stride),
                                   plane.copy_bytes,
                                   plane.tokens_per_block,
                                   plane.token_bytes});
        }
        const auto bytes = descriptors.size() * sizeof(kernels::BlockCopyPlane);
        layout           = torch::empty({static_cast<int64_t>(bytes)},
                              torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_index));
        // Initialization only: descriptors remain immutable for this cache pool's lifetime.
#if USING_ROCM
        C10_HIP_CHECK(hipMemcpy(layout.data_ptr(), descriptors.data(), bytes, hipMemcpyHostToDevice));
#else
        C10_CUDA_CHECK(cudaMemcpy(layout.data_ptr(), descriptors.data(), bytes, cudaMemcpyHostToDevice));
#endif
    }

    ~Impl() {
        // Drain before the owned KV storage, pinned memory and device descriptors disappear.
        try {
            DeviceGuard guard(device_index);
            for (auto& slot : slots) {
                if (slot.in_flight) {
                    slot.done.synchronize();
                }
            }
        } catch (const std::exception& e) {
            TORCH_WARN("block ID copy teardown: ", e.what());
        }
    }

    void copy(const torch::Tensor& mappings, const torch::Tensor& valid_token_lengths) {
        TORCH_CHECK(mappings.device().is_cpu() && mappings.scalar_type() == torch::kInt32 && mappings.is_contiguous()
                        && mappings.dim() == 2 && mappings.size(1) == 3,
                    "block ID copy expects a contiguous CPU int32 [N,3] matrix");
        const auto count = mappings.size(0);
        TORCH_CHECK(count <= std::numeric_limits<int>::max(), "too many block copy mappings");
        const int32_t* lengths = nullptr;
        if (valid_token_lengths.defined()) {
            TORCH_CHECK(valid_token_lengths.device().is_cpu() && valid_token_lengths.scalar_type() == torch::kInt32
                            && valid_token_lengths.is_contiguous() && valid_token_lengths.dim() == 1
                            && valid_token_lengths.numel() == count,
                        "block copy valid lengths must be a contiguous CPU int32 [N] vector");
            lengths = valid_token_lengths.data_ptr<int32_t>();
        }
        if (count == 0) {
            return;
        }
        const auto* ids = mappings.data_ptr<int32_t>();
        {
            RTP_LLM_PROFILE_SCOPE("kv.block_id_copy.validate");
            for (int64_t i = 0; i < count; ++i) {
                TORCH_CHECK(ids[3 * i] == 0 && ids[3 * i + 1] >= 0 && ids[3 * i + 1] < block_count
                                && ids[3 * i + 2] >= 0 && ids[3 * i + 2] < block_count,
                            "invalid group or block ID in block copy mapping ",
                            i);
                TORCH_CHECK(!lengths || (lengths[i] >= 0 && uint32_t(lengths[i]) <= max_valid_tokens),
                            "invalid live token count in block copy mapping ",
                            i);
            }
        }
        std::lock_guard<std::mutex> lock(mutex);
        DeviceGuard                 guard(device_index);
        auto                        stream = currentStream(device_index);
#if USING_ROCM
        hipStreamCaptureStatus capture_status;
        C10_HIP_CHECK(hipStreamIsCapturing(stream.stream(), &capture_status));
        const bool outside_capture = capture_status == hipStreamCaptureStatusNone;
#else
        cudaStreamCaptureStatus capture_status;
        C10_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &capture_status));
        const bool outside_capture = capture_status == cudaStreamCaptureStatusNone;
#endif
        TORCH_CHECK(outside_capture, "enqueue block ID copy before GPU graph replay, outside capture");

        auto& slot = slots[next_slot];
        if (slot.in_flight && !slot.done.query()) {
            RTP_LLM_PROFILE_SCOPE("kv.block_id_copy.wait_workspace");
            slot.done.synchronize();
        }
        slot.in_flight = false;
        if (slot.capacity < static_cast<size_t>(count)) {
            RTP_LLM_PROFILE_SCOPE("kv.block_id_copy.grow_workspace");
            auto capacity = std::max<size_t>(256, slot.capacity);
            while (capacity < static_cast<size_t>(count)) {
                capacity *= 2;
            }
            auto host     = torch::empty({static_cast<int64_t>(capacity), 4},
                                     torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
            auto device   = torch::empty({static_cast<int64_t>(capacity), 4},
                                       torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_index));
            slot.host     = std::move(host);
            slot.device   = std::move(device);
            slot.capacity = capacity;
        }
        // Preserve copy ordering when callers switch streams. Forward on the
        // caller's stream follows this copy naturally; no CPU synchronization.
        if (last_slot >= 0 && last_stream != stream.stream()) {
            slots[last_slot].done.block(stream);
        }
        const auto bytes  = static_cast<size_t>(count) * 4 * sizeof(int32_t);
        auto*      packed = slot.host.data_ptr<int32_t>();
        for (int64_t i = 0; i < count; ++i) {
            packed[4 * i]     = ids[3 * i];
            packed[4 * i + 1] = ids[3 * i + 1];
            packed[4 * i + 2] = ids[3 * i + 2];
            packed[4 * i + 3] = lengths ? lengths[i] : 0;
        }
        {
            RTP_LLM_PROFILE_SCOPE("kv.block_id_copy.enqueue");
#if USING_ROCM
            C10_HIP_CHECK(hipMemcpyAsync(
                slot.device.data_ptr(), slot.host.data_ptr(), bytes, hipMemcpyHostToDevice, stream.stream()));
#else
            C10_CUDA_CHECK(cudaMemcpyAsync(
                slot.device.data_ptr(), slot.host.data_ptr(), bytes, cudaMemcpyHostToDevice, stream.stream()));
#endif
            kernels::invokeBlockIdCopy(static_cast<const kernels::BlockCopyPlane*>(layout.data_ptr()),
                                       static_cast<int>(owners.size()),
                                       slot.device.data_ptr<int32_t>(),
                                       static_cast<int>(count),
                                       stream.stream());
            slot.done.record(stream);
            slot.in_flight = true;
#if USING_ROCM
            C10_HIP_KERNEL_LAUNCH_CHECK();
#else
            C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
        }
        last_slot   = static_cast<int>(next_slot);
        last_stream = stream.stream();
        next_slot   = (next_slot + 1) % slots.size();
    }

    std::vector<GpuBlockCopyPlane> owners;
    torch::Tensor                  layout;
    std::array<Slot, 2>            slots;
    std::mutex                     mutex;
    int64_t                        block_count      = std::numeric_limits<int64_t>::max();
    uint32_t                       max_valid_tokens = std::numeric_limits<uint32_t>::max();
    int                            device_index     = 0;
    size_t                         next_slot        = 0;
    int                            last_slot        = -1;
    StreamHandle                   last_stream      = nullptr;
};
#else
struct GpuBlockCopy::Impl {};
#endif

GpuBlockCopy::GpuBlockCopy(std::unique_ptr<Impl> impl): impl_(std::move(impl)) {}
GpuBlockCopy::~GpuBlockCopy() = default;

std::unique_ptr<GpuBlockCopy> GpuBlockCopy::create(std::vector<GpuBlockCopyPlane> planes) {
#if USING_CUDA || USING_ROCM
    return std::unique_ptr<GpuBlockCopy>(new GpuBlockCopy(std::make_unique<Impl>(std::move(planes))));
#else
    return nullptr;
#endif
}

void GpuBlockCopy::copy(const torch::Tensor& cpu_mappings, const torch::Tensor& valid_token_lengths) {
#if USING_CUDA || USING_ROCM
    impl_->copy(cpu_mappings, valid_token_lengths);
#else
    TORCH_CHECK(false, "block ID copy requires CUDA or ROCm");
#endif
}

}  // namespace rtp_llm
