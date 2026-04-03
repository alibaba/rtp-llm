#include "rtp_llm/models_py/bindings/cuda/SplitKvCacheCopy.h"
#include "rtp_llm/models_py/bindings/kernels/sm_copy_kernel.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace rtp_llm {
namespace {

class SplitKvCopyState {
public:
    SplitKvCopyState()  = default;
    ~SplitKvCopyState() = default;

    SplitKvCopyState(const SplitKvCopyState&)            = delete;
    SplitKvCopyState& operator=(const SplitKvCopyState&) = delete;

    void releaseDeviceResources() {
        releaseAll();
    }

    bool run(const std::vector<torch::Tensor>& src,
             const std::vector<torch::Tensor>& dst,
             int                               layer_num,
             int64_t                           kv_stride,
             int64_t                           scale_stride,
             cudaStream_t                      stream) {
        if (layer_num <= 0 || src.size() != dst.size()) {
            return false;
        }
        const size_t L   = static_cast<size_t>(layer_num);
        const size_t tpi = 2u * L;
        const size_t n   = src.size();
        if (n % tpi != 0) {
            return false;
        }
        const size_t kv    = static_cast<size_t>(kv_stride);
        const size_t scale = static_cast<size_t>(scale_stride);
        if (kv + scale == 0) {
            return false;
        }

        const bool h2d = src[0].is_cpu();
        const bool d2h = src[0].is_cuda();
        if (!h2d && !d2h) {
            return false;
        }

        copy_stream_ = stream;

        // Split-KV path is only enabled from KVCacheMemoryConnector with uniform per-layer strides and layout;
        // avoid O(layer * blocks) validation here (hot path).
        int ptr_device = -1;
        if (h2d) {
            if (!dst[0].is_cuda()) {
                return false;
            }
            ptr_device = static_cast<int>(dst[0].get_device());
        } else {
            if (!src[0].is_cuda() || !dst[0].is_cpu()) {
                return false;
            }
            ptr_device = static_cast<int>(src[0].get_device());
        }

        at::cuda::CUDAGuard device_guard(ptr_device);
        check_cuda_value(cudaSetDevice(ptr_device));

        if (buffer_device_ >= 0 && ptr_device != buffer_device_) {
            releaseAll();
        }
        buffer_device_ = ptr_device;

        const size_t block_size      = kv * L + scale * L;
        const size_t ptr_table_bytes = L * sizeof(void*);

        const size_t       block_nums = n / tpi;
        std::vector<void*> h_kv(L);
        std::vector<void*> h_scale(L);

        ensureBuffers(block_size, ptr_table_bytes, stream);

        void* const d_staging     = staging_;
        void* const d_kv_table    = ptr0_;
        void* const d_scale_table = ptr1_;

        for (size_t b = 0; b < block_nums; ++b) {
            const size_t off = b * tpi;
            if (h2d) {
                for (size_t i = 0; i < L; ++i) {
                    h_kv[i]    = dst[off + 2 * i].data_ptr();
                    h_scale[i] = dst[off + 2 * i + 1].data_ptr();
                }
                check_cuda_value(
                    cudaMemcpyAsync(d_staging, src[off].data_ptr(), block_size, cudaMemcpyHostToDevice, stream));
                check_cuda_value(
                    cudaMemcpyAsync(d_kv_table, h_kv.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
                check_cuda_value(
                    cudaMemcpyAsync(d_scale_table, h_scale.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
                sDevMPS::launch_scatter_copy_split(d_staging,
                                                   reinterpret_cast<void**>(d_kv_table),
                                                   reinterpret_cast<void**>(d_scale_table),
                                                   kv,
                                                   scale,
                                                   layer_num,
                                                   /*block_num=*/0,
                                                   stream);
            } else {
                for (size_t i = 0; i < L; ++i) {
                    h_kv[i]    = src[off + 2 * i].data_ptr();
                    h_scale[i] = src[off + 2 * i + 1].data_ptr();
                }
                check_cuda_value(
                    cudaMemcpyAsync(d_kv_table, h_kv.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
                check_cuda_value(
                    cudaMemcpyAsync(d_scale_table, h_scale.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
                sDevMPS::launch_gather_copy_split(reinterpret_cast<const void**>(d_kv_table),
                                                  reinterpret_cast<const void**>(d_scale_table),
                                                  kv,
                                                  scale,
                                                  d_staging,
                                                  layer_num,
                                                  /*block_num=*/0,
                                                  stream);
                check_cuda_value(
                    cudaMemcpyAsync(dst[off].data_ptr(), d_staging, block_size, cudaMemcpyDeviceToHost, stream));
            }
        }

        check_cuda_value(cudaStreamSynchronize(stream));
        check_cuda_error();
        return true;
    }

private:
    void releaseAll() {
        if (buffer_device_ < 0) {
            staging_cap_   = 0;
            ptr_table_cap_ = 0;
            return;
        }
        check_cuda_value(cudaSetDevice(buffer_device_));
        if (copy_stream_) {
            check_cuda_value(cudaStreamSynchronize(copy_stream_));
            if (staging_) {
                check_cuda_value(cudaFreeAsync(staging_, copy_stream_));
                staging_ = nullptr;
            }
            if (ptr0_) {
                check_cuda_value(cudaFreeAsync(ptr0_, copy_stream_));
                ptr0_ = nullptr;
            }
            if (ptr1_) {
                check_cuda_value(cudaFreeAsync(ptr1_, copy_stream_));
                ptr1_ = nullptr;
            }
            check_cuda_value(cudaStreamSynchronize(copy_stream_));
        } else {
            if (staging_) {
                check_cuda_value(cudaFree(staging_));
                staging_ = nullptr;
            }
            if (ptr0_) {
                check_cuda_value(cudaFree(ptr0_));
                ptr0_ = nullptr;
            }
            if (ptr1_) {
                check_cuda_value(cudaFree(ptr1_));
                ptr1_ = nullptr;
            }
        }
        staging_cap_   = 0;
        ptr_table_cap_ = 0;
        buffer_device_ = -1;
        copy_stream_   = nullptr;
    }

    void ensureBuffers(size_t staging_bytes, size_t ptr_table_bytes, cudaStream_t stream) {
        check_cuda_value(cudaSetDevice(buffer_device_));
        if (staging_bytes > staging_cap_) {
            if (staging_) {
                check_cuda_value(cudaStreamSynchronize(stream));
                check_cuda_value(cudaFreeAsync(staging_, stream));
                check_cuda_value(cudaStreamSynchronize(stream));
                staging_     = nullptr;
                staging_cap_ = 0;
            }
            check_cuda_value(cudaMalloc(&staging_, staging_bytes));
            staging_cap_ = staging_bytes;
        }
        if (ptr_table_bytes > ptr_table_cap_) {
            if (ptr0_ || ptr1_) {
                check_cuda_value(cudaStreamSynchronize(stream));
                if (ptr0_) {
                    check_cuda_value(cudaFreeAsync(ptr0_, stream));
                    ptr0_ = nullptr;
                }
                if (ptr1_) {
                    check_cuda_value(cudaFreeAsync(ptr1_, stream));
                    ptr1_ = nullptr;
                }
                check_cuda_value(cudaStreamSynchronize(stream));
                ptr_table_cap_ = 0;
            }
            check_cuda_value(cudaMalloc(&ptr0_, ptr_table_bytes));
            check_cuda_value(cudaMalloc(&ptr1_, ptr_table_bytes));
            ptr_table_cap_ = ptr_table_bytes;
        }
    }

    void*        staging_{nullptr};
    void*        ptr0_{nullptr};
    void*        ptr1_{nullptr};
    size_t       staging_cap_{0};
    size_t       ptr_table_cap_{0};
    int          buffer_device_{-1};
    cudaStream_t copy_stream_{nullptr};
};

SplitKvCopyState& getState() {
    thread_local SplitKvCopyState state;
    return state;
}

}  // anonymous namespace

bool trySplitKvMultiCopy(const std::vector<torch::Tensor>& src,
                         const std::vector<torch::Tensor>& dst,
                         int                               layer_num,
                         int64_t                           kv_cache_stride_bytes,
                         int64_t                           kv_scale_stride_bytes,
                         cudaStream_t                      stream) {
    return getState().run(src, dst, layer_num, kv_cache_stride_bytes, kv_scale_stride_bytes, stream);
}

bool warmupSplitKvCopyKernels(cudaStream_t stream) {
    return sDevMPS::warmup_sm_copy_split_kernels(stream);
}

void releaseSplitKvCopyState() {
    getState().releaseDeviceResources();
}

}  // namespace rtp_llm
