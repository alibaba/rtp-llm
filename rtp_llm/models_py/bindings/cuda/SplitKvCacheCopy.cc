#include "rtp_llm/models_py/bindings/cuda/SplitKvCacheCopy.h"
#include "rtp_llm/models_py/bindings/MandatoryDrain.h"
#include "rtp_llm/models_py/bindings/common/kernels/sm_copy_kernel.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

#include <cstdlib>

namespace rtp_llm {

namespace {

class DeviceAllocation {
public:
    ~DeviceAllocation() {
        if (ptr_ == nullptr) {
            return;
        }
        const auto status = cudaFree(ptr_);
        if (status != cudaSuccess) {
            RTP_LLM_LOG_ERROR("failed to release copy scratch allocation, error=%d(%s)",
                              static_cast<int>(status),
                              cudaGetErrorString(status));
            std::abort();
        }
    }

    DeviceAllocation()                                  = default;
    DeviceAllocation(const DeviceAllocation&)            = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    void allocate(size_t bytes) {
        void* allocation = nullptr;
        check_cuda_value(cudaMalloc(&allocation, bytes));
        ptr_ = allocation;
    }

    void* get() const {
        return ptr_;
    }

    void release() {
        ptr_ = nullptr;
    }

private:
    void* ptr_{nullptr};
};

}  // namespace

bool splitKvMultiCopy(const std::vector<torch::Tensor>& src,
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

    const bool h2d = src[0].is_cpu() && dst[0].is_cuda();
    const bool d2h = src[0].is_cuda() && dst[0].is_cpu();
    if (!h2d && !d2h) {
        return false;
    }

    const size_t block_size      = kv * L + scale * L;
    const size_t ptr_table_bytes = L * sizeof(void*);
    const size_t block_nums      = n / tpi;

    DeviceAllocation staging;
    DeviceAllocation kv_table;
    DeviceAllocation sc_table;
    staging.allocate(block_size);
    kv_table.allocate(ptr_table_bytes);
    sc_table.allocate(ptr_table_bytes);

    std::vector<void*> h_kv(L);
    std::vector<void*> h_scale(L);

    cudaError_t drain_status = cudaSuccess;
    runWithMandatoryDrain(
        [&]() {
            for (size_t b = 0; b < block_nums; ++b) {
                const size_t off = b * tpi;
                if (h2d) {
                    for (size_t i = 0; i < L; ++i) {
                        h_kv[i]    = dst[off + 2 * i].data_ptr();
                        h_scale[i] = dst[off + 2 * i + 1].data_ptr();
                    }
                    check_cuda_value(cudaMemcpyAsync(
                        staging.get(), src[off].data_ptr(), block_size, cudaMemcpyHostToDevice, stream));
                    check_cuda_value(cudaMemcpyAsync(
                        kv_table.get(), h_kv.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
                    check_cuda_value(cudaMemcpyAsync(
                        sc_table.get(), h_scale.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
                    sDevMPS::launch_scatter_copy_split(staging.get(),
                                                       reinterpret_cast<void**>(kv_table.get()),
                                                       reinterpret_cast<void**>(sc_table.get()),
                                                       kv,
                                                       scale,
                                                       layer_num,
                                                       0,
                                                       stream);
                } else {
                    for (size_t i = 0; i < L; ++i) {
                        h_kv[i]    = src[off + 2 * i].data_ptr();
                        h_scale[i] = src[off + 2 * i + 1].data_ptr();
                    }
                    check_cuda_value(cudaMemcpyAsync(
                        kv_table.get(), h_kv.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
                    check_cuda_value(cudaMemcpyAsync(
                        sc_table.get(), h_scale.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
                    sDevMPS::launch_gather_copy_split(reinterpret_cast<const void**>(kv_table.get()),
                                                      reinterpret_cast<const void**>(sc_table.get()),
                                                      kv,
                                                      scale,
                                                      staging.get(),
                                                      layer_num,
                                                      0,
                                                      stream);
                    check_cuda_value(cudaMemcpyAsync(
                        dst[off].data_ptr(), staging.get(), block_size, cudaMemcpyDeviceToHost, stream));
                }
            }

            check_cuda_value(cudaFreeAsync(staging.get(), stream));
            staging.release();
            check_cuda_value(cudaFreeAsync(kv_table.get(), stream));
            kv_table.release();
            check_cuda_value(cudaFreeAsync(sc_table.get(), stream));
            sc_table.release();
        },
        [&]() {
            drain_status = cudaStreamSynchronize(stream);
            return drain_status == cudaSuccess;
        },
        [&]() {
            RTP_LLM_LOG_ERROR("split copy stream did not reach a terminal state, error=%d(%s)",
                              static_cast<int>(drain_status),
                              cudaGetErrorString(drain_status));
        });

    return true;
}

bool warmupSplitKvCopyKernels(cudaStream_t stream) {
    return sDevMPS::warmup_sm_copy_split_kernels(stream);
}

}  // namespace rtp_llm
