#include "rtp_llm/cpp/core/SplitKvCacheCopy.h"
#include "rtp_llm/cpp/kernels/SplitKvCopyKernel.h"
#include "rtp_llm/cpp/runtime/DeviceError.h"

namespace rtp_llm {

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

    void* staging  = nullptr;
    void* kv_table = nullptr;
    void* sc_table = nullptr;

    RTP_LLM_DEVICE_CHECK(cudaMalloc(&staging, block_size));
    RTP_LLM_DEVICE_CHECK(cudaMalloc(&kv_table, ptr_table_bytes));
    RTP_LLM_DEVICE_CHECK(cudaMalloc(&sc_table, ptr_table_bytes));

    std::vector<void*> h_kv(L);
    std::vector<void*> h_scale(L);

    for (size_t b = 0; b < block_nums; ++b) {
        const size_t off = b * tpi;
        if (h2d) {
            for (size_t i = 0; i < L; ++i) {
                h_kv[i]    = dst[off + 2 * i].data_ptr();
                h_scale[i] = dst[off + 2 * i + 1].data_ptr();
            }
            RTP_LLM_DEVICE_CHECK(
                cudaMemcpyAsync(staging, src[off].data_ptr(), block_size, cudaMemcpyHostToDevice, stream));
            RTP_LLM_DEVICE_CHECK(
                cudaMemcpyAsync(kv_table, h_kv.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
            RTP_LLM_DEVICE_CHECK(
                cudaMemcpyAsync(sc_table, h_scale.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
            sDevMPS::launch_scatter_copy_split(staging,
                                               reinterpret_cast<void**>(kv_table),
                                               reinterpret_cast<void**>(sc_table),
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
            RTP_LLM_DEVICE_CHECK(
                cudaMemcpyAsync(kv_table, h_kv.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
            RTP_LLM_DEVICE_CHECK(
                cudaMemcpyAsync(sc_table, h_scale.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
            sDevMPS::launch_gather_copy_split(reinterpret_cast<const void**>(kv_table),
                                              reinterpret_cast<const void**>(sc_table),
                                              kv,
                                              scale,
                                              staging,
                                              layer_num,
                                              0,
                                              stream);
            RTP_LLM_DEVICE_CHECK(
                cudaMemcpyAsync(dst[off].data_ptr(), staging, block_size, cudaMemcpyDeviceToHost, stream));
        }
    }

    RTP_LLM_DEVICE_CHECK(cudaFreeAsync(staging, stream));
    RTP_LLM_DEVICE_CHECK(cudaFreeAsync(kv_table, stream));
    RTP_LLM_DEVICE_CHECK(cudaFreeAsync(sc_table, stream));

    return true;
}

bool warmupSplitKvCopyKernels(cudaStream_t stream) {
    return sDevMPS::warmup_sm_copy_split_kernels(stream);
}

}  // namespace rtp_llm
