#include "rtp_llm/cpp/multimodal_processor/FeatureHashOp.h"
#include "rtp_llm/cpp/multimodal_processor/FeatureHash.h"

#if USING_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "rtp_llm/cpp/multimodal_processor/FeatureHashKernel.h"
#endif

namespace rtp_llm {

torch::Tensor getMultimodalFeatureHash(const torch::Tensor& embedding) {
    TORCH_CHECK(embedding.defined() && embedding.dim() >= 1 && embedding.size(0) > 0,
                "multimodal feature tensor is empty");
    const int64_t rows      = embedding.size(0);
    const int64_t row_bytes = embedding.numel() / rows * embedding.element_size();
    TORCH_CHECK(row_bytes > 0, "multimodal feature row is empty");
    auto hashes = torch::empty({rows}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
#if USING_CUDA
    if (embedding.is_cuda()) {
        const c10::cuda::CUDAGuard guard(embedding.device());
        auto                       emb        = embedding.contiguous();
        auto                       gpu_hashes = torch::empty({rows}, emb.options().dtype(torch::kInt32));
        const cudaStream_t         stream     = c10::cuda::getCurrentCUDAStream(emb.get_device());
        auto error = invokeFeatureHash(emb.data_ptr(), rows, row_bytes, gpu_hashes.data_ptr<int32_t>(), stream);
        if (error == cudaSuccess) {
            error = cudaMemcpyAsync(hashes.data_ptr<int32_t>(),
                                    gpu_hashes.data_ptr<int32_t>(),
                                    rows * sizeof(int32_t),
                                    cudaMemcpyDeviceToHost,
                                    stream);
        }
        if (error == cudaSuccess) {
            error = cudaStreamSynchronize(stream);
        }
        TORCH_CHECK(error == cudaSuccess, "failed to hash multimodal features on GPU: ", cudaGetErrorString(error));
        return hashes;
    }
#endif
    auto        emb    = embedding.to(torch::kCPU).contiguous();
    const auto* bytes  = static_cast<const uint8_t*>(emb.data_ptr());
    auto*       output = hashes.data_ptr<int32_t>();
    for (int64_t row = 0; row < rows; ++row) {
        output[row] = featureHashToTokenId(hashFeatureRowCpu(bytes + row * row_bytes, row_bytes));
    }
    return hashes;
}

}  // namespace rtp_llm
