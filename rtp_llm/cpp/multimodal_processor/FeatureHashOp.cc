#include "rtp_llm/cpp/multimodal_processor/FeatureHashOp.h"
#include "rtp_llm/cpp/multimodal_processor/FeatureHash.h"

#if USING_CUDA
#include <cstring>
#include <limits>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "rtp_llm/cpp/multimodal_processor/CudaPinnedCopy.h"
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

std::vector<torch::Tensor> getMultimodalFeatureHashes(const std::vector<torch::Tensor>& embeddings) {
    std::vector<torch::Tensor> outputs;
    outputs.reserve(embeddings.size());
#if USING_CUDA
    bool    same_cuda_device = embeddings.size() > 1;
    int64_t total_rows       = 0;
    for (const auto& embedding : embeddings) {
        TORCH_CHECK(embedding.defined() && embedding.dim() >= 1 && embedding.size(0) > 0,
                    "multimodal feature tensor is empty");
        const int64_t rows = embedding.size(0);
        TORCH_CHECK(embedding.numel() / rows > 0, "multimodal feature row is empty");
        TORCH_CHECK(rows <= std::numeric_limits<int64_t>::max() - total_rows, "multimodal feature batch is too large");
        total_rows += rows;
        same_cuda_device = same_cuda_device && embedding.is_cuda() && embedding.device() == embeddings.front().device();
    }
    if (same_cuda_device) {
        const c10::cuda::CUDAGuard guard(embeddings.front().device());
        const auto                 producer = c10::cuda::getCurrentCUDAStream(embeddings.front().get_device());
        auto hashes = torch::empty({total_rows}, embeddings.front().options().dtype(torch::kInt32));

        // Keep contiguous input storage alive until the packed hashes reach CPU.
        // Only the int32 outputs are packed, so input shapes and dtypes may vary.
        std::vector<torch::Tensor> inputs;
        inputs.reserve(embeddings.size());
        for (const auto& embedding : embeddings) {
            TORCH_CHECK(embedding.size(0) <= std::numeric_limits<int>::max(), "multimodal feature tensor has too many rows");
            inputs.push_back(embedding.contiguous());
        }
        torch::Tensor staging;
        int64_t       offset = 0;
        try {
            for (const auto& input : inputs) {
                const auto rows      = input.size(0);
                const auto row_bytes = input.numel() / rows * input.element_size();
                const auto error = invokeFeatureHash(
                    input.data_ptr(), rows, row_bytes, hashes.data_ptr<int32_t>() + offset, producer.stream());
                TORCH_CHECK(error == cudaSuccess,
                            "failed to hash multimodal features on GPU: ",
                            cudaGetErrorString(error));
                offset += rows;
            }
            staging = copyCudaTensorToPinnedHost(hashes, producer);
        } catch (...) {
            // A previous launch may still be reading an input from another
            // allocation stream. Keep every input alive through that launch.
            (void)cudaStreamSynchronize(producer.stream());
            throw;
        }

        offset = 0;
        for (const auto& input : inputs) {
            const auto rows = input.size(0);
            auto output = torch::empty({rows}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
            // Independent storage avoids retaining the staging allocation or
            // the other images' hashes in a single cache entry.
            std::memcpy(output.data_ptr<int32_t>(), staging.data_ptr<int32_t>() + offset, rows * sizeof(int32_t));
            outputs.push_back(std::move(output));
            offset += rows;
        }
        return outputs;
    }
#endif
    for (const auto& embedding : embeddings) {
        outputs.push_back(getMultimodalFeatureHash(embedding));
    }
    return outputs;
}

}  // namespace rtp_llm
