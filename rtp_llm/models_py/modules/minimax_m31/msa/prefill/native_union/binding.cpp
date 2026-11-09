#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

extern "C" int m31_fi_launch(const void*,
                             const void*,
                             const int*,
                             const int*,
                             const int*,
                             const int*,
                             void*,
                             int,
                             int,
                             int,
                             int,
                             int,
                             float,
                             float,
                             float,
                             int,
                             int,
                             cudaStream_t);

void run(torch::Tensor q,
         torch::Tensor pages,
         torch::Tensor topk,
         torch::Tensor cu_q,
         torch::Tensor table,
         torch::Tensor used_k,
         torch::Tensor output,
         double        scale) {
    TORCH_CHECK(q.is_cuda() && pages.is_cuda() && topk.is_cuda() && cu_q.is_cuda() && table.is_cuda()
                    && used_k.is_cuda() && output.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16, "Q and output must be BF16");
    TORCH_CHECK(pages.scalar_type() == at::kByte && topk.scalar_type() == at::kInt && cu_q.scalar_type() == at::kInt
                    && table.scalar_type() == at::kInt && used_k.scalar_type() == at::kInt,
                "bad index/page dtype");
    TORCH_CHECK(q.dim() == 3 && q.size(2) == 128 && q.size(1) % 16 == 0 && topk.dim() == 3
                    && topk.size(0) == q.size(1) / 16 && topk.size(1) == q.size(0) && topk.size(2) == 16,
                "bad Q/TopK geometry");
    TORCH_CHECK(output.sizes() == q.sizes(), "output shape must match Q");
    TORCH_CHECK(pages.dim() == 2 && pages.size(1) == (q.size(1) / 16) * 18432, "bad page geometry");
    TORCH_CHECK(cu_q.dim() == 1 && table.dim() == 2 && used_k.dim() == 1 && cu_q.numel() == used_k.numel() + 1
                    && table.size(0) == used_k.numel() && table.size(1) > 0,
                "bad request metadata geometry");
    TORCH_CHECK(q.device() == pages.device() && q.device() == topk.device() && q.device() == cu_q.device()
                    && q.device() == table.device() && q.device() == used_k.device() && q.device() == output.device(),
                "all tensors must share one device");
    TORCH_CHECK(q.is_contiguous() && pages.is_contiguous() && cu_q.is_contiguous() && table.is_contiguous()
                    && used_k.is_contiguous() && output.is_contiguous() && topk.stride(2) == 1,
                "noncontiguous input");
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
    int  status = m31_fi_launch(q.data_ptr(),
                               pages.data_ptr(),
                               topk.data_ptr<int>(),
                               cu_q.data_ptr<int>(),
                               table.data_ptr<int>(),
                               used_k.data_ptr<int>(),
                               output.data_ptr(),
                               q.size(0),
                               used_k.numel(),
                               table.size(1),
                               topk.stride(0),
                               topk.stride(1),
                               scale,
                               1.0f,
                               1.0f,
                               q.size(1) / 16,
                               q.get_device(),
                               stream);
    TORCH_CHECK(status == 0, "M3.1 FI candidate CUDA error: ", cudaGetErrorString((cudaError_t)status));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run);
}
