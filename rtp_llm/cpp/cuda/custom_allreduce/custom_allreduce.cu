// Self-contained custom all-reduce CUDA kernels for RTP-LLM.
// Adapted from vllm. No external dependencies beyond PyTorch + CUDA.
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include "rtp_llm/cpp/cuda/custom_allreduce/custom_allreduce.cuh"
#include "rtp_llm/cpp/cuda/custom_allreduce/custom_allreduce.h"

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

namespace rtp_llm {

int64_t custom_ar_init(const std::vector<fptr_t>& fake_ipc_ptrs, torch::Tensor& rank_data,
                       int64_t rank, bool full_nvlink) {
  int world_size = fake_ipc_ptrs.size();
  if (world_size > 8) throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0) throw std::invalid_argument("odd num gpus is not supported");
  if (rank < 0 || rank >= world_size) throw std::invalid_argument("invalid rank");

  custom_ar::Signal* ipc_ptrs[8];
  for (int i = 0; i < world_size; i++)
    ipc_ptrs[i] = reinterpret_cast<custom_ar::Signal*>(fake_ipc_ptrs[i]);
  return (fptr_t) new custom_ar::CustomAllreduce(
      ipc_ptrs, rank_data.data_ptr(), rank_data.numel(), rank, world_size, full_nvlink);
}

static bool _is_weak_contiguous(torch::Tensor& t) {
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() == t.numel() * t.element_size());
}

void custom_ar_all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                          fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<custom_ar::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size, cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }
  switch (out.scalar_type()) {
    case at::ScalarType::Float:
      fa->allreduce<float>(stream, reinterpret_cast<float*>(reg_buffer),
                           reinterpret_cast<float*>(out.data_ptr()), out.numel());
      break;
    case at::ScalarType::Half:
      fa->allreduce<half>(stream, reinterpret_cast<half*>(reg_buffer),
                          reinterpret_cast<half*>(out.data_ptr()), out.numel());
      break;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16:
      fa->allreduce<nv_bfloat16>(stream, reinterpret_cast<nv_bfloat16*>(reg_buffer),
                                  reinterpret_cast<nv_bfloat16*>(out.data_ptr()), out.numel());
      break;
#endif
    default:
      throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
  }
}

void custom_ar_dispose(fptr_t _fa) {
  delete reinterpret_cast<custom_ar::CustomAllreduce*>(_fa);
}

int64_t custom_ar_meta_size() {
  return sizeof(custom_ar::Signal);
}

void custom_ar_register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<custom_ar::CustomAllreduce*>(_fa);
  TORCH_CHECK(fake_ipc_ptrs.size() == (size_t)fa->world_size_);
  void* ipc_ptrs[8];
  for (size_t i = 0; i < fake_ipc_ptrs.size(); i++)
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  fa->register_buffer(ipc_ptrs);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> custom_ar_get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<custom_ar::CustomAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offsets);
}

void custom_ar_register_graph_buffers(
    fptr_t _fa, const std::vector<std::vector<int64_t>>& handles,
    const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<custom_ar::CustomAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (size_t i = 0; i < handles.size(); i++)
    bytes.emplace_back(handles[i].begin(), handles[i].end());
  fa->register_graph_buffers(bytes, offsets);
}

}  // namespace rtp_llm
