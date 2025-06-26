#include <torch/library.h>
#include "rtp_llm/cpp/th_op/GptInitParameter.h"
#include "rtp_llm/cpp/dataclass/LoadBalance.h"
#include "rtp_llm/cpp/dataclass/EngineScheduleInfo.h"
#include "rtp_llm/cpp/th_op/GptInitParameterRegister.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/RtpLLMOp.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/RtpNorm.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/RtpEmbeddingLookup.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/EmbeddingHandlerOp.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/FlashInferOp.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/FusedQKRmsNorm.h"

using namespace rtp_llm;

namespace torch_ext {

// TODO(wangyin.yx): organize these regsiter function into classified registration functions

PYBIND11_MODULE(libth_transformer, m) {
    rtp_llm::registerLoadBalanceInfo(m);
    rtp_llm::registerEngineScheduleInfo(m);
    register_parallelism_distributed_config(m);
    register_concurrency_config(m);
    register_fmha_config(m);
    register_kvcache_config(m);
    register_profiling_debug_logging_config(m);
    register_hwkernel_config(m);
    register_device_resource_config(m);
    register_sampler_config(m);
    register_moe_config(m);
    register_model_specific_config(m);
    register_speculative_execution_config(m);
    register_service_discovery_config(m);
    register_cache_store_config(m);
    register_scheduler_config(m);
    register_batch_decode_scheduler_config(m);
    register_fifo_scheduler_config(m);
    register_misc_config(m);
    register_attn_params(m),
    registerGptInitParameter(m);
    registerRtpLLMOp(m);
    registerRtpEmbeddingOp(m);
    registerEmbeddingHandler(m);
    registerDeviceOps(m);
    register_arpc_config(m);
    registerFlashInferOp(m);
}

TORCH_LIBRARY_FRAGMENT(libth_transformer, m) {
  m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, int cuda_stream) -> ()");
  m.impl("rmsnorm", torch::kCUDA, &rmsnorm);

  m.def("fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, int cuda_stream) -> ()");
  m.impl("fused_add_rmsnorm", torch::kCUDA, &fused_add_rmsnorm);

  m.def("silu_and_mul(Tensor! output, Tensor input, int cuda_stream) -> ()");
  m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  m.def("fused_qk_rmsnorm(Tensor! IO, Tensor q_gamma, Tensor k_gamma, float layernorm_eps, int q_group_num, int k_group_num, int m, int n, int norm_size, int cuda_stream) -> ()");
  m.impl("fused_qk_rmsnorm", torch::kCUDA, &FusedQKRMSNorm);

  m.def("layernorm(Tensor! output, Tensor input, Tensor weight, Tensor beta, float eps, int cuda_stream) -> ()");
  m.impl("layernorm", torch::kCUDA, &layernorm);

  m.def("fused_add_layernorm(Tensor! input, Tensor! residual, Tensor bias, Tensor weight, Tensor beta, float eps, int cuda_stream) -> ()");
  m.impl("fused_add_layernorm", torch::kCUDA, &fused_add_layernorm);

  m.def("embedding(Tensor! output, Tensor input, Tensor weight, int cuda_stream) -> ()");
  m.impl("embedding", torch::kCUDA, &embedding);
}

}  // namespace torch_ext
