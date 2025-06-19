#include "rtp_llm/cpp/th_op/GptInitParameter.h"
#include "rtp_llm/cpp/dataclass/LoadBalance.h"
#include "rtp_llm/cpp/dataclass/EngineScheduleInfo.h"
#include "rtp_llm/cpp/th_op/GptInitParameterRegister.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/RtpLLMOp.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/EmbeddingHandlerOp.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include <torch/library.h>
#include "3rdparty/flashinfer/flashinfer.h"

using namespace rtp_llm;

namespace torch_ext {
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
    registerGptInitParameter(m);
    registerRtpLLMOp(m);
    registerRtpEmbeddingOp(m);
    registerEmbeddingHandler(m);
    registerDeviceOps(m);
    register_arpc_config(m);
}

TORCH_LIBRARY(libth_transformer, m) {
  m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, int cuda_stream) -> ()");
  m.impl("rmsnorm", torch::kCUDA, &rmsnorm);
}

}  // namespace torch_ext
