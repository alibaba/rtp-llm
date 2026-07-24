#include "rtp_llm/cpp/runtime/Bootstrap.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "autil/StackTracer.h"
#include "autil/EnvUtil.h"
#include <cstdio>
#include <mutex>
#include <unistd.h>

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#elif USING_ROCM
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include "rtp_llm/models_py/bindings/rocm/hip_host_utils.h"
#endif

namespace rtp_llm::process {

namespace {
std::once_flag g_bootstrap_flag;
}  // anonymous namespace

void bootstrap(const BootstrapConfig& cfg) {
    std::call_once(g_bootstrap_flag, [&]() {
        setlinebuf(stdout);

        if (cfg.enable_stack_tracer) {
            autil::EnvUtil::setEnv("STACK_TRACER_LOG", "true");
            DECLARE_STACK_TRACER_FILE("rtp_llm_stack.log");
        }

#if USING_CUDA
        RTP_LLM_LOG_INFO("Initialize runtime. device_id=%zu", cfg.device_id);
        check_cuda_value(cudaSetDevice(cfg.device_id));
        at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream());
#elif USING_ROCM
        RTP_LLM_LOG_INFO("Initialize runtime (ROCm). device_id=%zu", cfg.device_id);
        ROCM_CHECK(hipSetDevice(cfg.device_id));
#endif
        RTP_LLM_LOG_INFO("Runtime bootstrap done");
    });
}

}  // namespace rtp_llm::process
