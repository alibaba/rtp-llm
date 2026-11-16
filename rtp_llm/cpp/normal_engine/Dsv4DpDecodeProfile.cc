#include "rtp_llm/cpp/normal_engine/Dsv4DpDecodeProfile.h"

#include <algorithm>
#include <limits>
#include <pybind11/embed.h>
#include "rtp_llm/cpp/cache/WarmUpResult.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/normal_engine/Dsv4DpDecodeRuntime.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#endif

namespace rtp_llm {

namespace py = pybind11;

namespace {

void reserveDsv4DecodeWorkspace(const EngineInitParams& params, std::optional<WarmUpResult>& warm_up_result);

}  // namespace

std::unique_ptr<DecodeReplicaCoordinator>
createDsv4DpDecodeCoordinator(const EngineInitParams& params, std::string_view resolved_moe_strategy, bool) {
    const auto& parallelism_config = params.parallelism_config;
    const auto& pd_sep_config      = params.pd_sep_config;
    const auto& sp_config          = params.sp_config;
    const auto& eplb_config        = params.eplb_config;
    const auto& kv_cache_config    = params.kv_cache_config;
    const auto& runtime_config     = params.runtime_config;
    if (resolved_moe_strategy != "sm120_graph_decode") {
        return nullptr;
    }
    RTP_LLM_CHECK_WITH_INFO(
        params.model_config_.model_type == "deepseek_v4" && parallelism_config.role_type == pd_sep_config.role_type
            && isDsv4DedicatedDecodeTopology(pd_sep_config.role_type,
                                             parallelism_config.pp_size,
                                             parallelism_config.tp_size,
                                             parallelism_config.dp_size,
                                             parallelism_config.ep_size,
                                             parallelism_config.world_size,
                                             parallelism_config.local_world_size,
                                             parallelism_config.prefill_cp_config.is_enabled()),
        "SM120 DSv4 dedicated decode requires single-host PP1/TP1 and DP=EP=world in {4, 8}, with local CP off");
#if USING_CUDA
    const auto [major, minor] = getComputeCapability(parallelism_config.local_rank);
    RTP_LLM_CHECK_WITH_INFO(major == 12 && minor == 0 && params.hw_kernel_config.enable_cuda_graph,
                            "sm120_graph_decode requires exact SM120 and enable_cuda_graph");
#else
    RTP_LLM_FAIL("sm120_graph_decode requires an exact SM120 CUDA device");
#endif
    RTP_LLM_CHECK_WITH_INFO(sp_config.type == SP_TYPE_NONE, "sm120_graph_decode supports ordinary decode only");
    RTP_LLM_CHECK_WITH_INFO(!eplb_config.enable_eplb() && kv_cache_config.multi_task_prompt_tokens.empty(),
                            "sm120_graph_decode requires fixed expert ownership and a dedicated decode worker "
                            "without local system prompts");
    RTP_LLM_CHECK_WITH_INFO(!runtime_config.use_batch_decode_scheduler
                                || runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type == 0,
                            "sm120_graph_decode requires the decode warmup for batch scheduling");
    constexpr int phase_width = 1;
    auto          buckets     = params.hw_kernel_config.decode_capture_batch_sizes;
    std::sort(buckets.begin(), buckets.end());
    RTP_LLM_CHECK_WITH_INFO(validDsv4DecodeCaptureBuckets(buckets, parallelism_config.world_size, phase_width)
                                && buckets.back() <= params.concurrency_config.concurrency_limit
                                && params.model_config_.max_seq_len > phase_width
                                && params.model_config_.max_seq_len <= std::numeric_limits<int>::max(),
                            "sm120_graph_decode requires an explicit, bounded decode capture catalog and context");
    return std::make_unique<Dsv4DpDecodeRuntime>(parallelism_config.world_size, reserveDsv4DecodeWorkspace);
}

namespace {

void reserveDsv4DecodeWorkspace(const EngineInitParams& params, std::optional<WarmUpResult>& warm_up_result) {
    const auto& buckets   = params.hw_kernel_config.decode_capture_batch_sizes;
    const auto  device_id = getDeviceId();
    const auto  before    = getGpuExecStatus().device_memory_status.available_bytes;
    size_t      workspace_bytes;
    {
        py::gil_scoped_acquire gil;
        workspace_bytes =
            params.py_model.attr("reserve_sm120_decode_workspace")(buckets, std::vector<int64_t>{1}, device_id)
                .cast<size_t>();
    }

    const auto after                   = getGpuExecStatus().device_memory_status.available_bytes;
    const auto cache_and_runtime_bytes = dsv4DecodeWorkspaceBudget(
        before, after, warm_up_result ? std::optional(warm_up_result->device_reserved_bytes) : std::nullopt);
    if (params.kv_cache_config.kv_cache_mem_mb > 0) {
        RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(params.kv_cache_config.kv_cache_mem_mb)
                                    <= cache_and_runtime_bytes / (1024 * 1024),
                                "explicit KV cache reservation leaves insufficient DSv4 decode workspace memory");
    }
    warm_up_result = WarmUpResult{cache_and_runtime_bytes, warm_up_result ? warm_up_result->max_used_memory : 0};
    RTP_LLM_LOG_INFO("DSv4 decode fixed EP workspace reserve=%zu MiB, remaining=%zu MiB",
                     workspace_bytes / (1024 * 1024),
                     cache_and_runtime_bytes / (1024 * 1024));
}

}  // namespace

}  // namespace rtp_llm
