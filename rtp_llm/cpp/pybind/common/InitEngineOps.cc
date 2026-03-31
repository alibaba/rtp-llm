#include "rtp_llm/cpp/utils/SignalUtils.h"
#include "rtp_llm/cpp/pybind/common/InitEngineOps.h"
#include "rtp_llm/cpp/core/ExecOps.h"
#include "absl/debugging/symbolize.h"
#include <cstdio>
#include <string>

namespace rtp_llm {

void initEngine(std::string py_ft_alog_file_path) {
    bool init_log_success = rtp_llm::initLogger(py_ft_alog_file_path);
    fflush(stdout);
    fflush(stderr);

    if (!init_log_success) {
        throw std::runtime_error("init logger failed");
    }

    RTP_LLM_LOG_INFO("install sighandler begin");
    if (!rtp_llm::installSighandler()) {
        throw std::runtime_error("install sighandler failed");
    }

    RTP_LLM_LOG_INFO("install sighandler success");
}

static auto init_engine_func = torch::RegisterOperators("rtp_llm::init_engine", &rtp_llm::initEngine);

}  // namespace rtp_llm

static torch::Tensor
rtp_llm_preprocess_gemm_weight_by_key(const std::string& key, torch::Tensor weight, bool use_arm_gemm_use_kai) {
    return rtp_llm::preprocessGemmWeightByKey(key, weight, use_arm_gemm_use_kai);
}

static torch::Tensor rtp_llm_preprocess_weight_scale(torch::Tensor weight, torch::Tensor scale) {
    return rtp_llm::preprocessWeightScale(weight, scale);
}

static void rtp_llm_mask_logits(torch::Tensor logits, torch::Tensor mask) {
    rtp_llm::runtimeMaskLogits(logits, mask);
}

static void rtp_llm_sync_and_check() {
    rtp_llm::runtimeSyncAndCheck();
}

static auto reg_preprocess_gemm_weight =
    torch::RegisterOperators("rtp_llm::preprocess_gemm_weight_by_key", &rtp_llm_preprocess_gemm_weight_by_key);
static auto reg_preprocess_weight_scale =
    torch::RegisterOperators("rtp_llm::preprocess_weight_scale", &rtp_llm_preprocess_weight_scale);
static auto reg_mask_logits    = torch::RegisterOperators("rtp_llm::mask_logits", &rtp_llm_mask_logits);
static auto reg_sync_and_check = torch::RegisterOperators("rtp_llm::sync_and_check", &rtp_llm_sync_and_check);
