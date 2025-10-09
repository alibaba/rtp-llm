#include "rtp_llm/cpp/utils/SignalUtils.h"
#include "rtp_llm/cpp/pybind/common/InitEngineOps.h"
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
