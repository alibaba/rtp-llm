#include "rtp_llm/cpp/utils/SignalUtils.h"
#include "rtp_llm/cpp/pybind/common/InitEngineOps.h"
#include <cstdio>
#include <string>
#include "absl/debugging/symbolize.h"

namespace torch_ext {

void initEngine(std::string py_ft_alog_file_path) {
    bool init_log_success = rtp_llm::initLogger(py_ft_alog_file_path);
    fflush(stdout);
    fflush(stderr);

    if (!init_log_success) {
        std::runtime_error("init logger failed");
    }

    absl::InitializeSymbolizer(nullptr);

    RTP_LLM_LOG_INFO("install sighandler begin");
    if (!rtp_llm::installSighandler()) {
        std::cerr << "install sighandler failed" << std::endl;
        std::runtime_error("install sighandler failed");
    }

    RTP_LLM_LOG_INFO("install sighandler success");
    return;
}

static auto init_engine_func = torch::RegisterOperators("rtp_llm::init_engine", &torch_ext::initEngine);

}  // namespace torch_ext
