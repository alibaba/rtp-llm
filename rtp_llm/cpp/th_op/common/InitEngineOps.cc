#include "rtp_llm/cpp/utils/SignalUtils.h"
#include "rtp_llm/cpp/th_op/common/InitEngineOps.h"
#include <cstdio>

namespace torch_ext {

void initEngine() {
    bool init_log_success = rtp_llm::initLogger();
    fflush(stdout);
    fflush(stderr);

    if (!init_log_success) {
        std::runtime_error("init logger failed");
    }


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
