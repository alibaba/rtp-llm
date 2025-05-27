#include <stdlib.h>
#include <mutex>
#include "rtp_llm/cpp/cuda/launch_utils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool getEnvEnablePDL() {
    static std::once_flag flag;
    static bool           enablePDL = true;

    std::call_once(flag, [&]() {
        char* env = getenv("DISABLE_PDL");
        if (env && std::string(env) == "1") {
            enablePDL = false;
        }
        RTP_LLM_LOG_INFO("RTPLLM_ENABLE_PDL: %d", int(enablePDL));
    });
    return enablePDL;
}

}
