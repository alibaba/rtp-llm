#include <stdlib.h>
#include <mutex>
#include "rtp_llm/cpp/cuda/launch_utils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {

bool getEnvEnablePDL() {
    static std::once_flag flag;
    static bool           enablePDL = true;

    std::call_once(flag, [&]() {
        enablePDL = !StaticConfig::user_disable_pdl;
        RTP_LLM_LOG_INFO("RTPLLM_ENABLE_PDL: %d", int(enablePDL));
    });
    return enablePDL;
}

}  // namespace rtp_llm
