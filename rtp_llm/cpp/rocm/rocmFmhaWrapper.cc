#include "rocmFmhaWrapper.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {
inline void logStubWarningOnce() {
    static bool warned = false;
    if (!warned) {
        warned = true;
        RTP_LLM_LOG_WARNING("rocmFmhaWrapper is using stub implementation; CK FMHA path is disabled.");
    }
}
}  // namespace

uint32_t rocmFmhaWrapper::runCKFmha(void*,
                                    void*,
                                    void*,
                                    void*,
                                    void*,
                                    size_t,
                                    size_t,
                                    size_t,
                                    void*,
                                    void*,
                                    void*,
                                    void*,
                                    void*,
                                    bool,
                                    bool) {
    logStubWarningOnce();
    return 0;
}

uint32_t rocmFmhaWrapper::runCKFmhaV2(void*,
                                      void*,
                                      void*,
                                      void*,
                                      void*,
                                      size_t,
                                      size_t,
                                      size_t,
                                      void*,
                                      void*,
                                      void*,
                                      void*,
                                      void*,
                                      size_t,
                                      bool,
                                      bool) {
    logStubWarningOnce();
    return 0;
}

uint32_t rocmFmhaWrapper::runCKFmhaMLA(void*,
                                       void*,
                                       void*,
                                       void*,
                                       void*,
                                       size_t,
                                       size_t,
                                       float,
                                       void*,
                                       void*,
                                       void*,
                                       void*,
                                       void*) {
    logStubWarningOnce();
    return 0;
}

}  // namespace rtp_llm
