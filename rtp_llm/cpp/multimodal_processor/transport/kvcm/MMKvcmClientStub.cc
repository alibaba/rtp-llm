#include "rtp_llm/cpp/multimodal_processor/transport/kvcm/MMKvcmClient.h"

namespace rtp_llm {

bool hasMMKvcmImplementation() {
    return false;
}

std::shared_ptr<MMKvcmClient> createMMKvcmClient(const MMKvcmConfig&) {
    return nullptr;
}

}  // namespace rtp_llm
