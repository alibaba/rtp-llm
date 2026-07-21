#include "rtp_llm/cpp/cache/events/NullPublisher.h"

namespace rtp_llm {

bool NullPublisher::start() noexcept {
    return true;
}

PublishResult NullPublisher::tryPublish(KVCacheEvent) noexcept {
    return PublishResult::DISABLED;
}

void NullPublisher::stop() noexcept {}

PublisherStatus NullPublisher::status() const noexcept {
    return {PublisherState::DISABLED, 0, 0, 0};
}

bool NullPublisher::enabled() const noexcept {
    return false;
}

}  // namespace rtp_llm
