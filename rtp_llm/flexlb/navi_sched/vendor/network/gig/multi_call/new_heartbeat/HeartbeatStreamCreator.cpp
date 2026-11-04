#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/HeartbeatStreamCreator.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/HeartbeatServerManager.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/HeartbeatServerStream.h"

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, HeartbeatStreamCreator);

HeartbeatStreamCreator::HeartbeatStreamCreator() {
}

HeartbeatStreamCreator::~HeartbeatStreamCreator() {
}

std::string HeartbeatStreamCreator::methodName() const {
    return GIG_HEARTBEAT_METHOD_NAME;
}

GigServerStreamPtr HeartbeatStreamCreator::create(const StreamRpcParam &param) {
    auto manager = getManager();
    if (!manager) {
        return nullptr;
    }
    auto stream = std::make_shared<HeartbeatServerStream>(manager);
    manager->addStream(stream);
    return std::dynamic_pointer_cast<GigServerStream>(stream);
}

void HeartbeatStreamCreator::setManager(const HeartbeatServerManagerPtr &manager) {
    autil::ScopedLock scope(_managerMutex);
    _manager = manager;
}

HeartbeatServerManagerPtr HeartbeatStreamCreator::getManager() const {
    autil::ScopedLock scope(_managerMutex);
    return _manager;
}

} // namespace multi_call
