#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigServerStreamCreator.h"
#include "autil/Lock.h"

namespace multi_call {

class HeartbeatServerManager;

class HeartbeatStreamCreator : public GigServerStreamCreator
{
public:
    HeartbeatStreamCreator();
    ~HeartbeatStreamCreator();

private:
    HeartbeatStreamCreator(const HeartbeatStreamCreator &);
    HeartbeatStreamCreator &operator=(const HeartbeatStreamCreator &);

public:
    std::string methodName() const override;
    GigServerStreamPtr create(const StreamRpcParam &param) override;

public:
    void setManager(const std::shared_ptr<HeartbeatServerManager> &manager);

private:
    std::shared_ptr<HeartbeatServerManager> getManager() const;

private:
    mutable autil::ThreadMutex _managerMutex;
    std::shared_ptr<HeartbeatServerManager> _manager;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(HeartbeatStreamCreator);

} // namespace multi_call
