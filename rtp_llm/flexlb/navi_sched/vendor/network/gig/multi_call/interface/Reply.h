#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/Response.h"

namespace multi_call {
class ChildNodeReply;

class Reply
{
public:
    Reply();
    virtual ~Reply();

private:
    Reply(const Reply &);
    Reply &operator=(const Reply &);

public:
    // virtual for ut
    virtual std::vector<ResponsePtr> getResponses(size_t &lackCount);
    virtual std::vector<ResponsePtr> getResponses(size_t &lackCount,
                                                  std::vector<LackResponseInfo> &lackInfos);
    virtual std::map<std::string, std::vector<ResponsePtr>> getResponseMap(size_t &lackCount);
    void setChildNodeReply(const std::shared_ptr<ChildNodeReply> &reply);

public:
    static std::string getErrorMessage(MultiCallErrorCode ec);

private:
    std::shared_ptr<ChildNodeReply> _childNodeReply;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(Reply);
} // namespace multi_call

