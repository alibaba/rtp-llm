#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigServerStream.h"

namespace multi_call {

class GigServerStreamCreator
{
public:
    GigServerStreamCreator();
    virtual ~GigServerStreamCreator();

private:
    GigServerStreamCreator(const GigServerStreamCreator &);
    GigServerStreamCreator &operator=(const GigServerStreamCreator &);

public:
    virtual std::string methodName() const = 0;
    virtual GigServerStreamPtr create(const StreamRpcParam &param) = 0;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(GigServerStreamCreator);

} // namespace multi_call

