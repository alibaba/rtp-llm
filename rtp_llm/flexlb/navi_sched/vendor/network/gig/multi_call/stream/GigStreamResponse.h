#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/Response.h"

namespace multi_call {

class GigStreamResponse : public Response
{
public:
    GigStreamResponse();
    ~GigStreamResponse();

private:
    GigStreamResponse(const GigStreamResponse &);
    GigStreamResponse &operator=(const GigStreamResponse &);

public:
    void init(void *data) override;
    size_t size() const override {
        return _totalSize;
    }
    void setTotalSize(size_t size) {
        _totalSize = size;
    }

private:
    size_t _totalSize = 0;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(GigStreamResponse);

} // namespace multi_call

