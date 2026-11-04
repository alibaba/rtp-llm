#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamBase.h"

namespace multi_call {

class GigStreamHandlerBase;

class GigServerStream : public GigStreamBase
{
public:
    GigServerStream();
    virtual ~GigServerStream();

private:
    GigServerStream(const GigServerStream &);
    GigServerStream &operator=(const GigServerStream &);

public:
    bool send(PartIdTy partId, bool eof, google::protobuf::Message *message) override;
    void sendCancel(PartIdTy partId, google::protobuf::Message *message) override;

public:
    std::shared_ptr<QueryInfo> getQueryInfo() const override;
    std::string getPeer() const override;
    int64_t getBeginTime() const;
    int64_t getBeginTimeNs() const;
private:
    void setHandler(const std::shared_ptr<GigStreamHandlerBase> &handler);
private:
    friend class GrpcServerStreamHandler;

private:
    std::shared_ptr<GigStreamHandlerBase> _handler;
};

MULTI_CALL_TYPEDEF_PTR(GigServerStream);

} // namespace multi_call

