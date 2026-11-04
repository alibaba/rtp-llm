#pragma once

#include <grpc++/generic/async_generic_service.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc/server/GrpcServerWorker.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/GigStreamHeader.pb.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamHandlerBase.h"

namespace multi_call {

class QueryInfo;
class GrpcServerWorker;

class GrpcServerStreamHandler : public GigStreamHandlerBase
{
public:
    GrpcServerStreamHandler(GrpcServerWorker *worker, const ServerCompletionQueueStatusPtr &cqs);
    ~GrpcServerStreamHandler();

private:
    GrpcServerStreamHandler(const GrpcServerStreamHandler &);
    GrpcServerStreamHandler &operator=(const GrpcServerStreamHandler &);

public:
    bool init() override;

public:
    QueryInfoPtr getQueryInfo() const;
    std::string getPeer() const;
    int64_t getBeginTime() const override {
        return _beginTime;
    }
    int64_t getBeginTimeNs() const override {
        return _beginTimeNs;
    }
private:
    void triggerNew();
    void initCallback(bool ok) override;
    bool receiveBeginQuery();
    bool receiveInitQuery();
    bool getServerContextMetaValue(const std::string &key, std::string &value) const;
    bool hasMetaKey(const std::string &key) const;
    GigStreamClosureBase *doSend(bool sendEof, grpc::ByteBuffer *message,
                                 GigStreamClosureBase *closure, bool &ok) override;
    GigStreamClosureBase *receiveNext() override;
    bool ignoreReceiveError() override;
    GigStreamClosureBase *finish() override;
    void doFinishCallback(bool ok) override;
    void clean() override;
    int64_t getLatency() const;
    void setQueryInfo(const std::shared_ptr<QueryInfo> &queryInfo);
    std::shared_ptr<QueryInfo> stealQueryInfo();

private:
    // receive
    GrpcServerWorker *_worker;
    ServerCompletionQueueStatusPtr _cqs;
    grpc::GenericServerContext _serverContext;
    grpc::GenericServerAsyncReaderWriter _grpcStream;
    bool _triggered;
    bool _finished;
    int64_t _beginTime;
    int64_t _beginTimeNs;
    mutable autil::ThreadMutex _queryInfoLock;
    std::shared_ptr<QueryInfo> _queryInfo;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(GrpcServerStreamHandler);

} // namespace multi_call

