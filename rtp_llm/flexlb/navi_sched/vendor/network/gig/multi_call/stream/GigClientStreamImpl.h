#pragma once

#include "absl/container/flat_hash_set.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/Caller.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamMessage.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamRpcInfo.h"
#include "autil/Lock.h"

namespace multi_call {

class SearchServiceResource;
class GigClientStream;
class GigStreamRequest;

class GigClientStreamImpl
{
public:
    GigClientStreamImpl(GigClientStream *stream);
    ~GigClientStreamImpl();

private:
    GigClientStreamImpl(const GigClientStreamImpl &);
    GigClientStreamImpl &operator=(const GigClientStreamImpl &);

public:
    bool init(const std::shared_ptr<ChildNodeReply> &reply,
              const std::vector<std::shared_ptr<SearchServiceResource>> &resourceVec,
              const CallerPtr &caller, bool disableRetry, bool forceStop,
              const absl::flat_hash_set<PartIdTy> &enablePartIds, bool allowLack);
    PartIdTy getPartCount() const;
    bool send(PartIdTy partId, bool eof, google::protobuf::Message *message);
    void sendCancel(PartIdTy partId, google::protobuf::Message *message);
    GigStreamRpcInfo getStreamRpcInfo(PartIdTy partId) const;

private:
    void abort();
    bool writeMessage(bool cancel, const GigStreamMessage &message);

private:
    GigClientStream *_stream;
    std::unordered_map<PartIdTy, std::shared_ptr<GigStreamRequest>> _requestMap;
    PartIdTy _partCount;
    std::shared_ptr<ChildNodeReply> _reply;
    CallerPtr _caller;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(GigClientStreamImpl);

} // namespace multi_call
