#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/ServerTopoMap.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigServerStream.h"

namespace multi_call {

class HeartbeatServerManager;
class NewHeartbeatRequest;
class NewHeartbeatResponse;

class HeartbeatServerStream : public GigServerStream
{
public:
    HeartbeatServerStream(const std::shared_ptr<HeartbeatServerManager> &manager);
    ~HeartbeatServerStream();

private:
    HeartbeatServerStream(const HeartbeatServerStream &);
    HeartbeatServerStream &operator=(const HeartbeatServerStream &);

public:
    google::protobuf::Message *newReceiveMessage(google::protobuf::Arena *arena) const override;
    bool receive(const multi_call::GigStreamMessage &message) override;
    void receiveCancel(const multi_call::GigStreamMessage &message,
                       multi_call::MultiCallErrorCode ec) override;
public:
    uint64_t getClientId() const {
        return _clientId;
    }
    bool flush(bool sync);
    void stop();
    void fillResponse(NewHeartbeatResponse &response);
    bool cancelled() const {
        return _cancelled;
    }
    bool isAllReplicaStopped(SignatureTy sig);

private:
    bool fillHash(const StringHashRequestDef &request, NewHeartbeatResponse &response) const;
    void updateClientSignatureMap(const NewHeartbeatRequest &request);
    SignatureMapPtr getClientSignatureMap() const;
    void setClientSignatureMap(const SignatureMapPtr &newMap);

private:
    uint64_t _clientId;
    ServerTopoMapPtr _serverTopoMap;
    mutable autil::ThreadMutex _signatureMutex;
    SignatureMapPtr _clientSignatureMap;
    mutable autil::ThreadMutex _flushMutex;
    bool _inited    : 1;
    bool _cancelled : 1;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(HeartbeatServerStream);

} // namespace multi_call
