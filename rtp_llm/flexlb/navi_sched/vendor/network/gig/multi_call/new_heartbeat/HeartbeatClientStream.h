#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/ClientTopoInfoMap.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigClientStream.h"

namespace multi_call {

class HeartbeatClientManagerNotifier;
class HostHeartbeatStats;
class StringTable;

class HeartbeatClientStream : public GigClientStream
{
public:
    HeartbeatClientStream(const std::shared_ptr<HostHeartbeatStats> &hostStats,
                          const ClientTopoInfoMapPtr &clientInfoMap, const HeartbeatSpecPtr &spec,
                          const std::shared_ptr<StringTable> &stringTable);
    ~HeartbeatClientStream();

private:
    HeartbeatClientStream(const HeartbeatClientStream &);
    HeartbeatClientStream &operator=(const HeartbeatClientStream &);

public:
    google::protobuf::Message *newReceiveMessage(google::protobuf::Arena *arena) const override;
    bool receive(const GigStreamMessage &message) override;
    void receiveCancel(const GigStreamMessage &message, MultiCallErrorCode ec) override;

public:
    bool tick();
    ClientTopoInfoMapPtr getClientMap() const;
    void disableProvider();

private:
    bool doTick();
    int64_t fillRequest(NewHeartbeatRequest &request);
    bool doReceive(const NewHeartbeatResponse &response, int64_t netLatencyUs, bool &finished,
                   bool &changed);
    void updateStringTable(const StringTableDef &def,
                           std::unordered_set<std::shared_ptr<std::string>> &strSet);
    void setClientMap(const ClientTopoInfoMapPtr &newMap);
    void setLackHashSet(const LackHashSetPtr &lackHashSet);
    ClientTopoInfoPtr getOldInfo(const ClientTopoInfoMapPtr &clientMap, SignatureTy topoSig) const;

private:
    static void fillLackHash(const LackHashSetPtr &lackHashSet, NewHeartbeatRequest &request);

private:
    std::shared_ptr<HostHeartbeatStats> _hostStats;
    HeartbeatSpecPtr _spec;
    std::shared_ptr<StringTable> _stringTable;
    autil::ThreadMutex _sendLock;
    std::shared_ptr<HeartbeatClientManagerNotifier> _notifier;
    mutable autil::ReadWriteLock _clientMapLock;
    ClientTopoInfoMapPtr _clientInfoMap;
    LackHashSetPtr _lackHashSet;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(HeartbeatClientStream);

} // namespace multi_call
