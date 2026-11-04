#pragma once

#include "navi/distribute/NaviStreamBase.h"
#include <multi_call/stream/GigClientStream.h>

namespace navi {

class RemoteSubGraphBase;

class NaviClientStream : public autil::ObjectTracer<NaviClientStream, true>,
                         public multi_call::GigClientStream,
                         public NaviStreamBase
{
public:
    NaviClientStream(const multi_call::StreamRpcParam &param,
                     const std::string &bizName,
                     const NaviLoggerPtr &logger);
    ~NaviClientStream();
private:
    NaviClientStream(const NaviClientStream &);
    NaviClientStream &operator=(const NaviClientStream &);
public:
    google::protobuf::Message *newReceiveMessage(
            google::protobuf::Arena *arena) const override;
    bool receive(const multi_call::GigStreamMessage &message) override;
    void receiveCancel(const multi_call::GigStreamMessage &message,
                       multi_call::MultiCallErrorCode ec) override;
    bool notifyReceive(multi_call::PartIdTy partId) override;
    void *beginReceive(multi_call::PartIdTy partId) override;
    void endReceive(void *handle, bool fake) override;
    void *beginSend(multi_call::PartIdTy partId) override;
    void endSend(void *handle, bool fake) override;
    void initMetric(NaviPerf *naviPerf,
                    CollectFlags flags,
                    const MallocTraceParam *mallocParam,
                    const std::shared_ptr<MemLeakCollector> &memLeakCollector);
};

NAVI_TYPEDEF_PTR(NaviClientStream);

}
