#pragma once

#include <multi_call/stream/GigServerStream.h>

#include "navi/distribute/NaviStreamBase.h"
#include "navi/engine/Navi.h"
#include "navi/engine/TaskQueue.h"

namespace navi {

class ServerRemoteGraph;
class NaviServerStreamCreator;

class NaviServerStream : public autil::ObjectTracer<NaviServerStream, true>,
                         public multi_call::GigServerStream,
                         public NaviStreamBase
{
public:
    NaviServerStream(const multi_call::StreamRpcParam &param,
                     NaviServerStreamCreator *creator,
                     const std::shared_ptr<NaviSnapshot> &snapshot);
    ~NaviServerStream();

private:
    NaviServerStream(const NaviServerStream &);
    NaviServerStream &operator=(const NaviServerStream &);

public:
    google::protobuf::Message *newReceiveMessage(
            google::protobuf::Arena *arena) const override;
    bool receive(const multi_call::GigStreamMessage &message) override;
    void receiveCancel(const multi_call::GigStreamMessage &message,
                       multi_call::MultiCallErrorCode ec) override;
    bool notifyReceive(multi_call::PartIdTy partId) override;
    void setTaskQueueHolder(TaskQueueHolder token);
    TaskQueueHolder stealTaskQueueHolder();
    void *beginReceive(multi_call::PartIdTy partId) override;
    void endReceive(void *handle, bool fake) override;
    void *beginSend(multi_call::PartIdTy partId) override;
    void endSend(void *handle, bool fake) override;
private:
    void initThreadPool();
    bool notifyInitReceive(multi_call::PartIdTy partId);
    bool runGraph(NaviMessage *request, const ArenaPtr &arena);

private:
    NaviServerStreamCreator *_creator = nullptr;
    std::shared_ptr<NaviSnapshot> _snapshot;
    bool _initMessage = true;
    bool _needGraphDef = true;
    TaskQueueHolder _taskQueueHolder;
};

NAVI_TYPEDEF_PTR(NaviServerStream);

}
