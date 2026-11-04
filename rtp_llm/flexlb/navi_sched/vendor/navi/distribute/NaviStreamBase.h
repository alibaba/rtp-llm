#pragma once

#include <multi_call/stream/GigStreamBase.h>

#include "autil/ObjectTracer.h"
#include "navi/engine/DomainHolder.h"
#include "navi/engine/GraphDomainRemoteBase.h"
#include "navi/engine/NaviThreadPool.h"
#include "navi/engine/TaskQueue.h"
#include "navi/distribute/StreamMetric.h"

namespace navi {

class MemLeakCollector;

class NaviStreamBase
{
public:
    NaviStreamBase(const NaviLoggerPtr &logger);
    ~NaviStreamBase();
private:
    NaviStreamBase(const NaviStreamBase &);
    NaviStreamBase &operator=(const NaviStreamBase &);
public:
    google::protobuf::Message *newReceiveMessageBase(
            google::protobuf::Arena *arena) const;
    bool receiveBase(const multi_call::GigStreamMessage &message);
    void receiveCancelBase(const multi_call::GigStreamMessage &message,
                           multi_call::MultiCallErrorCode ec);
    bool notifyReceiveBase(const multi_call::GigStreamBasePtr &stream, multi_call::PartIdTy partId);
    bool setDomain(const GraphDomainRemoteBasePtr &domain);
    bool initialized();
    void finish();
    void setThreadPool(const std::shared_ptr<NaviThreadPool> &threadPool);
    void *beginTaskQueue(multi_call::PartIdTy partId, const ScheduleInfo &schedInfo);
    void *beginReceiveQueue(multi_call::PartIdTy partId, const ScheduleInfo &schedInfo);
    void *beginReceive(multi_call::PartIdTy partId);
    void *beginSend(multi_call::PartIdTy partId);
    void endQueueHandle(void *handle);
    void endHandle(void *handle, bool fake);
    const StreamMetricPtr &getStreamMetric() const {
        return _streamMetric;
    }
protected:
    void initMetric(CollectFlags collectFlags,
                    NaviPartId partCount,
                    NaviPerf *naviPerf,
                    bool isClient,
                    const MallocTraceParam *mallocParam,
                    std::shared_ptr<MemLeakCollector> memLeakCollector);

protected:
    DECLARE_LOGGER();
    std::shared_ptr<NaviThreadPool> _threadPool;
    StreamMetricPtr _streamMetric;
    DomainHolder<GraphDomainRemoteBase> _domainHolder;
    bool _initialized;
    GraphDomainRemoteBase *_domainGdb = nullptr;
};

class NaviStreamReceiveItem : public NaviThreadPoolItemBase {
public:
    NaviStreamReceiveItem(const multi_call::GigStreamBasePtr &stream,
                          NaviStreamBase *streamBase,
                          multi_call::PartIdTy partId)
        : _stream(stream)
        , _streamBase(streamBase)
        , _partId(partId)
    {
    }

public:
    void process() override {
        if (_stream && _streamBase) {
            _inited = true;
            _handle = _streamBase->beginReceiveQueue(_partId, _schedInfo);
            // Let this worker run inline consumers even when the stream never becomes empty.
            if (!_stream->asyncReceive(_partId, 64)) {
                _stream->sendCancel(_partId, nullptr);
            }
        }
    }
    void destroy() override {
        if (_inited) {
            auto streamBase = _streamBase;
            auto stream = std::move(_stream);
            auto handle = _handle;
            delete this;
            streamBase->endQueueHandle(handle);
        } else if (_stream) {
            _stream->sendCancel(_partId, nullptr);
            _stream.reset();
            delete this;
        }
    }
private:
    bool _inited = false;
    void *_handle = nullptr;
    multi_call::GigStreamBasePtr _stream;
    NaviStreamBase *_streamBase = nullptr;
    multi_call::PartIdTy _partId;
};

}
