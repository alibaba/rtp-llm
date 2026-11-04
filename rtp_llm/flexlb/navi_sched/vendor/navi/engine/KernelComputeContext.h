#pragma once

#include "navi/common.h"
#include "navi/engine/Data.h"
#include "navi/engine/KernelInitContext.h"
#include "navi/engine/NaviError.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamRpcInfo.h"
#include <vector>

namespace navi {

class Node;
class KernelComputeContext;
struct InputSnapshot;
class GraphDef;

class GroupDatas {
public:
    GroupDatas();
private:
    GroupDatas(InputSnapshot *begin, InputSnapshot *end);
public:
    class Iterator {
    public:
        Iterator();
        Iterator(InputSnapshot *ptr);
        Iterator operator++();
        Iterator operator+(size_t offset) const;
        bool operator!=(const Iterator & rhs) const;
        const StreamData &operator*();
    private:
        InputSnapshot *_ptr;
    };
public:
    Iterator begin() const;
    Iterator end() const;
    size_t size() const;
    const StreamData &operator[](size_t index) const;
    bool eof() const;
private:
    InputSnapshot *_begin;
    InputSnapshot *_end;
private:
    friend class KernelComputeContext;
};

class InputPeekScope {
public:
    InputPeekScope(KernelComputeContext &ctx, PortIndex index);
    ~InputPeekScope();
public:
    bool ok() const {
        return _ok;
    }
    const StreamData &data() const {
        return _data;
    }
private:
    KernelComputeContext &_ctx;
    PortIndex _index;
    bool _ok = false;
    StreamData _data;
};

class GroupPeekScope {
public:
    GroupPeekScope(KernelComputeContext &ctx, IndexType group);
    ~GroupPeekScope();
public:
    bool ok() const {
        return _ok;
    }
    const GroupDatas &datas() const {
        return _datas;
    }
private:
    KernelComputeContext &_ctx;
    IndexType _group;
    bool _ok = false;
    GroupDatas _datas;
};

class KernelComputeContext : public KernelInitContext
{
private:
    KernelComputeContext(Node *node);
private:
    KernelComputeContext(const KernelComputeContext &);
    KernelComputeContext& operator=(const KernelComputeContext &);
public:
    // input
    bool getInput(PortIndex index, DataPtr &data, bool &eof);
    bool getInput(PortIndex index, StreamData &streamData, bool clear = false);
    bool getGroupInput(IndexType group, GroupDatas &datas);
    bool peekInput(PortIndex index, DataPtr &data, bool &eof);
    bool peekInput(PortIndex index, StreamData &streamData);
    bool peekGroupInput(IndexType group, GroupDatas &datas);
    bool clearInput(PortIndex index);
    bool clearGroupInput(IndexType group);
    bool setInputEof(PortIndex index);
    // output
    bool setOutput(PortIndex index, const DataPtr &data, bool eof);
    bool setOutput(PortIndex index, Data *data, bool eof);
    // misc
    void stopSchedule();
    bool deadLock() const;
    void setIgnoreDeadlock();
    ErrorCode fork(GraphDef *graphDef, const ForkGraphParam &param = {});
    int64_t getScheduleCount() const;
    void updateTraceLevel(const std::string &levelStr) const;
    bool updateTimeoutMs(int64_t timeoutMs) const;
    void updateThreadLimit(uint32_t threadLimit) const;
    void updateCollect(bool collectMetric, bool collectPerf) const;
    void collectTrace(std::vector<std::string> &traceVec) const;
    std::shared_ptr<multi_call::GigStreamRpcInfoMap> getRpcInfoMap() const;
    NaviErrorPtr getScopeError() const;
    ErrorHandleStrategy getScopeErrorHandleStrategy() const;
    void reportScopeError(const NaviErrorPtr &error);
    void reportScopeError(ErrorCode ec);
    int64_t getRemainTimeMs() const;
private:
    bool doGetGroupInput(IndexType group, EdgeGetType type, GroupDatas &datas);
private:
    friend class Node;
    friend class KernelTester;
    friend class ResourceCreateKernel;
    friend class PortMergeKernel;
private:
    using KernelInitContext::fork;
private:
    bool _deadLock : 1;
    bool _ignoreDeadlock : 1;
};

NAVI_TYPEDEF_PTR(KernelComputeContext);

}
