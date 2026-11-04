#pragma once

#include "navi/common.h"
#include "navi/engine/Resource.h"
#include "navi/engine/ResourceMap.h"
#include "navi/engine/RunGraphParams.h"
#include "navi/engine/TimeoutChecker.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/CollectFlags.h"
#include "autil/StringUtil.h"
#include <memory>

namespace navi {

class LocalSubGraph;
class Port;
class Node;

struct ForkGraphParam {
    bool collectMetric = false;
    bool collectPerf = false;
    bool errorAsEof = false;
    bool keepGraphDef = false;
    int64_t timeoutMs = DEFAULT_TIMEOUT_MS;
    std::vector<OverrideData> overrideDataVec;
    std::vector<NamedData> namedDataVec;
    std::map<GraphId, ResourceMapPtr> subResourceMaps;
};

class KernelInitContext
{
private:
    KernelInitContext(Node *node);
public:
    const TimeoutChecker *getTimeoutChecker() const;
    bool getInputNode(PortIndex index, std::string &node);
    bool getIORange(IoType ioType, const std::string &name, PortIndex &start,
                    PortIndex &end) const;
    bool getInputGroupSize(IndexType group, size_t &size);
    bool getOutputGroupSize(IndexType group, size_t &size);
    bool inputConnected(PortIndex index);
    bool outputConnected(PortIndex index);
    ErrorCode fork(GraphDef *graphDef, const ForkGraphParam &param = {});
    bool forkError() const;
    AsyncPipePtr createAsyncPipe(ActivateStrategy activateStrategy = AS_REQUIRED);
    bool registerPeriodicPipe(
        const AsyncPipePtr &asyncPipe,
        int64_t periodMs,
        int64_t repeat = -1);
    bool getPeriodicPipeActivationCount(
        const AsyncPipePtr &asyncPipe,
        int64_t &activatedCount,
        int64_t &remainingCount,
        int64_t &periodMs,
        int64_t &nextTickTimeMs) const;
    bool stopPeriodicAsyncPipe(const AsyncPipePtr &asyncPipe);
    ResourcePtr buildR(const std::string &name, KernelConfigContext ctx, ResourceMap &inputResourceMap) const;
    void cacheExpensive(bool value);
public:
    const std::string &getConfigPath() const;
    const std::string &getThisBizName() const;
    NaviPartId getThisPartId() const;
    NaviPartId getPartCount() const;
    int32_t getNodeIndex() const;
    KernelConfigContext getConfigContext() const;
    bool getBizPartInfo(const std::string &bizName, NaviPartId &partCount, std::vector<NaviPartId> &partIds) const;
    int64_t getSessionBeginTime() const;
    int64_t getRunGraphBeginTime() const;
private:
    friend class KernelComputeContext;
    friend class Node;
    friend class InputPortKernel;
    friend class OutputPortKernel;
    friend class TerminatorKernel;
    friend class ScopeTerminatorKernel;
    friend class PortSplitKernel;
    friend class PortMergeKernel;
    friend class ResourceCreateKernel;
    friend class ResourceSaveKernel;
    friend class ResourcePublishKernel;
    friend class NaviTesterInputKernel;
    friend class NaviTesterOutputKernel;
    friend class NaviTesterControlKernel;
    friend class KernelTester;
    friend class TFSendRecvKernel;
    friend class TFKernel;
protected:
    Node *_node;
    LocalSubGraph *_subGraph;
    bool _forkError: 1;
    bool _hasFork : 1;
    bool _hasOutput : 1;
};

NAVI_TYPEDEF_PTR(KernelInitContext);

}
