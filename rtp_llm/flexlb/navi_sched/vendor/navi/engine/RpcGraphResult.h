#pragma once

#include "navi/proto/GraphVis.pb.h"
#include "navi/engine/GraphResultBase.h"
#include "navi/engine/PartInfo.h"
#include "navi/engine/NaviError.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamRpcInfo.h"

namespace navi {

class SinglePartResult : public GraphResultBase {
public:
    SinglePartResult(NaviPartId partId)
        : _partId(partId)
    {
    }
public:
    NaviErrorPtr addResult(NaviMessage *message,
                           const multi_call::GigStreamRpcInfo *rpcInfo,
                           const NaviErrorPtr &error);
private:
    void doToProto(bool partial,
                   NaviMessage &naviMessage,
                   ResultToProtoParam *param) override;
    void markThisLocation(SingleResultLocationDef *singleLocation) const override;
    void doToVisProto(GraphVisNodeDef *visDef, ResultToProtoParam *param) override;
    void doCollectRpcInfo(multi_call::GigStreamRpcInfoMap &rpcInfoMap) override;
    void doCollectTrace(TraceCollector &collector,
                        SymbolTableDef &symbolTableDef) override;
private:
    NaviErrorPtr doAddResult(SerializedResultDef *resultDef,
                             const multi_call::GigStreamRpcInfo *rpcInfo,
                             const NaviErrorPtr &error);
private:
    static void addRpcInfo(const multi_call::GigStreamRpcInfo &rpcInfo,
                           RpcPartResultLocationDef *partLocation);
    static void collectPartRpcInfo(const RpcResultLocationDef &rpcLoc,
                                   const RpcPartResultLocationDef &partLocation,
                                   multi_call::GigStreamRpcInfoMap &rpcInfoMap);
    static multi_call::GigStreamRpcInfo parseSingleGigRpcInfoDef(
            const SingleGigRpcInfoDef &rpcInfoDef);
private:
    NaviPartId _partId = INVALID_NAVI_PART_ID;
    mutable autil::ThreadMutex _resultMutex;
    std::vector<std::shared_ptr<SerializedResultDef>> _results;
};

NAVI_TYPEDEF_PTR(SinglePartResult);

class RpcGraphResult : public GraphResultBase
{
public:
    void init(const std::string &from, NaviPartId fromPartId,
              const std::string &to, const PartInfo &partInfo,
              GraphId graphId);
    NaviErrorPtr addResult(NaviPartId partId, NaviMessage *message,
                           const multi_call::GigStreamRpcInfo *rpcInfo,
                           const NaviErrorPtr &error);
private:
    void markThisLocation(SingleResultLocationDef *singleLocation) const override;
private:
    std::string _fromBiz;
    NaviPartId _fromPartId = INVALID_NAVI_PART_ID;
    std::string _toBiz;
    NaviPartId _partCount = INVALID_NAVI_PART_COUNT;
    GraphId _graphId = INVALID_GRAPH_ID;
};

NAVI_TYPEDEF_PTR(RpcGraphResult);

}

