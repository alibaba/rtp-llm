#pragma once

#include "navi/engine/KernelMetric.h"
#include "navi/engine/EdgeMetric.h"

namespace navi {

class NaviResult;
class GraphMetricDef;
class NaviSymbolTable;
typedef std::shared_ptr<NaviSymbolTable> NaviSymbolTablePtr;
class StringHashTable;
struct ResultToProtoParam;

struct GraphMetricTime {
    int64_t queueUs = 0;
    int64_t computeUs = 0;
};

class GraphMetric
{
public:
    GraphMetric();
    ~GraphMetric();
private:
    GraphMetric(const GraphMetric &);
public:
    void updateBeginTime(int64_t beginTime, int64_t beginTimeNs) {
        _beginTime = std::min(_beginTime, beginTime);
        _endTime = _beginTime;
        _beginTimeMonoNs = std::min(_beginTimeMonoNs, beginTimeNs);
    }
    void setNaviSymbolTable(const NaviSymbolTablePtr &naviSymbolTable) {
        _naviSymbolTable = naviSymbolTable;
    }
    const NaviSymbolTablePtr &getNaviSymbolTable() const {
        return _naviSymbolTable;
    }
    void addKernelMetricPtr(const KernelMetricPtr &metric);
    void addEdgeMetricPtr(const EdgeMetricPtr &metric);
    void end();
    const GraphMetricTime &getGraphMetricTime() const;
    int64_t getLatency() const;
    void toProto(GraphMetricDef *metric,
                 StringHashTable *stringHashTable,
                 ResultToProtoParam *param);
    void show() const;
    void fillResult(NaviResult &result) const;
private:
    const KernelMetricPtr &getInitMetric() const {
        return _initMetric;
    }
    void fillKernelProto(GraphMetricDef *metricDef,
                         StackTable &stackTable,
                         int64_t &computeLatencyUs,
                         int64_t &queueLatencyUs);
    void fillEdgeProto(GraphMetricDef *metricDef, StringHashTable *stringHashTable);
private:
    friend class NaviResult;
private:
    NaviSymbolTablePtr _naviSymbolTable;
    int64_t _beginTime;
    int64_t _endTime;
    int64_t _beginTimeMonoNs;
    GraphMetricTime _graphMetricTime;
    std::vector<KernelMetricPtr> _kernelMetricVec;
    std::vector<EdgeMetricPtr> _edgeMetricVec;
    KernelMetricPtr _initMetric;
};

}
