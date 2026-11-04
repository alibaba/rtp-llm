#include "navi/engine/GraphMetric.h"
#include "navi/engine/NaviResult.h"
#include "navi/engine/StackTable.h"
#include "navi/engine/GraphResultBase.h"
#include "autil/TimeUtility.h"
#include "navi/perf/NaviSymbolTable.h"
#include "navi/proto/GraphVis.pb.h"
#include "navi/util/CommonUtil.h"

namespace navi {

GraphMetric::GraphMetric() {
    _beginTime = autil::TimeUtility::currentTime();
    _beginTimeMonoNs = CommonUtil::getTimelineTimeNs();
    _endTime = _beginTime;
}

GraphMetric::~GraphMetric() {
}

void GraphMetric::addKernelMetricPtr(const KernelMetricPtr &metric) {
    if (metric->isGraphInit()) {
        _initMetric = metric;
    } else {
        _kernelMetricVec.push_back(metric);
    }
}

void GraphMetric::addEdgeMetricPtr(const EdgeMetricPtr &metric) {
    _edgeMetricVec.push_back(metric);
}

void GraphMetric::end() {
    if (_endTime != _beginTime) {
        return;
    }
    _endTime = autil::TimeUtility::currentTime();
    GraphMetricTime time;
    for (const auto &metric : _kernelMetricVec) {
        time.queueUs += metric->queueLatencyUs();
        time.computeUs += metric->totalLatencyUs();
    }
    _graphMetricTime = time;
}

const GraphMetricTime &GraphMetric::getGraphMetricTime() const {
    return _graphMetricTime;
}

int64_t GraphMetric::getLatency() const {
    return _endTime - _beginTime;
}

void GraphMetric::fillKernelProto(GraphMetricDef *metricDef,
                                  StackTable &stackTable,
                                  int64_t &computeLatencyUs,
                                  int64_t &queueLatencyUs)
{
    for (const auto &metric : _kernelMetricVec) {
        metric->fillProto(metricDef->add_metrics(), stackTable);
    }
    _kernelMetricVec.clear();
}

void GraphMetric::fillEdgeProto(GraphMetricDef *metricDef,
                                StringHashTable *stringHashTable){
    for (const auto &metric : _edgeMetricVec) {
        metric->fillProto(metricDef->add_edge_metrics(), stringHashTable);
    }
}

void GraphMetric::toProto(GraphMetricDef *metricDef,
                          StringHashTable *stringHashTable,
                          ResultToProtoParam *param) {
    if (!_kernelMetricVec.empty()) {
        int64_t computeLatencyUs = 0;
        int64_t queueLatencyUs = 0;
        auto stackTablePtr = std::make_shared<StackTable>(metricDef->mutable_stack_table());
        auto &stackTable = *stackTablePtr;
        fillKernelProto(metricDef, stackTable, computeLatencyUs, queueLatencyUs);
        if (!param) {
            if (_initMetric) {
                _initMetric->fillProto(metricDef->add_metrics(), stackTable);
                _initMetric.reset();
            }
            if (_naviSymbolTable) {
                _naviSymbolTable->resolveSymbol(
                    stackTable,
                    metricDef->mutable_perf_symbol_table(),
                    stringHashTable);
            }
        } else {
            param->stackTable = stackTablePtr;
            param->initMetric = _initMetric;
            param->naviSymbolTable = _naviSymbolTable;
        }
    }
    if (!_edgeMetricVec.empty()) {
        fillEdgeProto(metricDef, stringHashTable);
    }
    auto graphMetricTime = getGraphMetricTime();
    metricDef->set_begin_time(_beginTime);
    metricDef->set_end_time(_endTime);
    metricDef->set_begin_time_mono_ns(_beginTimeMonoNs);
    metricDef->set_compute_latency_us(graphMetricTime.computeUs);
    metricDef->set_queue_latency_us(graphMetricTime.queueUs);
}

void GraphMetric::fillResult(NaviResult &result) const {
    result.beginTime = _beginTime;
    result.endTime = _endTime;
    result.queueUs = _graphMetricTime.queueUs;
    result.computeUs = _graphMetricTime.computeUs;
}

void GraphMetric::show() const {
    auto graphMetricTime = getGraphMetricTime();
    std::cout << "graph metric:" << std::endl;
    for (const auto &metric : _kernelMetricVec) {
        std::cout << metric->toString() << std::endl;
    }
    std::cout << "latency: " << getLatency() / 1000.0 << std::endl;
    std::cout << "total time: " << graphMetricTime.computeUs / 1000.0 << std::endl;
    std::cout << "total queue time: " << graphMetricTime.queueUs / 1000.0 << std::endl;
    std::cout << "graph metric end" << std::endl;
}

}
