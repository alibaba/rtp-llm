#pragma once

#include "navi/common.h"
#include "navi/engine/Data.h"
#include "navi/proto/GraphVis.pb.h"
#include <list>
namespace navi {

class Node;
class CreatorManager;
class EdgeMetricDef;
class EdgeMetricEventDef;
class StringHashTable;

struct EdgeMetricSetEvent {
public:
    void fillProto(EdgeSetEventDef *event, StringHashTable *stringHashTable) const;
public:
    bool success;
    int64_t timeNs;
    uint64_t dataAddr;
    bool eof;
    int32_t dataIndex;
    int32_t tid;
    int32_t ec;
    int64_t getSummaryLatencyUs = 0;
    std::string summaryStr;
    std::string typeId;
    std::string node;
    std::string kernel;
};

struct EdgeMetricGetEvent {
public:
    void fillProto(EdgeGetEventDef *event) const;
public:
    bool success;
    EdgeGetType type;
    int64_t timeNs;
    uint64_t dataAddr;
    bool eof;
    int32_t dataIndex;
    int32_t tid;
    int32_t slotId;
    std::string node;
    std::string kernel;
};

class EdgeMetric;
NAVI_TYPEDEF_PTR(EdgeMetric);

class EdgeMetric
{
public:
    EdgeMetric(GraphId graphId,
               const std::string &node,
               const std::string &port,
               const std::string &query,
               CreatorManager *creatorManager);
    ~EdgeMetric();
    EdgeMetric(const EdgeMetric &) = delete;
    EdgeMetric &operator=(const EdgeMetric &) = delete;
public:
    void fillProto(EdgeMetricDef *metric, StringHashTable *stringHashTable) const;
    void reportSetEvent(bool success, int64_t timeNs, const Data *data, bool eof,
                        const Node *callNode);
    void reportGetEvent(bool success, int64_t timeNs, IndexType slotId, EdgeGetType type,
                        const Data *data, bool eof, const Node *callNode);
    void addSlotName(size_t index, const std::string &node, const std::string &port);
private:
    GraphId _graphId = INVALID_GRAPH_ID;
    std::string _node;
    std::string _port;
    std::string _query;
    mutable bool _collected = false;
    CreatorManager *_creatorManager = nullptr;
    int64_t _dataIndex = -1;
    mutable autil::ThreadMutex _mutex;
    std::vector<std::pair<std::string, std::string>> _slotNames;
    std::shared_ptr<std::list<EdgeMetricSetEvent>> _setList;
    std::shared_ptr<std::list<EdgeMetricGetEvent>> _getList;
};

}

