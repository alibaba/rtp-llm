#include "navi/engine/EdgeMetric.h"
#include "navi/engine/NaviThreadPool.h"
#include "navi/engine/Node.h"
#include "navi/engine/SummaryBuilder.h"
#include "navi/engine/StringHashTable.h"

namespace navi {

void EdgeMetricSetEvent::fillProto(EdgeSetEventDef *event,
                                   StringHashTable *stringHashTable) const
{
    event->set_success(success);
    event->set_time_ns(timeNs);
    event->set_data_addr(dataAddr);
    event->set_eof(eof);
    event->set_data_index(dataIndex);
    event->set_tid(tid);
    event->set_ec(ec);
    event->set_summary_hash(stringHashTable->getHash(summaryStr));
    event->set_type_id(typeId);
    event->set_node_name(node);
    event->set_kernel_name(kernel);
    event->set_get_summary_latency_us(getSummaryLatencyUs);
}

void EdgeMetricGetEvent::fillProto(EdgeGetEventDef *event) const {
    event->set_success(success);
    event->set_time_ns(timeNs);
    event->set_data_addr(dataAddr);
    event->set_eof(eof);
    event->set_data_index(dataIndex);
    event->set_tid(tid);
    event->set_slot_id(slotId);
    event->set_node_name(node);
    event->set_kernel_name(kernel);
    event->set_type(type);
}

EdgeMetric::EdgeMetric(GraphId graphId,
                       const std::string &node,
                       const std::string &port,
                       const std::string &query,
                       CreatorManager *creatorManager)
    : _graphId(graphId)
    , _node(node)
    , _port(port)
    , _query(query)
    , _creatorManager(creatorManager)
    , _dataIndex(-1)
{
}

EdgeMetric::~EdgeMetric() {
}

void EdgeMetric::fillProto(EdgeMetricDef *metric,
                           StringHashTable *stringHashTable) const
{
    metric->set_graph_id(_graphId);
    auto inputDef = metric->mutable_input();
    inputDef->set_node_name(_node);
    inputDef->set_port_name(_port);
    metric->set_summary_query(_query);
    std::shared_ptr<std::list<EdgeMetricSetEvent>> setListPtr;
    std::shared_ptr<std::list<EdgeMetricGetEvent>> getListPtr;
    {
        autil::ScopedLock lock(_mutex);
        for (const auto &pair : _slotNames) {
            auto def = metric->add_slot_names();
            def->set_node_name(pair.first);
            def->set_port_name(pair.second);
        }
        setListPtr = std::move(_setList);
        getListPtr = std::move(_getList);
    }
    if (setListPtr) {
        const auto &setList = *setListPtr;
        for (const auto &event : setList) {
            event.fillProto(metric->add_set_events(), stringHashTable);
        }
    }
    if (getListPtr) {
        const auto &getList = *getListPtr;
        for (const auto &event : getList) {
            event.fillProto(metric->add_get_events());
        }
    }
    _collected = true;
}

void EdgeMetric::reportSetEvent(bool success, int64_t timeNs, const Data *data, bool eof,
                                const Node *callNode)
{
    if (_collected) {
        return;
    }
    EdgeMetricSetEvent event;
    event.success = success;
    event.timeNs = timeNs;
    event.dataAddr = (size_t)data;
    event.eof = eof;
    event.dataIndex = ++_dataIndex;
    if (unlikely(0 == current_thread_id)) {
        current_thread_id = (long)syscall(SYS_gettid);
    }
    event.tid = current_thread_id;
    auto ec = SEC_NOT_SUPPORT;
    if (NAVI_DATA_SUMMARY_IGNORE_DATA == _query) {
        ec = SEC_IGNORE_DATA;
        if (data) {
            event.typeId = data->getTypeId();
        }
    } else {
        if (data) {
            try {
                autil::ScopedTime2 scope;
                SummaryBuilder builder;
                ec = data->getSummary(_query, builder);
                event.summaryStr = builder.toString();
                event.getSummaryLatencyUs = scope.done_us();
            } catch (const autil::legacy::ExceptionBase &e) {
                ec = SEC_FAILED;
                event.summaryStr = "get summary failed, exception [" + e.GetMessage() + "]";
                if (NAVI_TLS_LOGGER) {
                    event.summaryStr += ", bt: " + NAVI_TLS_LOGGER->logger->getExceptionBt(e);
                }
                auto typeCreator = _creatorManager->getType(data->getTypeId());
                if (typeCreator && typeCreator->getSourceFile()) {
                    event.summaryStr.append(", see type file: " + std::string(typeCreator->getSourceFile()));
                }
            }
            event.typeId = data->getTypeId();
        } else {
            ec = SEC_NULL_DATA;
        }
    }
    event.ec = ec;
    if (callNode) {
        event.node = callNode->getName();
        event.kernel = callNode->getKernelName();
    }
    autil::ScopedLock lock(_mutex);
    if (!_setList) {
        _setList = std::make_shared<std::list<EdgeMetricSetEvent>>();
        _setList->emplace_back(std::move(event));
    } else {
        _setList->emplace_back(std::move(event));
    }
}

void EdgeMetric::reportGetEvent(bool success, int64_t timeNs, IndexType slotId, EdgeGetType type,
                                const Data *data, bool eof, const Node *callNode)
{
    if (_collected) {
        return;
    }
    EdgeMetricGetEvent event;
    event.success = success;
    event.type = type;
    event.timeNs = timeNs;
    event.dataAddr = (size_t)data;
    event.eof = eof;
    event.dataIndex = _dataIndex;
    if (unlikely(0 == current_thread_id)) {
        current_thread_id = (long)syscall(SYS_gettid);
    }
    event.tid = current_thread_id;
    event.slotId = slotId;
    if (callNode) {
        event.node = callNode->getName();
        event.kernel = callNode->getKernelName();
    }
    autil::ScopedLock lock(_mutex);
    if (!_getList) {
        _getList = std::make_shared<std::list<EdgeMetricGetEvent>>();
        _getList->emplace_back(std::move(event));
    } else {
        _getList->emplace_back(std::move(event));
    }
}

void EdgeMetric::addSlotName(size_t index, const std::string &node, const std::string &port) {
    autil::ScopedLock lock(_mutex);
    if (_slotNames.size() <= index) {
        _slotNames.resize(index + 1);
    }
    _slotNames[index] = std::make_pair(node, port);
}

}

