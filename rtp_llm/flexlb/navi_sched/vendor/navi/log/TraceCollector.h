#pragma once

#include "navi/common.h"
#include "navi/log/LoggingEvent.h"
#include "autil/DataBuffer.h"
#include <list>

namespace navi {

class GraphTraceDef;
class StringHashTable;

class TraceCollector
{
public:
    TraceCollector();
    ~TraceCollector();
private:
    TraceCollector(const TraceCollector &);
    TraceCollector &operator=(const TraceCollector &);
public:
    void setFormatPattern(const std::string &pattern);
    void append(LoggingEvent &event);
    void append(LoggingEvent &&event);
    void merge(TraceCollector &other);
    void format(std::vector<std::string> &traceVec) const;
    void toProto(GraphTraceDef *traceDef, StringHashTable *stringHashTable);
    void fromProto(const GraphTraceDef &traceDef,
                   const SymbolTableDef &symbolTableDef);
    bool empty() const;
    size_t eventCount() const;
private:
    std::string _formatPattern;
    std::list<LoggingEvent> _eventList;
};

}
