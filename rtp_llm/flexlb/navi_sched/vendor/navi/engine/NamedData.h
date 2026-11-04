#pragma once

#include "navi/common.h"
#include "navi/engine/Data.h"

namespace navi {

struct NamedData {
    NamedData()
        : graphId(INVALID_GRAPH_ID)
    {
    }
public:
    GraphId graphId;
    std::string name;
    DataPtr data;
};

typedef NaviUnorderedMap<std::string, NamedData> SubNamedDataMap;
typedef NaviUnorderedMap<GraphId, SubNamedDataMap> NamedDataMap;

}

