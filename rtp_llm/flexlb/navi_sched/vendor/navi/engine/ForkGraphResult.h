#pragma once

#include "navi/engine/GraphResult.h"

namespace navi {

class ForkGraphResult : public GraphResult {
public:
    ForkGraphResult(GraphId graphId, const std::string &bizName,
                    const std::string &node)
        : _graphId(graphId)
        , _bizName(bizName)
        , _node(node)
    {
    }
private:
    void markThisLocation(SingleResultLocationDef *singleLocation) const override;
private:
    GraphId _graphId = INVALID_GRAPH_ID;
    std::string _bizName;
    std::string _node;
};

NAVI_TYPEDEF_PTR(ForkGraphResult);

}

