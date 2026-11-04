#pragma once

#include "navi/builder/GraphBuilder.h"
#include "navi/engine/SubGraphBorder.h"
#include "navi/engine/SubGraphBase.h"

namespace navi {

class GraphBase;

class GraphBorder
{
public:
    GraphBorder(const NaviLoggerPtr &logger, GraphParam *param);
    ~GraphBorder();
private:
    GraphBorder(const GraphBorder &);
    GraphBorder &operator=(const GraphBorder &);
public:
    bool collectSubGraph(const SubGraphDef &subGraphDef,
                         const GraphDomainPtr &domain);
    bool weave();
    SubGraphBorderMapPtr getBorderMap(GraphId graphId) const;
public:
    friend std::ostream& operator<<(std::ostream& os, const GraphBorder& border);
private:
    SubGraphBorderPtr addSubBorder(NaviPartId partId,
                                   GraphId graphId,
                                   const BorderDef &borderDef,
                                   const GraphDomainPtr &domain);
    bool match(const BorderId &thisBorder, const BorderId &otherBorder);
private:
    DECLARE_LOGGER();
    GraphParam *_param;
    SubGraphBorderMap _borderMap;
    NaviUnorderedMap<GraphId, std::shared_ptr<SubGraphBorderMap>> _subGraphBorderMap;
};

NAVI_TYPEDEF_PTR(GraphBorder);

}
