#include "navi/engine/GraphBorder.h"
#include "navi/util/CommonUtil.h"

namespace navi {

GraphBorder::GraphBorder(const NaviLoggerPtr &logger, GraphParam *param)
    : _logger(this, "border", logger)
    , _param(param)
{
}

GraphBorder::~GraphBorder() {
}

bool GraphBorder::collectSubGraph(const SubGraphDef &subGraphDef,
                                  const GraphDomainPtr &domain)
{
    auto graphId = subGraphDef.graph_id();
    auto borderCount = subGraphDef.borders_size();
    const auto &location = subGraphDef.location();
    auto thisPartId = location.this_part_id();
    auto &partInfoDef = location.part_info();
    NAVI_LOG(SCHEDULE1, "collect sub graph, location [%s]", location.ShortDebugString().c_str());
    if (0 == borderCount) {
        if (subGraphDef.has_option() && subGraphDef.option().ignore_isolate()) {
            return true;
        }
        if (0 == subGraphDef.nodes_size()) {
            NAVI_LOG(ERROR, "sub graph [%d] has no nodes, biz [%s]", graphId, location.biz_name().c_str());
            return false;
        } else {
            NAVI_LOG(ERROR,
                     "graph has isolated sub graph [%d], check graph topo or "
                     "graph output, biz [%s]",
                     graphId,
                     location.biz_name().c_str());
            return false;
        }
    }
    _borderMap.reserve(borderCount);
    // auto backPressureEofStop = !location.gig_info().enable_probe();
    auto backPressureEofStop = true;
    for (int32_t index = 0; index < borderCount; index++) {
        const auto &borderDef = subGraphDef.borders(index);
        auto subBorderPtr = addSubBorder(thisPartId, graphId, borderDef, domain);
        if (!subBorderPtr->init(partInfoDef, borderDef, domain.get(), backPressureEofStop)) {
            return false;
        }
        domain->addSubBorder(subBorderPtr);
    }
    return true;
}

SubGraphBorderPtr GraphBorder::addSubBorder(NaviPartId partId,
                                            GraphId graphId,
                                            const BorderDef &borderDef,
                                            const GraphDomainPtr &domain)
{
    BorderId borderId(graphId, borderDef);
    auto it = _borderMap.find(borderId);
    if (_borderMap.end() != it) {
        return it->second;
    } else {
        SubGraphBorderPtr border(new SubGraphBorder(domain->getLogger().logger,
                                                    _param, partId, borderId));
        _borderMap.emplace(borderId, border);
        auto it = _subGraphBorderMap.find(graphId);
        if (_subGraphBorderMap.end() != it) {
            it->second->emplace(borderId, border);
        } else {
            auto subMap = std::make_shared<SubGraphBorderMap>();
            subMap->emplace(borderId, border);
            _subGraphBorderMap.emplace(graphId, subMap);
        }
        return border;
    }
}

SubGraphBorderMapPtr GraphBorder::getBorderMap(GraphId graphId) const {
    auto it = _subGraphBorderMap.find(graphId);
    if (_subGraphBorderMap.end() != it) {
        return it->second;
    } else {
        return nullptr;
    }
}

std::ostream& operator<<(std::ostream& os, const GraphBorder& border) {
    os << &border << "{" << std::endl;
    for (const auto &pair : border._borderMap) {
        os << *pair.second;
    }
    os << std::endl <<"}";
    return os;
}

bool GraphBorder::weave() {
    for (const auto &thisPair : _borderMap) {
        const auto &thisBorderId = thisPair.first;
        for (const auto &otherPair : _borderMap) {
            const auto &otherBorderId = otherPair.first;
            if (!match(thisBorderId, otherBorderId)) {
                continue;
            }
            if (thisBorderId.ioType == IOT_INPUT) {
                continue;
            }
            if (!thisPair.second->linkTo(*otherPair.second)) {
                return false;
            }
            break;
        }
    }
    for (const auto &pair : _borderMap) {
        if (!pair.second->initBackPressure(getBorderMap(pair.first.id))) {
            return false;
        }
    }
    bool ret = true;
    for (const auto &pair : _borderMap) {
        const auto &subBorder = *pair.second;
        if (!subBorder.linked()) {
            NAVI_LOG(ERROR, "border peer sub graph [%d] not exist, border [%s]",
                     pair.first.peer, subBorder.getBorderIdStr().c_str());
            ret = false;
        }
    }
    return ret;
}

bool GraphBorder::match(const BorderId &thisBorder,
                        const BorderId &otherBorder)
{
    auto otherIoType = CommonUtil::getReverseIoType(thisBorder.ioType);
    return otherIoType == otherBorder.ioType
        && thisBorder.id == otherBorder.peer
        && thisBorder.peer == otherBorder.id;
}

}
