#include "navi/engine/SummaryBuilder.h"

namespace navi {

SummaryTableColumnPtr SummaryTableBuilder::addColumn(const std::string &name) {
    auto it = _columnMap.find(name);
    if (_columnMap.end() == it) {
        auto column = std::make_shared<SummaryTableColumn>(name);
        _columnMap[name] = column;
        _columnVec.push_back(column);
        return column;
    } else {
        return it->second;
    }
}

SummaryTableBuilderPtr SummaryBuilder::newTable(const std::string &name) {
    auto it = _tableMap.find(name);
    if (_tableMap.end() == it) {
        auto builder = std::make_shared<SummaryTableBuilder>(name);
        _tableMap[name] = builder;
        _tableVec.push_back(builder);
        return builder;
    } else {
        return it->second;
    }
}

std::string SummaryBuilder::toString() const {
    return autil::legacy::FastToJsonString(*this);
}

}

