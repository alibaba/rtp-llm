#include "navi/log/LogBtFilter.h"

namespace navi {

void LogBtFilterParam::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("file", file);
    json.Jsonize("line", line);
}

void LogBtFilter::addParam(const LogBtFilterParam &param) {
    _filterMap[param.line].insert(param.file);
}

bool LogBtFilter::pass(const LoggingEvent &event) {
    auto lineIt = _filterMap.find(event.line);
    if (_filterMap.end() == lineIt) {
        return false;
    }
    const auto &fileSet = lineIt->second;
    return fileSet.end() != fileSet.find(event.file);
}

}

