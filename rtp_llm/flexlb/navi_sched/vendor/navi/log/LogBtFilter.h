#pragma once

#include "autil/legacy/jsonizable.h"
#include "navi/log/LoggingEvent.h"

namespace navi {

struct LogBtFilterParam : public autil::legacy::Jsonizable {
public:
    LogBtFilterParam(const std::string &file_, int line_)
        : file(file_)
        , line(line_)
    {
    }
    LogBtFilterParam()
        : line(-1)
    {
    }
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
public:
    std::string file;
    int line;
};

class LogBtFilter {
public:
    void addParam(const LogBtFilterParam &param);
    bool pass(const LoggingEvent &event);
private:
    NaviUnorderedMap<int, NaviUnorderedSet<std::string> > _filterMap;
};

}

