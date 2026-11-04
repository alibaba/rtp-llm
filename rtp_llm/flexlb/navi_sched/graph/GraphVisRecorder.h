#pragma once

#include "navi/engine/NaviResult.h"
#include "navi/proto/GraphVis.pb.h"
#include <autil/Lock.h>
#include <string>

namespace navi_sched {

class GraphVisRecorder {
public:
    explicit GraphVisRecorder(const std::string &fileName);

public:
    void record(const navi::NaviResultPtr &naviResult);
    void save() const;
    static void save(const navi::NaviResultPtr &naviResult, const std::string &fileName);

private:
    static void write(const navi::GraphVisDef &vis, const std::string &fileName);

private:
    std::string _fileName;
    mutable autil::ThreadMutex _mutex;
    navi::GraphVisDef _vis;
};

NAVI_TYPEDEF_PTR(GraphVisRecorder);

} // namespace navi_sched
