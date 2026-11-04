#include "rtp_llm/flexlb/navi_sched/graph/GraphVisRecorder.h"
#include <fstream>

namespace navi_sched {

GraphVisRecorder::GraphVisRecorder(const std::string &fileName) : _fileName(fileName)
{
    save();
}

void GraphVisRecorder::record(const navi::NaviResultPtr &naviResult)
{
    if (!naviResult) {
        return;
    }
    auto visProto = naviResult->getVisProto();
    if (!visProto) {
        return;
    }
    autil::ScopedLock lock(_mutex);
    for (const auto &node : visProto->nodes()) {
        *_vis.add_nodes() = node;
    }
}

void GraphVisRecorder::save() const
{
    autil::ScopedLock lock(_mutex);
    write(_vis, _fileName);
}

void GraphVisRecorder::save(const navi::NaviResultPtr &naviResult,
                            const std::string &fileName)
{
    if (!naviResult) {
        return;
    }
    auto visProto = naviResult->getVisProto();
    if (!visProto) {
        return;
    }
    static autil::ThreadMutex mutex;
    autil::ScopedLock lock(mutex);
    write(*visProto, fileName);
}

void GraphVisRecorder::write(const navi::GraphVisDef &vis, const std::string &fileName)
{
    std::ofstream output(fileName, std::ios::binary | std::ios::trunc);
    if (output.is_open()) {
        output << vis.SerializeAsString();
    }
}

} // namespace navi_sched
