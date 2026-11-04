#include "navi/engine/NaviHistory.h"
#include "navi/engine/NaviHostInfo.h"
#include "navi/log/NaviLogger.h"
#include "autil/CompressionUtil.h"
#include "autil/FileUtil.h"

namespace navi {

NaviHistory::NaviHistory() {
}

NaviHistory::~NaviHistory() {
    _loopThread.reset();
    auto beginTimeUs = autil::TimeUtility::currentTime();
    _updateHistory.addControlEvent(NHET_EXIT, beginTimeUs, beginTimeUs);
    saveToFile();
}

bool NaviHistory::init(const std::string &naviName) {
    auto beginTimeUs = autil::TimeUtility::currentTime();
    auto ret = doInit(naviName);
    _updateHistory.addControlEvent(NHET_START, beginTimeUs, autil::TimeUtility::currentTime());
    _loopThread =
        autil::LoopThread::createLoopThread(std::bind(&NaviHistory::workLoop, this), 1000 * 1000, "navi_history");
    return ret;
}

bool NaviHistory::doInit(const std::string &naviName) {
    _fileName = ("./" + naviName + ".navi_history");
    auto content = NaviHostInfo::getFileContent(_fileName);
    if (content.empty()) {
        NAVI_KERNEL_LOG(INFO, "history file [%s] not exist or empty", _fileName.c_str());
        return true;
    }
    std::string decompressed;
    if (!autil::CompressionUtil::decompress(content, autil::CompressType::LZ4, decompressed, nullptr)) {
        NAVI_KERNEL_LOG(INFO, "decompress history file [%s] failed", _fileName.c_str());
        return true;
    }
    NaviHistoryDef historyDef;
    if (!historyDef.ParseFromString(decompressed)) {
        NAVI_KERNEL_LOG(INFO, "parse history file [%s] failed", _fileName.c_str());
        return true;
    }
    fromProto(historyDef);
    auto queryCounts = _queryHistory.getQueryCount();
    NAVI_KERNEL_LOG(INFO,
                    "load history file [%s] success, file compress size: %lu, decompress size: %lu, update event: %lu, "
                    "normal query: %lu, error query: %lu",
                    _fileName.c_str(),
                    content.size(),
                    decompressed.size(),
                    _updateHistory.getEventCount(),
                    queryCounts.first,
                    queryCounts.second);
    return true;
}

void NaviHistory::updateConfig(const HistoryConfig &config) {
    _updateHistory.updateConfig(config.update);
    _queryHistory.updateConfig(config.query);
}

void NaviHistory::toProto(bool shortQueryHistory, NaviHistoryDef &historyDef) const {
    auto &symbolTable = *historyDef.mutable_symbol_table();
    _updateHistory.toProto(*historyDef.mutable_update(), symbolTable);
    _queryHistory.toProto(shortQueryHistory, *historyDef.mutable_query(), symbolTable);
}

void NaviHistory::fromProto(const NaviHistoryDef &historyDef) {
    const auto &symbolTable = historyDef.symbol_table();
    _updateHistory.fromProto(historyDef.update(), symbolTable);
    _queryHistory.fromProto(historyDef.query(), symbolTable);
}

void NaviHistory::saveToFile() {
    if (_fileName.empty()) {
        return;
    }
    NaviHistoryDef historyDef;
    {
        toProto(false, historyDef);
    }
    auto content = historyDef.SerializeAsString();
    std::string compressed;
    if (!autil::CompressionUtil::compress(content, autil::CompressType::LZ4, compressed, nullptr)) {
        return;
    }
    if (!autil::FileUtil::writeFileAtomically(_fileName, compressed)) {
        return;
    }
}

void NaviHistory::workLoop() {
    _queryHistory.workLoop();
}

void NaviHistory::addUpdateEvent(const std::string &configStr,
                                 const NaviErrorPtr &error,
                                 int64_t beginTimeUs,
                                 int64_t endTimeUs)
{
    auto updated = _updateHistory.addUpdateEvent(configStr, error, beginTimeUs, endTimeUs);
    if (updated) {
        saveToFile();
    }
}

void NaviHistory::addControlEvent(UpdateHistoryEventType type, int64_t beginTimeUs, int64_t endTimeUs) {
    _updateHistory.addControlEvent(type, beginTimeUs, endTimeUs);
}

void NaviHistory::addQuery(const std::shared_ptr<NaviArpcRequestData> &requestData,
                           const RegistryArpcParam &arpcParam,
                           http_arpc::HTTPRPCServerClosure *httpClosure,
                           const std::shared_ptr<NaviResult> &naviResult,
                           const std::string &queryKey,
                           google::protobuf::Message *response,
                           bool needDebug)
{
    _queryHistory.addQuery(requestData, arpcParam, httpClosure, naviResult, queryKey, response, needDebug);
}

}

