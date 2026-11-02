#include "aios/navi_sched/test/simulator/TrafficReplayR.h"

#include "aios/navi_sched/data/BlockHashUtil.h"
#include "aios/navi_sched/data/MasterResponseToClientData.h"
#include "aios/navi_sched/debug/NaviSchedScheduleReplayJson.h"
#include "autil/StringUtil.h"
#include "autil/TimeUtility.h"
#include "navi/log/NaviLogger.h"
#include "navi/engine/Navi.h"
#include "navi/engine/NaviUserResult.h"
#include <autil/legacy/jsonizable.h>
#include <rapidjson/document.h>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <fstream>
#include <limits>

namespace navi_sched::test {
namespace {

using JsonWrapper = autil::legacy::Jsonizable::JsonWrapper;

struct RequestTimingJson : public autil::legacy::Jsonizable {
    NaviSchedRequestTiming value;
    RequestTimingJson() = default;
    explicit RequestTimingJson(const NaviSchedRequestTiming &input) : value(input) {}
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("batch_wait_ms", value.batchWaitMs);
        json.Jsonize("queue_wait_ms", value.queueWaitMs);
        json.Jsonize("execution_latency_ms", value.executionLatencyMs);
        json.Jsonize("client_total_latency_ms", value.clientTotalLatencyMs);
    }
};

struct LatencySample {
    uint64_t requestId = 0;
    uint64_t scheduleId = 0;
    double scheduleLoss = 0.0;
    int64_t scheduleLatencyUs = 0;
    NaviSchedRequestTiming requestTiming;
};

struct LatencySampleJson : public autil::legacy::Jsonizable {
    LatencySample value;
    LatencySampleJson() = default;
    LatencySampleJson(uint64_t requestId,
                      const NaviSchedRequestTiming &requestTiming,
                      uint64_t scheduleId, double scheduleLoss,
                      int64_t scheduleLatencyUs)
        : value{requestId, scheduleId, scheduleLoss,
                scheduleLatencyUs, requestTiming}
    {
    }
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("request_id", value.requestId);
        json.Jsonize("schedule_id", value.scheduleId);
        json.Jsonize("schedule_loss", value.scheduleLoss);
        json.Jsonize("schedule_latency_us", value.scheduleLatencyUs);
        RequestTimingJson requestTiming(value.requestTiming);
        json.Jsonize("request_timing", requestTiming);
        if (json.GetMode() != autil::legacy::Jsonizable::TO_JSON) {
            value.requestTiming = requestTiming.value;
        }
    }
};

const rapidjson::Value *findValue(const rapidjson::Value &object, const char *name)
{
    if (!object.IsObject()) {
        return nullptr;
    }
    auto it = object.FindMember(name);
    return object.MemberEnd() == it ? nullptr : &it->value;
}

bool getString(const rapidjson::Value &object, const char *name, std::string &value)
{
    auto field = findValue(object, name);
    if (!field || !field->IsString()) {
        return false;
    }
    value.assign(field->GetString(), field->GetStringLength());
    return true;
}

bool getInt64(const rapidjson::Value &object, const char *name, int64_t &value)
{
    auto field = findValue(object, name);
    if (!field) {
        return false;
    }
    if (field->IsInt64()) {
        value = field->GetInt64();
        return true;
    }
    if (field->IsUint64() && field->GetUint64() <= static_cast<uint64_t>(INT64_MAX)) {
        value = static_cast<int64_t>(field->GetUint64());
        return true;
    }
    if (!field->IsString()) {
        return false;
    }
    return autil::StringUtil::fromString(
        std::string(field->GetString(), field->GetStringLength()), value);
}

bool getTimestampMs(const rapidjson::Value &object, int64_t &value)
{
    if (getInt64(object, "timestamp_ms", value)) {
        return value >= 0;
    }
    auto field = findValue(object, "timestamp");
    if (!field) {
        return getInt64(object, "reqTime", value);
    }
    if (field->IsDouble()) {
        value = static_cast<int64_t>(std::llround(field->GetDouble() * 1000.0));
        return value >= 0;
    }
    return getInt64(object, "timestamp", value);
}

int64_t getNestedInt64(const rapidjson::Value &object, const char *objectName,
                       const char *fieldName)
{
    auto nested = findValue(object, objectName);
    int64_t value = 0;
    return nested && getInt64(*nested, fieldName, value) ? value : 0;
}

NaviSchedBlockHashesConstPtr getBlockHashes(const rapidjson::Value &object,
                                            int64_t promptTokenCount,
                                            int64_t blockSize)
{
    auto field = findValue(object, "input_block_hash_ids");
    if (!field) {
        field = findValue(object, "block_cache_keys");
    }
    if (!field) {
        field = findValue(object, "blockHashes");
    }
    if (!field || !field->IsArray()) {
        return nullptr;
    }
    auto hashes = std::make_shared<NaviSchedBlockHashes>();
    hashes->reserve(field->Size());
    for (const auto &value : field->GetArray()) {
        if (value.IsString()) {
            uint64_t hash = 0;
            std::string valueString(value.GetString(), value.GetStringLength());
            if (!autil::StringUtil::hexStrToUint64(valueString.c_str(), hash)) {
                return nullptr;
            }
            hashes->emplace_back(hash);
        } else if (value.IsInt64() && value.GetInt64() >= 0) {
            hashes->emplace_back(static_cast<uint64_t>(value.GetInt64()));
        } else if (value.IsUint64()) {
            hashes->emplace_back(value.GetUint64());
        } else {
            return nullptr;
        }
    }
    auto maxBlockCount = static_cast<size_t>(promptTokenCount / blockSize);
    if (hashes->size() > maxBlockCount) {
        hashes->pop_back();
    }
    return hashes->size() <= maxBlockCount ? hashes : nullptr;
}

bool getServiceName(const rapidjson::Value &object, std::string &serviceName)
{
    if (getString(object, "service_name", serviceName)) {
        return true;
    }
    auto field = findValue(object, "service_names");
    if (!field || !field->IsArray() || field->Empty() || !(*field)[0].IsString()) {
        return false;
    }
    serviceName.assign((*field)[0].GetString(), (*field)[0].GetStringLength());
    return true;
}

}

class TrafficReplayRequestClosure : public navi::NaviUserResultClosure {
public:
    TrafficReplayRequestClosure(TrafficReplayR *trafficReplayR, uint64_t requestId)
        : _trafficReplayR(trafficReplayR), _requestId(std::move(requestId))
    {
    }
    void run(navi::NaviUserResultPtr userResult,
             const std::shared_ptr<navi::ComputeScope> &) override
    {
        _trafficReplayR->finishRequest(this, userResult);
    }
    uint64_t getRequestId() const { return _requestId; }

private:
    TrafficReplayR *_trafficReplayR;
    uint64_t _requestId;
};

const std::string TrafficReplayR::RESOURCE_ID = "navi_sched.simulate_traffic_replay_r";

const std::string TrafficReplayBridgeR::RESOURCE_ID =
    "navi_sched.simulate_traffic_replay_bridge_r";

TrafficReplayBridgeR::TrafficReplayBridgeR()
    : navi::ExternalResource(RESOURCE_ID, navi::RS_RUN_GRAPH_EXTERNAL)
{
}

void TrafficReplayBridgeR::setTrafficReplayR(
    const std::shared_ptr<TrafficReplayR> &resource)
{
    autil::ScopedLock lock(_condition);
    _trafficReplayR = resource;
    _condition.broadcast();
}

void TrafficReplayBridgeR::clearTrafficReplayR(const TrafficReplayR *resource)
{
    autil::ScopedLock lock(_condition);
    if (_trafficReplayR.get() == resource) {
        _trafficReplayR.reset();
    }
}

std::shared_ptr<TrafficReplayR> TrafficReplayBridgeR::waitTrafficReplayR(
    int64_t timeoutUs)
{
    autil::ScopedLock lock(_condition);
    while (!_trafficReplayR) {
        if (ETIMEDOUT == _condition.wait(timeoutUs)) {
            return nullptr;
        }
    }
    return _trafficReplayR;
}

void TrafficReplayR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_KERNEL);
}

bool TrafficReplayR::config(navi::ResourceConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "traffic_log", _config.trafficLog, _config.trafficLog);
    NAVI_JSONIZE(ctx, "target_biz_name", _config.targetBizName, _config.targetBizName);
    NAVI_JSONIZE(ctx, "master_biz_name", _config.masterBizName, _config.masterBizName);
    NAVI_JSONIZE(ctx, "service_name", _config.serviceName, _config.serviceName);
    NAVI_JSONIZE(ctx, "client_count", _config.clientCount, _config.clientCount);
    NAVI_JSONIZE(ctx, "client_index", _config.clientIndex, _config.clientIndex);
    NAVI_JSONIZE(ctx, "time_compression", _config.timeCompression,
                 _config.timeCompression);
    NAVI_JSONIZE(ctx, "request_timeout_ms", _config.requestTimeoutMs,
                 _config.requestTimeoutMs);
    NAVI_JSONIZE(ctx, "start_delay_ms", _config.startDelayMs, _config.startDelayMs);
    NAVI_JSONIZE(ctx, "block_size", _config.blockSize, _config.blockSize);
    NAVI_JSONIZE(ctx, "collect_replay_sequences", _config.collectReplaySequences,
                 _config.collectReplaySequences);
    NAVI_JSONIZE(ctx, "replay_output_path", _config.replayOutputPath,
                 _config.replayOutputPath);
    NAVI_JSONIZE(ctx, "latency_output_path", _config.latencyOutputPath,
                 _config.latencyOutputPath);
    return !_config.trafficLog.empty() && !_config.targetBizName.empty() &&
           !_config.masterBizName.empty() && !_config.serviceName.empty() &&
           _config.clientCount > 0 && _config.clientIndex < _config.clientCount &&
           _config.timeCompression > 0.0 && _config.requestTimeoutMs > 0 &&
           _config.startDelayMs > 0 && _config.blockSize > 0 &&
           (_config.collectReplaySequences.empty() ||
            !_config.replayOutputPath.empty());
}

navi::ErrorCode TrafficReplayR::init(navi::ResourceInitContext &)
{
    if (!_trafficReplayBridgeR) {
        return navi::EC_INIT_RESOURCE;
    }
    if (!_config.latencyOutputPath.empty()) {
        _latencyOutput.open(_config.latencyOutputPath, std::ios::trunc);
        if (!_latencyOutput.good()) {
            NAVI_KERNEL_LOG(ERROR, "open traffic latency output failed, path [%s]",
                            _config.latencyOutputPath.c_str());
            return navi::EC_INIT_RESOURCE;
        }
    }
    if (!loadTrafficLog()) {
        return navi::EC_INIT_RESOURCE;
    }
    _trafficReplayBridgeR->setTrafficReplayR(
        std::dynamic_pointer_cast<TrafficReplayR>(shared_from_this()));
    return navi::EC_NONE;
}

TrafficReplayR::~TrafficReplayR()
{
    if (_trafficReplayBridgeR) {
        _trafficReplayBridgeR->clearTrafficReplayR(this);
    }
}

bool TrafficReplayR::takeNext(TrafficReplayRequest &request)
{
    if (_nextRequest >= _requests.size()) {
        return false;
    }
    request = _requests[_nextRequest++];
    return true;
}

bool TrafficReplayR::peekNext(TrafficReplayRequest &request) const
{
    if (_nextRequest >= _requests.size()) {
        return false;
    }
    request = _requests[_nextRequest];
    return true;
}

bool TrafficReplayR::empty() const { return _nextRequest >= _requests.size(); }

size_t TrafficReplayR::getRequestCount() const { return _requests.size(); }

int64_t TrafficReplayR::getFirstRequestTimeMs() const
{
    return _requests.empty() ? 0 : _requests.front().requestTimeMs;
}

const std::string &TrafficReplayR::getTargetBizName() const
{
    return _config.targetBizName;
}

const std::string &TrafficReplayR::getMasterBizName() const
{
    return _config.masterBizName;
}

double TrafficReplayR::getTimeCompression() const { return _config.timeCompression; }

int64_t TrafficReplayR::getRequestTimeoutMs() const { return _config.requestTimeoutMs; }

size_t TrafficReplayR::getClientIndex() const { return _config.clientIndex; }
size_t TrafficReplayR::getClientCount() const { return _config.clientCount; }
int64_t TrafficReplayR::getStartDelayMs() const { return _config.startDelayMs; }

int64_t TrafficReplayR::getBlockSize() const { return _config.blockSize; }

bool TrafficReplayR::shouldCollectReplay(const TrafficReplayRequest &request) const
{
    return std::binary_search(_collectReplaySequences.begin(),
                              _collectReplaySequences.end(), request.sequence);
}

bool TrafficReplayR::saveReplay(const NaviSchedScheduleReplayPtr &replay) const
{
    if (!replay) {
        return true;
    }
    autil::ScopedLock lock(_replayOutputLock);
    std::ofstream output(_config.replayOutputPath, std::ios::app);
    if (!output.good()) {
        NAVI_KERNEL_LOG(ERROR, "open traffic replay output failed, path [%s]",
                        _config.replayOutputPath.c_str());
        return false;
    }
    output << NaviSchedScheduleReplayJson::toJsonString(*replay) << '\n';
    return output.good();
}

void TrafficReplayR::logLatencyStats()
{
    std::vector<double> latencies;
    {
        autil::ScopedLock lock(_latencyOutputLock);
        if (_latencyStatsLogged) {
            return;
        }
        _latencyStatsLogged = true;
        _latencyOutput.flush();
        latencies = _clientTotalLatencies;
    }
    if (latencies.empty()) {
        NAVI_KERNEL_LOG(WARN, "traffic replay has no forward timing samples");
        return;
    }
    std::sort(latencies.begin(), latencies.end());
    double sum = 0.0;
    for (double latency : latencies) {
        sum += latency;
    }
    NAVI_KERNEL_LOG(
        INFO,
        "traffic replay latency distribution, count [%lu], min [%.3f], avg [%.3f], "
        "p50 [%.3f], p90 [%.3f], p95 [%.3f], p99 [%.3f], max [%.3f]",
        latencies.size(), latencies.front(), sum / latencies.size(),
        getPercentile(latencies, 0.50), getPercentile(latencies, 0.90),
        getPercentile(latencies, 0.95), getPercentile(latencies, 0.99), latencies.back());
}

bool TrafficReplayR::setStartPipe(const navi::AsyncPipePtr &startPipe)
{
    if (!startPipe) return false;
    autil::ScopedLock lock(_controlLock);
    if (_startPipe && _startPipe != startPipe) return false;
    _startPipe = startPipe;
    return true;
}

void TrafficReplayR::clearStartPipe(const navi::AsyncPipePtr &startPipe)
{
    autil::ScopedLock lock(_controlLock);
    if (_startPipe == startPipe) _startPipe.reset();
}

bool TrafficReplayR::start(int64_t startTimeMs, int64_t firstRequestTimeMs)
{
    navi::AsyncPipePtr startPipe;
    {
        autil::ScopedLock lock(_controlLock);
        if (_startTimeMs > 0 || startTimeMs <= 0 || firstRequestTimeMs < 0) {
            return false;
        }
        _startTimeMs = startTimeMs;
        _firstRequestTimeMs = firstRequestTimeMs;
        startPipe = _startPipe;
    }
    return startPipe && navi::EC_NONE == startPipe->setDataOverride(nullptr, false, true);
}

bool TrafficReplayR::getStartTime(int64_t &startTimeMs, int64_t &firstRequestTimeMs) const
{
    autil::ScopedLock lock(_controlLock);
    if (_startTimeMs <= 0) return false;
    startTimeMs = _startTimeMs;
    firstRequestTimeMs = _firstRequestTimeMs;
    return true;
}

bool TrafficReplayR::setControlPipe(const navi::AsyncPipePtr &controlPipe)
{
    if (!controlPipe) {
        return false;
    }
    autil::ScopedLock lock(_controlLock);
    if (_controlPipe && _controlPipe != controlPipe) {
        return false;
    }
    _controlPipe = controlPipe;
    return true;
}

void TrafficReplayR::clearControlPipe(const navi::AsyncPipePtr &controlPipe)
{
    autil::ScopedLock lock(_controlLock);
    if (_controlPipe == controlPipe) {
        _controlPipe.reset();
    }
}

bool TrafficReplayR::setCompletionPipe(const navi::AsyncPipePtr &completionPipe)
{
    if (!completionPipe) return false;
    autil::ScopedLock lock(_controlLock);
    if (_completionPipe && _completionPipe != completionPipe) return false;
    _completionPipe = completionPipe;
    return true;
}

void TrafficReplayR::clearCompletionPipe(const navi::AsyncPipePtr &completionPipe)
{
    autil::ScopedLock lock(_controlLock);
    if (_completionPipe == completionPipe) _completionPipe.reset();
}

void TrafficReplayR::setNavi(navi::Navi *navi)
{
    autil::ScopedLock lock(_controlLock);
    _navi = navi;
}

void TrafficReplayR::clearNavi(navi::Navi *navi)
{
    autil::ScopedLock lock(_controlLock);
    if (_navi == navi) _navi = nullptr;
}

bool TrafficReplayR::startRequest(navi::GraphDef *graphDef,
                                  const navi::RunGraphParams &params, uint64_t requestId)
{
    if (!graphDef || 0 == requestId) return false;
    auto closure = std::make_shared<TrafficReplayRequestClosure>(this, requestId);
    navi::Navi *navi = nullptr;
    {
        autil::ScopedLock lock(_controlLock);
        navi = _navi;
        if (!navi) return false;
        _requestStartTimesUs[requestId] = autil::TimeUtility::currentTime();
        _closures.emplace_back(closure);
        ++_inflightRequestCount;
    }
    navi::ResourceMap resourceMap;
    if (!resourceMap.addResource(_trafficReplayBridgeR->shared_from_this())) {
        return false;
    }
    return bool(navi->runLocalGraphAsync(graphDef, params, resourceMap, closure.get()));
}

int64_t TrafficReplayR::getInflightRequestCount() const
{
    autil::ScopedLock lock(_controlLock);
    return _inflightRequestCount;
}

void TrafficReplayR::finishRequest(TrafficReplayRequestClosure *closure,
                                   const navi::NaviUserResultPtr &userResult)
{
    bool eof = false;
    while (userResult && !eof) {
        navi::NaviUserData userData;
        userResult->nextData(userData, eof);
    }
    navi::AsyncPipePtr completionPipe;
    std::shared_ptr<TrafficReplayRequestClosure> closurePtr;
    {
        autil::ScopedLock lock(_controlLock);
        auto it = _closures.begin();
        for (; it != _closures.end(); ++it) {
            if (it->get() == closure) {
                break;
            }
        }
        if (_closures.end() == it) {
            return;
        }
        closurePtr = *it;
        _closures.erase(it);
        _requestStartTimesUs.erase(closurePtr->getRequestId());
        if (_inflightRequestCount > 0) --_inflightRequestCount;
        completionPipe = _completionPipe;
    }
    if (completionPipe) completionPipe->setDataOverride(nullptr, false, true);
}

void TrafficReplayR::recordRequestTiming(uint64_t requestId,
                                         const NaviSchedRequestTiming &requestTiming,
                                         uint64_t scheduleId,
                                         double scheduleLoss,
                                         int64_t scheduleLatencyUs)
{
    auto timing = requestTiming;
    {
        autil::ScopedLock lock(_controlLock);
        auto it = _requestStartTimesUs.find(requestId);
        if (_requestStartTimesUs.end() != it) {
            timing.clientTotalLatencyMs =
                std::max(0.0, static_cast<double>(autil::TimeUtility::currentTime() -
                                                  it->second) /
                                  1000.0) *
                _config.timeCompression;
            _requestStartTimesUs.erase(it);
        }
    }
    autil::ScopedLock lock(_latencyOutputLock);
    _clientTotalLatencies.emplace_back(timing.clientTotalLatencyMs);
    if (!_latencyOutput.is_open()) {
        return;
    }
    _latencyOutput << autil::legacy::FastToJsonString(
                          LatencySampleJson(
                              requestId, timing, scheduleId,
                              scheduleLoss, scheduleLatencyUs))
                   << '\n';
    if (!_latencyOutput.good()) {
        NAVI_KERNEL_LOG(ERROR, "write traffic latency output failed, path [%s]",
                        _config.latencyOutputPath.c_str());
    }
}

double TrafficReplayR::getPercentile(const std::vector<double> &values, double percentile)
{
    auto index = static_cast<size_t>(std::ceil(percentile * values.size()));
    return values[std::min(values.size() - 1, index - 1)];
}

bool TrafficReplayR::finish()
{
    navi::AsyncPipePtr controlPipe;
    {
        autil::ScopedLock lock(_controlLock);
        controlPipe = _controlPipe;
    }
    return !controlPipe || navi::EC_NONE == controlPipe->setEof();
}

bool TrafficReplayR::loadTrafficLog()
{
    std::ifstream input(_config.trafficLog);
    if (!input.good()) {
        NAVI_KERNEL_LOG(ERROR, "read traffic replay log failed, path [%s]",
                        _config.trafficLog.c_str());
        return false;
    }
    _requests.clear();
    _nextRequest = 0;
    _collectReplaySequences.clear();
    if (!_config.collectReplaySequences.empty()) {
        for (const auto &value :
             autil::StringUtil::split(_config.collectReplaySequences, ",")) {
            uint64_t sequence = 0;
            if (!autil::StringUtil::fromString(value, sequence) ||
                sequence > std::numeric_limits<size_t>::max()) {
                NAVI_KERNEL_LOG(ERROR, "invalid collect replay sequence [%s]",
                                value.c_str());
                return false;
            }
            _collectReplaySequences.emplace_back(sequence);
        }
        std::sort(_collectReplaySequences.begin(), _collectReplaySequences.end());
        _collectReplaySequences.erase(
            std::unique(_collectReplaySequences.begin(), _collectReplaySequences.end()),
            _collectReplaySequences.end());
    }
    size_t sequence = 0;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && !appendRecord(line, sequence)) return false;
        ++sequence;
    }
    if (!input.eof()) return false;
    std::stable_sort(
        _requests.begin(), _requests.end(),
        [](const TrafficReplayRequest &left, const TrafficReplayRequest &right) {
            return left.requestTimeMs < right.requestTimeMs;
        });
    if (_requests.empty()) {
        NAVI_KERNEL_LOG(ERROR, "traffic replay log has no valid request, path [%s]",
                        _config.trafficLog.c_str());
        return false;
    }
    std::vector<TrafficReplayRequest> requests;
    requests.reserve((_requests.size() + _config.clientCount - 1) / _config.clientCount);
    for (size_t index = 0; index < _requests.size(); ++index) {
        auto &request = _requests[index];
        request.sequence = index;
        request.requestId = index + 1;
        request.clientIndex = index % _config.clientCount;
        if (request.clientIndex == _config.clientIndex) {
            requests.emplace_back(std::move(request));
        }
    }
    _requests = std::move(requests);
    if (_requests.empty()) {
        NAVI_KERNEL_LOG(
            ERROR, "traffic replay client has no assigned request, clientIndex [%lu]",
            _config.clientIndex);
        return false;
    }
    return true;
}

bool TrafficReplayR::appendRecord(const std::string &json, size_t sequence)
{
    rapidjson::Document document;
    document.Parse(json.c_str(), json.size());
    if (document.HasParseError() || !document.IsObject()) {
        NAVI_KERNEL_LOG(ERROR, "invalid traffic replay record, sequence [%lu]", sequence);
        return false;
    }
    TrafficReplayRequest request;
    if (!getTimestampMs(document, request.requestTimeMs) || request.requestTimeMs < 0) {
        NAVI_KERNEL_LOG(ERROR, "traffic replay record misses timestamp, sequence [%lu]",
                        sequence);
        return false;
    }
    request.requestId = sequence + 1;
    getServiceName(document, request.serviceName);
    if (!_config.serviceName.empty() && _config.serviceName != request.serviceName) {
        return true;
    }
    getInt64(document, "input_length", request.promptTokenCount);
    if (request.promptTokenCount <= 0) {
        request.promptTokenCount = getNestedInt64(document, "usage", "inputTokens");
    }
    if (request.promptTokenCount <= 0) {
        getInt64(document, "prompt_token_count", request.promptTokenCount);
    }
    if (request.promptTokenCount <= 0) {
        NAVI_KERNEL_LOG(ERROR,
                        "traffic replay record misses prompt token count, sequence [%lu]",
                        sequence);
        return false;
    }
    request.blockHashes =
        getBlockHashes(document, request.promptTokenCount, _config.blockSize);
    if (!request.blockHashes) {
        NAVI_KERNEL_LOG(ERROR,
                        "traffic replay record has invalid input_block_hash_ids, "
                        "sequence [%lu]",
                        sequence);
        return false;
    }
    request.sequence = sequence;
    _requests.emplace_back(std::move(request));
    return true;
}

REGISTER_RESOURCE(TrafficReplayR);
REGISTER_RESOURCE(TrafficReplayBridgeR);

}
