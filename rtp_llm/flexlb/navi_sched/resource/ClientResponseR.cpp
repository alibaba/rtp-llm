#include "rtp_llm/flexlb/navi_sched/resource/ClientResponseR.h"
#include "autil/TimeUtility.h"
#include <chrono>
#include <vector>

namespace navi_sched {

const std::string ClientResponseR::RESOURCE_ID = "navi_sched.client_response_r";

ClientResponseR::ClientResponseR()
    : navi::ExternalResource(RESOURCE_ID, navi::RS_EXTERNAL)
{
}

bool ClientResponseR::begin(uint64_t requestId, bool expectReplay,
                            bool expectResponseStream,
                            bool hasClosureCompletion)
{
    if (requestId == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(_mutex);
    auto state = std::make_shared<RequestState>();
    state->expectReplay = expectReplay;
    state->expectResponseStream = expectResponseStream;
    state->closureCompletionFinished = !hasClosureCompletion;
    return !_stopped && _states.emplace(requestId, std::move(state)).second;
}

ClientResponseR::RequestStatePtr ClientResponseR::find(uint64_t requestId) const
{
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _states.find(requestId);
    return it == _states.end() ? nullptr : it->second;
}

bool ClientResponseR::attach(uint64_t requestId,
                             const navi::NaviUserResultPtr &userResult)
{
    auto state = find(requestId);
    if (!state || !userResult) {
        return false;
    }
    navi::NaviUserResultPtr resultToAbort;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->graphFinished || state->userResult) {
            return false;
        }
        state->userResult = userResult;
        if (state->abortRequested && !state->abortIssued) {
            state->abortIssued = true;
            resultToAbort = userResult;
        }
    }
    if (resultToAbort) {
        resultToAbort->abort(navi::EC_ABORT);
    }
    return true;
}

bool ClientResponseR::publishDecision(
    uint64_t requestId, const MasterResponseToClientDataPtr &decision)
{
    auto state = find(requestId);
    if (!state || !decision || decision->version != NAVI_SCHED_DATA_VERSION ||
        decision->requestId == 0) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->decision || state->graphFinished) {
            return false;
        }
        state->decision = decision->makeCopy();
        state->decision->requestId = requestId;
        state->decisionTimeUs = autil::TimeUtility::currentTime();
    }
    state->condition.notify_all();
    return true;
}

bool ClientResponseR::publishReplay(
    uint64_t requestId, const MasterResponseToClientDataPtr &replay)
{
    auto state = find(requestId);
    if (!state || !replay || replay->version != NAVI_SCHED_DATA_VERSION ||
        !replay->replay || replay->replay->version != NAVI_SCHED_DATA_VERSION) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->replay || state->graphFinished) {
            return false;
        }
        state->replay = replay->replay;
    }
    state->condition.notify_all();
    return true;
}

bool ClientResponseR::publishResponse(
    uint64_t requestId, const AgentResponseToClientDataPtr &response)
{
    if (!response || !response->valid() ||
        AgentResponseKind::ACK == response->kind) {
        return false;
    }
    auto state = find(requestId);
    if (!state) {
        return false;
    }
    {
        std::lock_guard<std::mutex> stateLock(state->mutex);
        if (state->responseFinished || state->graphFinished) {
            return false;
        }
        if (!state->expectResponseStream) {
            state->responseFinished = response->terminal;
            return true;
        }
        std::lock_guard<std::mutex> lock(_mutex);
        if (_stopped) {
            return false;
        }
        state->responseFinished = response->terminal;
        auto output = std::make_shared<AgentResponseToClientData>();
        output->requestId = requestId;
        output->kind = response->kind;
        output->frames = response->frames;
        output->terminal = response->terminal;
        output->errorMessage = response->errorMessage;
        output->requestTiming = response->requestTiming;
        _responses.emplace_back(std::move(output));
    }
    _responseCondition.notify_one();
    return true;
}

ClientResponseR::DecisionResult ClientResponseR::waitDecision(
    uint64_t requestId, int64_t timeoutMs)
{
    DecisionResult result;
    auto state = find(requestId);
    if (!state) {
        result.errorCode = navi::EC_ABORT;
        result.errorMessage = "client response state not found";
        return result;
    }
    std::unique_lock<std::mutex> lock(state->mutex);
    const auto ready = [state] {
        return (state->decision && (!state->expectReplay || state->replay)) ||
               state->graphFinished || state->stopRequested;
    };
    if (timeoutMs <= 0 ||
        !state->condition.wait_for(lock, std::chrono::milliseconds(timeoutMs), ready)) {
        result.errorCode = navi::EC_TIMEOUT;
        result.errorMessage = "timed out waiting for NaviSched decision";
        state->decisionConsumed = true;
        return result;
    }
    if (state->stopRequested) {
        result.errorCode = navi::EC_ABORT;
        result.errorMessage = state->errorMessage;
    } else if (state->decision && (!state->expectReplay || state->replay)) {
        result.errorCode = navi::EC_NONE;
        result.decision = state->decision;
        result.replay = state->replay;
    } else {
        result.errorCode = state->errorCode == navi::EC_NONE ? navi::EC_NO_DATA
                                                             : state->errorCode;
        result.errorMessage = state->errorMessage.empty()
                                  ? "schedule graph returned no decision"
                                  : state->errorMessage;
    }
    state->decisionConsumed = true;
    const bool eraseState = state->graphFinished;
    lock.unlock();
    if (eraseState) {
        std::lock_guard<std::mutex> mapLock(_mutex);
        auto it = _states.find(requestId);
        if (it != _states.end() && it->second == state) {
            _states.erase(it);
        }
    }
    return result;
}

AgentResponseToClientDataPtr ClientResponseR::takeNextResponse(int64_t timeoutMs)
{
    std::unique_lock<std::mutex> lock(_mutex);
    if (_responses.empty() && !_stopped && timeoutMs > 0) {
        _responseCondition.wait_for(
            lock, std::chrono::milliseconds(timeoutMs),
            [this] { return _stopped || !_responses.empty(); });
    }
    if (_responses.empty()) {
        return nullptr;
    }
    auto response = std::move(_responses.front());
    _responses.pop_front();
    return response;
}

bool ClientResponseR::abort(uint64_t requestId)
{
    auto state = find(requestId);
    if (!state) {
        return false;
    }
    navi::NaviUserResultPtr userResult;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->graphFinished) {
            return false;
        }
        state->abortRequested = true;
        if (state->userResult && !state->abortIssued) {
            state->abortIssued = true;
            userResult = state->userResult;
        }
    }
    if (userResult) {
        userResult->abort(navi::EC_ABORT);
    }
    return true;
}

ClientResponseR::DecisionResult ClientResponseR::finishGraph(
    uint64_t requestId, navi::ErrorCode errorCode,
    const std::string &errorMessage, bool consumeDecision)
{
    DecisionResult result;
    auto state = find(requestId);
    if (!state) {
        result.errorCode = navi::EC_ABORT;
        result.errorMessage = "client response state not found";
        return result;
    }
    bool synthesizeError = false;
    bool eraseState = false;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->graphFinished = true;
        state->userResult.reset();
        if (!state->stopRequested) {
            state->errorCode = errorCode;
            state->errorMessage = errorMessage;
        }
        synthesizeError = state->expectResponseStream && state->decision &&
                          !state->decision->target.address.empty() &&
                          !state->responseFinished;
        if (state->stopRequested) {
            result.errorCode = navi::EC_ABORT;
            result.errorMessage = state->errorMessage;
        } else if (state->decision && (!state->expectReplay || state->replay)) {
            result.errorCode = navi::EC_NONE;
            result.decision = state->decision;
            result.replay = state->replay;
            result.decisionTimeUs = state->decisionTimeUs;
        } else {
            result.errorCode = errorCode == navi::EC_NONE ? navi::EC_NO_DATA
                                                           : errorCode;
            result.errorMessage = errorMessage.empty()
                                      ? "schedule graph returned no decision"
                                      : errorMessage;
        }
        if (consumeDecision) {
            state->decisionConsumed = true;
        }
        eraseState = state->decisionConsumed && state->closureCompletionFinished;
    }
    state->condition.notify_all();
    if (eraseState) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _states.find(requestId);
        if (it != _states.end() && it->second == state) {
            _states.erase(it);
        }
    }
    if (synthesizeError) {
        auto response = std::make_shared<AgentResponseToClientData>();
        response->requestId = requestId;
        response->kind = AgentResponseKind::ERROR;
        response->terminal = true;
        response->errorMessage = errorMessage.empty()
                                     ? "NaviSched response graph ended before terminal output"
                                     : errorMessage;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (!_stopped) {
                _responses.emplace_back(std::move(response));
            }
        }
        _responseCondition.notify_one();
    }
    return result;
}

void ClientResponseR::finishClosureCompletion(uint64_t requestId)
{
    auto state = find(requestId);
    if (!state) {
        return;
    }
    bool eraseState = false;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->closureCompletionFinished = true;
        eraseState = state->graphFinished && state->decisionConsumed;
    }
    state->condition.notify_all();
    if (eraseState) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _states.find(requestId);
        if (it != _states.end() && it->second == state) {
            _states.erase(it);
        }
    }
}

void ClientResponseR::finishAll()
{
    std::vector<navi::NaviUserResultPtr> results;
    std::vector<RequestStatePtr> states;
    {
        std::unique_lock<std::mutex> lock(_mutex);
        if (_finishCompleted) {
            return;
        }
        if (_stopped) {
            _finishCondition.wait(lock, [this] { return _finishCompleted; });
            return;
        }
        _stopped = true;
        states.reserve(_states.size());
        for (const auto &[_, state] : _states) {
            states.emplace_back(state);
        }
    }
    results.reserve(states.size());
    for (const auto &state : states) {
        std::lock_guard<std::mutex> stateLock(state->mutex);
        if (state->graphFinished) {
            continue;
        }
        state->stopRequested = true;
        state->abortRequested = true;
        state->errorCode = navi::EC_ABORT;
        state->errorMessage = "NaviSched client stopped";
        if (state->userResult && !state->abortIssued) {
            state->abortIssued = true;
            results.emplace_back(state->userResult);
        }
        state->condition.notify_all();
    }
    _responseCondition.notify_all();
    for (const auto &result : results) {
        result->abort(navi::EC_ABORT);
    }
    for (const auto &state : states) {
        std::unique_lock<std::mutex> stateLock(state->mutex);
        state->condition.wait(stateLock,
                              [state] {
                                  return state->graphFinished &&
                                         state->closureCompletionFinished;
                              });
    }
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _states.clear();
        _finishCompleted = true;
    }
    _finishCondition.notify_all();
}

size_t ClientResponseR::requestCount() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _states.size();
}

navi::SummaryErrorCode ClientResponseR::getSummary(
    const std::string &, navi::SummaryBuilder &builder) const
{
    std::lock_guard<std::mutex> lock(_mutex);
    auto table = builder.newTable("NaviSchedClientResponseOverview");
    TABLE_ADD_COLUMN_VALUE(table, "active_request_count", _states.size());
    TABLE_ADD_COLUMN_VALUE(table, "queued_response_count", _responses.size());
    TABLE_ADD_COLUMN_VALUE(table, "stopped", _stopped);
    return navi::SEC_OK;
}

REGISTER_RESOURCE(ClientResponseR);

} // namespace navi_sched
