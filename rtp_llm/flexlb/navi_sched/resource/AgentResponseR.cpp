#include "aios/navi_sched/resource/AgentResponseR.h"
#include "aios/navi_sched/data/AgentResponseToClientData.h"

namespace navi_sched {

const std::string AgentResponseR::RESOURCE_ID = "navi_sched.agent_response_r";

AgentResponseR::AgentResponseR()
    : navi::ExternalResource(RESOURCE_ID, navi::RS_EXTERNAL)
{
}

bool AgentResponseR::registerRequest(uint64_t requestId,
                                     const navi::AsyncPipePtr &pipe)
{
    if (requestId == 0 || !pipe) {
        _rejectedCount.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    std::lock_guard<std::mutex> lock(_mutex);
    if (_stopped ||
        !_entries.emplace(requestId, std::make_shared<Entry>(pipe)).second) {
        _rejectedCount.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    _registeredCount.fetch_add(1, std::memory_order_relaxed);
    return true;
}

AgentResponseR::EntryPtr AgentResponseR::find(uint64_t requestId) const
{
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(requestId);
    return it == _entries.end() ? nullptr : it->second;
}

bool AgentResponseR::erase(uint64_t requestId, const EntryPtr &entry)
{
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(requestId);
    if (it == _entries.end() || it->second != entry) {
        return false;
    }
    _entries.erase(it);
    return true;
}

bool AgentResponseR::deliverLocked(
    const EntryPtr &entry,
    const std::shared_ptr<AgentResponseToClientData> &response,
    bool &pipeFailed)
{
    pipeFailed = false;
    if (!entry || !response || entry->finished || !entry->pipe) {
        return false;
    }
    const bool delivered = navi::EC_NONE == entry->pipe->setData(response);
    if (!delivered) {
        entry->finished = true;
        entry->pending.clear();
        entry->pipe->setEof();
        _rejectedCount.fetch_add(1, std::memory_order_relaxed);
        pipeFailed = true;
        return false;
    }
    _publishedCount.fetch_add(1, std::memory_order_relaxed);
    if (response->terminal) {
        entry->finished = true;
        entry->pending.clear();
        if (navi::EC_NONE != entry->pipe->setEof()) {
            pipeFailed = true;
        }
        _terminalCount.fetch_add(1, std::memory_order_relaxed);
    }
    return !pipeFailed;
}

bool AgentResponseR::acknowledge(uint64_t requestId,
                                 const NaviSchedRequestTiming &requestTiming,
                                 bool terminal)
{
    auto entry = find(requestId);
    if (!entry) {
        _rejectedCount.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    auto response = std::make_shared<AgentResponseToClientData>();
    response->requestId = requestId;
    response->kind = AgentResponseKind::ACK;
    response->requestTiming = requestTiming;
    response->terminal = terminal;

    bool pipeFailed = false;
    bool finished = false;
    bool delivered = false;
    {
        std::lock_guard<std::mutex> lock(entry->mutex);
        if (entry->acknowledged || entry->finished) {
            _rejectedCount.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        entry->acknowledged = true;
        delivered = deliverLocked(entry, response, pipeFailed);
        while (delivered && !entry->finished && !entry->pending.empty()) {
            auto pending = std::move(entry->pending.front());
            entry->pending.pop_front();
            delivered = deliverLocked(entry, pending, pipeFailed);
        }
        finished = entry->finished;
    }
    if (finished) {
        erase(requestId, entry);
    }
    if (pipeFailed) {
        _cancelledCount.fetch_add(1, std::memory_order_relaxed);
        notifyCancelled(requestId);
    }
    return delivered;
}

bool AgentResponseR::publish(uint64_t requestId, std::vector<std::string> frames,
                             bool terminal)
{
    if (frames.empty()) {
        _rejectedCount.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    auto entry = find(requestId);
    if (!entry) {
        _rejectedCount.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    auto response = std::make_shared<AgentResponseToClientData>();
    response->requestId = requestId;
    response->kind = AgentResponseKind::OUTPUT;
    response->frames = std::move(frames);
    response->terminal = terminal;

    bool pipeFailed = false;
    bool finished = false;
    bool delivered = true;
    {
        std::lock_guard<std::mutex> lock(entry->mutex);
        if (entry->finished || entry->terminalPending) {
            _rejectedCount.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (!entry->acknowledged) {
            entry->pending.emplace_back(std::move(response));
            entry->terminalPending = terminal;
            return true;
        }
        delivered = deliverLocked(entry, response, pipeFailed);
        finished = entry->finished;
    }
    if (finished) {
        erase(requestId, entry);
    }
    if (pipeFailed) {
        _cancelledCount.fetch_add(1, std::memory_order_relaxed);
        notifyCancelled(requestId);
    }
    return delivered;
}

bool AgentResponseR::fail(uint64_t requestId, const std::string &errorMessage)
{
    if (errorMessage.empty()) {
        _rejectedCount.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    auto entry = find(requestId);
    if (!entry) {
        _rejectedCount.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    auto response = std::make_shared<AgentResponseToClientData>();
    response->requestId = requestId;
    response->kind = AgentResponseKind::ERROR;
    response->terminal = true;
    response->errorMessage = errorMessage;

    bool pipeFailed = false;
    bool delivered = false;
    {
        std::lock_guard<std::mutex> lock(entry->mutex);
        if (entry->finished) {
            _rejectedCount.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        entry->pending.clear();
        entry->terminalPending = true;
        delivered = deliverLocked(entry, response, pipeFailed);
    }
    erase(requestId, entry);
    if (pipeFailed) {
        _cancelledCount.fetch_add(1, std::memory_order_relaxed);
        notifyCancelled(requestId);
    }
    return delivered;
}

bool AgentResponseR::cancel(uint64_t requestId, const navi::AsyncPipePtr &pipe)
{
    auto entry = find(requestId);
    if (!entry || entry->pipe != pipe) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(entry->mutex);
        if (entry->finished) {
            return false;
        }
        entry->finished = true;
        entry->pending.clear();
        entry->pipe->setEof();
    }
    erase(requestId, entry);
    _cancelledCount.fetch_add(1, std::memory_order_relaxed);
    notifyCancelled(requestId);
    return true;
}

void AgentResponseR::notifyCancelled(uint64_t requestId)
{
    CancelHandler handler;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        handler = _cancelHandler;
    }
    if (handler) {
        handler(requestId);
    }
}

void AgentResponseR::finishAll(const std::string &errorMessage)
{
    std::vector<std::pair<uint64_t, EntryPtr>> entries;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _stopped = true;
        entries.reserve(_entries.size());
        for (auto &value : _entries) {
            entries.emplace_back(value.first, std::move(value.second));
        }
        _entries.clear();
    }
    for (const auto &[requestId, entry] : entries) {
        auto response = std::make_shared<AgentResponseToClientData>();
        response->requestId = requestId;
        response->kind = AgentResponseKind::ERROR;
        response->terminal = true;
        response->errorMessage = errorMessage;
        std::lock_guard<std::mutex> lock(entry->mutex);
        if (!entry->finished && entry->pipe) {
            entry->pending.clear();
            entry->terminalPending = true;
            bool pipeFailed = false;
            deliverLocked(entry, response, pipeFailed);
        }
    }
}

void AgentResponseR::setCancelHandler(const CancelHandler &handler)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _cancelHandler = handler;
}

size_t AgentResponseR::requestCount() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _entries.size();
}

navi::SummaryErrorCode AgentResponseR::getSummary(
    const std::string &, navi::SummaryBuilder &builder) const
{
    auto table = builder.newTable("NaviSchedAgentResponseOverview");
    TABLE_ADD_COLUMN_VALUE(table, "active_request_count", requestCount());
    TABLE_ADD_COLUMN_VALUE(
        table, "registered_count", _registeredCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(
        table, "published_count", _publishedCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(
        table, "terminal_count", _terminalCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(
        table, "cancelled_count", _cancelledCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(
        table, "rejected_count", _rejectedCount.load(std::memory_order_relaxed));
    return navi::SEC_OK;
}

REGISTER_RESOURCE(AgentResponseR);

}
