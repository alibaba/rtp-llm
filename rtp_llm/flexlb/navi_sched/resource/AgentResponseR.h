#pragma once

#include "rtp_llm/flexlb/navi_sched/common/NaviSchedScheduleData.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Resource.h"
#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace navi_sched {

class AgentResponseR : public navi::ExternalResource {
public:
    using CancelHandler = std::function<void(uint64_t)>;

    AgentResponseR();

public:
    bool registerRequest(uint64_t requestId, const navi::AsyncPipePtr &pipe);
    bool acknowledge(uint64_t requestId,
                     const NaviSchedRequestTiming &requestTiming,
                     bool terminal = false);
    bool publish(uint64_t requestId, std::vector<std::string> frames, bool terminal);
    bool fail(uint64_t requestId, const std::string &errorMessage);
    bool cancel(uint64_t requestId, const navi::AsyncPipePtr &pipe);
    void finishAll(const std::string &errorMessage);
    void setCancelHandler(const CancelHandler &handler);
    size_t requestCount() const;
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

public:
    static const std::string RESOURCE_ID;

private:
    struct Entry {
        explicit Entry(navi::AsyncPipePtr responsePipe)
            : pipe(std::move(responsePipe))
        {
        }

        std::mutex mutex;
        const navi::AsyncPipePtr pipe;
        std::deque<std::shared_ptr<class AgentResponseToClientData>> pending;
        bool acknowledged = false;
        bool terminalPending = false;
        bool finished = false;
    };

    using EntryPtr = std::shared_ptr<Entry>;

    EntryPtr find(uint64_t requestId) const;
    bool deliverLocked(
        const EntryPtr &entry,
        const std::shared_ptr<class AgentResponseToClientData> &response,
        bool &pipeFailed);
    bool erase(uint64_t requestId, const EntryPtr &entry);
    void notifyCancelled(uint64_t requestId);

private:
    mutable std::mutex _mutex;
    std::unordered_map<uint64_t, EntryPtr> _entries;
    CancelHandler _cancelHandler;
    bool _stopped = false;
    std::atomic<uint64_t> _registeredCount{0};
    std::atomic<uint64_t> _publishedCount{0};
    std::atomic<uint64_t> _terminalCount{0};
    std::atomic<uint64_t> _cancelledCount{0};
    std::atomic<uint64_t> _rejectedCount{0};
};

NAVI_TYPEDEF_PTR(AgentResponseR);

} // namespace navi_sched
