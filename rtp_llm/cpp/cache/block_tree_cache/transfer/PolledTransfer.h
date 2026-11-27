#pragma once

#include "block_tree_cache.h"
#include "MultiRankBlockTransferEngine.h"

#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace rtp_llm {
class PolledRankBroadcast;

// Value snapshots for executors that pull work on the next scheduler tick.
// They contain host page identities, never native TreeNode or buffer pointers.
struct PolledPage {
    btc_position          position{};
    std::string           key;
    std::vector<btc_page> source;
    std::vector<btc_page> target;
    std::string           remote;
};
struct PolledWork {
    uint64_t                operation;
    btc_work_kind           kind;
    uint64_t                request;
    std::vector<uint32_t>   participants;
    std::vector<PolledPage> pages;
};
struct PolledOutcome {
    btc_result result;
    bool       safe;
    using Objects = std::map<std::pair<uint32_t, uint64_t>, std::string>;
    Objects objects;
    bool    committed = false;
};
struct PolledLoadLookup {
    btc_work_kind                kind;
    btc_position                 position;
    const std::vector<btc_page>* target;
};

class PolledTransfer {
public:
    using Routes = std::map<btc_work_kind, std::vector<uint32_t>>;
    using Done   = std::function<void(PolledOutcome)>;

    PolledTransfer(uint64_t instance, Routes routes);
    uint64_t                submit(btc_work_kind kind, uint64_t request, std::vector<PolledPage> pages, Done done);
    RankTransferRequest     prepareCopy(btc_work_kind                         kind,
                                        uint64_t                              request,
                                        std::vector<PolledPage>               pages,
                                        std::shared_ptr<TransferBufferSafety> proof,
                                        std::function<void(bool)>             on_safe);
    std::vector<PolledWork> poll();
    size_t                  size() const;
    std::vector<uint64_t>   findLoads(const std::vector<PolledLoadLookup>& loads) const;
    void                    accept(btc_operation operation, uint32_t actor);
    void                    complete(const btc_receipt* receipts, size_t count);
    void                    cancelUnexposed();
    std::vector<PolledPage> beginDisk(btc_operation id, btc_work_kind& kind);
    void                    endDisk(btc_operation id);
    bool                    supports(btc_work_kind kind) const {
        return routes_.count(kind) != 0;
    }
    uint64_t instance() const {
        return instance_;
    }

private:
    struct Participant {
        bool       accepted  = false;
        bool       safe      = false;
        btc_result result    = 0;
        bool       committed = false;
    };
    struct Pending {
        PolledWork                           work;
        std::map<uint32_t, Participant>      participants;
        Done                                 done;
        PolledOutcome::Objects               objects;
        bool                                 exposed         = false;
        bool                                 failed_reported = false;
        bool                                 disk_running    = false;
        std::shared_ptr<PolledRankBroadcast> ranks;
        std::function<void()> notification(bool all_safe);
    };

    uint64_t                    enqueue(btc_work_kind                        kind,
                                        uint64_t                             request,
                                        std::vector<PolledPage>              pages,
                                        Done                                 done,
                                        std::shared_ptr<PolledRankBroadcast> ranks);
    const uint64_t              instance_;
    const Routes                routes_;
    mutable std::mutex          mutex_;
    uint64_t                    next_ = 0;
    std::map<uint64_t, Pending> pending_;
};
}  // namespace rtp_llm
