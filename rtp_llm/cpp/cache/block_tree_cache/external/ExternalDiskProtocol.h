#pragma once

// ExternalBlockTreeDriver's coordinator protocol. The wire carries logical
// Disk IDs; each worker already maps those IDs into its own DiskBlockPool.
struct ExternalDiskPage {
    size_t       group{0};
    size_t       row{0};
    BlockIdxType slot{NULL_BLOCK_IDX};
    BlockIdxType host_slot{NULL_BLOCK_IDX};
    bool         active{false};
};

struct ExternalDiskJobView {
    uint64_t            id;
    const BlockIdxType* disk_ids;
    const BlockIdxType* host_ids;
    size_t              count;
};

struct ExternalDiskReclaimView {
    Key                     request;
    ExternalDiskPage        incoming;
    const ExternalDiskPage* order;
    size_t                  count;
    bool                    has_victim;
    ExternalDiskPage        victim;
    size_t                  slots_before;
    size_t                  slots_after;
};

struct ExternalDiskJobsView {
    const ExternalDiskJobView*     jobs;
    size_t                         job_count;
    const ExternalDiskReclaimView* reclaims;
    size_t                         reclaim_count;
};

struct ExternalDiskJob {
    uint64_t                  id;
    std::vector<BlockIdxType> disk_ids;
    std::vector<BlockIdxType> host_ids;

    ExternalDiskJobView view() const {
        return {id, disk_ids.data(), host_ids.empty() ? nullptr : host_ids.data(), disk_ids.size()};
    }
};

struct ExternalDiskReclaim {
    std::string                     request;
    ExternalDiskPage                incoming;
    std::vector<ExternalDiskPage>   order;
    std::optional<ExternalDiskPage> victim;
    size_t                          slots_before{0};
    size_t                          slots_after{0};

    ExternalDiskReclaimView view() const {
        return {{request.data(), request.size()},
                incoming,
                order.data(),
                order.size(),
                victim.has_value(),
                victim.value_or(ExternalDiskPage{}),
                slots_before,
                slots_after};
    }
};

struct ExternalDiskJobs {
    std::vector<ExternalDiskJob>     jobs;
    std::vector<ExternalDiskReclaim> reclaims;

    ExternalDiskJobsView view() {
        transformViews(jobs, job_views_);
        transformViews(reclaims, reclaim_views_);
        return {job_views_.data(), job_views_.size(), reclaim_views_.data(), reclaim_views_.size()};
    }

private:
    template<typename Record, typename View>
    static void transformViews(const std::vector<Record>& records, std::vector<View>& views) {
        views.reserve(records.size());
        for (const auto& item : records) {
            views.push_back(item.view());
        }
    }

    std::vector<ExternalDiskJobView>     job_views_;
    std::vector<ExternalDiskReclaimView> reclaim_views_;
};

enum class ExternalDiskEventKind : int { COMMIT, RETRY, FAILED };

struct ExternalDiskEventView {
    int                     kind;
    Key                     request;
    const ExternalDiskPage* pages;
    size_t                  page_count;
    const BlockIdxType*     disk_ids;
    size_t                  disk_id_count;
};

struct ExternalDiskEvent {
    ExternalDiskEventKind           kind;
    std::string                     request;
    std::vector<ExternalDiskPage>   pages;
    std::vector<BlockIdxType>       disk_ids;

    ExternalDiskEventView view() const {
        return {static_cast<int>(kind),
                {request.data(), request.size()},
                pages.data(),
                pages.size(),
                disk_ids.data(),
                disk_ids.size()};
    }
};

struct ExternalDiskSettlementView {
    bool                         done;
    const ExternalDiskEventView* events;
    size_t                       count;
};

struct ExternalDiskSettlement {
    bool                           done{false};
    std::vector<ExternalDiskEvent> events;

    ExternalDiskSettlementView view() {
        views_.reserve(events.size());
        for (const auto& event : events) {
            views_.push_back(event.view());
        }
        return {done, views_.data(), views_.size()};
    }

private:
    std::vector<ExternalDiskEventView> views_;
};
