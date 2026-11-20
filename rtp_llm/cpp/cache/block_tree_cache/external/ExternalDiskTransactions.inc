// This fragment is included in ExternalBlockTreeDriver. Host and Disk
// references, eviction and publication use the same metadata lock.

public:
    void configureDisk(size_t capacity, size_t ranks, bool trace) {
        const std::lock_guard lock(mutex_);
        if (disk_capacity_ != 0 || capacity == 0 || ranks == 0
            || capacity > static_cast<size_t>(std::numeric_limits<BlockIdxType>::max())
            || std::any_of(slots_.begin(), slots_.end(), [](const auto& entry) {
                   return std::get<1>(entry.first) == kDiskTier;
               })) {
            throw std::invalid_argument("invalid BTC Disk transaction configuration");
        }
        disk_capacity_ = capacity;
        disk_ranks_    = ranks;
        disk_trace_    = trace;
    }

    uint64_t pinDiskLoad(const DiskResidency* pages, size_t count) {
        const std::lock_guard   lock(mutex_);
        if (count == 0) {
            return 0;
        }
        std::set<DiskResidency>    seen;
        std::vector<DiskResidency> distinct;
        distinct.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto& page = pages[i];
            validate(page.group, kDiskTier, page.slot);
            if (disk_capacity_ == 0 || slots_.count({page.group, kDiskTier, page.slot}) == 0) {
                throw std::runtime_error("BTC Disk source is no longer published");
            }
            if (seen.insert(page).second) {
                distinct.push_back(page);
            }
        }

        const auto id = nextDiskTransactionLocked();
        std::vector<DiskResidency> pinned;
        pinned.reserve(distinct.size());
        try {
            for (const auto& page : distinct) {
                ++disk_load_refs_[page];
                pinned.push_back(page);
            }
            disk_reads_.emplace(id, pinned);
        } catch (...) {
            for (const auto& page : pinned) {
                dropDiskLoadRefLocked(page);
            }
            throw;
        }
        return id;
    }

    void completeDiskLoad(uint64_t lease, bool success) {
        const std::lock_guard lock(mutex_);
        if (lease == 0) {
            return;
        }
        const auto current = disk_reads_.find(lease);
        if (current == disk_reads_.end()) {
            return;  // Completion retries must not drop a different lease.
        }
        try {
            if (success) {
                for (const auto& page : current->second) {
                    auto* node = slots_.at({page.group, kDiskTier, page.slot});
                    recordDiskAccess(node, page.group);
                }
            }
        } catch (...) {
            for (const auto& page : current->second) {
                dropDiskLoadRefLocked(page);
            }
            disk_reads_.erase(current);
            throw;
        }
        for (const auto& page : current->second) {
            dropDiskLoadRefLocked(page);
        }
        disk_reads_.erase(current);
    }

    void retryDiskWrites(const DiskResidency* sources, size_t count, std::string_view request) {
        const std::lock_guard lock(mutex_);
        for (size_t i = 0; i < count; ++i) {
            const auto& source = sources[i];
            validate(source.group, kHostTier, source.slot);
            const auto  host = slots_.find({source.group, kHostTier, source.slot});
            if (host == slots_.end()) {
                continue;
            }
            const Projection projection{host->second->cache_key, source.group};
            const auto retry = disk_retry_.find(projection);
            if (retry == disk_retry_.end() || retry->second.host_slot != source.slot) {
                continue;
            }
            queueDiskWriteLocked(host->second, source.group, source.slot, retry->second.row, request, true);
        }
    }

    ExternalDiskJobs takeDiskJobs() {
        const std::lock_guard lock(mutex_);
        ExternalDiskJobs      result;
        if (disk_capacity_ == 0) {
            return result;
        }
        result.jobs.reserve(2);
        const size_t free = disk_capacity_ - disk_occupied_;
        if (free != 0 && !disk_pending_.empty()) {
            admitDiskWritesLocked(free, result);
        }

        const size_t demand = std::min(disk_pending_.size(), disk_capacity_);
        if (demand > disk_retiring_) {
            const auto incoming = disk_pending_.front();
            const size_t needed = demand - disk_retiring_;
            for (size_t i = 0; i < needed; ++i) {
                if (!reclaimDiskForWriteLocked(incoming, result)) {
                    break;
                }
            }
        }
        if (!disk_releases_.empty()) {
            const auto id = nextDiskTransactionLocked();
            result.jobs.push_back({id, disk_releases_, {}});
            disk_jobs_.emplace(id, DiskTransaction{{}, disk_releases_, disk_ranks_, false});
            disk_releases_.clear();
        }
        return result;
    }

    ExternalDiskSettlement settleDiskJob(uint64_t id, size_t acknowledgements, bool failure) {
        const std::lock_guard  lock(mutex_);
        ExternalDiskSettlement result;
        const auto entry = disk_jobs_.find(id);
        if (entry == disk_jobs_.end()) {
            throw std::invalid_argument("unknown BTC Disk job receipt");
        }
        auto& job = entry->second;
        if (acknowledgements == 0 || acknowledgements > job.pending) {
            throw std::invalid_argument("invalid BTC Disk job receipt count");
        }
        job.failed |= failure;
        if (acknowledgements < job.pending) {
            job.pending -= acknowledgements;
            return result;
        }

        if (!job.writes.empty()) {
            settleDiskWriteLocked(job, result);
        } else if (job.failed) {
            // A failed release cannot make physical capacity reusable.
            throw std::runtime_error("BTC Disk release was not confirmed on every rank");
        } else {
            disk_occupied_ -= job.disk_ids.size();
            disk_retiring_ -= job.disk_ids.size();
        }
        disk_jobs_.erase(entry);
        result.done = true;
        return result;
    }

    bool diskHasPendingWork() {
        const std::lock_guard lock(mutex_);
        return !disk_pending_.empty() || !disk_releases_.empty() || !disk_jobs_.empty();
    }

    bool diskResetBlocked() {
        const std::lock_guard lock(mutex_);
        return !disk_jobs_.empty() || !disk_reads_.empty();
    }

    void resetDisk() {
        const std::lock_guard lock(mutex_);
        resetDiskLocked();
    }

private:
    struct DiskOffer {
        Projection   projection;
        BlockIdxType host_slot;
        size_t       row;
        std::string  request;
        bool         retry;
    };

    struct DiskTransaction {
        std::vector<DiskOffer>    writes;
        std::vector<BlockIdxType> disk_ids;
        size_t                    pending;
        bool                      failed;
    };

    using PendingDiskOffers = std::list<DiskOffer>;

    uint64_t nextDiskTransactionLocked() {
        if (next_disk_transaction_ == std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error("BTC Disk transaction IDs exhausted");
        }
        return next_disk_transaction_++;
    }

    void dropDiskLoadRefLocked(const DiskResidency& page) {
        const auto ref = disk_load_refs_.find(page);
        if (--ref->second == 0) {
            disk_load_refs_.erase(ref);
        }
    }

    void afterHostStoreSuccessLocked(
        TreeNode* node, size_t group, BlockIdxType slot, size_t row, std::string_view request) {
        queueDiskWriteLocked(node, group, slot, row, request, false);
    }

    void queueDiskWriteLocked(
        TreeNode* node, size_t group, BlockIdxType slot, size_t row, std::string_view request, bool retry) {
        if (disk_capacity_ == 0) {
            return;
        }
        const Projection projection{node->cache_key, group};
        const auto& copies = copies_.at(projection);
        if (copies.slots[kDiskTier] != NULL_BLOCK_IDX || disk_pending_keys_.count(projection)
            || disk_writing_.count(projection)) {
            return;
        }
        const auto inserted = disk_pending_keys_.emplace(projection, disk_pending_.end()).first;
        try {
            inserted->second = disk_pending_.insert(disk_pending_.end(), {projection, slot, row, std::string(request), retry});
        } catch (...) {
            disk_pending_keys_.erase(inserted);
            throw;
        }
    }

    void beforeHostUnpublishDiskLocked(TreeNode* node, size_t group, BlockIdxType slot) {
        const Projection projection{node->cache_key, group};
        const auto pending = disk_pending_keys_.find(projection);
        if (pending != disk_pending_keys_.end() && pending->second->host_slot == slot) {
            disk_pending_.erase(pending->second);
            disk_pending_keys_.erase(pending);
        }
        const auto retry = disk_retry_.find(projection);
        if (retry != disk_retry_.end() && retry->second.host_slot == slot) {
            disk_retry_.erase(retry);
        }
    }

    void admitDiskWritesLocked(size_t free, ExternalDiskJobs& result) {
        const size_t count = std::min(free, disk_pending_.size());
        constexpr auto max_slot = static_cast<uint64_t>(std::numeric_limits<BlockIdxType>::max());
        if (next_disk_slot_ > max_slot || count - 1 > max_slot - next_disk_slot_) {
            throw std::overflow_error("BTC logical Disk slots exhausted");
        }
        const auto id = nextDiskTransactionLocked();
        ExternalDiskJob external{id, {}, {}};
        DiskTransaction transaction{{}, {}, disk_ranks_, false};
        external.disk_ids.reserve(count);
        external.host_ids.reserve(count);
        transaction.writes.reserve(count);
        auto offer = disk_pending_.begin();
        for (size_t index = 0; index < count; ++index, ++offer) {
            external.disk_ids.push_back(static_cast<BlockIdxType>(next_disk_slot_ + index));
            external.host_ids.push_back(offer->host_slot);
            transaction.writes.push_back(*offer);
        }
        transaction.disk_ids = external.disk_ids;
        const auto entry = disk_jobs_.emplace(id, std::move(transaction)).first;

        size_t pinned = 0;
        try {
            // Match the original batch touch order: the first Host row is newest.
            for (auto it = entry->second.writes.rbegin(); it != entry->second.writes.rend(); ++it) {
                const auto& write = *it;
                const auto host  = slots_.find({write.projection.second, kHostTier, write.host_slot});
                if (host == slots_.end() || host->second->cache_key != write.projection.first) {
                    throw std::logic_error("BTC Disk write source changed before Host pin");
                }
                const auto [where, inserted] = disk_writing_.insert(write.projection);
                if (!inserted) {
                    throw std::logic_error("BTC Disk write queued twice");
                }
                try {
                    pinHostSlotLocked(write.projection.second, write.host_slot);
                } catch (...) {
                    disk_writing_.erase(where);
                    throw;
                }
                ++pinned;
            }
        } catch (...) {
            for (size_t i = 0; i < pinned; ++i) {
                const auto& write = entry->second.writes[count - 1 - i];
                unpinHostSlotLocked(write.projection.second, write.host_slot);
                disk_writing_.erase(write.projection);
            }
            disk_jobs_.erase(entry);
            throw;
        }
        for (const auto& write : entry->second.writes) {
            disk_pending_keys_.erase(write.projection);
            disk_pending_.pop_front();
        }
        next_disk_slot_ += count;
        disk_occupied_ += count;
        result.jobs.push_back(std::move(external));
    }

    ExternalDiskPage publishedDiskPageLocked(DiskResidency page) const {
        size_t row = 0;
        for (auto* node = slots_.at({page.group, kDiskTier, page.slot}); node->parent && node->parent->parent;
             node = node->parent) {
            ++row;
        }
        return {page.group, row, page.slot, NULL_BLOCK_IDX, disk_load_refs_.count(page) != 0};
    }

    bool reclaimDiskForWriteLocked(const DiskOffer& incoming, ExternalDiskJobs& result) {
        const auto plan = selectDiskVictimLocked({}, disk_trace_);
        if (disk_trace_) {
            ExternalDiskReclaim event;
            event.request      = incoming.request;
            event.incoming     = {incoming.projection.second, incoming.row, incoming.host_slot, incoming.host_slot, false};
            event.slots_before = disk_visible_;
            event.slots_after  = disk_visible_ - (plan.victim.has_value() ? 1 : 0);
            for (const auto& page : plan.order) {
                event.order.push_back(publishedDiskPageLocked(page));
            }
            if (plan.victim) {
                event.victim = publishedDiskPageLocked(*plan.victim);
            }
            result.reclaims.push_back(std::move(event));
        }
        if (!plan.victim) {
            return false;
        }
        retirePublishedDiskLocked(*plan.victim);
        return true;
    }

    void retirePublishedDiskLocked(DiskResidency page) {
        disk_releases_.push_back(page.slot);
        eraseSlot(slots_.find({page.group, kDiskTier, page.slot}));
        ++disk_retiring_;
        --disk_visible_;
    }

    void settleDiskWriteLocked(const DiskTransaction& job, ExternalDiskSettlement& result) {
        if (job.failed) {
            disk_releases_.reserve(disk_releases_.size() + job.disk_ids.size());
            result.events.push_back({ExternalDiskEventKind::FAILED, {}, {}, job.disk_ids});
        }
        for (size_t index = 0; index < job.writes.size(); ++index) {
            const auto& write = job.writes[index];
            const auto  group = write.projection.second;
            const auto  disk = job.disk_ids[index];
            auto* node = slots_.at({group, kHostTier, write.host_slot});
            if (node->cache_key != write.projection.first) {
                throw std::logic_error("BTC Disk write lost its pinned Host source");
            }
            if (job.failed) {
                disk_retry_.insert_or_assign(write.projection, write);
                disk_releases_.push_back(disk);
                ++disk_retiring_;
            } else {
                auto& copies = copies_.at(write.projection);
                if (copies.slots[kDiskTier] != NULL_BLOCK_IDX) {
                    throw std::logic_error("BTC Disk write target is already published");
                }
                slots_.emplace(SlotKey{group, kDiskTier, disk}, node);
                copies.slots[kDiskTier] = disk;
                recordDiskAccess(node, group);
                refresh(node, group, copies);
                ++disk_visible_;
                disk_retry_.erase(write.projection);
                if (disk_trace_) {
                    const ExternalDiskPage page{group, write.row, disk, write.host_slot, false};
                    result.events.push_back({ExternalDiskEventKind::COMMIT, write.request, {page}, {disk}});
                    if (write.retry) {
                        result.events.push_back({ExternalDiskEventKind::RETRY, write.request, {page}, {disk}});
                    }
                }
            }
            disk_writing_.erase(write.projection);
            unpinHostSlotLocked(group, write.host_slot);
        }
    }

    void resetDiskLocked() {
        if (disk_capacity_ == 0) {
            return;
        }
        if (!disk_jobs_.empty() || !disk_reads_.empty()) {
            throw std::runtime_error("BTC Disk jobs must finish before reset");
        }
        disk_pending_.clear();
        disk_pending_keys_.clear();
        disk_retry_.clear();
        std::vector<DiskResidency> visible;
        visible.reserve(disk_visible_);
        for (const auto& [slot, node] : slots_) {
            const auto [group, tier, id] = slot;
            if (tier == kDiskTier) {
                visible.push_back({group, id});
            }
        }
        disk_releases_.reserve(disk_releases_.size() + visible.size());
        for (const auto& page : visible) {
            retirePublishedDiskLocked(page);
        }
    }

    size_t                                     disk_capacity_{0};
    size_t                                     disk_ranks_{0};
    size_t                                     disk_occupied_{0};
    size_t                                     disk_visible_{0};
    size_t                                     disk_retiring_{0};
    bool                                       disk_trace_{false};
    uint64_t                                   next_disk_slot_{1};
    uint64_t                                   next_disk_transaction_{1};
    PendingDiskOffers                          disk_pending_;
    std::map<Projection, PendingDiskOffers::iterator> disk_pending_keys_;
    std::set<Projection>                       disk_writing_;
    std::map<Projection, DiskOffer>            disk_retry_;
    std::map<DiskResidency, size_t>             disk_load_refs_;
    std::map<uint64_t, std::vector<DiskResidency>> disk_reads_;
    std::vector<BlockIdxType>                  disk_releases_;
    std::map<uint64_t, DiskTransaction>        disk_jobs_;
