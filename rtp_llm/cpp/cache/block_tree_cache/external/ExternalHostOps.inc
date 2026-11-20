// Included in ExternalBlockTreeDriver. All Host metadata is governed by mutex_.
    void configureHost(size_t capacity) {
        const std::lock_guard lock(mutex_);
        if (host_pool_) {
            if (host_pool_->totalBlocksNum() != capacity) {
                throw std::logic_error("cannot resize the BTC Host row pool");
            }
            return;
        }
        for (const auto& entry : slots_) {
            if (std::get<1>(entry.first) == kHostTier) {
                throw std::logic_error("configure BTC Host before publishing Host rows");
            }
        }
        std::vector<EvictionHeap> heaps;
        heaps.reserve(groups_);
        for (size_t group = 0; group < groups_; ++group) {
            heaps.emplace_back(EvictionPolicy::LRU);
        }
        host_pool_ = std::make_unique<ExternalHostPool>(capacity);
        host_heaps_ = std::move(heaps);
    }

    std::unique_ptr<HostStorePlan> prepareHostStore(const Key*                hashes,
                                                   size_t                    hash_count,
                                                   const HostStoreCandidate* candidates,
                                                   size_t                    candidate_count,
                                                   Key                       request_id) {
        const std::lock_guard lock(mutex_);
        requireHostPool();

        auto plan = std::make_unique<HostStorePlan>();
        HostStore transaction;
        std::set<std::pair<size_t, std::string_view>> offered;
        std::set<BlockIdxType> excluded;
        size_t last_row = 0;
        for (size_t i = 0; i < candidate_count; ++i) {
            const auto& candidate = candidates[i];
            checkPublication(candidate.group, candidate.row, kHostTier, 0, hash_count);
            const auto name = hashes[candidate.row].view();
            const auto slot = hostSlotLocked(candidate.group, name);
            if (slot != NULL_BLOCK_IDX) {
                excluded.insert(slot);
                continue;
            }
            if (!offered.emplace(candidate.group, name).second
                || pending_host_keys_.count({candidate.group, std::string(name)}) != 0) {
                continue;
            }
            transaction.pages.push_back({candidate.group, candidate.row, NULL_BLOCK_IDX, 0});
            plan->reservations.push_back({i, NULL_BLOCK_IDX});
            last_row = std::max(last_row, candidate.row);
        }
        const auto needed = transaction.pages.size();
        if (needed == 0) {
            return plan;
        }
        if (needed > host_pool_->availableBlocksNum()) {
            return nullptr;
        }

        const size_t fresh = std::min(needed, host_pool_->freeBlocksNum());
        std::vector<HostResidency> victims;
        victims.reserve(needed - fresh);
        while (fresh + victims.size() < needed) {
            const auto victim = chooseHostVictimLocked(excluded);
            if (!victim) {
                return nullptr;
            }
            excluded.insert(victim->slot);
            victims.push_back(*victim);
        }

        transaction.hashes.reserve(last_row + 1);
        for (size_t row = 0; row <= last_row; ++row) {
            transaction.hashes.emplace_back(hashes[row].view());
        }
        transaction.request_id = request_id.view();
        plan->victim_hashes.reserve(victims.size());
        plan->evicted.reserve(victims.size());
        BlockIdList victim_blocks;
        victim_blocks.reserve(victims.size());
        for (const auto& victim : victims) {
            const auto* node = slots_.at({victim.group, kHostTier, victim.slot});
            const auto& hash = plan->victim_hashes.emplace_back(key_names_.at(node->cache_key)->first);
            plan->evicted.push_back({victim.group, {hash.data(), hash.size()}, victim.slot});
            victim_blocks.push_back(victim.slot + 1);
        }

        plan->ticket = nextHostTicketLocked();
        auto& pending = host_stores_.emplace(plan->ticket, std::move(transaction)).first->second;
        std::optional<BlockIdList> allocated;
        bool victims_pinned = false;
        try {
            for (const auto& page : pending.pages) {
                pending_host_keys_.emplace(page.group, pending.hashes[page.row]);
            }
            allocated = host_pool_->malloc(fresh);
            if (!allocated) {
                throw std::logic_error("BTC Host allocator lost unreserved capacity");
            }
            host_pool_->incTreeRef(*allocated, BlockTreeRefType::EVICT);
            // The EVICT hold transfers a victim's slot to this transaction.
            // eraseSlot drops its old CACHE hold and publication under mutex_.
            host_pool_->incTreeRef(victim_blocks, BlockTreeRefType::EVICT);
            victims_pinned = true;
            for (const auto& victim : victims) {
                eraseSlot(slots_.find({victim.group, kHostTier, victim.slot}));
            }
            for (size_t i = 0; i < needed; ++i) {
                const auto block = i < fresh ? (*allocated)[i] : victim_blocks[i - fresh];
                pending.pages[i].slot = block - 1;
                plan->reservations[i].slot = block - 1;
            }
            // Earlier inputs were at the front of the previous Host LRU.
            // Assign their age now so out-of-order ACKs cannot reverse it.
            for (auto page = pending.pages.rbegin(); page != pending.pages.rend(); ++page) {
                page->admission_seq = ++host_clock_;
            }
        } catch (...) {
            if (victims_pinned) {
                host_pool_->decTreeRef(victim_blocks, BlockTreeRefType::EVICT);
            }
            if (allocated) {
                host_pool_->decTreeRef(*allocated, BlockTreeRefType::EVICT);
            }
            retireHostStoreLocked(plan->ticket);
            throw;
        }
        return plan;
    }

    bool settleHostStore(uint64_t ticket, bool success) {
        const std::lock_guard lock(mutex_);
        const auto it = host_stores_.find(ticket);
        if (it == host_stores_.end()) {
            return false;
        }
        auto& pending = it->second;
        if (!success) {
            releaseHostStoreBlocksLocked(pending);
            retireHostStoreLocked(ticket);
            return true;
        }

        std::vector<Key> hashes;
        std::vector<Binding> publications;
        BlockIdList blocks;
        hashes.reserve(pending.hashes.size());
        publications.reserve(pending.pages.size());
        blocks.reserve(pending.pages.size());
        for (const auto& hash : pending.hashes) {
            hashes.push_back({hash.data(), hash.size()});
        }
        for (const auto& page : pending.pages) {
            publications.push_back({page.group, page.row, kHostTier, page.slot});
            blocks.push_back(page.slot + 1);
        }

        host_pool_->incTreeRef(blocks, BlockTreeRefType::CACHE);
        try {
            publishLocked(hashes.data(), publications.data(), publications.size());
            for (const auto& page : pending.pages) {
                auto* node = slots_.at({page.group, kHostTier, page.slot});
                upsertHostRecencyLocked(node, page.group, page.admission_seq);
            }
        } catch (...) {
            for (const auto& page : pending.pages) {
                const auto slot = slots_.find({page.group, kHostTier, page.slot});
                if (slot == slots_.end()) {
                    host_pool_->decTreeRef(page.slot + 1, BlockTreeRefType::CACHE);
                } else {
                    eraseSlot(slot);
                }
            }
            releaseHostStoreBlocksLocked(pending);
            retireHostStoreLocked(ticket);
            throw;
        }

        releaseHostStoreBlocksLocked(pending);
        for (const auto& page : pending.pages) {
            pending_host_keys_.erase({page.group, pending.hashes[page.row]});
        }
        auto committed = std::move(pending);
        host_stores_.erase(it);
        for (const auto& page : committed.pages) {
            auto* node = slots_.at({page.group, kHostTier, page.slot});
            afterHostStoreSuccessLocked(node, page.group, page.slot, page.row, committed.request_id);
        }
        return true;
    }

    uint64_t pinHostLoad(const HostResidency* pages, size_t count) {
        const std::lock_guard lock(mutex_);
        if (count == 0) {
            return 0;
        }
        requireHostPool();
        std::set<HostResidency> seen;
        std::vector<HostResidency> unique;
        BlockIdList blocks;
        for (size_t i = 0; i < count; ++i) {
            if (seen.insert(pages[i]).second) {
                checkHostSlotLocked(pages[i].group, pages[i].slot);
                unique.push_back(pages[i]);
                blocks.push_back(pages[i].slot + 1);
            }
        }
        const auto ticket = nextHostTicketLocked();
        const auto lease = host_loads_.emplace(ticket, std::move(blocks)).first;
        host_pool_->incTreeRef(lease->second, BlockTreeRefType::LOAD);
        try {
            for (auto page = unique.rbegin(); page != unique.rend(); ++page) {
                recordHostAccessLocked(slots_.at({page->group, kHostTier, page->slot}), page->group);
            }
        } catch (...) {
            host_pool_->decTreeRef(lease->second, BlockTreeRefType::LOAD);
            host_loads_.erase(lease);
            throw;
        }
        return ticket;
    }

    bool releaseHostLoad(uint64_t ticket) {
        const std::lock_guard lock(mutex_);
        if (ticket == 0) {
            return true;
        }
        const auto lease = host_loads_.find(ticket);
        if (lease == host_loads_.end()) {
            return false;
        }
        host_pool_->decTreeRef(lease->second, BlockTreeRefType::LOAD);
        host_loads_.erase(lease);
        return true;
    }

    BlockIdxType hostSlot(size_t group, Key hash) {
        const std::lock_guard lock(mutex_);
        if (group >= groups_) {
            throw std::invalid_argument("invalid BTC Host group");
        }
        return hostSlotLocked(group, hash.view());
    }

    void resetHost() {
        const std::lock_guard lock(mutex_);
        resetHostLocked();
    }

private:
    struct HostStore {
        struct Page {
            size_t       group;
            size_t       row;
            BlockIdxType slot;
            uint64_t     admission_seq;
        };

        std::vector<Page>        pages;
        std::vector<std::string> hashes;
        std::string              request_id;
    };

    void requireHostPool() const {
        if (!host_pool_) {
            throw std::logic_error("BTC Host row pool is not configured");
        }
    }

    uint64_t nextHostTicketLocked() {
        if (next_host_ticket_ == std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error("BTC Host ticket space exhausted");
        }
        return ++next_host_ticket_;
    }

    BlockIdxType hostSlotLocked(size_t group, std::string_view name) const {
        const auto key = key_ids_.find(name);
        if (key == key_ids_.end()) {
            return NULL_BLOCK_IDX;
        }
        const auto copy = copies_.find({key->second, group});
        return copy == copies_.end() ? NULL_BLOCK_IDX : copy->second.slots[kHostTier];
    }

    TreeNode* checkHostSlotLocked(size_t group, BlockIdxType slot) {
        requireHostPool();
        validate(group, kHostTier, slot);
        const auto current = slots_.find({group, kHostTier, slot});
        if (current == slots_.end()) {
            throw std::logic_error("BTC Host load requires a published row");
        }
        return current->second;
    }

    void pinHostSlotLocked(size_t group, BlockIdxType slot) {
        auto* node = checkHostSlotLocked(group, slot);
        recordHostAccessLocked(node, group);
        host_pool_->incTreeRef(slot + 1, BlockTreeRefType::LOAD);
    }

    void unpinHostSlotLocked(size_t group, BlockIdxType slot) {
        checkHostSlotLocked(group, slot);
        host_pool_->decTreeRef(slot + 1, BlockTreeRefType::LOAD);
    }

    void recordHostAccessLocked(TreeNode* node, size_t group) {
        upsertHostRecencyLocked(node, group, ++host_clock_);
    }

    void upsertHostRecencyLocked(TreeNode* node, size_t group, uint64_t sequence) {
        CandidateMeta meta{};
        meta.admission_seq = meta.last_access_seq = sequence;
        host_heaps_[group].upsert(node, meta);
    }

    std::optional<HostResidency> chooseHostVictimLocked(const std::set<BlockIdxType>& excluded) const {
        std::optional<std::pair<EvictionEntry, size_t>> best;
        const auto order = [](const EvictionEntry& entry, size_t group) {
            return std::make_tuple(entry.primary_key, entry.secondary_key, entry.cache_key, group);
        };
        for (size_t group = 0; group < groups_; ++group) {
            const auto candidate = host_heaps_[group].best([&](TreeNode* node) {
                const auto slot = copies_.at({node->cache_key, group}).slots[kHostTier];
                return excluded.count(slot) == 0 && host_pool_->canReclaimCacheBlock(slot + 1);
            });
            if (candidate && (!best || order(*candidate, group) < order(best->first, best->second))) {
                best.emplace(*candidate, group);
            }
        }
        if (!best) {
            return std::nullopt;
        }
        const auto& [entry, group] = *best;
        return HostResidency{group, copies_.at({entry.node->cache_key, group}).slots[kHostTier]};
    }

    void releaseHostStoreBlocksLocked(const HostStore& pending) {
        for (const auto& page : pending.pages) {
            host_pool_->decTreeRef(page.slot + 1, BlockTreeRefType::EVICT);
        }
    }

    void retireHostStoreLocked(uint64_t ticket) {
        const auto it = host_stores_.find(ticket);
        for (const auto& page : it->second.pages) {
            pending_host_keys_.erase({page.group, it->second.hashes[page.row]});
        }
        host_stores_.erase(it);
    }

    void resetHostLocked() {
        if (!host_pool_) {
            return;
        }
        if (!host_stores_.empty() || !host_loads_.empty()
            || host_pool_->referencedBlocksNum(BlockTreeRefType::LOAD) != 0) {
            throw std::logic_error("BTC Host reset requires every transfer to settle");
        }
        for (auto it = slots_.begin(); it != slots_.end();) {
            if (std::get<1>(it->first) == kHostTier) {
                eraseSlot(it++);
            } else {
                ++it;
            }
        }
    }

    std::unique_ptr<ExternalHostPool>        host_pool_;
    std::vector<EvictionHeap>                host_heaps_;
    std::map<uint64_t, HostStore>            host_stores_;
    std::map<uint64_t, BlockIdList>          host_loads_;
    std::set<std::pair<size_t, std::string>> pending_host_keys_;
    uint64_t                               host_clock_{0};
    uint64_t                               next_host_ticket_{0};

public:
