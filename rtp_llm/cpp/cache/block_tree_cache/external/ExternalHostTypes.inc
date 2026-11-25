// Included in the external tree driver's implementation namespace.

// Host backing is rank-local. This pool owns the shared row numbers and refs.
class ExternalHostPool final: public IBlockPool {
public:
    explicit ExternalHostPool(size_t capacity): IBlockPool(makeConfig(capacity)) {
        markInitialized();
        reserveAllReleasedBlocksNoLock();
    }

    size_t blockSizeBytes() const override {
        return 0;
    }

private:
    static std::shared_ptr<const BlockPoolConfigBase> makeConfig(size_t capacity) {
        if (capacity == 0 || capacity >= static_cast<size_t>(std::numeric_limits<BlockIdxType>::max())) {
            throw std::invalid_argument("invalid BTC Host row capacity");
        }
        auto config = std::make_shared<BlockPoolConfigBase>();
        config->pool_type = BlockPoolType::HOST;
        config->pool_name = "external host";
        config->physical_block_count = capacity + 1;  // IBlockPool reserves row zero.
        return config;
    }
};

struct HostResidency {
    size_t       group;
    BlockIdxType slot;

    bool operator<(const HostResidency& other) const {
        return std::tie(group, slot) < std::tie(other.group, other.slot);
    }
};

struct HostStoreCandidate {
    size_t group;
    size_t row;
};

struct HostStoreReservation {
    size_t       candidate;
    BlockIdxType slot;
};

struct HostStoreVictim {
    size_t       group;
    Key          key;
    BlockIdxType slot;
};

struct HostStoreView {
    uint64_t                    ticket;
    const HostStoreReservation* reservations;
    size_t                      reservation_count;
    const HostStoreVictim*      evicted;
    size_t                      evicted_count;
};

struct HostStorePlan {
    uint64_t                          ticket{};
    std::vector<HostStoreReservation> reservations;
    std::vector<std::string>          victim_hashes;
    std::vector<HostStoreVictim>      evicted;

    HostStoreView view() const {
        return {ticket, reservations.data(), reservations.size(), evicted.data(), evicted.size()};
    }
};
