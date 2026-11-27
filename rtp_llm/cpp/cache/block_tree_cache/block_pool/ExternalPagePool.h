#pragma once

#include "block_tree_cache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"

#include <list>
#include <map>
#include <unordered_map>

namespace rtp_llm {

// The host allocator supplies the device buffer and one hold per imported page.
// Native groups use ordinary IBlockPool indices and tree references. Only the
// mapping to the host's physical identity survives until RELEASE is acknowledged.
class ExternalPagePool final: public IBlockPool {
public:
    ExternalPagePool();
    size_t blockSizeBytes() const override {
        return 0;
    }

    // Reserves one temporary LOAD ref. On rejection the host still owns its hold.
    std::optional<BlockIdxType> prepare(const btc_page& page, bool require_unused_slot);
    void                        finishPrepare(BlockIdxType block, bool adopted);
    btc_page                    describe(BlockIdxType block) const;
    std::vector<BlockIdxType>   resolve(const btc_page& page) const;
    struct PhysicalResolution {
        std::vector<BlockIdList> provided;
        std::vector<BlockIdList> companions;
        size_t                   returning{0};  // eligible physical slots waiting only for RELEASE
    };
    PhysicalResolution          resolvePhysical(const btc_page* pages, size_t count, uint64_t pool) const;
    void                        invalidate(const BlockIdList& blocks);
    bool                        canPublish(BlockIdxType block) const override;
    std::vector<btc_page>       takeReleased();
    void                        acknowledgeRelease(const btc_page& page);
    bool                        hasOutstandingHolds() const;
    void                        referenceRequestBlocks(const BlockIdList& blocks) override;
    void                        unreferenceRequestBlocks(const BlockIdList& blocks) override;

private:
    struct Imported {
        btc_page     page;
        BlockIdxType block;
        bool         adopted     = false;
        bool         invalidated = false;
    };
    using Records = std::list<Imported>;
    using Slot    = std::pair<uint64_t, uint64_t>;
    struct Occupants {
        uint64_t generation;
        size_t   holds;
    };

    bool                                                onLastTreeRefNoLock(BlockIdxType block) override;
    void                                                forgetNoLock(Records& records, Records::iterator record);
    Records                                             live_, released_, awaiting_ack_;
    std::map<Slot, Occupants>                           slots_;
    std::unordered_map<uint64_t, Records::iterator>     by_hold_;
    std::unordered_map<BlockIdxType, Records::iterator> by_block_;
};
}  // namespace rtp_llm
