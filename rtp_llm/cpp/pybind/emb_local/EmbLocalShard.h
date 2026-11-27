#pragma once

#include "rtp_llm/cpp/pybind/emb_local/EmbEvictionWriter.h"

#include <limits>

namespace rtp_llm::emb_local {

class Shard {
public:
    using TensorList = std::vector<at::Tensor>;
    using Streams    = std::optional<c10::cuda::CUDAStream>;
    struct Eviction {
        std::string                 key;
        std::string                 remote_key;
        uint64_t                    generation;
        Tier                        tier;
        Audit                       audit;
        std::shared_ptr<const Tree> tree;
        TensorList                  tensors;
    };

    struct Mutations {
        int64_t  evicted         = 0;
        int64_t  demoted         = 0;
        uint64_t cleared_through = 0;
        int64_t  snapshot_bytes  = 0;
        std::vector<Eviction> snapshots;
    };

    Shard(int64_t gpu_bytes, int64_t cpu_bytes, int device, bool notify_cpu = false, bool notify_gpu = false,
          int64_t snapshot_limit = 0, std::shared_ptr<EvictionWriter> writer = {}):
        device_(device), limits_{cpu_bytes, gpu_bytes}, notify_{notify_cpu, notify_gpu}, snapshot_limit_(snapshot_limit),
        writer_(std::move(writer)) {
        TORCH_CHECK(gpu_bytes >= 0 && cpu_bytes >= 0 && snapshot_limit >= 0, "invalid local EMB capacity");
        arenas_[tierIndex(Tier::Cpu)] = makeArena(Tier::Cpu, cpu_bytes);
        arenas_[tierIndex(Tier::Gpu)] = makeArena(Tier::Gpu, gpu_bytes);
    }

    Index::Lookup claim(const std::string& key) {
        return index_.claim(key);
    }
    Index::Lookup lookup(const std::string& key, bool touch) {
        return index_.lookup(key, touch);
    }

    template<typename Action>
    bool withCurrent(const std::string& key, const Cell& cell, Action action) {
        return index_.withCurrent(key, cell, std::move(action));
    }

    bool erase(const std::string& key, uint64_t generation) {
        return static_cast<bool>(index_.remove(key, generation));
    }

    void clear() {
        index_.clear(false);
    }

    auto residency(int64_t limit) const {
        return index_.residency(limit);
    }
    Index::Charge charge() const {
        return index_.charge();
    }

    int64_t droppedSnapshots() const {
        return dropped_snapshots_.load(std::memory_order_relaxed);
    }

    void close() {
        // Invalidate any work copying outside the index lock before waiting
        // for a publication already in flight.
        index_.clear(true);
        std::array<std::shared_ptr<Arena>, 2> closed_arenas;
        {
            std::lock_guard transfer(transfer_);
            closed_ = true;
            std::lock_guard pool_lock(pools_);
            for (size_t pos = 0; pos < arenas_.size(); ++pos) {
                if (arenas_[pos]) {
                    arenas_[pos]->stop();
                    old_arenas_[pos].emplace_back(arenas_[pos]);
                }
            }
            closed_arenas.swap(arenas_);
        }
    }

    Mutations resize(int64_t gpu_bytes, int64_t cpu_bytes, Streams stream) {
        TORCH_CHECK(gpu_bytes >= 0 && cpu_bytes >= 0, "invalid local EMB capacity");
        std::lock_guard transfer(transfer_);
        if (closed_) {
            return {};
        }
        const Bytes next_limits{cpu_bytes, gpu_bytes};
        auto        replacement = arenas_;
        bool        rebuilt     = false;
        for (auto tier : {Tier::Cpu, Tier::Gpu}) {
            const auto pos = tierIndex(tier);
            if (next_limits[pos] > (arenas_[pos] ? arenas_[pos]->stats().capacity : 0)) {
                replacement[pos] = makeArena(tier, next_limits[pos]);
                rebuilt          = true;
            }
        }
        Mutations evicted;
        limits_ = next_limits;
        if (rebuilt) {
            const auto [retired_through, retired_count] = index_.clear(false);
            evicted.cleared_through                     = retired_through;
            evicted.evicted                             = retired_count;
            std::lock_guard pool_lock(pools_);
            for (size_t pos = 0; pos < arenas_.size(); ++pos) {
                if (replacement[pos] != arenas_[pos]) {
                    if (arenas_[pos]) {
                        arenas_[pos]->stop();
                        old_arenas_[pos].emplace_back(arenas_[pos]);
                    }
                    arenas_[pos] = std::move(replacement[pos]);
                }
            }
        } else {
            for (auto tier : {Tier::Gpu, Tier::Cpu}) {
                const auto pos = tierIndex(tier);
                while (index_.charge().bytes[pos] > limits_[pos]) {
                    if (!evict(tier, nullptr, nullptr, stream, evicted)) {
                        break;
                    }
                }
            }
        }
        return evicted;
    }

    std::pair<bool, Mutations> publish(const std::string&                         key,
                                       uint64_t                                   generation,
                                       const TensorList&                          tensors,
                                       std::shared_ptr<const Tree>                tree,
                                       int64_t                                    tokens,
                                       const std::vector<std::shared_ptr<Ready>>& ready,
                                       Streams                                    stream) {
        // Producer waiting does not own any cache capacity or cache mutex.
        for (const auto& event : ready) {
            event->wait();
        }
        auto layout = layoutFor(tensors);
        setOffsets(layout);
        std::lock_guard transfer(transfer_);
        const auto      existing = index_.lookup(key, false);
        if (existing.generation != generation || generation == 0 || existing.resident) {
            return {};
        }
        if (layoutBytes(layout)[tierIndex(Tier::Gpu)] > limits_[tierIndex(Tier::Gpu)]) {
            layout = placeOnCpu(std::move(layout));
        }
        Mutations  evicted;
        const bool published =
            install(key, generation, {}, tensors, std::move(layout), tokens, stream, evicted, {}, std::move(tree));
        return {published, std::move(evicted)};
    }

    TensorList read(const std::shared_ptr<const Resident>& record, Streams stream) {
        TORCH_CHECK(record, "local EMB read requires a resident record");
        return copyOut(record, stream);
    }

    std::pair<bool, Mutations> tryPromote(const std::string&                     key,
                                          const std::shared_ptr<const Resident>& expected,
                                          const TensorList&                      restored,
                                          Streams                                stream) {
        // The caller already owns its copy. Promotion can be skipped while
        // another producer or promotion holds the transfer mutex.
        std::unique_lock transfer(transfer_, std::try_to_lock);
        if (!transfer || !expected || expected->tier != Tier::Cpu || index_.lookup(key, false).resident != expected) {
            return {};
        }
        auto layout = layoutFor(restored);
        setOffsets(layout);
        if (layoutBytes(layout)[tierIndex(Tier::Gpu)] == 0) {
            return {};
        }
        Mutations  evicted;
        const bool published = install(
            key, expected->generation, expected, restored, std::move(layout), expected->tokens, stream, evicted);
        return {published, std::move(evicted)};
    }

    Arena::Stats stats(Tier tier) {
        std::lock_guard pool_lock(pools_);
        const auto      pos     = tierIndex(tier);
        const auto&     arena   = arenas_[pos];
        auto            result  = arena ? arena->stats() : Arena::Stats{};
        auto&           retired = old_arenas_[pos];
        retired.erase(std::remove_if(retired.begin(),
                                     retired.end(),
                                     [&](const auto& owner) {
                                         const auto previous = owner.lock();
                                         if (!previous) {
                                             return true;
                                         }
                                         const auto usage = previous->stats();
                                         result.retired += usage.retired;
                                         result.quarantined += usage.quarantined;
                                         return false;
                                     }),
                      retired.end());
        return result;
    }

private:
    using Layout = std::vector<Slice>;
    using Bytes  = std::array<int64_t, 2>;

    std::shared_ptr<Arena> makeArena(Tier tier, int64_t capacity) const {
        if (capacity == 0) {
            return {};
        }
        if (tier == Tier::Gpu) {
            return std::make_shared<Arena>(capacity, at::Device(at::kCUDA, device_), false);
        }
        return std::make_shared<Arena>(capacity, at::Device(at::kCPU), device_ >= 0);
    }

    Layout layoutFor(const TensorList& tensors) const {
        Layout result;
        result.reserve(tensors.size());
        for (const auto& tensor : tensors) {
            TORCH_CHECK(tensor.layout() == at::kStrided && !tensor.is_quantized()
                            && (tensor.device().is_cpu() || (tensor.is_cuda() && tensor.get_device() == device_)),
                        "unsupported local EMB tensor");
            result.push_back({0,
                              static_cast<int64_t>(tensor.nbytes()),
                              tensor.scalar_type(),
                              tensor.sizes().vec(),
                              tensor.device(),
                              tensor.is_cuda() ? Tier::Gpu : Tier::Cpu});
        }
        return result;
    }

    static Layout placeOnCpu(Layout layout) {
        for (auto& slice : layout) {
            slice.placement = Tier::Cpu;
        }
        return layout;
    }

    static Bytes layoutBytes(const Layout& layout) {
        Bytes bytes{};
        for (const auto& slice : layout) {
            auto& end = bytes[tierIndex(slice.placement)];
            end       = std::max(end, slice.offset + slice.nbytes);
        }
        return bytes;
    }

    static void setOffsets(Layout& layout) {
        Bytes bytes{};
        for (auto& slice : layout) {
            auto&      end       = bytes[tierIndex(slice.placement)];
            const auto alignment = static_cast<int64_t>(c10::elementSize(slice.type));
            const auto padding   = (alignment - end % alignment) % alignment;
            TORCH_CHECK(slice.nbytes >= 0 && end <= std::numeric_limits<int64_t>::max() - padding - slice.nbytes,
                        "local EMB layout size overflow");
            slice.offset = end + padding;
            end          = slice.offset + slice.nbytes;
        }
    }

    // Mutation is serialized by transfer_. Index locks protect only lookup,
    // LRU and publish/erase. A request can copy out while a writer is blocked
    // on a CUDA event or evicting a different key.
    bool install(const std::string&                     key,
                 uint64_t                               generation,
                 const std::shared_ptr<const Resident>& expected,
                 const TensorList&                      tensors,
                 Layout                                 layout,
                 int64_t                                tokens,
                 Streams                                stream,
                 Mutations&                             evicted,
                 const std::string*                     protected_key = nullptr,
                 std::shared_ptr<const Tree>            tree          = {}) {
        setOffsets(layout);
        const auto bytes = layoutBytes(layout);
        if (bytes == Bytes{} || bytes[0] > limits_[0] || bytes[1] > limits_[1]) {
            return false;
        }
        Bytes alignment{8, 8};
        for (const auto& slice : layout) {
            auto& tier_alignment = alignment[tierIndex(slice.placement)];
            tier_alignment       = std::max(tier_alignment, static_cast<int64_t>(c10::elementSize(slice.type)));
        }
        auto next = std::make_shared<Resident>(Resident{generation,
                                                        bytes[1] ? Tier::Gpu : Tier::Cpu,
                                                        {},
                                                        bytes,
                                                        tokens,
                                                        std::move(layout),
                                                        expected ? expected->tree : std::move(tree)});
        // Ensure GPU victims are demoted before reserving CPU capacity for
        // the incoming mixed object.
        for (auto tier : {Tier::Gpu, Tier::Cpu}) {
            const auto pos = tierIndex(tier);
            if (bytes[pos] == 0) {
                continue;
            }
            const auto replace_bytes = expected ? expected->bytes[pos] : 0;
            for (;;) {
                const auto resident_bytes = index_.charge().bytes[pos];
                if (resident_bytes - replace_bytes <= limits_[pos] - bytes[pos]) {
                    next->storage[pos] = arenas_[pos]->reserve(bytes[pos], alignment[pos]);
                    if (next->storage[pos]) {
                        break;
                    }
                }
                // Retired blocks can still be owned by a previous reader.
                if (!arenas_[pos]->canReserveAfterEviction(bytes[pos], alignment[pos])
                    || !evict(tier, &key, protected_key, stream, evicted)) {
                    return false;
                }
            }
        }
        copyAndWait(tensors, views(*next), stream, expected ? expected->storage : Blocks{}, next->storage);
        return index_.publish(key, next, expected);
    }

    bool
    evict(Tier tier, const std::string* key, const std::string* protected_key, Streams stream, Mutations& evicted) {
        auto victim = index_.oldest(tier, key, protected_key, false);
        if (!victim && tier == Tier::Cpu) {
            victim = index_.oldest(Tier::Gpu, key, protected_key, true);
        }
        if (!victim) {
            return false;
        }
        auto& [victim_key, record] = *victim;
        // Only GPU pressure attempts to move a GPU entry to CPU. CPU
        // pressure evicts a mixed GPU entry after exhausting the CPU LRU.
        if (tier == Tier::Gpu && limits_[tierIndex(Tier::Cpu)] != 0) {
            try {
                auto cold   = placeOnCpu(record->slices);
                auto source = views(*record);
                if (install(victim_key,
                            record->generation,
                            record,
                            source,
                            std::move(cold),
                            record->tokens,
                            stream,
                            evicted,
                            key)) {
                    ++evicted.demoted;
                    return true;
                }
            } catch (const c10::Error&) {
                // Demotion is optional. The GPU victim can still be detached.
            }
        }
        if (auto removed = index_.remove(victim_key, record->generation, record.get())) {
            ++evicted.evicted;
            if (writer_ || notify_[tierIndex(record->tier)]) {
                // The index removal selected this exact generation. Only detached
                // tensors may outlive evict(): retaining its cell pins the slab
                // interval that the incoming publication is about to reserve.
                try {
                    auto meta = removed.cell->evictedMetadata();
                    if (meta && writer_) {
                        writer_->submit(meta->first, meta->second, record, stream);
                    } else if (meta) {
                        const auto room = snapshot_limit_ - evicted.snapshot_bytes;
                        if (record->bytes[0] > room || record->bytes[1] > room - record->bytes[0]) {
                            ++dropped_snapshots_;
                        } else {
                            evicted.snapshots.push_back({victim_key, std::move(meta->first), record->generation,
                                                         record->tier, std::move(meta->second), record->tree,
                                                         copyOut(record, stream)});
                            evicted.snapshot_bytes += record->bytes[0] + record->bytes[1];
                        }
                    }
                } catch (const std::exception&) {
                    // Opportunistic remote caching must not fail a local mutation.
                    ++dropped_snapshots_;
                }
            }
        }
        return true;
    }

    const int                                        device_;
    Bytes                                            limits_;
    const std::array<bool, 2>                        notify_;
    const int64_t                                    snapshot_limit_;
    const std::shared_ptr<EvictionWriter>              writer_;
    std::atomic<int64_t>                             dropped_snapshots_{0};
    std::mutex                                       transfer_;
    std::mutex                                       pools_;
    bool                                             closed_ = false;
    std::array<std::shared_ptr<Arena>, 2>            arenas_;
    std::array<std::vector<std::weak_ptr<Arena>>, 2> old_arenas_;
    Index                                            index_;
};

}  // namespace rtp_llm::emb_local
