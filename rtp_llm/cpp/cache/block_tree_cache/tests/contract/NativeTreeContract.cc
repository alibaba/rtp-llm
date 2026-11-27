#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>

namespace {
using namespace rtp_llm;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::abort();
    }
}

class MemoryPool final: public IBlockPool {
public:
    explicit MemoryPool(size_t slots, bool growable = false): IBlockPool(config(slots, growable)), words_(slots + 1) {
        markInitialized();
    }
    size_t blockSizeBytes() const override { return sizeof(uint64_t); }
    uint64_t& word(BlockIdxType block) { return words_.at(block); }

private:
    static std::shared_ptr<BlockPoolConfigBase> config(size_t slots, bool growable) {
        auto result = std::make_shared<BlockPoolConfigBase>();
        result->physical_block_count = slots + 1;
        result->growable = growable;
        return result;
    }
    std::vector<uint64_t> words_;
};

template<typename Group>
std::shared_ptr<Group> makeGroup(size_t id, GroupSetGeometry geometry, GroupSetPools pools) {
    auto group = std::make_shared<Group>(std::move(pools));
    group->initialize(id, geometry, {"group-" + std::to_string(id)});
    return group;
}

using Rows = std::vector<std::vector<GroupSetResource>>;

std::shared_ptr<HostBlockPool> hostPool(uint64_t& bytes) {
    auto config = std::make_shared<HostBlockPoolConfig>();
    config->pool_type = BlockPoolType::HOST;
    config->pool_name = "native Host test";
    config->physical_block_count = 2;
    config->payload_bytes = config->stride_bytes = sizeof(bytes);
    config->alignment = alignof(uint64_t);
    auto pool = std::make_shared<HostBlockPool>(std::move(config));
    check(pool->initBacking(&bytes, sizeof(bytes), 0), "initialize native Host backing");
    return pool;
}

void opaque_keys_and_native_group_windows() {
    const auto device = std::make_shared<MemoryPool>(12);
    const auto full = makeGroup<FullGroupSet>(0, {CacheGroupType::FULL, 8, 0}, {{device}, {}, {}});
    const auto linear = makeGroup<LinearGroupSet>(1, {CacheGroupType::LINEAR, 16, 0}, {{device}, {}, {}});
    const auto sliding = makeGroup<SWAGroupSet>(2, {CacheGroupType::SWA, 8, 17}, {{device}, {}, {}});
    BlockTree tree({full, linear, sliding}, 8);

    const auto slots = device->malloc(9).value();
    Rows rows(6, std::vector<GroupSetResource>(3));
    for (size_t row = 0; row < rows.size(); ++row) {
        rows[row][0].device_blocks = {slots[row]};
    }
    // Two groups share both one pool and an actual physical allocation.
    rows[5][1].device_blocks = {slots[5]};
    for (size_t row = 3; row < rows.size(); ++row) {
        rows[row][2].device_blocks = {slots[row + 3]};
    }
    const std::string low_bits("x\0bcdefg", 8);
    const auto first = low_bits + "older";
    const auto other = low_bits + "other";
    BlockTreeKeys path{BlockTreeKey(first), BlockTreeKey("b"), BlockTreeKey("c"), BlockTreeKey("d"),
                       BlockTreeKey("e"), BlockTreeKey("f")};
    tree.insertNode(path, rows, false, false);
    const auto nodes = tree.findNode(path);
    check(nodes.size() == 6 && nodes.front()->key().bytes() == first, "full binary key survives publication");
    check(tree.findNode(BlockTreeKeys{BlockTreeKey(other)}).empty(), "same low bits do not alias an existing key");
    check(tree.cacheKeys().empty(), "opaque node keys cannot appear as legacy integer IDs");
    check(device->treeRefCount(slots[5]) == 2, "shared physical allocation has two logical references");
    check(tree.previousGroupPage(nodes[5], 1) == nodes[3], "16-token group advances two 8-token key rows");
    check(!tree.isLeafAtTier(nodes[4], 0, Tier::DEVICE), "FULL parent with a Device child is not a leaf");

    auto full_validator = full->createMatchValidator();
    auto linear_validator = linear->createMatchValidator();
    auto sliding_validator = sliding->createMatchValidator();
    check(full->computeReuseBlockCount(6) == 6 && linear->computeReuseBlockCount(3) == 1
              && sliding->computeReuseBlockCount(6) == 3,
          "native group rules calculate complete pages using their own geometry");
    for (size_t row = 0; row < nodes.size(); ++row) {
        check(full_validator->validate(nodes[row]->group_set_resources[0]), "FULL prefix remains contiguous");
        if ((row + 1) % 2 == 0) {
            check(linear_validator->validate(nodes[row]->group_set_resources[1]) == (row == 5),
                  "LINEAR accepts a later checkpoint when earlier positions are missing");
        }
        check(sliding_validator->validate(nodes[row]->group_set_resources[2]) == (row == 5),
              "SWA reconnects only after every page in its rounded-up window is present");
    }
}

void retained_copy_has_independent_physical_lifetime() {
    const auto device = std::make_shared<MemoryPool>(1);
    uint64_t host_bytes = 0;
    const auto host = hostPool(host_bytes);
    const auto group = makeGroup<FullGroupSet>(0, {CacheGroupType::FULL, 1, 0}, {{device}, host, {}});
    check(group->hostPool() == host && group->pools().host == host,
          "native GroupSet exposes the same mapped Host allocator");
    const auto device_slot = device->malloc().value();
    const auto host_slot = host->malloc().value();
    const auto device_page = device->pageIdentity(device_slot).value();
    check(host_slot == device_slot && !host->isCurrentPage(device_page), "equal slots in distinct pools don't alias");
    constexpr uint64_t value = 0x6a5c90f25ULL;
    device->word(device_slot) = value;
    *static_cast<uint64_t*>(host->blockBuffer(host_slot).addr) = device->word(device_slot);

    {
        BlockTree tree({group});
        Rows rows(1, std::vector<GroupSetResource>(1));
        rows[0][0].device_blocks = {device_slot};
        const auto original = tree.insertNode({177}, rows, true, false).path.front();
        check(tree.retainHostReplica(original, 0, host_slot), "confirmed Host copy is recorded with Device data");
        check(tree.insertNode({177}, rows, false, false).accepted_resource_count == 0,
              "duplicate Device publication does not adopt another resource");
        auto& resource = tree.findNode({177}).front()->group_set_resources[0];
        check(resource.getTopTier() == Tier::DEVICE && resource.host_block == host_slot,
              "ordinary duplicate publication preserves the explicit Host copy");
        resource.evictFromTier(Tier::DEVICE);
        group->unreferenceBlocks({0, Tier::DEVICE, {{original, {device_slot}}}}, BlockTreeRefType::CACHE);
        check(!device->isCurrentPage(device_page), "eviction releases the old Device identity");
        const auto overwritten = device->malloc().value();
        check(overwritten == device_slot && !device->isCurrentPage(device_page), "allocation generation protects slot reuse");
        device->word(overwritten) = std::numeric_limits<uint64_t>::max();
        check(tree.findNode({177}).front() == original && resource.getTopTier() == Tier::HOST
                  && *static_cast<uint64_t*>(host->blockBuffer(resource.host_block).addr) == value,
              "same native node serves the intact Host copy after overwriting Device bytes");
    }
    check(host->freeBlocksNum() == 1, "tree destruction releases only the retained Host ownership");
}

void final_frozen_reference_retires_the_native_allocation() {
    MemoryPool device(1);
    bool refused_device = false;
    try {
        device.setBlockRetirementCallback([](BlockPoolPage) noexcept {});
    } catch (const std::invalid_argument&) {
        refused_device = true;
    }
    check(refused_device, "Device pool retains its original release semantics");

    uint64_t host_bytes = 0;
    const auto host = hostPool(host_bytes);
    std::optional<BlockPoolPage> retired;
    host->setBlockRetirementCallback([&](BlockPoolPage page) noexcept { retired = page; });
    size_t notified = 0;
    host->setCapacityChangeCallback([&] {
        check(host->freeBlocksNum() == 1, "capacity callback runs after unlocking the pool");
        ++notified;
    });
    const auto block = host->malloc().value();
    const auto identity = host->pageIdentity(block).value();
    bool rejected = false;
    try {
        host->releaseRetiredBlock(identity);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected && !retired && !host->malloc(), "fresh allocation without references is not retired");
    host->incTreeRef(block, BlockTreeRefType::LOAD);
    const auto release_frozen = host->prepareFrozenReferenceRelease({block});
    host->freezeBlock(block);
    check(host->decTreeRef(block, BlockTreeRefType::LOAD) == 0 && !retired,
          "ordinary release leaves the frozen allocation pinned");
    release_frozen();
    check(retired && retired->pool == host.get() && retired->slot == block
              && retired->generation == identity.generation && !notified && !host->malloc(),
          "final frozen release hands the same unavailable generation to its owner");
    check(host->isAllocated(block) && host->isCurrentPage(identity) && host->treeRefCount(block) == 0
              && !host->canReclaimCacheBlock(block),
          "retired slot stays visible for acknowledgment without becoming reusable");
    host->releaseRetiredBlock(*retired);
    check(notified == 1 && host->malloc().value() == block && !host->isCurrentPage(identity),
          "acknowledgment returns the slot for its next native generation");
}

void growable_native_pool_preserves_live_slots() {
    MemoryPool pool(0, true);
    check(!pool.malloc(), "growable external adapter starts with no guessed Device capacity");
    pool.appendCapacity(1);
    const auto first = pool.malloc().value();
    pool.incTreeRef(first, BlockTreeRefType::CACHE);
    const auto first_identity = pool.pageIdentity(first).value();
    for (size_t i = 0; i < 32; ++i) {
        pool.appendCapacity(1);
    }
    check(pool.totalBlocksNum() == 33 && pool.freeBlocksNum() == 32 && pool.isCurrentPage(first_identity),
          "growing external slots preserves prior Device generations");
    check(pool.decTreeRef(first, BlockTreeRefType::CACHE) == 1, "external allocation retires after its final hold");
}

}  // namespace

int main() {
    opaque_keys_and_native_group_windows();
    retained_copy_has_independent_physical_lifetime();
    final_frozen_reference_retires_the_native_allocation();
    growable_native_pool_preserves_live_slots();
}
