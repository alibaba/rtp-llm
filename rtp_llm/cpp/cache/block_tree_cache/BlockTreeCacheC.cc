#include "BlockTreeCacheCInternal.h"
#include "block_pool/DiskBlockIO.h"
#include "block_pool/HostBlockPool.h"
#include "group_set/FullGroupSet.h"
#include "group_set/LinearGroupSet.h"
#include "group_set/PrefixCoverage.h"
#include "group_set/SWAGroupSet.h"
#include "storage_backend/PolledStorage.h"
#include "transfer/BlockTransferDispatcher.h"
#include "transfer/MultiRankBlockTransferEngine.h"

#include <algorithm>
#include <limits>
#include <random>
#include <set>
#include <unordered_set>

using namespace rtp_llm;
using namespace rtp_llm::c_api;

namespace {
thread_local std::string last_error;

uint64_t instanceId() {
    std::random_device random;
    uint64_t           value = 0;
    while (!value)
        value = (uint64_t(random()) << 32) ^ random();
    return value;
}

CacheGroupType groupType(btc_rule rule) {
    switch (rule) {
        case BTC_FULL:
            return CacheGroupType::FULL;
        case BTC_LINEAR:
            return CacheGroupType::LINEAR;
        case BTC_SWA:
            return CacheGroupType::SWA;
        default:
            throw std::invalid_argument("Unknown cache group rule");
    }
}

EvictionPolicy policy(uint32_t value) {
    switch (value) {
        case BTC_EVICT_LRU:
            return EvictionPolicy::LRU;
        case BTC_EVICT_LFU:
            return EvictionPolicy::LFU;
        case BTC_EVICT_FIFO:
            return EvictionPolicy::FIFO;
        default:
            throw std::invalid_argument("Unknown cache eviction policy");
    }
}

PolledTransfer::Routes routes(const btc_config& config) {
    if (config.route_count && !config.routes)
        throw std::invalid_argument("Missing executor routes");
    PolledTransfer::Routes result;
    for (size_t i = 0; i < config.route_count; ++i) {
        const auto& r = config.routes[i];
        if (!r.count || !r.participants || r.kind < BTC_WORK_HOST_TO_DEVICE || r.kind > BTC_WORK_DISK_TO_DEVICE
            || !result.emplace(r.kind, std::vector<uint32_t>(r.participants, r.participants + r.count)).second) {
            throw std::invalid_argument("Invalid or duplicated executor route");
        }
    }
    result.try_emplace(BTC_WORK_RELEASE, std::vector<uint32_t>{0});
    return result;
}

btc_work_kind transferKind(Tier source, Tier target) {
    if (source == Tier::HOST && target == Tier::DEVICE)
        return BTC_WORK_HOST_TO_DEVICE;
    if (source == Tier::DEVICE && target == Tier::HOST)
        return BTC_WORK_DEVICE_TO_HOST;
    if (source == Tier::HOST && target == Tier::DISK)
        return BTC_WORK_HOST_TO_DISK;
    if (source == Tier::DISK && target == Tier::HOST)
        return BTC_WORK_DISK_TO_HOST;
    if (source == Tier::DISK && target == Tier::DEVICE)
        return BTC_WORK_DISK_TO_DEVICE;
    throw std::invalid_argument("Unsupported native transfer direction");
}

RankTransferRequest prepareWorkerCopy(btc_cache& cache, const TransferTask& task) {
    const auto              descs = task.descriptors();
    const auto&             first = descs.front();
    const auto              kind  = transferKind(first.source_tier, first.target_tier);
    std::vector<PolledPage> pages;
    pages.reserve(descs.size());
    for (const auto& desc : descs) {
        assert(desc.source_tier == first.source_tier && desc.target_tier == first.target_tier);
        PolledPage page{{cache.groups.at(desc.group_set_id).id, (desc.path_index + 1) * cache.key_tokens},
                        desc.logical_key,
                        {},
                        {},
                        {}};
        for (auto block : desc.source_blocks)
            page.source.push_back(cache.describePage(desc.group_set_id, desc.source_tier, block));
        for (auto block : desc.target_blocks)
            page.target.push_back(cache.describePage(desc.group_set_id, desc.target_tier, block));
        pages.push_back(std::move(page));
    }
    return cache.transfers->prepareCopy(
        kind, task.requestId().value_or(0), std::move(pages), task.completionProof(), [&cache, trace = task.traceId()](bool late) {
            if (!cache.native) return;
            if (late && cache.native->localTransferCompletion(trace) == TransferTaskRegistry::State::BUFFER_SAFE) {
                for (const auto& frozen : cache.native->frozenTasks()) {
                    const auto* numeric = std::get_if<uint64_t>(&frozen->execution().trace_id);
                    if (numeric && *numeric == trace) frozen->unfreeze();
                }
            }
            cache.native->forgetLocalTransferCompletion(trace);
        });
}

template<class T>
btc_bytes view(const T& value) {
    return {value.data(), value.size()};
}

// Native block zero is reserved. Each C pool exports a zero-based range,
// shared across groups or starting after the preceding group's quota.
btc_page localPage(uint64_t pool, uint64_t start, BlockPoolPage page) {
    return {pool, start + static_cast<uint64_t>(page.slot - 1), page.generation, 0};
}

BlockIdxType localBlock(uint64_t start, uint64_t slot) {
    if (slot < start || slot - start >= std::numeric_limits<BlockIdxType>::max())
        throw std::invalid_argument("Local slot is outside its pool range");
    return static_cast<BlockIdxType>(slot - start + 1);
}

template<typename Config>
std::shared_ptr<Config> localConfig(BlockPoolType type, uint64_t capacity) {
    if (capacity >= static_cast<uint64_t>(std::numeric_limits<BlockIdxType>::max()))
        throw std::invalid_argument("Invalid native local pool capacity");
    auto config = std::make_shared<Config>();
    config->pool_name = type == BlockPoolType::HOST ? "Host" : "Disk";
    config->pool_type = type;
    config->physical_block_count = capacity + 1;
    return config;
}

std::shared_ptr<HostBlockPool> makeHostPool(uint64_t start, uint64_t capacity, const btc_host_region* region) {
    if (!capacity)
        return {};
    auto config = localConfig<HostBlockPoolConfig>(BlockPoolType::HOST, capacity);
    if (region) {
        if (!region->memory.data || !region->payload_bytes || region->stride_bytes < region->payload_bytes)
            throw std::invalid_argument("Invalid external Host layout");
        const size_t rows = region->memory.size / region->stride_bytes;
        if (start > rows || capacity > rows - start)
            throw std::invalid_argument("External Host region is too small for its slot ranges");
        config->payload_bytes = region->payload_bytes;
        config->stride_bytes = region->stride_bytes;
        config->alignment = 1;
    }
    auto pool = std::make_shared<HostBlockPool>(std::move(config));
    if (region)
        pool->initBacking(region->memory.data, region->memory.size, start * region->stride_bytes);
    else
        pool->initDelegated();
    return pool;
}

// Disk bytes live in per-slot files or in the external executor. Only the
// native allocation and reference counts belong to this handle.
class DiskSlots final: public IBlockPool {
public:
    explicit DiskSlots(uint64_t capacity): IBlockPool(localConfig<BlockPoolConfigBase>(BlockPoolType::DISK, capacity)) {
        markInitialized();
    }
    size_t blockSizeBytes() const override { return 0; }
};

template<class Ack>
void submitRelease(PolledTransfer& transport, std::vector<btc_page> release, Ack ack) {
    if (release.empty())
        return;
    std::vector<PolledPage> pages;
    pages.reserve(release.size());
    for (const auto& page : release)
        pages.push_back({{}, {}, {page}, {}, {}});
    transport.submit(BTC_WORK_RELEASE,
                     0,
                     std::move(pages),
                     [&transport, release = std::move(release), ack = std::move(ack)](PolledOutcome out) {
                         if (!out.safe)
                             return;
                         if (out.result == BTC_SUCCESS) {
                             for (const auto& page : release)
                                 ack(page);
                         } else {
                             submitRelease(transport, release, ack);
                         }
                     });
}
}  // namespace

namespace rtp_llm::c_api {
// Only handles requesting RELEASE allocate these queues. The pool callback
// keeps its queue alive; the cache's groups own the pool while polling/acking.
struct LocalReleases {
    LocalReleases(IBlockPool& pool, uint64_t start): pool(pool), start(start) {
        pending.reserve(pool.totalBlocksNum());
    }
    IBlockPool& pool;
    uint64_t start;
    std::mutex mutex;
    std::vector<btc_page> pending;
};

btc_status rememberError(btc_status code, const char* message) {
    last_error = message;
    return code;
}
std::string copyBytes(btc_bytes bytes) {
    if (!bytes.size)
        return {};
    if (!bytes.data)
        throw std::invalid_argument("Missing bytes");
    return std::string(static_cast<const char*>(bytes.data), bytes.size);
}
BlockTreeKeys copyKeys(btc_keys keys) {
    if (keys.count && !keys.keys)
        throw std::invalid_argument("Missing prefix keys");
    BlockTreeKeys result;
    result.reserve(keys.count);
    for (size_t i = 0; i < keys.count; ++i) {
        const auto& key = keys.keys[i];
        if (!key.data || !key.size)
            throw std::invalid_argument("Empty prefix key");
        result.emplace_back(std::string_view(static_cast<const char*>(key.data), key.size));
    }
    return result;
}

bool PreparedPages::prepare(const btc_binding* bindings, size_t count, bool load) {
    if (count && !bindings)
        throw std::invalid_argument("Missing bindings");
    size_t members = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto& binding = bindings[i];
        const auto& group   = cache_.groups[cache_.groupIndex(binding.position.group)];
        if (!binding.position.end_token || binding.position.end_token % group.page_tokens || !binding.pages
            || binding.page_count != group.set->pools().device.size())
            throw std::invalid_argument("Invalid group binding");
        if (binding.page_count > SIZE_MAX - members)
            throw std::invalid_argument("Too many binding members");
        members += binding.page_count;
    }
    prepared_.reserve(members);
    std::map<std::pair<uint64_t, uint64_t>, uint64_t> destinations;
    for (size_t i = 0; i < count; ++i) {
        const auto& binding    = bindings[i];
        const auto& group      = cache_.groups[cache_.groupIndex(binding.position.group)];
        auto [entry, inserted] = rows.try_emplace({group.id, binding.position.end_token});
        if (!inserted)
            throw std::invalid_argument("Duplicate logical position");
        entry->second.reserve(binding.page_count);
        for (size_t j = 0; j < binding.page_count; ++j) {
            const auto& page = binding.pages[j];
            if (!group.member_pools.empty() && group.member_pools[j] != page.pool)
                throw std::invalid_argument("Physical pool does not match group member");
            bool exclusive = false;
            if (load) {
                auto [pos, fresh] =
                    destinations.emplace(std::make_pair(page.pool, page.slot), binding.position.end_token);
                if (pos->second != binding.position.end_token)
                    return false;
                exclusive = fresh;
            }
            auto id = cache_.device->prepare(page, exclusive);
            if (!id)
                return false;
            prepared_.push_back(*id);
            entry->second.push_back(*id);
        }
    }
    return true;
}
PreparedPages::~PreparedPages() {
    for (auto id : prepared_)
        cache_.device->finishPrepare(id, adopted_);
}
}  // namespace rtp_llm::c_api

btc_cache::btc_cache(const btc_config& config, const btc_host_region* host_region):
    instance(instanceId()),
    key_tokens(config.key_tokens),
    name_space(copyBytes(config.name_space)),
    device(std::make_shared<ExternalPagePool>()),
    transfers(std::make_shared<PolledTransfer>(instance, routes(config))),
    requests(makeRequests()) {
    if (!key_tokens || name_space.empty() || !config.group_count
        || !config.groups || config.tiered_eviction > 1 || config.release_local_pages > 1
        || config.external_disk_io > 1) {
        throw std::invalid_argument("Invalid BlockTreeCache configuration");
    }
    const auto trackRelease = [&](const std::shared_ptr<IBlockPool>& pool, uint64_t id, uint64_t start) {
        if (!config.release_local_pages || !pool)
            return;
        auto queue = std::make_shared<LocalReleases>(*pool, start);
        pool->setBlockRetirementCallback([queue, id](BlockPoolPage page) noexcept {
            std::lock_guard lock(queue->mutex);
            queue->pending.push_back(localPage(id, queue->start, page));
        });
        local_releases.push_back(std::move(queue));
    };
    const auto shared_host = makeHostPool(0, config.host_slots, host_region);
    const auto shared_disk = config.disk_slots ? std::make_shared<DiskSlots>(config.disk_slots) : nullptr;
    trackRelease(shared_host, BTC_HOST_POOL, 0);
    trackRelease(shared_disk, BTC_DISK_POOL, 0);
    uint64_t               host_capacity = config.host_slots, disk_capacity = config.disk_slots;
    std::vector<btc_group> ordered(config.groups, config.groups + config.group_count);
    std::sort(ordered.begin(), ordered.end(), [](auto a, auto b) { return a.id < b.id; });
    std::vector<GroupSetPtr> sets;
    for (const auto& spec : ordered) {
        if (!group_indices.emplace(spec.id, groups.size()).second || !spec.page_tokens || spec.page_tokens % key_tokens
            || !spec.members || (spec.rule != BTC_SWA && spec.window_tokens) || (spec.host_slots && shared_host)
            || (spec.disk_slots && shared_disk) || spec.host_slots > UINT64_MAX - host_capacity
            || spec.disk_slots > UINT64_MAX - disk_capacity) {
            throw std::invalid_argument("Invalid BlockTreeCache group layout");
        }
        const uint64_t host_start = shared_host ? 0 : host_capacity;
        const uint64_t disk_start = shared_disk ? 0 : disk_capacity;
        auto host = shared_host;
        auto disk = shared_disk;
        if (spec.host_slots) {
            host = makeHostPool(host_start, spec.host_slots, host_region);
            trackRelease(host, BTC_HOST_POOL, host_start);
        }
        if (spec.disk_slots) {
            disk = std::make_shared<DiskSlots>(spec.disk_slots);
            trackRelease(disk, BTC_DISK_POOL, disk_start);
        }
        if (disk && !host)
            throw std::invalid_argument("Disk requires Host for this group");
        host_capacity += spec.host_slots;
        disk_capacity += spec.disk_slots;
        GroupSetPools pools{std::vector<std::shared_ptr<IBlockPool>>(spec.members, device), host, disk};
        const auto    type = groupType(spec.rule);
        GroupSetPtr   set;
        if (type == CacheGroupType::FULL)
            set = std::make_shared<FullGroupSet>(std::move(pools));
        else if (type == CacheGroupType::LINEAR)
            set = std::make_shared<LinearGroupSet>(std::move(pools));
        else
            set = std::make_shared<SWAGroupSet>(std::move(pools));
        std::vector<std::string> tags;
        std::vector<uint64_t>    member_pools;
        for (size_t i = 0; i < spec.members; ++i) {
            tags.push_back(std::to_string(spec.id) + "." + std::to_string(i));
            if (spec.member_pools) {
                if (spec.member_pools[i] >= BTC_DISK_POOL)
                    throw std::invalid_argument("Reserved device pool identity");
                member_pools.push_back(spec.member_pools[i]);
            }
        }
        BackingLayout backing;
        if (host && host_region) {
            backing = {host->payloadBytes(), host->payloadBytes(), host->strideBytes(), false};
        }
        set->initialize(groups.size(), GroupSetGeometry{type, spec.page_tokens, spec.window_tokens}, std::move(tags),
                        backing);
        sets.push_back(set);
        groups.push_back({spec.id, spec.page_tokens, std::move(member_pools), set, host_start, disk_start});
    }
    const auto route = [&](btc_work_kind k) { return transfers->supports(k); };
    if ((host_region && !host_capacity)
        || (host_capacity != 0) != route(BTC_WORK_HOST_TO_DEVICE) || (disk_capacity != 0) != route(BTC_WORK_HOST_TO_DISK)
        || (disk_capacity != 0) != (route(BTC_WORK_DISK_TO_HOST) || route(BTC_WORK_DISK_TO_DEVICE))
        || (route(BTC_WORK_DISK_TO_DEVICE) && !config.external_disk_io) || (config.external_disk_io && !disk_capacity)
        || (!host_capacity
            && (route(BTC_WORK_DEVICE_TO_HOST) || route(BTC_WORK_REMOTE_TO_HOST) || route(BTC_WORK_HOST_TO_REMOTE)))
        || route(BTC_WORK_REMOTE_QUERY) != (route(BTC_WORK_REMOTE_TO_HOST) || route(BTC_WORK_REMOTE_TO_DEVICE))
        || (config.tiered_eviction && (!route(BTC_WORK_DEVICE_TO_HOST) || disk_capacity
                                      || route(BTC_WORK_DEVICE_TO_REMOTE) || route(BTC_WORK_REMOTE_TO_DEVICE)
                                      || route(BTC_WORK_REMOTE_QUERY) != route(BTC_WORK_HOST_TO_REMOTE)))) {
        throw std::invalid_argument("Missing transfer route or local capacity");
    }
    if (disk_capacity && !config.external_disk_io) {
        auto directory = copyBytes(config.disk_directory);
        if (directory.empty() || directory.find('\0') != std::string::npos)
            throw std::invalid_argument("Invalid Disk path");
        disk_files = std::make_unique<DiskFiles>(directory);
    }
    BlockTreeCacheConfig options;
    options.enable_host_cache = host_capacity != 0;
    options.enable_disk_cache = disk_capacity != 0;
    options.enable_remote_cache =
        route(BTC_WORK_REMOTE_QUERY) || route(BTC_WORK_DEVICE_TO_REMOTE) || route(BTC_WORK_HOST_TO_REMOTE);
    options.write_remote_on_host_pressure = route(BTC_WORK_HOST_TO_REMOTE);
    options.external_device_capacity = true;
    options.watermark_device = options.watermark_host = options.watermark_disk = {1.0, 1.0};
    if (config.tiered_eviction) {
        options.watermark_device = {kDefaultDeviceLowWatermarkRatio, kDefaultDeviceHighWatermarkRatio};
        options.watermark_host = {kDefaultMemoryLowWatermarkRatio, kDefaultMemoryHighWatermarkRatio};
    }
    options.device_eviction_policy                                             = policy(config.device_eviction_policy);
    options.full_prefix_scan_interval_ms                                       = 0;
    if (options.enable_remote_cache)
        storage = makeHostStorage(*this, config);
    auto workers = std::make_unique<BlockTreeTaskPool>(1, 0, "btc_host");
    task_pool    = workers.get();
    RankTransferTransport wire;
    wire.prepare = [this](const TransferTask& task) { return prepareWorkerCopy(*this, task); };
    auto coordinator = std::make_shared<MultiRankBlockTransferEngine>(sets, std::move(wire));
    TransferTransport transport;
    // The local dispatcher registry tracks physical custody of exported pages.
    // Rank delivery and result reduction are owned by the same coordinator as RTP.
    transport.execute_local = [coordinator](TransferTask task) { return coordinator->execute(std::move(task)); };
    auto dispatcher         = std::make_unique<BlockTransferDispatcher>(sets, std::move(transport));
    native                  = std::make_unique<BlockTreeCache>(std::make_unique<BlockTree>(std::move(sets), key_tokens),
                                              std::move(options),
                                              storage,
                                              std::move(dispatcher),
                                              std::move(workers),
                                              std::make_shared<CacheObserver>());
    if (!native->init())
        throw std::runtime_error("Failed to start native BlockTreeCache");
}

btc_cache::~btc_cache() = default;
size_t btc_cache::groupIndex(uint32_t id) const {
    return group_indices.at(id);
}
btc_page btc_cache::describePage(size_t group, Tier tier, BlockIdxType block) const {
    if (tier == Tier::DEVICE)
        return device->describe(block);
    const auto& layout = groups.at(group);
    const auto& pools = layout.set->pools();
    const IBlockPool* pool = tier == Tier::HOST ? pools.host.get() : pools.disk.get();
    if (!pool || (tier != Tier::HOST && tier != Tier::DISK))
        throw std::logic_error("Invalid native local page");
    const auto page = pool->pageIdentity(block);
    if (!page)
        throw std::logic_error("Native local slot no longer allocated");
    return tier == Tier::HOST ? localPage(BTC_HOST_POOL, layout.host_start, *page)
                              : localPage(BTC_DISK_POOL, layout.disk_start, *page);
}
void btc_cache::event(uint64_t request, btc_event_kind kind, uint64_t tokens) {
    std::lock_guard lock(event_mutex);
    events.push_back({++cursor, request, kind, tokens});
}
void btc_cache::flush() {
    if (native)
        native->expireRemotePrefetches();
    if (task_pool)
        task_pool->waitForIdle();
    queueReleases();
}
void btc_cache::queueReleases() {
    submitRelease(
        *transfers, device->takeReleased(), [pool = device](const btc_page& page) { pool->acknowledgeRelease(page); });
    for (const auto& queue : local_releases) {
        std::lock_guard lock(queue->mutex);
        submitRelease(*transfers, queue->pending, [this, queue](const btc_page& page) {
            queue->pool.releaseRetiredBlock({&queue->pool, localBlock(queue->start, page.slot), page.generation});
            notifyLocalCapacity(*this);
        });
        queue->pending.clear();
    }
}

struct btc_batch {
    explicit btc_batch(btc_cache& cache): name_space(cache.name_space), records(cache.transfers->poll()) {
        {
            std::lock_guard lock(cache.event_mutex);
            events.assign(cache.events.begin(), cache.events.end());
        }
        pages.resize(records.size());
        for (size_t i = 0; i < records.size(); ++i) {
            const auto& w    = records[i];
            auto&       list = pages[i];
            for (const auto& p : w.pages)
                list.push_back({p.position,
                                view(p.key),
                                p.source.data(),
                                p.source.size(),
                                p.target.data(),
                                p.target.size(),
                                view(p.remote)});
            work.push_back({{cache.instance, w.operation},
                            w.kind,
                            w.request,
                            view(name_space),
                            w.participants.data(),
                            w.participants.size(),
                            list.data(),
                            list.size()});
        }
    }
    std::string                             name_space;
    std::vector<PolledWork>                 records;
    std::vector<btc_event>                  events;
    std::vector<std::vector<btc_work_page>> pages;
    std::vector<btc_work>                   work;
};

uint32_t btc_abi_version(void) {
    return BTC_ABI_VERSION;
}
const char* btc_last_error(void) {
    return last_error.c_str();
}
size_t btc_required_pages(btc_rule rule, uint64_t tokens, uint64_t window, uint64_t end) {
    if (!tokens || !end || end % tokens || rule < BTC_FULL || rule > BTC_SWA || (rule != BTC_SWA && window))
        return 0;
    return reusableBlockCount(groupType(rule), tokens, window, end / tokens);
}
static const btc_config& checkedConfig(const btc_config* config) {
    if (require(config).abi_version != BTC_ABI_VERSION)
        throw std::invalid_argument("Unsupported BlockTreeCache ABI version");
    return *config;
}
btc_status btc_open(const btc_config* config, btc_cache** output) {
    return status([&] {
        require(output) = nullptr;
        *output         = new btc_cache(checkedConfig(config));
        return BTC_OK;
    });
}
btc_status btc_open_with_host_region(const btc_config* config, const btc_host_region* region, btc_cache** output) {
    return status([&] {
        require(output) = nullptr;
        *output = new btc_cache(checkedConfig(config), &require(region));
        return BTC_OK;
    });
}
btc_status btc_host_slot_buffer(btc_cache* handle, uint32_t group, uint64_t slot, btc_buffer* out) {
    return locked(handle, [&](btc_cache& cache) {
        auto& output = require(out);
        output = {};
        const auto& layout = cache.groups.at(cache.groupIndex(group));
        const auto host = layout.set->hostPool();
        if (!host)
            throw std::invalid_argument("Group has no Host slots");
        const auto buffer = host->blockBuffer(localBlock(layout.host_start, slot));
        output = {buffer.addr, buffer.payload_bytes};
        return BTC_OK;
    });
}
btc_status btc_host_capacity(btc_cache* handle, uint32_t group, size_t* total, size_t* free_slots) {
    return locked(handle, [&](btc_cache& cache) {
        const auto& pool = cache.groups.at(cache.groupIndex(group)).set->pools().host;
        if (!pool || !total || !free_slots)
            throw std::invalid_argument("Expected a Host pool and two capacity outputs");
        *total = pool->totalBlocksNum();
        *free_slots = pool->freeBlocksNum();
        return BTC_OK;
    });
}
btc_status btc_installed_device_pages(btc_cache* handle, btc_keys keys, const btc_binding* bindings,
                                      size_t count, uint8_t* installed) {
    return locked(handle, [&](btc_cache& cache) -> btc_status {
        if (!cache.native)
            return BTC_CLOSED;
        if (count && (!bindings || !installed))
            throw std::invalid_argument("Missing device bindings or output");
        const auto path = copyKeys(keys);
        cache.native->visitTreeLocked([&](const BlockTree& tree) {
            const auto nodes = tree.findNode(path);
            for (size_t i = 0; i < count; ++i) {
                const auto& binding = bindings[i];
                const auto group = cache.groupIndex(binding.position.group);
                const auto& layout = cache.groups[group];
                const auto end = binding.position.end_token;
                if (!end || end % layout.page_tokens || end / cache.key_tokens > path.size() || !binding.pages
                    || binding.page_count != layout.set->pools().device.size())
                    throw std::invalid_argument("Invalid installed device binding");
                const auto row = end / cache.key_tokens - 1;
                installed[i] = false;
                if (row >= nodes.size())
                    continue;
                const auto& value = nodes[row]->group_set_resources[group];
                if (!value.transfer_detached && value.hasCompleteDeviceValue()
                    && layout.set->canPublishDevice(value.device_blocks)) {
                    installed[i] = std::equal(value.device_blocks.begin(), value.device_blocks.end(),
                                              binding.pages, binding.pages + binding.page_count,
                                              [&](auto block, const btc_page& page) {
                                                  return samePage(cache.device->describe(block), page);
                                              });
                }
            }
        });
        return BTC_OK;
    });
}
static btc_status publish(btc_cache*         handle,
                          btc_keys           keys,
                          uint64_t           committed,
                          const btc_binding* bindings,
                          size_t             count,
                          bool               resident,
                          size_t*            resident_count) {
    return locked(handle, [&](btc_cache& c) -> btc_status {
        if (c.draining)
            return BTC_CLOSED;
        auto       path   = copyKeys(keys);
        const auto length = std::min(static_cast<uint64_t>(path.size()), committed / c.key_tokens);
        path.erase(path.begin() + length, path.end());
        PreparedPages pages(c);
        if (!pages.prepare(bindings, count, false))
            return BTC_RETRY;
        std::vector<std::vector<GroupSetResource>> slots(path.size(), std::vector<GroupSetResource>(c.groups.size()));
        for (auto& [pos, members] : pages.rows) {
            if (pos.second / c.key_tokens > path.size())
                throw std::invalid_argument("Published page exceeds committed keys");
            slots[pos.second / c.key_tokens - 1][c.groupIndex(pos.first)].device_blocks = std::move(members);
        }
        const auto length_resident = c.native->insert(path, slots, Tier::DEVICE, resident);
        pages.adopt();
        if (resident_count)
            *resident_count = length_resident;
        return BTC_OK;
    });
}
btc_status btc_publish(btc_cache* c, btc_keys keys, uint64_t committed, const btc_binding* bindings, size_t n) {
    return publish(c, keys, committed, bindings, n, false, nullptr);
}
btc_status btc_publish_resident(
    btc_cache* c, btc_keys keys, uint64_t committed, const btc_binding* bindings, size_t n, size_t* resident) {
    if (!resident)
        return rememberError(BTC_INVALID, "Missing resident count");
    *resident = 0;
    return publish(c, keys, committed, bindings, n, true, resident);
}
btc_status btc_poll(btc_cache* c, btc_batch** output) {
    return locked(c, [&](auto& cache) {
        require(output) = nullptr;
        cache.flush();
        *output = new btc_batch(cache);
        return BTC_OK;
    });
}
btc_batch_view btc_batch_get(const btc_batch* b) {
    return b ? btc_batch_view{b->work.data(), b->work.size(), b->events.data(), b->events.size()} : btc_batch_view{};
}
void btc_batch_free(btc_batch* b) {
    delete b;
}
btc_status btc_pending_work(btc_cache* c, size_t* count) {
    return locked(c, [&](auto& cache) {
        cache.flush();
        require(count) = cache.transfers->size();
        return BTC_OK;
    });
}
btc_status btc_accept(btc_cache* c, btc_operation id, uint32_t actor) {
    return locked(c, [&](auto& cache) {
        cache.transfers->accept(id, actor);
        return BTC_OK;
    });
}
btc_status btc_complete(btc_cache* c, const btc_receipt* receipts, size_t count) {
    return locked(c, [&](auto& cache) {
        cache.transfers->complete(receipts, count);
        cache.flush();
        return BTC_OK;
    });
}
btc_status btc_ack_events(btc_cache* c, uint64_t cursor) {
    return locked(c, [&](auto& cache) {
        std::lock_guard lock(cache.event_mutex);
        if (cursor > cache.cursor)
            throw std::invalid_argument("Unknown cache event cursor");
        while (!cache.events.empty() && cache.events.front().cursor <= cursor)
            cache.events.pop_front();
        return BTC_OK;
    });
}
static BlockIdList physicalPages(btc_cache& cache, uint64_t pool, const btc_page* pages, size_t count) {
    if (count && !pages)
        throw std::invalid_argument("Missing device page array");
    BlockIdList list;
    for (size_t i = 0; i < count; ++i) {
        if (pool != BTC_HOST_POOL && pool != pages[i].pool)
            throw std::invalid_argument("Unexpected device pool");
        auto blocks = cache.device->resolve(pages[i]);
        list.insert(list.end(), blocks.begin(), blocks.end());
    }
    return list;
}
// Normalize aliases once for both synchronous reclaim and the watermark policy.
struct DeviceSlots {
    std::vector<BlockIdList> all;
    BlockIdList eligible_blocks;
    size_t eligible_slots;
    size_t returning;
};

static DeviceSlots deviceSlots(btc_cache& cache, uint64_t pool, const btc_page* pages, size_t n, bool collect_blocks) {
    auto resolved = cache.device->resolvePhysical(pages, n, pool);
    DeviceSlots slots{{}, {}, 0, resolved.returning};
    std::unordered_set<BlockIdxType> seen;
    for (size_t i = 0; i < n; ++i) {
        auto& aliases = resolved.provided[i];
        if (collect_blocks)
            slots.eligible_blocks.insert(slots.eligible_blocks.end(), aliases.begin(), aliases.end());
        if (pages[i].pool == pool && !aliases.empty() && seen.insert(aliases.front()).second)
            slots.all.push_back(std::move(aliases));
    }
    slots.eligible_slots = slots.all.size();
    for (auto& aliases : resolved.companions)
        slots.all.push_back(std::move(aliases));
    return slots;
}

static bool deviceBalanceNeeded(btc_cache& cache, uint64_t pool, size_t capacity, size_t free_slots) {
    if (pool >= BTC_DISK_POOL || !capacity || free_slots > capacity || !cache.native
        || cache.native->config().watermark_device.low_ratio == 1.0)
        throw std::invalid_argument("Expected a tiered cache and a physical Device capacity");
    auto active = cache.device_watermark_rounds.find(pool);
    return (active != cache.device_watermark_rounds.end() && active->second)
           || cache.native->config().watermark_device.reachedHigh(capacity, capacity - free_slots);
}
btc_status btc_device_balance_needed(btc_cache* c, uint64_t pool, size_t total, size_t free_slots, uint8_t* needed) {
    return locked(c, [&](auto& cache) {
        require(needed) = deviceBalanceNeeded(cache, pool, total, free_slots);
        return BTC_OK;
    });
}
btc_status btc_balance_device(btc_cache* c, uint64_t pool, size_t total, size_t free_slots,
                              const btc_page* pages, size_t n) {
    return locked(c, [&](auto& cache) -> btc_status {
        if (cache.draining)
            return BTC_CLOSED;
        if (n && !pages)
            throw std::invalid_argument("Missing eligible Device pages");
        for (size_t i = 0; i < n; ++i) {
            if (pages[i].pool != pool)
                throw std::invalid_argument("Eligible Device page is from another pool");
        }
        if (deviceBalanceNeeded(cache, pool, total, free_slots)) {
            const auto slots = deviceSlots(cache, pool, pages, n, false);
            cache.native->balanceDevicePages(*cache.device, slots.all, slots.eligible_slots, total, free_slots,
                                             slots.returning, cache.device_watermark_rounds[pool]);
        }
        return BTC_OK;
    });
}
static btc_status releaseDevice(btc_cache* c, uint64_t pool, size_t pressure, const btc_page* pages, size_t n,
                                bool copy_only, bool demote, bool cascade) {
    return locked(c, [&](auto& cache) {
        if (n && !pages)
            throw std::invalid_argument("Missing device page array");
        if (cache.native && pressure && n) {
            const auto slots = deviceSlots(cache, pool, pages, n, cascade);
            cache.native->releaseDevicePages(*cache.device, slots.all, slots.eligible_slots, slots.eligible_blocks,
                                             pressure, copy_only, demote, cascade);
        }
        return BTC_OK;
    });
}
btc_status btc_reclaim(btc_cache* c, uint64_t pool, size_t pressure, const btc_page* pages, size_t n) {
    return releaseDevice(c, pool, pressure, pages, n, false, false, false);
}
btc_status btc_reclaim_related(btc_cache* c, uint64_t pool, size_t pressure, const btc_page* pages, size_t n) {
    return releaseDevice(c, pool, pressure, pages, n, false, false, true);
}
btc_status btc_demote(btc_cache* c, uint64_t pool, size_t pressure, const btc_page* pages, size_t n, uint32_t remove) {
    if (remove > 1)
        return rememberError(BTC_INVALID, "Invalid demotion mode");
    return releaseDevice(c, pool, pressure, pages, n, !remove, true, false);
}
btc_status btc_invalidate(btc_cache* c, const btc_page* pages, size_t n) {
    return locked(c, [&](auto& cache) {
        if (cache.native) {
            auto blocks = physicalPages(cache, BTC_HOST_POOL, pages, n);
            cache.device->invalidate(blocks);
            cache.native->invalidateDevicePages(*cache.device, blocks);
        }
        return BTC_OK;
    });
}
btc_status btc_reclaim_local(btc_cache* c, uint64_t pool, uint32_t group, size_t pressure) {
    return locked(c, [&](auto& cache) {
        if (pool != BTC_HOST_POOL && pool != BTC_DISK_POOL)
            throw std::invalid_argument("Invalid local pool");
        if (cache.native)
            cache.native->reclaimLocalPages(
                pool == BTC_HOST_POOL ? Tier::HOST : Tier::DISK, cache.groupIndex(group), pressure);
        return BTC_OK;
    });
}
btc_status btc_set_local_eviction_policy(btc_cache* c, uint64_t pool, uint32_t value) {
    return locked(c, [&](auto& cache) {
        if (pool != BTC_HOST_POOL && pool != BTC_DISK_POOL)
            throw std::invalid_argument("Invalid local pool");
        if (cache.native)
            cache.native->setLocalEvictionPolicy(pool == BTC_HOST_POOL ? Tier::HOST : Tier::DISK, policy(value));
        return BTC_OK;
    });
}
btc_status btc_drain(btc_cache* c) {
    return locked(c, [&](auto& cache) -> btc_status {
        if (!cache.draining) {
            cache.draining = true;
            cancelRequests(cache);
        }
        cache.flush();
        cache.transfers->cancelUnexposed();
        cache.flush();
        if (cache.transfers->size())
            return BTC_PENDING;
        cache.native.reset();
        cache.task_pool = nullptr;
        cache.queueReleases();
        return cache.transfers->size() || cache.device->hasOutstandingHolds() ? BTC_PENDING : BTC_OK;
    });
}
btc_status btc_close(btc_cache* c) {
    const auto result = btc_drain(c);
    if (result == BTC_OK)
        delete c;
    return result;
}
btc_status btc_execute_disk(btc_cache* c, btc_operation id, const btc_buffer* buffers, size_t count) {
    return status([&] {
        auto& cache = require(c);
        if (!cache.disk_files || (count && !buffers))
            throw std::invalid_argument("Missing internal Disk or buffers");
        btc_work_kind kind;
        auto          pages = cache.transfers->beginDisk(id, kind);
        try {
            if (pages.size() != count)
                throw std::invalid_argument("Wrong Disk page count");
            for (size_t i = 0; i < count; ++i) {
                const bool  write = kind == BTC_WORK_HOST_TO_DISK;
                const auto& disk  = write ? pages[i].target : pages[i].source;
                cache.disk_files->copy(write, disk.at(0).slot, buffers[i].data, buffers[i].size, id.operation);
            }
        } catch (...) {
            cache.transfers->endDisk(id);
            throw;
        }
        cache.transfers->endDisk(id);
        return BTC_OK;
    });
}
