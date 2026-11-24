#include "rtp_llm/cpp/cache/block_tree_cache/remote_mock/RemoteMockAdapter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/remote_mock/RemoteMockWriteback.h"

#include <list>
#include <string_view>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <future>
#include <limits>
#include <iomanip>
#include <sstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unistd.h>

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/AlignedHostMemory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyGeometry.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpClient.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/remote_mock_control.pb.h"
#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockFormat.h"
#include "rtp_llm/cpp/utils/BackgroundProfilingScope.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace {
using Clock = std::chrono::steady_clock;
void require(bool ok, std::string_view message) {
    if (!ok)
        throw std::invalid_argument("remote mock: " + std::string(message));
}
uint64_t mixPosition(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}
void setSession(remote_mock::Session* out, const RemoteMockSession& session) {
    out->set_server_epoch(session.server_epoch);
    out->set_session_id(session.session_id);
    out->set_generation(session.generation);
}
RemoteMockSession getSession(const remote_mock::Session& session) {
    return {session.server_epoch(), session.session_id(), session.generation()};
}
ErrorInfo failed(const std::string& message) {
    return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "remote mock: " + message);
}
ErrorInfo admissionFailure(const std::exception& error) noexcept {
    try {
        return failed(error.what());
    } catch (const std::bad_alloc&) {
        return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, std::string{});
    }
}
RemoteMockLimits getLimits(const remote_mock::Limits& input) {
    RemoteMockLimits limits;
#define LIMIT(name) limits.name = input.name()
    LIMIT(total_bytes);
    LIMIT(source_bytes);
    LIMIT(write_slot_bytes);
    LIMIT(read_staging_bytes);
    LIMIT(write_staging_bytes);
    LIMIT(write_slots);
    LIMIT(max_rpc_bytes);
    LIMIT(max_blocks_per_rpc);
    LIMIT(max_read_operations);
    LIMIT(max_ranks);
    LIMIT(max_groups_per_rank);
    LIMIT(max_sources_per_group);
    LIMIT(max_members_per_group);
    LIMIT(replay_window);
    LIMIT(max_delay_ms);
#undef LIMIT
    return limits;
}
// FIX1 service charge, including decoded request and simultaneous metadata.
uint64_t serviceCharge(const google::protobuf::Message& request,
                       uint64_t                         payload,
                       uint32_t                         blocks,
                       const RemoteMockLimits&          limits) {
    const uint64_t metadata = request.SpaceUsedLong() + request.ByteSizeLong() + uint64_t(8192) * blocks + 8192;
    auto           charge   = remoteMockStagingCharge(payload, blocks, limits, metadata);
    require(charge.ok(), "RPC metadata/payload charge overflows or exceeds limits");
    return *charge;
}
void fillReadEnvelope(CacheLoadRequest& request, const std::string& id, uint32_t timeout) {
    request.set_requestid(id);
    request.set_client_ip(autil::NetUtil::getBindIp());
    request.set_request_send_start_time_us(std::numeric_limits<int64_t>::max());
    request.set_partition_count(1);
    request.set_partition_id(0);
    request.set_timeout_ms(timeout - 10);
}
}  // namespace

struct RemoteMockAdapter::Impl {
    struct Record {
        RemoteMockLoadTarget target;
        size_t               offset{0};
        size_t               received{0};
    };
    struct Operation {
        uint64_t                           id{0};
        RemoteMockLoadPlan                 plan;
        DoneCallback                       callback;
        std::atomic<bool>                  cancelled{false};
        size_t                             next_target{0};
        size_t                             rpc_sequence{0};
        std::vector<Record>                records;
        std::shared_ptr<AlignedHostMemory> staging;
        std::atomic<bool>                  finished{false};
    };
    // Every transport-visible addr aliases this one owner. Registry removal and
    // outer RPC drain cannot release admission while a callback blockRead still
    // holds a borrowed BlockBuffer. Drop backing before notifying credit release.
    struct HostBackingLease {
        std::vector<std::shared_ptr<void>> owners;
        std::function<void()>              released;
        ~HostBackingLease() {
            owners.clear();
            if (released)
                released();
        }
    };
    struct HostOperation {
        uint64_t                                  id{0}, sequence{0};
        uint32_t                                  slot{0};
        std::string                               request_id;
        std::shared_ptr<RemoteStoreTask>          task;
        std::shared_ptr<UserBufferRegistration>   registration;
        std::vector<std::shared_ptr<BlockBuffer>> buffers;
        DoneCallback                              callback;
        std::mutex                                mutex;
        ErrorInfo         result{ErrorCode::EXECUTION_EXCEPTION, "Host transfer has not completed"};
        bool              published{false}, rpc_drained{false}, cleanup_started{false}, cleanup_done{false};
        bool              backing_drained{true}, finished{false};
        std::atomic<bool> cancelled{false};
    };
    KVCacheConfig                                                config;
    ParallelismConfig                                            parallel;
    std::vector<GroupSetPtr>                                     groups;
    std::shared_ptr<PerRankBlockTransferEngine>                  transfer;
    std::shared_ptr<NormalCacheStore>                            store;
    TcpClient                                                    control;
    std::shared_ptr<arpc::RPCChannelBase>                        channel;
    std::string                                                  address;
    uint32_t                                                     port{0};
    RemoteMockSession                                            identity;
    RemoteMockLimits                                             limits;
    std::map<std::string, std::string>                           service_limits;
    RemoteMockRankCredits                                        credit;
    std::vector<RemoteMockLayout>                                layouts;
    std::vector<RemoteMockSourceDescriptor>                      sources;
    uint64_t                                                     layout_hash{0};
    uint64_t                                                     max_record{0};
    std::unique_ptr<BlockTreeTaskPool>                           worker;
    mutable std::mutex                                           mutex;
    std::condition_variable                                      idle;
    bool                                                         accepting{true};
    std::unordered_map<uint64_t, std::shared_ptr<Operation>>     active;
    std::unordered_map<uint64_t, std::shared_ptr<HostOperation>> host_active;
    std::vector<bool>                                            write_slots;
    uint64_t                                                     next_write_sequence{1};
    std::list<std::function<void()>>                             drained;
    uint64_t                                                     next_operation{1};
    RemoteMockAdapterStats                                       counters;
    std::unique_ptr<RemoteMockWriteback>                         writeback;

    uint32_t remaining(Clock::time_point deadline) const {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
        require(ms > 10, "operation deadline expired");
        return static_cast<uint32_t>(std::min<int64_t>(ms, config.mock_remote_cache_timeout_ms));
    }
    template<class Request, class Method>
    remote_mock::ControlResponse call(const Request& request, Method method, Clock::time_point deadline) {
        arpc::ANetRPCController controller;
        controller.SetExpireTime(remaining(deadline));
        remote_mock::ControlResponse               response;
        remote_mock::RemoteMockControlService_Stub stub(channel.get());
        (stub.*method)(&controller, &request, &response, nullptr);
        require(!controller.Failed(), "control RPC failed: " + controller.ErrorText());
        return response;
    }
    remote_mock::ControlResponse health(Clock::time_point deadline) {
        return call(remote_mock::HealthRequest{}, &remote_mock::RemoteMockControlService_Stub::health, deadline);
    }
    void     initialize();
    void     probe(Clock::time_point deadline);
    uint64_t transferHost(const RemoteMockHostTransfer& plan, DoneCallback callback);
    void     finishHost(const std::shared_ptr<HostOperation>& op);
    void     settleHost(const std::shared_ptr<HostOperation>& op);
    void     closeSession() noexcept;
    void     pump(const std::shared_ptr<Operation>& op);
    void     read(const std::shared_ptr<Operation>& op);
    void     copied(const std::shared_ptr<Operation>& op);
    void     finish(const std::shared_ptr<Operation>& op, ErrorInfo error);
    void     enqueue(std::function<void()> callback) {
        if (!worker->submitCompletion(std::move(callback)))
            throw std::runtime_error("remote mock completion worker stopped before drain");
    }
    void checkOperation(const Operation& op) const {
        require(!op.cancelled.load(), "operation cancelled");
        remaining(op.plan.deadline);
    }
};

void RemoteMockAdapter::Impl::initialize() {
    config.validateMockCacheConfig();
    require(config.mock_remote_cache, "adapter requires mock_remote_cache=true");
    require(transfer && !groups.empty(), "no transfer engine or GroupSets");
    require(parallel.tp_size > 0 && parallel.tp_size <= UINT32_MAX && parallel.tp_rank >= 0
                && parallel.tp_rank < parallel.tp_size && parallel.world_rank >= 0,
            "invalid rank geometry");
    require(parallel.world_size == parallel.tp_size, "one TP group per mock session is required");
    require(parallel.world_rank == parallel.tp_rank, "world rank must identify the same member of this TP group");
    require(parallel.tp_size - 1 <= 65535 - config.mock_remote_cache_callback_port,
            "callback base+rank port range overflow");
    require(config.mock_remote_cache_timeout_ms > 10 && config.mock_remote_cache_timeout_ms <= INT32_MAX,
            "TCP timeout must fit signed RPC milliseconds and exceed client 10ms margin");
    const auto colon    = config.mock_remote_cache_address.rfind(':');
    address             = config.mock_remote_cache_address.substr(0, colon);
    port                = static_cast<uint32_t>(std::stoul(config.mock_remote_cache_address.substr(colon + 1)));
    const auto deadline = Clock::now() + std::chrono::milliseconds(config.mock_remote_cache_timeout_ms);
    require(control.init(1), "control TcpClient initialization failed");
    channel = control.getChannel(address, port);
    require(channel != nullptr, "control endpoint unreachable");
    auto effective = health(deadline);
    require(effective.error_code() == EC_SUCCESS, effective.error_message());
    limits         = getLimits(effective.limits());
    service_limits = {
        {"service_reads_per_connection", std::to_string(effective.reads_per_connection())},
        {"service_writes_per_connection", std::to_string(effective.writes_per_connection())},
        {"service_max_callback_endpoints", std::to_string(effective.max_callback_endpoints())},
        {"service_max_timeout_ms", std::to_string(effective.max_timeout_ms())},
    };
    require(uint64_t(parallel.tp_size) <= limits.max_ranks, "expected ranks exceeds service limit");
    require(config.mock_remote_cache_timeout_ms <= effective.max_timeout_ms(), "timeout exceeds service limit");
    require(config.mock_remote_cache_read_delay_ms + 10 < config.mock_remote_cache_timeout_ms
                && config.mock_remote_cache_write_delay_ms + 10 < config.mock_remote_cache_timeout_ms,
            "configured delay leaves no RPC progress time");
    require(config.mock_remote_cache_read_delay_ms <= limits.max_delay_ms
                && config.mock_remote_cache_write_delay_ms <= limits.max_delay_ms,
            "delay exceeds service limit");
    require(limits.write_slots >= parallel.tp_size && limits.max_read_operations >= parallel.tp_size,
            "every rank requires a write slot and read operation credit");
    require(effective.max_callback_endpoints() >= parallel.tp_size, "insufficient callback endpoint credits");
    credit.local_read_bytes    = (uint64_t(config.mock_remote_cache_staging_size_mb) << 20) / 2;
    credit.local_write_bytes   = (uint64_t(config.mock_remote_cache_staging_size_mb) << 20) - credit.local_read_bytes;
    credit.service_read_bytes  = limits.read_staging_bytes / parallel.tp_size;
    credit.service_write_bytes = limits.write_staging_bytes / parallel.tp_size;
    credit.first_write_slot    = uint64_t(limits.write_slots) * parallel.tp_rank / parallel.tp_size;
    credit.write_slots =
        uint64_t(limits.write_slots) * (parallel.tp_rank + 1) / parallel.tp_size - credit.first_write_slot;

    remote_mock::InitRankRequest request;
    request.set_rank(parallel.tp_rank);
    request.set_expected_ranks(parallel.tp_size);
    request.set_read_delay_ms(config.mock_remote_cache_read_delay_ms);
    request.set_write_delay_ms(config.mock_remote_cache_write_delay_ms);
    require(groups.size() <= limits.max_groups_per_rank, "too many GroupSets");
    uint64_t    source_bytes = 0;
    std::string geometry;
    for (size_t i = 0; i < groups.size(); ++i) {
        require(groups[i] && groups[i]->groupSetId() == i, "noncanonical GroupSet identity");
        const auto&      group = *groups[i];
        RemoteMockLayout layout;
        layout.group             = i;
        layout.pool              = i;
        layout.payload_bytes     = group.payloadBytes();
        layout.storage_bytes     = group.storageBytes();
        layout.stride_bytes      = (layout.storage_bytes + 15) & ~uint64_t(15);
        layout.crc_enabled       = group.crcEnabled();
        layout.crc_footer_offset = layout.crc_enabled ? layout.storage_bytes - 4 : 0;
        layout.source_count      = config.mock_remote_cache_source_blocks;
        const auto& first        = group.group(group.groupTags().front());
        layout.dtype             = "dtype:" + std::to_string(static_cast<int>(first.spec->memoryLayoutDType()));
        require(first.cacheKeyTokenStride() <= UINT32_MAX, "token stride exceeds wire capacity");
        layout.tokens_per_block = first.cacheKeyTokenStride();
        layout.cp_world_size    = parallel.prefill_cp_config.kv_cache_sharded ? parallel.tp_size : 1;
        layout.cp_rank          = parallel.prefill_cp_config.kv_cache_sharded ? parallel.tp_rank : 0;
        layout.attention_window = first.policy.sliding_window_size;
        for (const auto& tag : group.groupTags()) {
            geometry += tag + ":" + group.group(tag).spec->fingerprint() + ";";
            if (group.group(tag).spec->memoryLayoutDType() != first.spec->memoryLayoutDType())
                layout.dtype = "mixed";
        }
        for (const auto& tile :
             getDeviceHostCopyTiles(group, std::vector<BlockIdxType>(group.devicePools().size(), 0))) {
            layout.member_offsets.push_back(tile.host_offset);
            geometry += std::to_string(tile.member_group_id) + ":" + std::to_string(tile.local_layer_index) + ":"
                        + std::to_string(tile.buffer.size_bytes) + ";";
        }
        auto valid = validateRemoteMockLayout(layout, limits);
        require(valid.ok(), std::string(valid.message()));
        require(layout.stride_bytes <= limits.source_bytes / layout.source_count,
                "source record bank exceeds service source budget");
        require(source_bytes <= limits.source_bytes - layout.stride_bytes * layout.source_count,
                "rank source bank sum exceeds service budget");
        source_bytes += layout.stride_bytes * layout.source_count;
        require(source_bytes <= limits.source_bytes, "rank source banks exceed service source budget");
        max_record = std::max(max_record, layout.stride_bytes);
        auto out   = request.add_groups();
#define FIELD(name) out->set_##name(layout.name)
        FIELD(group);
        FIELD(pool);
        FIELD(payload_bytes);
        FIELD(storage_bytes);
        FIELD(stride_bytes);
        FIELD(crc_enabled);
        FIELD(crc_format_version);
        FIELD(crc_footer_offset);
        FIELD(source_count);
        FIELD(alignment);
        FIELD(dtype);
        FIELD(tokens_per_block);
        FIELD(cp_world_size);
        FIELD(cp_rank);
        FIELD(attention_window);
#undef FIELD
        for (const auto offset : layout.member_offsets)
            out->add_member_offsets(offset);
        layouts.push_back(std::move(layout));
    }
    geometry += request.SerializeAsString();
    layout_hash = 1469598103934665603ULL;
    for (const auto byte : geometry)
        layout_hash = (layout_hash ^ uint8_t(byte)) * 1099511628211ULL;
    require(layout_hash != 0, "zero layout hash");
    request.set_layout_hash(layout_hash);

    CacheLoadRequest sizing;
    fillReadEnvelope(sizing, std::string(96, 'r'), config.mock_remote_cache_timeout_ms);
    auto singleton = sizing.add_blocks();
    singleton->set_key(std::string(kRemoteMockMaxKeyBytes, 'r'));
    singleton->set_len(UINT32_MAX);
    const auto overhead = serviceCharge(sizing, 1, 1, limits) - 2;
    require(credit.service_read_bytes > overhead + 2 && credit.service_write_bytes >= 29186,
            "fair per-rank service staging cannot fit a valid singleton");
    require(max_record < credit.local_read_bytes && max_record <= credit.local_write_bytes,
            "staging halves cannot hold one complete record");
    credit.read_operations =
        std::min<uint64_t>({uint64_t(config.mock_remote_cache_max_inflight),
                            limits.max_read_operations / uint64_t(parallel.tp_size),
                            effective.reads_per_connection(),
                            credit.service_read_bytes / (overhead + 2),
                            (credit.local_read_bytes - credit.local_read_bytes / 8) / (max_record + overhead + 2)});
    require(credit.read_operations > 0, "no per-rank read progress credits");
    const uint64_t local_share       = credit.local_read_bytes / credit.read_operations;
    const uint64_t local_allowance   = local_share - local_share / 8;
    const uint64_t service_allowance = credit.service_read_bytes / credit.read_operations;
    require(local_allowance > max_record + overhead && service_allowance > overhead,
            "record plus metadata exceeds per-operation read staging");
    credit.max_slice_bytes = std::min<uint64_t>(
        {limits.max_rpc_bytes, (service_allowance - overhead) / 2, (local_allowance - max_record - overhead) / 2});
    require(credit.max_slice_bytes > 0, "no payload fits negotiated staging credits");
    credit.max_batch_records = std::min<uint64_t>(
        {8, limits.max_blocks_per_rpc, (local_allowance - overhead - 2 * credit.max_slice_bytes) / max_record});
    require(credit.max_batch_records > 0, "no record batch fits local staging");

    CacheTransferRequest write_sizing;
    write_sizing.set_request_id(std::string(96, 'w'));
    write_sizing.set_client_ip(autil::NetUtil::getBindIp());
    write_sizing.set_client_port(65535);
    write_sizing.set_rdma_client_port(0);
    write_sizing.set_timeout_ms(config.mock_remote_cache_timeout_ms - 10);
    auto info = write_sizing.mutable_transfer_info();
    info->set_partition_count(1);
    info->set_partition_id(0);
    info->add_local_keys(std::string(96, 'w'));
    info->add_remote_keys(std::string(kRemoteMockMaxKeyBytes, 'w'));
    auto write_block = write_sizing.add_blocks();
    write_block->set_key(std::string(96, 'w'));
    write_block->set_len(UINT32_MAX);
    write_block->set_addr(UINT64_MAX);
    const auto write_overhead = serviceCharge(write_sizing, 1, 1, limits) - 2;
    credit.write_operations   = std::min<uint64_t>({uint64_t(config.mock_remote_cache_max_inflight),
                                                    credit.write_slots,
                                                    effective.writes_per_connection(),
                                                    credit.service_write_bytes / (write_overhead + 2),
                                                    credit.local_write_bytes / (max_record + 16 + write_overhead + 2)});
    require(credit.write_operations > 0, "no per-rank write progress credits including full record retention");
    credit.max_write_slice_bytes = std::min<uint64_t>(
        {limits.max_rpc_bytes,
         limits.write_slot_bytes,
         (credit.service_write_bytes / credit.write_operations - write_overhead) / 2,
         (credit.local_write_bytes / credit.write_operations - max_record - 16 - write_overhead) / 2});
    require(credit.max_write_slice_bytes > 0, "no write payload fits fair staging credits");

    if (parallel.tp_rank == 0) {
        require(!effective.has_session(), "another engine session is active");
        require(effective.last_generation() < UINT64_MAX, "session generation exhausted");
        identity = {effective.server_epoch(),
                    mixPosition(uint64_t(currentTimeUs()) ^ uint64_t(getpid())),
                    effective.last_generation() + 1};
        if (!identity.session_id)
            identity.session_id = 1;
    } else {
        std::mutex                   wait_mutex;
        std::condition_variable      wait_cv;
        std::unique_lock<std::mutex> wait_lock(wait_mutex);
        while (!effective.has_session()) {
            remaining(deadline);
            wait_cv.wait_for(wait_lock, std::chrono::milliseconds(5));
            effective = health(deadline);
        }
        require(!effective.ready() && effective.expected_ranks() == parallel.tp_size,
                "discovered session does not match this initializing TP group");
        identity = getSession(effective.session());
    }
    setSession(request.mutable_session(), identity);
    RTP_LLM_LOG_INFO("remote mock rank=%ld source_bytes=%lu local_staging=%lu/%lu service_staging=%lu/%lu "
                     "read_ops=%u write_slots=%u slice=%u",
                     parallel.tp_rank,
                     source_bytes,
                     credit.local_read_bytes,
                     credit.local_write_bytes,
                     credit.service_read_bytes,
                     credit.service_write_bytes,
                     credit.read_operations,
                     credit.write_slots,
                     credit.max_slice_bytes);
    auto registered = call(request, &remote_mock::RemoteMockControlService_Stub::initRank, deadline);
    require(registered.error_code() == EC_SUCCESS, registered.error_message());
    require(getSession(registered.session()) == identity && registered.sources_size() == int(groups.size()),
            "registration returned wrong identity/source count");
    sources.resize(groups.size());
    std::vector<bool> seen(groups.size(), false);
    for (const auto& source : registered.sources()) {
        require(source.group() < groups.size() && source.rank() == parallel.tp_rank && !seen[source.group()]
                    && source.pool() == layouts[source.group()].pool && source.layout_hash() == layout_hash
                    && source.source_count() == layouts[source.group()].source_count,
                "registration source descriptor mismatch");
        seen[source.group()]    = true;
        sources[source.group()] = {source.rank(),
                                   source.group(),
                                   source.pool(),
                                   source.source_count(),
                                   source.layout_hash(),
                                   source.record_crc32c()};
    }
    std::mutex                   wait_mutex;
    std::condition_variable      wait_cv;
    std::unique_lock<std::mutex> wait_lock(wait_mutex);
    while (!registered.ready()) {
        remaining(deadline);
        wait_cv.wait_for(wait_lock, std::chrono::milliseconds(5));
        registered = health(deadline);
        require(registered.has_session() && getSession(registered.session()) == identity,
                "collective initialization rolled back");
    }
    require(registered.expected_ranks() == parallel.tp_size && registered.registered_ranks() == parallel.tp_size,
            "READY rank count mismatch");
    require(registered.read_delay_ms() == config.mock_remote_cache_read_delay_ms
                && registered.write_delay_ms() == config.mock_remote_cache_write_delay_ms,
            "effective session delay mismatch");
    CacheStoreInitParams tcp;
    tcp.rdma_mode                    = false;
    tcp.listen_port                  = config.mock_remote_cache_callback_port + parallel.world_rank;
    tcp.device_id                    = groups.front()->devicePools().front()->deviceIndex();
    tcp.thread_count                 = 2;
    tcp.queue_size                   = std::max<uint32_t>(16, credit.read_operations + credit.write_operations);
    tcp.messager_io_thread_count     = 1;
    tcp.messager_worker_thread_count = 2;
    store                            = NormalCacheStore::createNormalCacheStore(tcp);
    require(store != nullptr, "dedicated TCP CacheStore/callback listener initialization failed");
    write_slots.assign(credit.write_slots, false);
    probe(deadline);
    worker = std::make_unique<BlockTreeTaskPool>(1, credit.read_operations, "RemoteMockCrc");
    require(worker->start(), "CRC continuation worker initialization failed");
}

void RemoteMockAdapter::Impl::probe(Clock::time_point deadline) {
    auto bytes       = std::make_shared<AlignedHostMemory>(16, 16, "remote_mock_callback_probe");
    bytes->data()[0] = 0xa7;
    RemoteMockHostTransfer plan{{{0, 0, 0, std::shared_ptr<void>(bytes, bytes->data()), 1}}, deadline, 32};
    auto                   complete = std::make_shared<std::promise<ErrorInfo>>();
    auto                   result   = complete->get_future();
    const auto operation = transferHost(plan, [complete](ErrorInfo error) { complete->set_value(std::move(error)); });
    require(operation != 0, "callback probe admission rejected");
    if (result.wait_until(deadline) != std::future_status::ready) {
        std::shared_ptr<RemoteStoreTask> task;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto                        found = host_active.find(operation);
            if (found != host_active.end()) {
                std::lock_guard<std::mutex> operation_lock(found->second->mutex);
                task = found->second->task;
            }
        }
        if (task)
            task->cancel(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "callback probe deadline expired"));
    }
    // This callback includes producer completion and the existing bounded RPC drain.
    // The external startup owner waits; no scheduler/network worker does so.
    require(result.wait_for(std::chrono::milliseconds(config.mock_remote_cache_timeout_ms))
                == std::future_status::ready,
            "callback probe failed to drain by its bounded transport deadline");
    const auto                   error = result.get();
    std::unique_lock<std::mutex> lock(mutex);
    require(idle.wait_until(lock, deadline, [&] { return host_active.find(operation) == host_active.end(); }),
            "callback probe cleanup exceeded startup deadline");
    require(error.ok(), "callback endpoint probe failed: " + error.ToString());
}

uint64_t RemoteMockAdapter::Impl::transferHost(const RemoteMockHostTransfer& plan, DoneCallback callback) {
    if (!callback)
        return 0;
    std::shared_ptr<HostOperation>    op;
    std::shared_ptr<HostBackingLease> backing;
    bool                              admitted = false;
    try {
        const auto timeout = remaining(plan.deadline);
        require(!plan.slices.empty() && plan.slices.size() <= limits.max_blocks_per_rpc,
                "invalid Host transfer slice count");
        uint64_t bytes = 0;
        for (const auto& slice : plan.slices) {
            require(slice.group < layouts.size() && slice.bytes && slice.length > 0, "invalid Host transfer slice");
            require(slice.offset <= layouts[slice.group].storage_bytes
                        && slice.length <= layouts[slice.group].storage_bytes - slice.offset,
                    "Host transfer slice exceeds its frozen record");
            bytes += slice.length;
            require(bytes <= limits.max_rpc_bytes && bytes <= limits.write_slot_bytes,
                    "Host transfer exceeds RPC or write-slot capacity");
        }
        require(plan.retained_bytes >= bytes
                    && plan.retained_bytes <= credit.local_write_bytes / credit.write_operations,
                "Host transfer must charge its full retained allocation");
        op = std::make_shared<HostOperation>();
        {
            std::lock_guard<std::mutex> lock(mutex);
            require(accepting
                        && static_cast<size_t>(std::count(write_slots.begin(), write_slots.end(), true))
                               < credit.write_operations,
                    "Host transfer admission stopped or exhausted");
            const auto slot = std::find(write_slots.begin(), write_slots.end(), false);
            require(slot != write_slots.end() && next_operation != UINT64_MAX && next_write_sequence != UINT64_MAX,
                    "Host transfer identity/slot exhausted");
            op->slot                    = slot - write_slots.begin();
            op->id                      = next_operation;
            op->sequence                = next_write_sequence;
            auto&            request_id = op->request_id;
            constexpr size_t max_digits = std::numeric_limits<uint64_t>::digits10 + 1;
            request_id.reserve(sizeof("rmw---") - 1 + 3 * max_digits);
            request_id.append("rmw-")
                .append(std::to_string(identity.session_id))
                .append("-")
                .append(std::to_string(parallel.tp_rank))
                .append("-")
                .append(std::to_string(op->sequence));
            host_active.emplace(op->id, op);
            // All potentially allocating admission work completed under the lock.
            op->callback = std::move(callback);
            *slot        = true;
            ++next_operation;
            ++next_write_sequence;
            admitted = true;
        }
        backing           = std::make_shared<HostBackingLease>();
        backing->released = [this, weak = std::weak_ptr<HostOperation>(op)] {
            if (auto retained = weak.lock()) {
                {
                    std::lock_guard<std::mutex> lock(retained->mutex);
                    retained->backing_drained = true;
                }
                settleHost(retained);
            }
        };
        op->backing_drained = false;  // Not published to transport yet.
        backing->owners.reserve(plan.slices.size());
        for (const auto& slice : plan.slices)
            backing->owners.push_back(slice.bytes);
        auto request = std::make_shared<RemoteStoreRequest>(
            op->request_id, op->request_id, address, port, 0, currentTimeUs() + int64_t(timeout) * 1000, 1, 0, 1, 0);
        CacheTransferRequest envelope;
        envelope.set_request_id(op->request_id);
        envelope.set_client_ip(autil::NetUtil::getBindIp());
        envelope.set_client_port(config.mock_remote_cache_callback_port + parallel.world_rank);
        envelope.set_rdma_client_port(0);
        envelope.set_timeout_ms(timeout - 10);
        auto info = envelope.mutable_transfer_info();
        info->set_partition_count(1);
        info->set_partition_id(0);
        for (size_t index = 0; index < plan.slices.size(); ++index) {
            const auto&   slice = plan.slices[index];
            const auto    local = op->request_id + "-" + std::to_string(index);
            RemoteMockKey target;
            target.write        = true;
            target.session      = identity;
            target.layout_hash  = layout_hash;
            target.rank         = parallel.tp_rank;
            target.group        = slice.group;
            target.pool         = layouts[slice.group].pool;
            target.index        = credit.first_write_slot + op->slot;
            target.operation    = op->sequence;
            target.record_index = slice.record_index;
            target.offset       = slice.offset;
            target.length       = slice.length;
            {
                BackgroundProfilingScope crc("cache.remote_mock.write.crc");
                target.checksum = remoteMockCrc32c(static_cast<const uint8_t*>(slice.bytes.get()), slice.length);
            }
            const auto remote = encodeRemoteMockKey(target);
            require(std::none_of(request->buffer_pairs.begin(),
                                 request->buffer_pairs.end(),
                                 [&](const auto& entry) { return entry.second == remote; }),
                    "duplicate Host target range");
            request->buffer_pairs[local] = remote;
            info->add_local_keys(local);
            info->add_remote_keys(remote);
            auto block = envelope.add_blocks();
            block->set_key(local);
            block->set_len(slice.length);
            block->set_addr(reinterpret_cast<uintptr_t>(slice.bytes.get()));
            op->buffers.push_back(std::make_shared<BlockBuffer>(
                local, std::shared_ptr<void>(backing, slice.bytes.get()), slice.length, false, true));
        }
        const auto charge = serviceCharge(envelope, bytes, plan.slices.size(), limits);
        require(charge <= credit.service_write_bytes / credit.write_operations
                    && charge <= credit.local_write_bytes / credit.write_operations
                    && plan.retained_bytes <= credit.local_write_bytes / credit.write_operations - charge,
                "Host transfer exceeds fair dynamic metadata/staging credits");
        op->registration = store->regUserBuffersScoped(op->buffers);
        require(op->registration != nullptr, "Host snapshot registration failed");
        RemoteStoreTask::DoneCallback done = [op](const ErrorInfo& error) {
            std::lock_guard<std::mutex> lock(op->mutex);
            // Preserve the status code even if copying its optional message fails.
            try {
                op->result = error;
            } catch (const std::bad_alloc&) {
                op->result = ErrorInfo(error.code(), std::string{});
            }
        };
        RemoteStoreTask::DrainedCallback drained = [this, op] {
            {
                std::lock_guard<std::mutex> lock(op->mutex);
                op->rpc_drained = true;
            }
            finishHost(op);
        };
        // Construct both hooks before task registration. Attach them together
        // on the concrete task before the watch or cancellation can observe it.
        auto task = store->submitRemoteStoreTaskWithCallbacks(
            request,
            std::make_shared<CacheStoreRemoteStoreMetricsCollector>(nullptr, op->buffers.size()),
            std::move(done),
            std::move(drained));
        if (!task)
            throw std::runtime_error("Host transfer task rejected");
        {
            std::lock_guard<std::mutex> lock(op->mutex);
            op->task = task;
        }
        if (op->cancelled.load())
            task->cancel(ErrorInfo(ErrorCode::CANCELLED, "Host transfer cancelled before publication"));
        auto published = std::make_shared<RequestBlockBuffer>(op->request_id);
        published->addBlocks(op->buffers);
        BackgroundProfilingScope submit("cache.remote_mock.write.submit");
        store->store(published, [this, op](bool ok, CacheStoreErrorCode) {
            if (!ok)
                op->task->cancel(failed("Host snapshot publication failed"));
            {
                std::lock_guard<std::mutex> lock(op->mutex);
                op->published = true;
            }
            finishHost(op);
        });
        return op->id;
    } catch (const std::exception& error) {
        if (admitted) {
            std::shared_ptr<RemoteStoreTask> task;
            {
                std::lock_guard<std::mutex> lock(op->mutex);
                task          = op->task;
                op->result    = admissionFailure(error);
                op->published = true;
                if (!task)
                    op->rpc_drained = true;
            }
            if (task)
                task->cancel(ErrorInfo(ErrorCode::CANCELLED, std::string{}));
            finishHost(op);
        } else {
            try {
                callback(admissionFailure(error));
            } catch (...) {
                RTP_LLM_LOG_WARNING("remote mock Host rejection callback threw");
            }
        }
        return 0;
    }
}

void RemoteMockAdapter::Impl::finishHost(const std::shared_ptr<HostOperation>& op) {
    ErrorInfo                        result;
    std::shared_ptr<RemoteStoreTask> task;
    {
        std::lock_guard<std::mutex> lock(op->mutex);
        if (op->cleanup_started || !op->published || !op->rpc_drained)
            return;
        op->cleanup_started = true;
        result              = std::move(op->result);
        task                = std::move(op->task);
    }
    if (op->registration)
        op->registration->reset();
    if (task) {
        store->releaseRemoteStoreTask(task);
        if (!store->releaseScopedRequest(op->request_id))
            result = failed("Host request cleanup failed");
    }
    // These releases can synchronously destroy the backing lease and reenter
    // settleHost. Never hold the operation/adapter mutex while dropping owners.
    op->buffers.clear();
    task.reset();
    {
        std::lock_guard<std::mutex> lock(op->mutex);
        op->result       = std::move(result);
        op->cleanup_done = true;
    }
    settleHost(op);
}

void RemoteMockAdapter::Impl::settleHost(const std::shared_ptr<HostOperation>& op) {
    ErrorInfo result;
    {
        std::lock_guard<std::mutex> lock(op->mutex);
        if (op->finished || !op->cleanup_done || !op->backing_drained)
            return;
        op->finished = true;
        result       = std::move(op->result);
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        // Backing aliases and registration have drained. Make the slot reusable
        // before continuation admission, but retain host_active through callback return.
        write_slots[op->slot] = false;
    }
    auto callback = std::move(op->callback);
    try {
        BackgroundProfilingScope ack("cache.remote_mock.write.ack_callback");
        callback(std::move(result));
    } catch (...) {
        RTP_LLM_LOG_WARNING("remote mock Host transfer callback threw");
    }
    callback = nullptr;
    std::list<std::function<void()>> ready;
    {
        std::lock_guard<std::mutex> lock(mutex);
        host_active.erase(op->id);
        if (active.empty() && host_active.empty())
            ready.swap(drained);
    }
    idle.notify_all();
    for (auto& callback : ready) {
        try {
            callback();
        } catch (...) {
            RTP_LLM_LOG_WARNING("remote mock Host drain callback threw");
        }
    }
}

void RemoteMockAdapter::Impl::closeSession() noexcept {
    if (!channel || !identity.session_id)
        return;
    try {
        remote_mock::SessionRequest request;
        setSession(request.mutable_session(), identity);
        call(request,
             &remote_mock::RemoteMockControlService_Stub::close,
             Clock::now() + std::chrono::milliseconds(config.mock_remote_cache_timeout_ms));
    } catch (...) {
        RTP_LLM_LOG_WARNING("remote mock session close failed");
    }
}

void RemoteMockAdapter::Impl::finish(const std::shared_ptr<Operation>& op, ErrorInfo error) {
    if (op->finished.exchange(true))
        return;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (error.ok())
            ++counters.read_completed;
        else
            ++counters.read_failed;
    }
    auto callback = std::move(op->callback);
    try {
        callback(std::move(error));
    } catch (...) {
        RTP_LLM_LOG_WARNING("remote mock terminal callback threw");
    }
    callback = nullptr;
    op->staging.reset();
    op->records.clear();
    op->plan.target_owner.reset();
    std::list<std::function<void()>> ready;
    {
        std::lock_guard<std::mutex> lock(mutex);
        active.erase(op->id);
        if (active.empty() && host_active.empty())
            ready.swap(drained);
    }
    idle.notify_all();
    for (auto& callback : ready) {
        try {
            callback();
        } catch (...) {
            RTP_LLM_LOG_WARNING("remote mock drain callback threw");
        }
    }
}

void RemoteMockAdapter::Impl::pump(const std::shared_ptr<Operation>& op) {
    try {
        checkOperation(*op);
        if (op->next_target == op->plan.targets.size()) {
            finish(op, ErrorInfo::OkStatus());
            return;
        }
        op->records.clear();
        const auto group  = op->plan.targets[op->next_target].source.group;
        const auto stride = layouts[group].stride_bytes;
        for (; op->next_target < op->plan.targets.size() && op->records.size() < credit.max_batch_records;
             ++op->next_target) {
            const auto& target = op->plan.targets[op->next_target];
            if (target.source.group != group)
                break;
            if (std::any_of(op->records.begin(), op->records.end(), [&](const Record& record) {
                    return record.target.source.index == target.source.index;
                }))
                break;  // Repeated source still needs a distinct TCP load; no payload reuse.
            op->records.push_back({target, op->records.size() * stride, 0});
        }
        op->staging = std::make_shared<AlignedHostMemory>(op->records.size() * stride, 16, "remote_mock_read");
        read(op);
    } catch (const std::exception& error) {
        finish(op, failed(error.what()));
    } catch (...) {
        finish(op, failed("unexpected load exception"));
    }
}

void RemoteMockAdapter::Impl::read(const std::shared_ptr<Operation>& op) {
    try {
        checkOperation(*op);
        const auto timeout = remaining(op->plan.deadline);
        const auto id      = "rm-" + std::to_string(identity.session_id) + "-" + std::to_string(parallel.tp_rank) + "-"
                        + std::to_string(op->id) + "-" + std::to_string(op->rpc_sequence++);
        auto             request = std::make_shared<RequestBlockBuffer>(id);
        CacheLoadRequest envelope;
        fillReadEnvelope(envelope, id, timeout);
        uint64_t                                 bytes = 0;
        std::vector<std::pair<size_t, uint32_t>> ranges;
        for (size_t i = 0; i < op->records.size(); ++i) {
            const auto& record  = op->records[i];
            const auto& source  = record.target.source;
            const auto  storage = layouts[source.group].storage_bytes;
            if (record.received == storage)
                continue;
            if (bytes >= credit.max_slice_bytes || ranges.size() >= limits.max_blocks_per_rpc)
                break;
            const auto length = uint32_t(std::min<uint64_t>(storage - record.received, credit.max_slice_bytes - bytes));
            RemoteMockKey key;
            key.session        = source.session;
            key.layout_hash    = source.layout_hash;
            key.rank           = source.rank;
            key.group          = source.group;
            key.pool           = source.pool;
            key.index          = source.index;
            key.offset         = record.received;
            key.length         = length;
            key.checksum       = source.checksum;
            const auto encoded = encodeRemoteMockKey(key);
            auto       block   = envelope.add_blocks();
            block->set_key(encoded);
            block->set_len(length);
            const auto charge = serviceCharge(envelope, bytes + length, ranges.size() + 1, limits);
            const auto local  = op->records.size() * layouts[source.group].stride_bytes + charge
                               + credit.local_read_bytes / credit.read_operations / 8;
            if (charge > credit.service_read_bytes / credit.read_operations
                || local > credit.local_read_bytes / credit.read_operations) {
                envelope.mutable_blocks()->RemoveLast();
                require(!ranges.empty(), "negotiated singleton charge does not fit");
                break;
            }
            request->addBlock(encoded,
                              std::shared_ptr<void>(op, op->staging->data() + record.offset + record.received),
                              length,
                              false,
                              true);
            ranges.emplace_back(i, length);
            bytes += length;
        }
        if (ranges.empty()) {
            copied(op);
            return;
        }
        BackgroundProfilingScope submit("cache.remote_mock.load.submit");
        store->load(
            request,
            [this,
             op,
             ranges = std::move(ranges),
             bytes,
             post_error = failed("TCP load continuation allocation failed")](
                bool ok, CacheStoreErrorCode code) mutable noexcept {
                try {
                    BackgroundProfilingScope complete("cache.remote_mock.load.completion");
                    if (ok) {
                        std::lock_guard<std::mutex> lock(mutex);
                        counters.read_payload_bytes += bytes;
                    }
                    enqueue([this, op, ranges = std::move(ranges), ok, code] {
                        if (!ok) {
                            finish(op, failed("TCP load failed: " + CacheStoreErrorCodeToString(code)));
                            return;
                        }
                        for (const auto& [index, length] : ranges)
                            op->records[index].received += length;
                        read(op);
                    });
                } catch (...) {
                    // The TCP terminal follows its final Host memcpy and cannot
                    // touch these views again. No H2D has been submitted for
                    // these bytes. The operation owns all request staging.
                    finish(op, std::move(post_error));
                }
            },
            address,
            port,
            0,
            timeout,
            1,
            0);
    } catch (const std::exception& error) {
        finish(op, failed(error.what()));
    } catch (...) {
        finish(op, failed("unexpected read continuation exception"));
    }
}

void RemoteMockAdapter::Impl::copied(const std::shared_ptr<Operation>& op) {
    try {
        checkOperation(*op);
        {
            BackgroundProfilingScope crc("cache.remote_mock.crc");
            for (const auto& record : op->records) {
                const auto& source = record.target.source;
                const auto& layout = layouts[source.group];
                const auto* data   = op->staging->data() + record.offset;
                require(record.received == layout.storage_bytes, "record reassembly incomplete");
                require(remoteMockCrc32c(data, layout.storage_bytes) == source.checksum, "whole-record CRC mismatch");
                const auto status = validateRemoteMockRecord(layout, data, layout.storage_bytes);
                require(status.ok(), "packed cache record CRC mismatch");
            }
        }
        std::vector<TransferDescriptor> descriptors;
        std::vector<HostBufferView>     hosts;
        for (const auto& record : op->records) {
            const auto         group = record.target.source.group;
            TransferDescriptor desc;
            desc.group_set_id  = group;
            desc.source_tier   = Tier::HOST;
            desc.target_tier   = Tier::DEVICE;
            desc.target_blocks = record.target.device_blocks;
            descriptors.push_back(std::move(desc));
            hosts.push_back(
                {op->staging->data() + record.offset, layouts[group].payload_bytes, layouts[group].stride_bytes});
        }
        transfer->executeHostBuffers(
            TransferTask(std::move(descriptors), std::chrono::milliseconds(remaining(op->plan.deadline))),
            std::move(hosts),
            op,
            [this, op, post_error = failed("physical completion continuation allocation failed")](
                TransferResult result) mutable noexcept {
                try {
                    enqueue([this, op, result = std::move(result)] {
                        if (!result.ok()) {
                            finish(op, result.status);
                            return;
                        }
                        op->staging.reset();
                        pump(op);
                    });
                } catch (...) {
                    finish(op, std::move(post_error));
                }
            });
    } catch (const std::exception& error) {
        finish(op, failed(error.what()));
    } catch (...) {
        finish(op, failed("unexpected CRC/H2D exception"));
    }
}

RemoteMockAdapter::RemoteMockAdapter(std::unique_ptr<Impl> impl): impl_(std::move(impl)) {}
std::shared_ptr<RemoteMockAdapter>
RemoteMockAdapter::create(const KVCacheConfig&                        config,
                          const ParallelismConfig&                    parallel,
                          std::vector<GroupSetPtr>                    groups,
                          std::shared_ptr<PerRankBlockTransferEngine> transfer_engine) {
    auto impl      = std::make_unique<Impl>();
    impl->config   = config;
    impl->parallel = parallel;
    impl->groups   = std::move(groups);
    impl->transfer = std::move(transfer_engine);
    auto adapter   = std::shared_ptr<RemoteMockAdapter>(new RemoteMockAdapter(std::move(impl)));
    try {
        adapter->impl_->initialize();
    } catch (...) {
        adapter->impl_->closeSession();
        throw;
    }
    return adapter;
}
RemoteMockAdapter::~RemoteMockAdapter() {
    stopAccepting();
    impl_->writeback.reset();  // Drains queued captures and inter-RPC continuations first.
    {
        std::unique_lock<std::mutex> lock(impl_->mutex);
        impl_->idle.wait(lock, [&] { return impl_->active.empty() && impl_->host_active.empty(); });
    }
    if (impl_->worker) {
        impl_->worker->waitForIdle();
        impl_->worker.reset();
    }
    if (impl_->parallel.tp_rank == 0)
        impl_->closeSession();
}
RemoteMockSourceHandle RemoteMockAdapter::freezeSource(size_t group, uint64_t admission, uint64_t logical) const {
    require(group < impl_->sources.size(), "unknown source GroupSet");
    require(admission <= UINT64_MAX - logical, "frozen source position overflow");
    const auto& source = impl_->sources[group];
    return {impl_->identity,
            source.layout_hash,
            source.rank,
            source.group,
            source.pool,
            uint32_t(mixPosition(admission + logical) % source.source_count),
            source.record_checksum,
            admission,
            logical};
}
uint64_t RemoteMockAdapter::load(const RemoteMockLoadPlan& plan, DoneCallback callback) {
    if (!callback)
        return 0;
    std::shared_ptr<Impl::Operation> op;
    bool                             admitted = false;
    try {
        require(plan.target_owner && !plan.targets.empty(), "load requires targets and a target owner");
        impl_->remaining(plan.deadline);
        const auto allowance = impl_->credit.local_read_bytes / impl_->credit.read_operations;
        require(plan.targets.size() <= allowance / 4096, "load target metadata exceeds bounded admission capacity");
        uint64_t metadata_bytes = 0;
        for (const auto& target : plan.targets) {
            metadata_bytes +=
                2 * sizeof(RemoteMockLoadTarget) + 2 * target.device_blocks.size() * sizeof(BlockIdxType) + 512;
            require(metadata_bytes <= allowance / 8, "load target metadata exceeds its reserved staging share");
            const auto& source = target.source;
            require(source.group < impl_->groups.size(), "unknown target GroupSet");
            const auto expected = freezeSource(source.group, source.admission_offset, source.logical_index);
            require(source.session == expected.session && source.layout_hash == expected.layout_hash
                        && source.rank == expected.rank && source.pool == expected.pool
                        && source.index == expected.index && source.checksum == expected.checksum,
                    "stale or altered frozen source handle");
            const auto& pools = impl_->groups[source.group]->devicePools();
            require(target.device_blocks.size() == pools.size(), "target member count mismatch");
            for (size_t i = 0; i < pools.size(); ++i)
                require(pools[i]->validBlock(target.device_blocks[i]), "invalid target device block");
        }
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            require(impl_->accepting && impl_->active.size() < impl_->credit.read_operations,
                    "read admission stopped or exhausted");
            require(impl_->next_operation != UINT64_MAX, "operation identity exhausted");
            op       = std::make_shared<Impl::Operation>();
            op->id   = impl_->next_operation;
            op->plan = plan;
            impl_->active.emplace(op->id, op);
            op->callback = std::move(callback);
            ++impl_->next_operation;
            admitted = true;
            ++impl_->counters.read_submitted;
        }
        impl_->enqueue([impl = impl_.get(), op] { impl->pump(op); });
        return op->id;
    } catch (const std::exception& error) {
        if (admitted) {
            impl_->finish(op, admissionFailure(error));
        } else {
            try {
                callback(admissionFailure(error));
            } catch (...) {
                RTP_LLM_LOG_WARNING("remote mock load rejection callback threw");
            }
        }
        return 0;
    }
}
uint64_t RemoteMockAdapter::transferHost(const RemoteMockHostTransfer& plan, DoneCallback callback) {
    return impl_->transferHost(plan, std::move(callback));
}
bool RemoteMockAdapter::cancel(uint64_t operation) {
    std::shared_ptr<RemoteStoreTask> host;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto                  read = impl_->active.find(operation);
        if (read != impl_->active.end()) {
            read->second->cancelled.store(true);
            return true;
        }
        const auto write = impl_->host_active.find(operation);
        if (write == impl_->host_active.end())
            return false;
        write->second->cancelled.store(true);
        std::lock_guard<std::mutex> operation_lock(write->second->mutex);
        host = write->second->task;
    }
    if (host)
        host->cancel(ErrorInfo(ErrorCode::CANCELLED, "remote mock Host transfer cancelled"));
    return true;
}
void RemoteMockAdapter::stopAccepting() {
    if (impl_->writeback)
        impl_->writeback->stopAccepting();
    std::vector<std::shared_ptr<RemoteStoreTask>> hosts;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->accepting = false;
        for (const auto& [_, op] : impl_->active)
            op->cancelled.store(true);
        for (const auto& [_, op] : impl_->host_active) {
            op->cancelled.store(true);
            std::lock_guard<std::mutex> operation_lock(op->mutex);
            if (op->task)
                hosts.push_back(op->task);
        }
    }
    for (const auto& task : hosts)
        task->cancel(ErrorInfo(ErrorCode::CANCELLED, "remote mock adapter stopped"));
}
void RemoteMockAdapter::initializeWriteback(const std::shared_ptr<RemoteMockAdapter>& self,
                                            bool                                      owns_device_allocations,
                                            size_t                                    max_descriptors) {
    require(self.get() == this && !impl_->writeback, "remote writer role attachment is startup-only");
    impl_->writeback = std::make_unique<RemoteMockWriteback>(
        self, impl_->transfer.get(), impl_->groups, owns_device_allocations, max_descriptors);
}
bool RemoteMockAdapter::writeback(TransferTask task, DoneCallback callback) {
    if (!impl_->writeback) {
        if (callback)
            callback(failed("remote writer was not attached"));
        return false;
    }
    return impl_->writeback->submit(std::move(task), std::move(callback));
}
size_t RemoteMockAdapter::maxWritebackDescriptors() const {
    return impl_->writeback ? impl_->writeback->maxDescriptors() : 0;
}
RemoteMockWritebackStats RemoteMockAdapter::writebackStats() const {
    return impl_->writeback ? impl_->writeback->stats() : RemoteMockWritebackStats{};
}
std::function<void()> RemoteMockAdapter::prepareDrain(std::function<void()> callback) {
    std::list<std::function<void()>> node;
    node.emplace_back(std::move(callback));
    auto finish = [this, node = std::move(node)]() mutable {
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            if (!impl_->active.empty() || !impl_->host_active.empty()) {
                impl_->drained.splice(impl_->drained.end(), node);
                return;
            }
        }
        node.front()();
    };
    if (impl_->writeback)
        return impl_->writeback->prepareDrain(std::move(finish));
    return finish;
}
void RemoteMockAdapter::drain(std::function<void()> callback) {
    if (callback)
        prepareDrain(std::move(callback))();
}
RemoteMockSession RemoteMockAdapter::session() const {
    return impl_->identity;
}
RemoteMockRankCredits RemoteMockAdapter::credits() const {
    return impl_->credit;
}
RemoteMockRuntimeSnapshot RemoteMockAdapter::runtimeSnapshot() const {
    RemoteMockRuntimeSnapshot result;
    result.options    = impl_->service_limits;
    const auto option = [&](const char* name, const auto& value) {
        std::ostringstream text;
        text << std::boolalpha << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
        result.options.emplace(name, text.str());
    };
#define OPTION(name) option(#name, impl_->config.name)
    OPTION(mock_remote_cache);
    OPTION(mock_remote_cache_address);
    OPTION(mock_remote_cache_hit_ratio);
    OPTION(mock_remote_cache_source_blocks);
    OPTION(mock_remote_cache_read_delay_ms);
    OPTION(mock_remote_cache_write_delay_ms);
    OPTION(mock_remote_cache_callback_port);
    OPTION(mock_remote_cache_staging_size_mb);
    OPTION(mock_remote_cache_max_inflight);
    OPTION(mock_remote_cache_timeout_ms);
    OPTION(mock_memory_cache);
    OPTION(mock_memory_cache_hit_ratio);
    OPTION(mock_disk_cache);
    OPTION(mock_disk_cache_hit_ratio);
#undef OPTION
    result.callback_endpoint =
        autil::NetUtil::getBindIp() + ":"
        + std::to_string(impl_->config.mock_remote_cache_callback_port + impl_->parallel.world_rank);
    result.session        = impl_->identity;
    result.layout_hash    = impl_->layout_hash;
    result.rank           = impl_->parallel.tp_rank;
    result.expected_ranks = impl_->parallel.tp_size;
    result.limits         = impl_->limits;
    result.credits        = impl_->credit;
    for (const auto& layout : impl_->layouts) {
        RemoteMockLayout compact;
        compact.group              = layout.group;
        compact.pool               = layout.pool;
        compact.payload_bytes      = layout.payload_bytes;
        compact.storage_bytes      = layout.storage_bytes;
        compact.stride_bytes       = layout.stride_bytes;
        compact.crc_enabled        = layout.crc_enabled;
        compact.crc_format_version = layout.crc_format_version;
        compact.crc_footer_offset  = layout.crc_footer_offset;
        compact.source_count       = layout.source_count;
        compact.alignment          = layout.alignment;
        compact.dtype              = layout.dtype;
        compact.tokens_per_block   = layout.tokens_per_block;
        compact.cp_world_size      = layout.cp_world_size;
        compact.cp_rank            = layout.cp_rank;
        compact.attention_window   = layout.attention_window;
        result.layouts.push_back(std::move(compact));
    }
    return result;
}
RemoteMockLimits RemoteMockAdapter::limits() const {
    return impl_->limits;
}
const std::vector<RemoteMockLayout>& RemoteMockAdapter::layouts() const {
    return impl_->layouts;
}
RemoteMockAdapterStats RemoteMockAdapter::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->counters;
}
}  // namespace rtp_llm
