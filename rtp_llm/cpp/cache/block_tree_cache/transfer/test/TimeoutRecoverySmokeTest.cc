#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTaskRecovery.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {
using namespace std::chrono_literals;
using namespace block_transfer_engine_test;
using Clock = std::chrono::steady_clock;
using State = TransferTaskRegistry::State;
int64_t monoUs(Clock::time_point value) {
    return std::chrono::duration_cast<std::chrono::microseconds>(value.time_since_epoch()).count();
}
const char* stateName(State state) {
    return state == State::BUFFER_SAFE ? "BUFFER_SAFE" : state == State::RUNNING ? "RUNNING" :
           state == State::REJECTED ? "REJECTED" : "UNKNOWN";
}
constexpr size_t kBytes = 64 * 1024;
constexpr size_t kRequests = 32;
constexpr size_t kBlocksPerRequest = 16;
constexpr size_t kBlocks = kRequests * kBlocksPerRequest;

// Pure observation around actual O_DIRECT pread/pwrite. No injected waits,
// barriers, errors, deadline changes, or synthetic completion proofs.
class ObservedPosixIO final: public DiskBlockIO {
public:
    struct Event {
        uint64_t offset;
        bool write;
        Clock::time_point begin, end;
        DiskBlockIOStatus status;
    };
    DiskBlockIOStatus openAndPreallocate(const std::string& path, size_t bytes, bool buffered) override {
        EXPECT_FALSE(buffered);
        return io_.openAndPreallocate(path, bytes, buffered);
    }
    void close() override { io_.close(); }
    std::string debugString() const override { return io_.debugString(); }
    DiskBlockIOStatus read(uint64_t offset, void* dst, size_t bytes) override {
        const auto begin = Clock::now();
        const auto result = io_.read(offset, dst, bytes);
        record({offset, false, begin, Clock::now(), result});
        return result;
    }
    DiskBlockIOStatus write(uint64_t offset, const void* src, size_t bytes) override {
        const auto begin = Clock::now();
        const auto result = io_.write(offset, src, bytes);
        record({offset, true, begin, Clock::now(), result});
        return result;
    }
    DiskBlockIOStatus read(const std::vector<DiskRead>& items) override {
        for (const auto& item : items) {
            const auto result = read(item.offset, item.buffer, item.bytes);
            if (result != DiskBlockIOStatus::OK) return result;
        }
        return DiskBlockIOStatus::OK;
    }
    DiskBlockIOStatus write(const std::vector<DiskWrite>& items) override {
        for (const auto& item : items) {
            const auto result = write(item.offset, item.buffer, item.bytes);
            if (result != DiskBlockIOStatus::OK) return result;
        }
        return DiskBlockIOStatus::OK;
    }
    std::vector<Event> events() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_;
    }
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.clear();
    }
private:
    void record(Event event) {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(event);
    }
    BlockTreePosixDiskBlockIO io_;
    mutable std::mutex mutex_;
    std::vector<Event> events_;
};

struct SmokeWorker {
    SmokeWorker(size_t rank, bool async, const std::string& directory): rank(rank) {
        host = makeHostPool(kBytes, kBlocks, true);
        device = makeTestDevicePool({{kBytes, 0}}, kBlocks, "timeout_device_rank_" + std::to_string(rank));
        auto storage = std::make_unique<ObservedPosixIO>();
        io = storage.get();
        disk = makeDiskPool(kBytes, kBlocks, directory, std::move(storage),
                            "natural_timeout_rank_" + std::to_string(rank), false, true);
        auto topology = makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, kBytes)});
        groups = {makeTestGroupSet(0, topology, topology->groupTags(), {device}, host, disk, true)};
        DiskIOExecutorConfig config;
        config.read_thread_count = 1;
        config.write_thread_count = 1;
        engine = std::make_shared<PerRankBlockTransferEngine>(
            groups, true, DeviceHostCopyOptions{}, 4, kBlocksPerRequest, 1, kRequests * 4,
            nullptr, rank, 1, async, config);
        dispatcher = std::make_shared<BlockTransferDispatcher>(engine);
        for (size_t i = 0; i < kBlocks; ++i) {
            hosts.push_back(poolMalloc(*host));
            disks.push_back(poolMalloc(*disk));
            devices.push_back(poolMalloc(*device));
            device->incTreeRef(devices.back(), BlockTreeRefType::CACHE);
            host->incTreeRef(hosts.back(), BlockTreeRefType::CACHE);
            disk->incTreeRef(disks.back(), BlockTreeRefType::CACHE);
            const auto buffer = host->blockBuffer(hosts.back());
            std::memset(buffer.addr, pattern(i), kBytes);
            encodeTestRecord(buffer.addr, kBytes);
            EXPECT_EQ(io->write(disk->blockOffset(disks.back()), buffer.addr, disk->strideBytes()), DiskBlockIOStatus::OK);
        }
        io->clear();
    }
    unsigned char pattern(size_t index) const { return static_cast<unsigned char>(31 + rank + index % 173); }
    std::vector<TransferDescriptor> descriptors(bool write, size_t request) const {
        std::vector<TransferDescriptor> result;
        for (size_t i = request * kBlocksPerRequest; i < (request + 1) * kBlocksPerRequest; ++i)
            result.push_back(write ? TransferDescriptor::hostToDisk(0, hosts[i], disks[i]) :
                                     TransferDescriptor::diskToDevice(0, disks[i], {devices[i]}));
        return result;
    }
    size_t rank;
    ObservedPosixIO* io{nullptr};
    std::shared_ptr<HostBlockPool> host;
    DiskBlockPoolPtr disk;
    DeviceBlockPoolPtr device;
    std::vector<GroupSetPtr> groups;
    std::vector<BlockIdxType> hosts, disks, devices;
    std::shared_ptr<PerRankBlockTransferEngine> engine;
    std::shared_ptr<BlockTransferDispatcher> dispatcher;
};

class SmokeService final: public RpcService::Service {
public:
    struct Observation {
        const char* stage;
        uint64_t trace;
        int64_t request;
        int64_t mono_us;
        int64_t deadline_wall_us;
        bool cancelled;
        const char* reason;
    };
    struct Completed {
        uint64_t trace;
        int64_t request;
        Clock::time_point rpc_deadline;
        TransferResult result;
    };
    explicit SmokeService(SmokeWorker& worker): worker_(worker) {}
    grpc::Status ExecuteFunction(grpc::ServerContext* server_context, const FunctionRequestPB* request,
                                 FunctionResponsePB* response) override {
        if (request->has_completion_query()) {
            auto* result = response->mutable_completion_query();
            result->set_rank(worker_.rank);
            for (const auto& key : request->completion_query().keys()) {
                auto* entry = result->add_entries();
                *entry->mutable_key() = key;
                const auto state = worker_.dispatcher->queryLocalCompletion(key.transfer_trace_id());
                entry->set_state(state == State::BUFFER_SAFE ? TRANSFER_COMPLETION_BUFFER_SAFE :
                                 state == State::RUNNING ? TRANSFER_COMPLETION_RUNNING : TRANSFER_COMPLETION_UNKNOWN);
                entry->set_admission_closed(state == State::REJECTED);
            }
            return grpc::Status::OK;
        }
        if (!request->has_mem_request()) return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "missing transfer");
        std::vector<TransferDescriptor> descriptors;
        const auto& wire = request->mem_request();
        observe("rpc_enter", *server_context, *request);
        if (!BlockTransferRequestConverter::decodeTransfer(wire, descriptors, worker_.groups)) {
            observe("rpc_rejected", *server_context, *request, "decode_failed");
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "invalid transfer");
        }
        const auto rpc_deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(
            server_context->deadline() - std::chrono::system_clock::now());
        TransferTask task(std::move(descriptors), std::chrono::milliseconds(wire.timeout_ms()), wire.transfer_request_id());
        task.setTraceId(wire.transfer_trace_id());
        observe("worker_dispatch", *server_context, *request);
        auto context = worker_.dispatcher->executePerRank(std::move(task));
        context->waitDone();
        auto result = context->result();
        BlockTransferRequestConverter::encodeTransferResult(*response->mutable_mem_response(), result);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            completed_.push_back({wire.transfer_trace_id(), wire.transfer_request_id(), rpc_deadline, std::move(result)});
        }
        observe("rpc_exit", *server_context, *request);
        cv_.notify_all();
        return grpc::Status::OK;
    }
    bool waitCompleted(uint64_t trace) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, 5s, [&] {
            return std::any_of(completed_.begin(), completed_.end(), [trace](const auto& item) { return item.trace == trace; });
        });
    }
    std::vector<Completed> completed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return completed_;
    }
    std::vector<Observation> observations() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return observations_;
    }
    size_t droppedObservations() const { return dropped_observations_.load(); }
private:
    void observe(const char* stage, const grpc::ServerContext& context, const FunctionRequestPB& request,
                 const char* reason = "") noexcept {
        try {
            const auto& wire = request.mem_request();
            const Observation event{stage, wire.transfer_trace_id(), wire.transfer_request_id(), monoUs(Clock::now()),
                std::chrono::duration_cast<std::chrono::microseconds>(context.deadline().time_since_epoch()).count(),
                context.IsCancelled(), reason};
            {
                std::lock_guard<std::mutex> lock(mutex_);
                observations_.push_back(event);
            }
            RTP_LLM_LOG_INFO("transfer_lifecycle event=native_%s trace=%llu request=%lld rank=%zu registry=%s "
                             "mono_us=%lld deadline_wall_us=%lld cancelled=%d reason=%s",
                             stage, static_cast<unsigned long long>(event.trace), static_cast<long long>(event.request),
                             worker_.rank, worker_.dispatcher->taskRegistry().epoch().c_str(),
                             static_cast<long long>(event.mono_us), static_cast<long long>(event.deadline_wall_us),
                             event.cancelled, reason);
        } catch (...) {
            // Missing diagnostic events must be visible, never treated as proof
            // that this RPC did not enter the worker.
            dropped_observations_.fetch_add(1);
        }
    }
    SmokeWorker& worker_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<Completed> completed_;
    std::vector<Observation> observations_;
    std::atomic<size_t> dropped_observations_{0};
};

struct SmokeServer {
    explicit SmokeServer(SmokeWorker& worker): service(worker) {}
    bool start() {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service);
        server = builder.BuildAndStart();
        return server && port > 0;
    }
    ~SmokeServer() { shutdown(); }
    void shutdown() {
        if (server) {
            server->Shutdown(std::chrono::system_clock::now() + 5s);
            server->Wait();
            server.reset();
        }
    }
    std::string address() const { return "127.0.0.1:" + std::to_string(port); }
    SmokeService service;
    std::unique_ptr<grpc::Server> server;
    int port{0};
};

struct BusinessCompletion {
    std::promise<TransferResult> promise;
    std::atomic<size_t> callbacks{0};
};

class LocalTimeoutRecoverySmokeTest: public ::testing::TestWithParam<bool> {
protected:
    void SetUp() override {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
            GTEST_SKIP() << "CUDA is required for the DISK -> DEVICE read route";
        // Each rank uses small device/host/disk pools; no model weights are allocated.
        std::vector<std::string> addresses;
        for (size_t rank = 0; rank < 2; ++rank) {
            workers_.push_back(std::make_unique<SmokeWorker>(rank, GetParam(), directory_.path));
            ASSERT_EQ(workers_.back()->engine->asyncEnabled(), GetParam());
            servers_.push_back(std::make_unique<SmokeServer>(*workers_.back()));
            ASSERT_TRUE(servers_.back()->start());
            addresses.push_back(servers_.back()->address());
        }
        broadcast_ = std::make_shared<BroadcastManager>(addresses);
        ASSERT_TRUE(broadcast_->init());
        auto multi = std::make_shared<MultiRankBlockTransferEngine>(workers_[0]->groups, broadcast_);
        coordinator_ = std::make_unique<BlockTransferDispatcher>(workers_[0]->engine, multi);
    }
    void TearDown() override {
        for (auto& worker : workers_) worker->engine->waitForIdle();
        coordinator_.reset();
        // Same shutdown as destruction, but keep the observation buffers alive
        // until the final snapshot. After Wait no new handler can enter.
        for (auto& server : servers_) server->shutdown();
        for (auto& worker : workers_) worker->engine->waitForIdle();
        dumpUnreleasedEvidence("server_drained");
        servers_.clear();
    }
    struct EvidenceRequest {
        uint64_t trace;
        int64_t request;
        size_t index;
        bool write;
    };
    void rememberUnreleased(const std::vector<std::shared_ptr<FrozenTask>>& records,
                            int64_t first_request, bool write) {
        unreleased_evidence_.clear();
        for (const auto& record : records) {
            if (record->snapshot().state == FrozenTask::State::RELEASED) continue;
            const auto request = record->execution().request_id;
            if (!request || *request < first_request || *request >= first_request + static_cast<int64_t>(kRequests)) {
                ADD_FAILURE() << "cannot map diagnostic trace to this round's disk blocks: " << record->debugString();
                continue;
            }
            unreleased_evidence_.push_back({std::get<uint64_t>(record->execution().trace_id), *request,
                                           static_cast<size_t>(*request - first_request), write});
        }
    }
    void dumpUnreleasedEvidence(const char* phase) const {
        if (unreleased_evidence_.empty()) return;
        for (size_t rank = 0; rank < workers_.size(); ++rank) {
            const auto& worker = *workers_[rank];
            const auto& service = servers_[rank]->service;
            const auto observations = service.observations();
            const auto completed = service.completed();
            const auto io_events = worker.io->events();
            for (const auto& request : unreleased_evidence_) {
                std::vector<uint64_t> offsets;
                for (size_t i = request.index * kBlocksPerRequest; i < (request.index + 1) * kBlocksPerRequest; ++i)
                    offsets.push_back(worker.disk->blockOffset(worker.disks[i]));
                std::ostringstream detail;
                size_t entered = 0, io_count = 0;
                for (const auto& event : observations) {
                    if (event.trace != request.trace) continue;
                    if (std::string(event.stage) == "rpc_enter") ++entered;
                    detail << "native_timeout_event phase=" << phase << " trace=" << request.trace << " rank=" << rank
                           << " stage=" << event.stage << " mono_us=" << event.mono_us
                           << " deadline_wall_us=" << event.deadline_wall_us << " cancelled=" << event.cancelled
                           << " reason=" << event.reason << '\n';
                }
                for (const auto& event : io_events) {
                    if (event.write != request.write
                        || std::find(offsets.begin(), offsets.end(), event.offset) == offsets.end()) continue;
                    ++io_count;
                    detail << "native_timeout_io phase=" << phase << " trace=" << request.trace << " rank=" << rank
                           << " offset=" << event.offset << " write=" << event.write
                           << " begin_mono_us=" << monoUs(event.begin) << " end_mono_us=" << monoUs(event.end)
                           << " status=" << static_cast<int>(event.status) << '\n';
                }
                size_t completed_count = 0;
                for (const auto& done : completed) {
                    if (done.trace != request.trace) continue;
                    ++completed_count;
                    detail << "native_timeout_worker_result phase=" << phase << " trace=" << request.trace
                           << " rank=" << rank << " code=" << static_cast<int>(done.result.status.code())
                           << " buffer_safe=" << done.result.buffer_safe
                           << " rpc_deadline_mono_us=" << monoUs(done.rpc_deadline)
                           << " message=\"" << done.result.status.ToString() << "\"\n";
                }
                const auto state = worker.dispatcher->taskRegistry().queryLocal(
                    std::vector<uint64_t>{request.trace}).front();
                std::cout << "native_timeout_evidence phase=" << phase << " engine=" << (GetParam() ? "workloop" : "legacy")
                          << " trace=" << request.trace << " request=" << request.request << " rank=" << rank
                          << " addr=" << servers_[rank]->address() << " registry=" << worker.dispatcher->taskRegistry().epoch()
                          << " state=" << stateName(state) << " rpc_enter_count=" << entered
                          << " rpc_completed_count=" << completed_count << " io_count=" << io_count
                          << " dropped_observations=" << service.droppedObservations()
                          << " snapshot_mono_us=" << monoUs(Clock::now()) << '\n' << detail.str() << std::flush;
            }
        }
    }
    void submit(bool write, size_t request, int64_t request_id, std::chrono::milliseconds timeout,
                const std::shared_ptr<BusinessCompletion>& completion) {
        TransferTask task(workers_[0]->descriptors(write, request), timeout, request_id);
        coordinator_->runTransfer(
            std::move(task),
            [completion](TransferResult result) {
                if (completion->callbacks.fetch_add(1) != 0) {
                    ADD_FAILURE() << "business completion called more than once";
                    return;
                }
                completion->promise.set_value(std::move(result));
            }, write ? "evict" : "load");
    }
    size_t recover(const std::vector<std::shared_ptr<FrozenTask>>& records) {
        FrozenTaskRecovery recovery(
            [this] { return coordinator_->frozenTasks(); },
            [this](const auto& execution, int timeout) { return TransferCompletionQuery(broadcast_).query(execution, timeout); });
        EXPECT_TRUE(recovery.start(10, 1000));
        const auto deadline = Clock::now() + 10s;
        // This waits only for production completion/recovery; no producer is
        // delayed, paused or otherwise controlled by the smoke test.
        std::mutex mutex;
        std::condition_variable cv;
        std::unique_lock<std::mutex> lock(mutex);
        while (Clock::now() < deadline && std::any_of(records.begin(), records.end(), [](const auto& record) {
            return record->snapshot().state != FrozenTask::State::RELEASED;
        })) cv.wait_for(lock, 10ms);
        recovery.stop();
        size_t released = 0;
        for (const auto& record : records) {
            if (record->snapshot().state == FrozenTask::State::RELEASED) ++released;
            EXPECT_EQ(record->snapshot().state, FrozenTask::State::RELEASED) << record->debugString();
            const auto query = record->recoverySnapshot().last_query;
            EXPECT_NE(query, nullptr);
            if (query) {
                EXPECT_TRUE(query->all_safe) << record->debugString();
            }
            EXPECT_EQ(record->failure().status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
        }
        EXPECT_TRUE(coordinator_->frozenTasks().empty());
        for (const auto& worker : workers_) {
            EXPECT_EQ(worker->host->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
            EXPECT_EQ(worker->disk->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
            EXPECT_EQ(worker->device->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
        }
        return released;
    }
    void longTimeoutControl(bool write) {
        // Ensure a stale destination cannot satisfy the data assertion.
        for (const auto& worker : workers_)
            for (size_t i = 0; i < kBlocksPerRequest; ++i) {
                const auto host = worker->host->blockBuffer(worker->hosts[i]);
                if (write) {
                    std::memset(host.addr, worker->pattern(i) + 1, kBytes);
                    encodeTestRecord(host.addr, kBytes);
                } else {
                    const auto target = worker->device->convertIndexToBuffer(0, worker->devices[i]).front();
                    ASSERT_EQ(cudaMemset(target.addr, 0xEE, kBytes), cudaSuccess);
                }
            }
        auto completion = std::make_shared<BusinessCompletion>();
        auto future = completion->promise.get_future();
        submit(write, 0, 100000, 10s, completion);
        ASSERT_EQ(future.wait_for(15s), std::future_status::ready);
        ASSERT_TRUE(future.get().ok());
        EXPECT_EQ(completion->callbacks.load(), 1u);
        EXPECT_TRUE(coordinator_->frozenTasks().empty());
        for (const auto& worker : workers_)
            for (size_t i = 0; i < kBlocksPerRequest; ++i) {
                const auto buffer = worker->host->blockBuffer(worker->hosts[i]);
                if (write) {
                    std::memset(buffer.addr, 0xEE, kBytes);
                    ASSERT_EQ(worker->io->read(worker->disk->blockOffset(worker->disks[i]), buffer.addr, worker->disk->strideBytes()),
                              DiskBlockIOStatus::OK);
                } else {
                    const auto target = worker->device->convertIndexToBuffer(0, worker->devices[i]).front();
                    ASSERT_EQ(cudaMemcpy(buffer.addr, target.addr, kBytes, cudaMemcpyDeviceToHost), cudaSuccess);
                }
                const auto* data = static_cast<const unsigned char*>(buffer.addr);
                const auto expected = static_cast<unsigned char>(worker->pattern(i) + (write ? 1 : 0));
                EXPECT_TRUE(std::all_of(data, data + kBytes, [expected](unsigned char value) { return value == expected; }));
            }
    }
    void exercise(bool write) {
        size_t queue_timeouts = 0, io_timeouts = 0, frozen_count = 0, released_count = 0;
        // RPC-paced HOST->DISK writes normally reach the independent disk
        // executor before this owner's initial-plan queue can expire. Exercise
        // that queue separately in WorkLoopWriteQueueTimeoutSmokeTest with
        // concurrent local producers; retain the RPC/in-flight I/O checks here.
        const bool require_rpc_queue_timeout = !(GetParam() && write);
        int64_t round = 0;
        for (const int budget : {1, 2, 5, 10}) {
            SCOPED_TRACE(::testing::Message() << "timeout_ms=" << budget << " write=" << write);
            for (const auto& worker : workers_) worker->io->clear();
            std::vector<std::shared_ptr<BusinessCompletion>> completions;
            std::vector<std::future<TransferResult>> futures;
            const auto first_request = (++round) * static_cast<int64_t>(kRequests);
            for (size_t request = 0; request < kRequests; ++request) {
                auto completion = std::make_shared<BusinessCompletion>();
                futures.push_back(completion->promise.get_future());
                completions.push_back(completion);
                submit(write, request, first_request + request, std::chrono::milliseconds(budget), completion);
            }
            for (size_t request = 0; request < kRequests; ++request) {
                ASSERT_EQ(futures[request].wait_for(15s), std::future_status::ready);
                const auto result = futures[request].get();
                EXPECT_TRUE(result.ok() || result.status.code() == ErrorCode::DEADLINE_EXCEEDED
                            || result.status.code() == ErrorCode::TRANSFER_COMPLETION_UNKNOWN)
                    << result.status.ToString();
                EXPECT_EQ(completions[request]->callbacks.load(), 1u);
            }
            const auto records = coordinator_->frozenTasks();
            frozen_count += records.size();
            released_count += recover(records);
            rememberUnreleased(records, first_request, write);
            for (const auto& worker : workers_) worker->engine->waitForIdle();
            dumpUnreleasedEvidence("recovery_wait_end");
            for (size_t rank = 0; rank < workers_.size(); ++rank) {
                const auto events = workers_[rank]->io->events();
                for (const auto& event : events) {
                    EXPECT_EQ(event.status, DiskBlockIOStatus::OK);
                }
                for (const auto& record : records) {
                    if (record->snapshot().state == FrozenTask::State::RELEASED) {
                        const auto trace = std::get<uint64_t>(record->execution().trace_id);
                        const auto state = workers_[rank]->dispatcher->taskRegistry()
                                               .queryLocal(std::vector<uint64_t>{trace}).front();
                        // A fenced request may never enter the handler. Its registry
                        // tombstone prevents a later arrival from starting any I/O.
                        if (state != State::REJECTED) {
                            ASSERT_TRUE(servers_[rank]->service.waitCompleted(trace));
                        }
                    }
                }
                const auto worker_results = servers_[rank]->service.completed();
                for (const auto& done : worker_results) {
                    if (done.request < first_request || done.request >= first_request + static_cast<int64_t>(kRequests)) continue;
                    const auto& message = done.result.status.ToString();
                    if (done.result.status.code() == ErrorCode::DEADLINE_EXCEEDED
                        && (message.find("WorkLoop queue") != std::string::npos
                            || message.find("TE worker queue") != std::string::npos)) {
                        ++queue_timeouts;
                        EXPECT_TRUE(done.result.buffer_safe);
                    }
                }
                for (const auto& record : records) {
                    ASSERT_TRUE(record->execution().request_id.has_value());
                    const auto request = static_cast<size_t>(*record->execution().request_id - first_request);
                    ASSERT_LT(request, kRequests);
                    const auto first_offset = workers_[rank]->disk->blockOffset(workers_[rank]->disks[request * kBlocksPerRequest]);
                    const auto end_offset = first_offset + kBlocksPerRequest * kBytes;
                    const auto trace = std::get<uint64_t>(record->execution().trace_id);
                    if (workers_[rank]->dispatcher->taskRegistry()
                            .queryLocal(std::vector<uint64_t>{trace}).front() == State::REJECTED) {
                        EXPECT_FALSE(std::any_of(events.begin(), events.end(), [&](const auto& event) {
                            return event.write == write && event.offset >= first_offset && event.offset < end_offset;
                        })) << "fenced trace started I/O: " << trace << " rank=" << rank;
                    }
                    const auto done = std::find_if(worker_results.begin(), worker_results.end(),
                                                   [trace](const auto& item) { return item.trace == trace; });
                    if (done != worker_results.end()
                        && std::any_of(events.begin(), events.end(), [&](const auto& event) {
                            return event.write == write && event.offset >= first_offset && event.offset < end_offset
                                   && event.begin < done->rpc_deadline && done->rpc_deadline < event.end;
                        })) ++io_timeouts;
                    const auto diagnostics = record->failure().execution_diagnostics;
                    ASSERT_NE(diagnostics, nullptr);
                    EXPECT_TRUE(std::any_of(diagnostics->ranks.begin(), diagnostics->ranks.end(), [](const auto& row) {
                        return row.grpc_code && *row.grpc_code == grpc::StatusCode::DEADLINE_EXCEEDED;
                    })) << record->debugString();
                }
            }
            std::cout << "natural local timeout smoke: engine=" << (GetParam() ? "workloop" : "legacy")
                      << " direction=" << (write ? "write" : "read") << " timeout_ms=" << budget
                      << " queue=" << queue_timeouts << " in_flight_io=" << io_timeouts
                      << " frozen=" << frozen_count << " released=" << released_count << std::endl;
            ASSERT_EQ(released_count, frozen_count) << "natural timeout left frozen tasks; see trace/query diagnostics";
            if ((!require_rpc_queue_timeout || queue_timeouts > 0) && io_timeouts > 0) break;
        }
        RecordProperty("natural_queue_timeouts", static_cast<int>(queue_timeouts));
        RecordProperty("natural_io_timeouts", static_cast<int>(io_timeouts));
        RecordProperty("recovered_frozen_tasks", static_cast<int>(released_count));
        if (require_rpc_queue_timeout) {
            EXPECT_GT(queue_timeouts, 0u) << "1/2/5/10 ms scan did not hit a real transfer queue timeout";
        }
        EXPECT_GT(io_timeouts, 0u) << "1/2/5/10 ms scan did not expire while real disk I/O was in flight";
        EXPECT_GT(frozen_count, 0u) << "no frozen execution was exercised";
        longTimeoutControl(write);
        for (const auto& worker : workers_) {
            worker->engine->waitForIdle();
            for (size_t i = 0; i < kBlocks; ++i) {
                worker->host->decTreeRef(worker->hosts[i], BlockTreeRefType::CACHE);
                worker->disk->decTreeRef(worker->disks[i], BlockTreeRefType::CACHE);
                worker->device->decTreeRef(worker->devices[i], BlockTreeRefType::CACHE);
            }
            EXPECT_EQ(worker->host->freeBlocksNum(), kBlocks);
            EXPECT_EQ(worker->disk->freeBlocksNum(), kBlocks);
            EXPECT_EQ(worker->device->freeBlocksNum(), kBlocks);
        }
    }
    TempDirGuard directory_{"natural_local_timeout_smoke"};
    std::vector<std::unique_ptr<SmokeWorker>> workers_;
    std::vector<std::unique_ptr<SmokeServer>> servers_;
    std::shared_ptr<BroadcastManager> broadcast_;
    std::unique_ptr<BlockTransferDispatcher> coordinator_;
    std::vector<EvidenceRequest> unreleased_evidence_;
};

TEST_P(LocalTimeoutRecoverySmokeTest, NaturalReadTimeoutsRecoverAndLongDeadlineSucceeds) { exercise(false); }
TEST_P(LocalTimeoutRecoverySmokeTest, NaturalWriteTimeoutsRecoverAndLongDeadlineSucceeds) { exercise(true); }
INSTANTIATE_TEST_SUITE_P(Engines, LocalTimeoutRecoverySmokeTest, ::testing::Bool(), [](const auto& info) {
    return info.param ? "WorkLoop" : "Legacy";
});
}  // namespace
}  // namespace rtp_llm
