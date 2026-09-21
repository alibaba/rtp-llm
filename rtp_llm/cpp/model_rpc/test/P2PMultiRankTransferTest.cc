#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <random>
#include <numeric>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include <grpc++/grpc++.h>

#include "autil/NetUtil.h"
#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorDecode.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorPrefill.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerPrefillRead.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeRead.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerDecodeRead.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorResourceStore.h"
#include "rtp_llm/cpp/cache/connector/p2p/test/MockGenerateStream.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpKVCacheReceiver.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

namespace rtp_llm {
namespace {

constexpr int kRanks = 2;
constexpr int kLayers = 2;
constexpr int kBlocks = 257; // Up to 64 concurrent requests with four blocks each.
constexpr int kWaitMs = 10000;
constexpr int kRequestMs = 120000;

void check(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

void checkCuda(cudaError_t error) {
    check(error == cudaSuccess, cudaGetErrorString(error));
}

std::string env(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "";
}

int setting(const char* name, int fallback, int minimum, int maximum) {
    const auto text = env(name);
    if (text.empty()) return fallback;
    size_t parsed = 0;
    const int value = std::stoi(text, &parsed);
    check(parsed == text.size() && value >= minimum && value <= maximum, std::string("invalid ") + name);
    return value;
}

struct ProcessSample {
    int64_t rss_kib = 0, cpu_us = 0;
    int threads = 0, fds = 0;
};

ProcessSample sampleProcess() {
    ProcessSample result;
    std::ifstream status("/proc/self/status");
    check(status.good(), "cannot read process status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) result.rss_kib = std::stoll(line.substr(6));
        if (line.rfind("Threads:", 0) == 0) result.threads = std::stoi(line.substr(8));
    }
    auto* directory = ::opendir("/proc/self/fd");
    check(directory != nullptr, "cannot inspect file descriptors");
    while (auto* entry = ::readdir(directory)) if (entry->d_name[0] != '.') ++result.fds;
    ::closedir(directory);
    rusage usage{};
    check(::getrusage(RUSAGE_SELF, &usage) == 0, "cannot sample CPU time");
    result.cpu_us = (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1000000
                    + usage.ru_utime.tv_usec + usage.ru_stime.tv_usec;
    return result;
}

template<class Predicate>
bool until(Predicate predicate, int timeout_ms = kWaitMs) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= end) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return true;
}

// Only test configuration and observations use this socket. KV bytes travel
// through the production TCP backend; StartLoad and rank broadcasts use gRPC.
struct Fd {
    int value = -1;
    ~Fd() { if (value >= 0) ::close(value); }
};

sockaddr_in address(const std::string& text) {
    const auto colon = text.rfind(':');
    check(colon != std::string::npos, "expected IPv4:port");
    const auto port = std::stoi(text.substr(colon + 1));
    check(port > 0 && port <= 65535, "invalid control port");
    sockaddr_in result{};
    result.sin_family = AF_INET;
    result.sin_port = htons(port);
    check(::inet_pton(AF_INET, text.substr(0, colon).c_str(), &result.sin_addr) == 1, "invalid control IP");
    return result;
}

void io(int fd, void* data, size_t size, bool sending, int timeout_ms = kWaitMs) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    auto* bytes = static_cast<char*>(data);
    while (size) {
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - std::chrono::steady_clock::now()).count();
        check(remaining > 0, "control deadline exceeded");
        pollfd p{fd, static_cast<short>(sending ? POLLOUT : POLLIN), 0};
        const auto ready = ::poll(&p, 1, static_cast<int>(remaining));
        if (ready < 0 && errno == EINTR) continue;
        check(ready > 0, "control poll failed");
        const auto n = sending ? ::send(fd, bytes, size, MSG_NOSIGNAL | MSG_DONTWAIT) :
                                 ::recv(fd, bytes, size, MSG_DONTWAIT);
        if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        check(n > 0, "control connection closed");
        bytes += n;
        size -= static_cast<size_t>(n);
    }
}

struct Command {
    char op = 'S';  // A: allocate, R: register, P: publish, S: snapshot, Q: finish.
    int64_t id = 0;
    int32_t blocks = 2;
};

struct Report {
    uint32_t magic = 0x4d523032;
    int32_t grpc_ports[kRanks]{};
    int32_t load_entered = 0, load_done = 0;
    int32_t handles[kRanks]{};
    int32_t handles_done[kRanks]{};
    int32_t free_blocks[kRanks]{};
    int32_t source_requests = 0, registrations = 0, deadline_entries = 0, resources = 0;
    int32_t expired_records = 0, computed_buffers = 0, source_markers = 0;
    ProcessSample process;
};

std::string key(int64_t id) { return "multi_rank_" + std::to_string(id); }

// The gate queues an actual CUDA wait on rank 1's transfer thread. It does not
// fabricate task completion or lease status. Release always precedes teardown.
class CopyGate {
public:
    void init() {
        checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        checkCuda(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
        checkCuda(cudaLaunchHostFunc(stream_, [](void* opaque) {
            auto* self = static_cast<CopyGate*>(opaque);
            std::unique_lock<std::mutex> lock(self->mutex_);
            self->cv_.wait(lock, [&] { return self->released_; });
        }, this));
        checkCuda(cudaEventRecord(event_, stream_));
    }
    ~CopyGate() {
        release();
        if (stream_) cudaStreamSynchronize(stream_);
        if (event_) cudaEventDestroy(event_);
        if (stream_) cudaStreamDestroy(stream_);
    }
    void release() {
        { std::lock_guard<std::mutex> lock(mutex_); released_ = true; }
        cv_.notify_all();
    }
    cudaEvent_t event() const { return event_; }
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool released_ = false;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t event_ = nullptr;
};

class Converter: public LayerBlockConverter {
public:
    explicit Converter(std::shared_ptr<SingleTypeKVCacheAllocator> allocator): allocator_(std::move(allocator)) {}
    std::vector<BlockInfo> convertIndexToBuffer(int layer, const std::string& tag, int block,
                                               int partitions, int partition) const override {
        return allocator_->convertIndexToBufferByTag(layer, tag, block, partitions, partition);
    }
    std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const override { return {}; } // TCP needs no MR.
private:
    std::shared_ptr<SingleTypeKVCacheAllocator> allocator_;
};

class KickoffGate {
public:
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        entered_ = true;
        cv_.notify_all();
        cv_.wait(lock, [&] { return released_; });
    }
    bool entered() {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(5), [&] { return entered_; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool entered_ = false, released_ = false;
};

class RankService: public RpcService::Service {
public:
    P2PConnector* connector = nullptr;
    std::atomic<int> load_entered{0}, load_done{0}, handles{0}, handles_done{0}, cancels{0}, lease_queries{0}, reads_done{0};
    std::atomic<bool> fail_read{false};

    grpc::Status StartLoad(grpc::ServerContext* ctx, const P2PConnectorStartLoadRequestPB* request,
                           P2PConnectorStartLoadResponsePB* response) override {
        ++load_entered;
        connector->handleRead(*request, *response, [ctx] { return ctx->IsCancelled(); });
        ++load_done;
        return grpc::Status::OK;
    }
    grpc::Status ExecuteFunction(grpc::ServerContext*, const FunctionRequestPB* request,
                                 FunctionResponsePB* response) override {
        const auto type = request->p2p_request().type();
        if (type == P2PConnectorBroadcastType::HANDLE_READ) ++handles;
        if (type == P2PConnectorBroadcastType::CANCEL_READ) ++cancels;
        if (type == P2PConnectorBroadcastType::QUERY_LEASE_STATUS) ++lease_queries;
        if (type == P2PConnectorBroadcastType::READ && fail_read) {
            // Model an RPC failure after its server-side work was accepted.
            // Keep the real worker alive so cancel/lease broadcasts must drain it.
            std::lock_guard<std::mutex> thread_lock(thread_mutex_);
            check(!read_thread_.joinable(), "only one injected READ per session");
            read_thread_ = std::thread([this, copy = *request] {
                FunctionResponsePB ignored;
                connector->executeFunction(copy, ignored);
                ++reads_done;
            });
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return release_failure_; });
            return {grpc::StatusCode::UNAVAILABLE, "injected nonzero-rank READ failure"};
        }
        connector->executeFunction(*request, *response);
        if (type == P2PConnectorBroadcastType::HANDLE_READ) ++handles_done;
        if (type == P2PConnectorBroadcastType::READ) ++reads_done;
        return grpc::Status::OK;
    }
    void releaseFailure() {
        { std::lock_guard<std::mutex> lock(mutex_); release_failure_ = true; }
        cv_.notify_all();
    }
    void join() {
        std::lock_guard<std::mutex> lock(thread_mutex_);
        if (read_thread_.joinable()) read_thread_.join();
    }
private:
    std::mutex mutex_, thread_mutex_;
    std::condition_variable cv_;
    bool release_failure_ = false;
    std::thread read_thread_;
};

class Rank {
public:
    explicit Rank(const std::string& host):
        config(test::makeSimpleMhaCacheConfig(kLayers, kBlocks, 4, DataType::TYPE_FP16, 2, 16)),
        allocator(std::make_shared<SingleTypeKVCacheAllocator>(config, AllocationType::DEVICE)) {
        check(allocator->init(), "allocator init failed");
        baseline = allocator->freeBlocksNum();
        grpc::ServerBuilder builder;
        builder.AddListeningPort(host + ":0", grpc::InsecureServerCredentials(), &grpc_port);
        builder.RegisterService(&service);
        server = builder.BuildAndStart();
        check(server && grpc_port > 0, "rank gRPC bind failed");
        transfer_port = autil::NetUtil::randomPort();
        check(transfer_port > 0, "transfer port allocation failed");
    }
    ~Rank() {
        service.releaseFailure();
        server->Shutdown(std::chrono::system_clock::now());
        server->Wait();
        service.join();
        connector.reset();
    }
    void init(RoleType role, int rank, const std::string& host, const std::vector<std::unique_ptr<Rank>>& ranks,
              int load_ms) {
        P2PConnectorConfig c;
        c.role_type = role;
        c.tp_rank = rank;
        auto& s = c.scheduler_config;
        s.role_type = role;
        s.parallelism_config.tp_size = kRanks;
        s.parallelism_config.tp_rank = rank;
        s.parallelism_config.world_size = kRanks;
        s.parallelism_config.world_rank = rank;
        s.parallelism_config.role_type = role;
        s.topology = config.topologyPtr();
        s.load_cache_timeout_ms = load_ms;
        s.p2p_lease_query_timeout_ms = 60000;
        for (const auto& r : ranks) {
            s.worker_grpc_addrs.push_back(host + ":" + std::to_string(r->grpc_port));
            s.worker_addrs.push_back(host + ":" + std::to_string(r->transfer_port) + ":" + std::to_string(r->grpc_port));
        }
        auto& w = c.worker_config;
        w.tp_size = kRanks;
        w.tp_rank = rank;
        w.topology = s.topology;
        w.layer_all_num = kLayers;
        w.load_cache_timeout_ms = load_ms;
        w.transfer_backend_config.cache_store_listen_port = transfer_port;
        w.transfer_backend_config.messager_worker_thread_count = 1;
        w.transfer_backend_config.messager_io_thread_count = 1;
        connector = std::make_unique<P2PConnector>(c, std::make_shared<Converter>(allocator), nullptr);
        check(connector->init(), "connector init failed");
        service.connector = connector.get();
    }
    KVCacheResourcePtr allocate(size_t count) {
        const auto pool = allocator->getDeviceBlockPool();
        const auto blocks = pool->malloc(count);
        check(blocks.has_value(), "GPU pool allocation failed");
        pool->incRef(*blocks);
        KVCacheResource source;
        source.initGroups(config.topologyPtr());
        source.mutableBlockIds(0).assign(*blocks);
        for (size_t i = 0; i < count; ++i) source.cacheKeys().push_back(1000 + i);
        auto ref = allocator->incrKVCacheRef(source, source.cacheKeys(), true);
        pool->decRef(*blocks);
        check(ref != nullptr, "GPU resource reference failed");
        return ref;
    }
    uint8_t value(int64_t id, int rank, int layer, size_t block, size_t offset) const {
        return (id * 17 + rank * 37 + layer * 53 + block * 71 + offset) % 251;
    }
    void fill(const KVCacheResource& resource, int64_t id, int rank, bool poison = false) {
        for (int l = 0; l < kLayers; ++l) {
            for (size_t b = 0; b < resource.blocks(0).size(); ++b) {
                for (const auto& info : allocator->convertIndexToBuffer(l, resource.blocks(0)[b])) {
                    std::vector<uint8_t> bytes(info.size_bytes);
                    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = poison ? 0xff : value(id, rank, l, b, i);
                    checkCuda(cudaMemcpy(info.addr, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
                }
            }
        }
        checkCuda(cudaStreamSynchronize(nullptr));
    }
    void verify(const KVCacheResource& resource, int64_t id, int rank) {
        for (int l = 0; l < kLayers; ++l) {
            for (size_t b = 0; b < resource.blocks(0).size(); ++b) {
                for (const auto& info : allocator->convertIndexToBuffer(l, resource.blocks(0)[b])) {
                    std::vector<uint8_t> bytes(info.size_bytes);
                    checkCuda(cudaMemcpy(bytes.data(), info.addr, bytes.size(), cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < bytes.size(); ++i) {
                        check(bytes[i] == value(id, rank, l, b, i), "payload mismatch rank=" + std::to_string(rank)
                            + " layer=" + std::to_string(l) + " block=" + std::to_string(b)
                            + " byte=" + std::to_string(i));
                    }
                }
            }
        }
    }
    bool inCopy() {
        auto receiver = std::dynamic_pointer_cast<transfer::tcp::TcpKVCacheReceiver>(connector->decode_->worker_->receiver_);
        const auto store = receiver->getTransferTaskStore();
        std::shared_lock<std::shared_mutex> lock(store->mutex_);
        for (const auto& [name, task] : store->task_map_) {
            std::shared_lock<std::shared_mutex> task_lock(task->mutex_);
            if (task->transferring_ && !task->done_) return true;
        }
        return false;
    }
    void gateCopy(CopyGate& gate) {
        auto receiver = std::dynamic_pointer_cast<transfer::tcp::TcpKVCacheReceiver>(connector->decode_->worker_->receiver_);
        auto promise = std::make_shared<std::promise<cudaError_t>>();
        auto future = promise->get_future();
        check(receiver->transfer_service_->worker_thread_pool_->pushTask([promise, event = gate.event()] {
            promise->set_value(cudaStreamWaitEvent(getNoBlockCopyStream(0).stream(), event, 0));
        }) == autil::ThreadPoolBase::ERROR_NONE, "cannot prime H2D gate");
        check(future.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "H2D gate prime timed out");
        checkCuda(future.get());
    }
    bool hasCompletedTask(const std::string& request_key) {
        auto* worker = connector->decode_->worker_.get();
        std::shared_ptr<P2PWorkerDecodeRead::ReadTaskGroup> group;
        {
            std::lock_guard<std::mutex> lock(worker->read_tasks_mutex_);
            auto it = worker->read_tasks_.find(request_key);
            if (it == worker->read_tasks_.end()) return false;
            group = it->second;
        }
        for (const auto& task : group->tasks) {
            if (task->done() && task->success()) return true;
        }
        return false;
    }
    bool readsDrained() {
        auto* worker = connector->decode_->worker_.get();
        std::scoped_lock lock(worker->read_tasks_mutex_, worker->lease_map_mutex_);
        return worker->read_tasks_.empty() && worker->lease_map_.empty();
    }
    size_t pendingCancels(bool expired_only = false) {
        auto* worker = connector->decode_->worker_.get();
        std::lock_guard<std::mutex> lock(worker->read_tasks_mutex_);
        if (!expired_only) return worker->pending_cancel_keys_.size();
        return std::count_if(worker->pending_cancel_keys_.begin(), worker->pending_cancel_keys_.end(),
            [](const auto& entry) { return entry.second < currentTimeMs() - 2000; });
    }
    void drainSenders() {
        // The fixture keeps P source allocations until every sender thread has
        // passed its previous work. This deliberately does not rely on the
        // production P-side cancellation lifetime guarantee under discussion.
        struct Drain {
            std::mutex mutex;
            std::condition_variable cv;
            int entered = 0;
            bool release = false;
        };
        auto state = std::make_shared<Drain>();
        auto cleanup = std::shared_ptr<void>(nullptr, [state](void*) {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->release = true;
            state->cv.notify_all();
        });
        auto& worker = connector->prefill_->worker_;
        const int threads = worker->config_.p2p_prefill_sender_thread_count;
        for (int i = 0; i < threads; ++i) {
            check(worker->async_sender_pool_->pushTask([state] {
                std::unique_lock<std::mutex> lock(state->mutex);
                ++state->entered;
                state->cv.notify_all();
                state->cv.wait(lock, [&] { return state->release; });
            }, false, false) == autil::ThreadPoolBase::ERROR_NONE, "cannot queue sender drain barrier");
        }
        std::unique_lock<std::mutex> lock(state->mutex);
        check(state->cv.wait_for(lock, std::chrono::seconds(5), [&] { return state->entered == threads; }),
            "sender threads did not drain");
    }
    CacheConfig config;
    std::shared_ptr<SingleTypeKVCacheAllocator> allocator;
    KVCacheResourcePtr resource;
    size_t baseline = 0;
    int grpc_port = 0, transfer_port = 0;
    RankService service;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<P2PConnector> connector;
};

std::vector<std::unique_ptr<Rank>> makeRanks(RoleType role, const std::string& host, int load_ms) {
    std::vector<std::unique_ptr<Rank>> ranks;
    for (int r = 0; r < kRanks; ++r) ranks.push_back(std::make_unique<Rank>(host));
    for (int r = 0; r < kRanks; ++r) ranks[r]->init(role, r, host, ranks, load_ms);
    return ranks;
}

class P2PMultiRankPeer: public DeviceTestBase {};

TEST_F(P2PMultiRankPeer, DISABLED_Serve) {
    const auto host = env("P2P_MULTI_HOST");
    ASSERT_FALSE(host.empty()) << "set P2P_MULTI_HOST to the reachable host IPv4 address";
    const auto addr = address(env("P2P_MULTI_LISTEN"));
    Fd listener;
    listener.value = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    ASSERT_GE(listener.value, 0);
    int reuse = 1;
    ASSERT_EQ(::setsockopt(listener.value, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)), 0);
    ASSERT_EQ(::bind(listener.value, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)), 0);
    ASSERT_EQ(::listen(listener.value, 1), 0);
    const auto count = env("P2P_MULTI_SESSIONS");
    const int sessions = count.empty() ? 10 : std::stoi(count);
    ASSERT_GT(sessions, 0);
    for (int session = 0; session < sessions; ++session) {
        pollfd pending{listener.value, POLLIN, 0};
        ASSERT_GT(::poll(&pending, 1, 180000), 0) << "no Decode controller connected";
        Fd control;
        control.value = ::accept4(listener.value, nullptr, nullptr, SOCK_CLOEXEC);
        ASSERT_GE(control.value, 0);
        auto ranks = makeRanks(RoleType::PREFILL, host, kRequestMs);
        struct SourceRequest {
            std::vector<KVCacheResourcePtr> resources;
            int64_t deadline = 0;
        };
        std::map<int64_t, SourceRequest> sources;
        int64_t id = 0, deadline = 0;
        auto report = [&](bool diagnostics = false) {
            Report out;
            for (int r = 0; r < kRanks; ++r) {
                out.grpc_ports[r] = ranks[r]->grpc_port;
                out.handles[r] = ranks[r]->service.handles.load();
                out.handles_done[r] = ranks[r]->service.handles_done.load();
                out.free_blocks[r] = ranks[r]->allocator->freeBlocksNum();
            }
            out.load_entered = ranks[0]->service.load_entered.load();
            out.load_done = ranks[0]->service.load_done.load();
            out.source_requests = sources.size();
            const auto store = ranks[0]->connector->streamStore();
            {
                std::lock_guard<std::mutex> lock(store->resource_map_mutex_);
                out.registrations = store->request_states_.size();
                out.deadline_entries = store->deadline_index_.size();
                out.resources = store->resource_map_.size();
                for (const auto& [request_key, state] : store->request_states_) {
                    if (state.request_deadline_ms < currentTimeMs() - 2000) ++out.expired_records;
                }
            }
            for (const auto& rank : ranks) {
                const auto buffers = rank->connector->prefill_->worker_->getComputedBuffersStore();
                std::lock_guard<std::mutex> lock(buffers->computed_buffers_mutex_);
                out.computed_buffers += buffers->computed_buffers_.size();
                out.source_markers += buffers->removed_requests_.size();
                for (const auto& [request_key, expiry] : buffers->removed_requests_) {
                    if (expiry < currentTimeMs() - 2000) ++out.expired_records;
                }
            }
            if (diagnostics) out.process = sampleProcess();
            io(control.value, &out, sizeof(out), true);
        };
        report();
        while (true) {
            Command command;
            io(control.value, &command, sizeof(command), false, 180000);
            if (command.op == 'Q') break;
            // Lowercase operations address independent concurrent requests.
            if (command.op == 'f') {
                for (const auto& rank : ranks) {
                    check(rank->service.handles == rank->service.handles_done, "HANDLE_READ still active");
                    rank->drainSenders();
                }
            } else if (command.op == 'a') {
                check(command.blocks > 0 && command.blocks <= 4, "invalid batch block count");
                check(!sources.count(command.id), "duplicate batch request ID");
                SourceRequest source;
                for (int r = 0; r < kRanks; ++r) {
                    auto resource = ranks[r]->allocate(command.blocks);
                    ranks[r]->fill(*resource, command.id, r);
                    source.resources.push_back(std::move(resource));
                }
                sources.emplace(command.id, std::move(source));
            } else if (command.op == 'r') {
                auto& source = sources.at(command.id);
                auto store = ranks[0]->connector->streamStore();
                source.deadline = store->requestDeadline(key(command.id), kRequestMs);
                auto meta = std::make_shared<MockMeta>();
                meta->setUniqueKey(key(command.id));
                meta->setRequestId(command.id);
                meta->setDeadlineMs(source.deadline);
                check(store->addResource(meta, source.resources[0]), "batch registration failed");
                PrefillResultStore::SideChannelData payload;
                payload.has_first_token = true;
                payload.first_token_id = command.id;
                store->publishPrefillPayload(key(command.id), source.deadline, std::move(payload));
            } else if (command.op == 'p' || command.op == '1') {
                auto& source = sources.at(command.id);
                check(source.deadline > 0, "publish before registration");
                for (int r = 0; r < kRanks; ++r) {
                    for (int l = 0; l < (command.op == '1' ? 1 : kLayers); ++l) {
                        check(ranks[r]->connector->writeByLayerTag(l,
                                                                   ranks[r]->config.topologyPtr()->groups()[0].tag,
                                                                   source.resources[r],
                                                                   command.id,
                                                                   nullptr,
                                                                   source.deadline),
                              "batch publication failed");
                    }
                }
            } else if (command.op == 'd') {
                auto store = ranks[0]->connector->streamStore();
                {
                    std::lock_guard<std::mutex> lock(store->resource_map_mutex_);
                    check(!store->resource_map_.count(key(command.id)), "production still holds batch resource");
                }
                check(sources.erase(command.id) == 1, "unknown batch request cleanup");
            } else if (command.op == 'A') {
                check(ranks[0]->service.load_entered == ranks[0]->service.load_done, "previous StartLoad still running");
                id = command.id;
                for (int r = 0; r < kRanks; ++r) {
                    ranks[r]->resource.reset();
                    ranks[r]->resource = ranks[r]->allocate(2);
                    ranks[r]->fill(*ranks[r]->resource, id, r);
                }
            } else if (command.op == 'R') {
                auto store = ranks[0]->connector->streamStore();
                deadline = store->requestDeadline(key(id), kRequestMs);
                auto meta = std::make_shared<MockMeta>();
                meta->setUniqueKey(key(id));
                meta->setRequestId(id);
                meta->setDeadlineMs(deadline);
                check(store->addResource(meta, ranks[0]->resource), "Prefill resource registration failed");
                PrefillResultStore::SideChannelData payload;
                payload.has_first_token = true;
                payload.first_token_id = id;
                store->publishPrefillPayload(key(id), deadline, std::move(payload));
            } else if (command.op == 'P') {
                for (const auto& rank : ranks) {
                    for (int l = 0; l < kLayers; ++l) {
                        check(
                            rank->connector->writeByLayerTag(
                                l, rank->config.topologyPtr()->groups()[0].tag, rank->resource, id, nullptr, deadline),
                            "Prefill publication failed");
                    }
                }
            } else {
                check(command.op == 'S', "invalid control command");
            }
            report(command.op == 'S');
        }
        ranks.clear();
        char ack = 'Q';
        io(control.value, &ack, 1, true);
    }
}

class P2PMultiRankTransferTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        host_ = env("P2P_MULTI_HOST");
        const auto peer = env("P2P_MULTI_PEER");
        check(!host_.empty() && !peer.empty(), "set P2P_MULTI_HOST and P2P_MULTI_PEER; see test guide");
        peer_host_ = peer.substr(0, peer.rfind(':'));
        check(peer_host_ != host_ && peer_host_ != "127.0.0.1", "this suite requires two different hosts");
        const auto addr = address(peer);
        control_.value = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
        check(control_.value >= 0, "control socket failed");
        const auto result = ::connect(control_.value, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr));
        check(result == 0 || errno == EINPROGRESS, "control connect failed");
        pollfd pending{control_.value, POLLOUT, 0};
        check(::poll(&pending, 1, kWaitMs) > 0, "control connect timed out");
        int error = 0;
        socklen_t size = sizeof(error);
        check(::getsockopt(control_.value, SOL_SOCKET, SO_ERROR, &error, &size) == 0 && error == 0, "control connect error");
        receive();
    }
    void TearDown() override {
        // Release GPU and RPC gates before cancelling/joining even on assertion failure.
        if (gate_) gate_->release();
        if (queue_gate_) queue_gate_->release();
        for (const auto& request : batch_) {
            if (request.context && !request.context->done()) ranks_[0]->connector->cancelRead(request.context);
        }
        EXPECT_TRUE(until([&] {
            for (const auto& request : batch_) {
                if (request.context && (!request.context->done() || request.context->resourceHoldPending())) return false;
            }
            return true;
        }, 30000));
        batch_.clear();
        for (const auto& rank : ranks_) rank->service.releaseFailure();
        if (!context_) context_ = retired_context_.lock();
        if (context_ && !context_->done()) ranks_[0]->connector->cancelRead(context_);
        if (context_) {
            EXPECT_TRUE(until([&] { return context_->done() && !context_->resourceHoldPending(); }, 30000));
        }
        context_.reset();
        ranks_.clear();
        gate_.reset();
        if (control_.value >= 0) {
            try {
                Command q;
                q.op = 'Q';
                io(control_.value, &q, sizeof(q), true);
                char ack = 0;
                io(control_.value, &ack, 1, false, 30000);
                EXPECT_EQ(ack, 'Q');
            } catch (const std::exception& e) { ADD_FAILURE() << e.what(); }
        }
        DeviceTestBase::TearDown();
    }
    Report receive() {
        Report r;
        io(control_.value, &r, sizeof(r), false, 30000);
        check(r.magic == Report{}.magic, "control protocol mismatch");
        report_ = r;
        return r;
    }
    Report command(char op, int64_t id = 0, int blocks = 2) {
        Command c;
        c.op = op;
        c.id = id;
        c.blocks = blocks;
        io(control_.value, &c, sizeof(c), true);
        return receive();
    }
    void prepare(int load_ms = 60000) {
        ranks_ = makeRanks(RoleType::DECODE, host_, load_ms);
        allocateRequest();
    }
    void allocateRequest() {
        command('A', id_);
        for (int r = 0; r < kRanks; ++r) {
            // Drain the free list before allocating so both independent rank pools
            // choose the same physical IDs even after rank 1's competing allocation.
            ranks_[r]->allocate(ranks_[r]->baseline).reset();
            ranks_[r]->resource = ranks_[r]->allocate(2);
            ranks_[r]->fill(*ranks_[r]->resource, id_, r, true);
        }
        check(ranks_[0]->resource->blocks(0) == ranks_[1]->resource->blocks(0), "rank block IDs must match");
    }
    void start(bool registered = true) {
        if (registered) { command('R'); command('P'); }
        auto meta = std::make_shared<MockMeta>();
        meta->setUniqueKey(key(id_));
        meta->setRequestId(id_);
        meta->setDeadlineMs(currentTimeMs() + kRequestMs);
        meta->setPrefillAddr(peer_host_, report_.grpc_ports[0]);
        meta->setPrefillTpSize(kRanks);
        meta->setPrefillCpSize(1);
        // Model rank-synchronous allocator ownership without a model/NCCL loop.
        // Only the real rank-zero async context may retain this collective ref.
        std::vector<KVCacheResourcePtr> held;
        for (const auto& rank : ranks_) held.push_back(rank->resource);
        auto* root = held[0].get();
        KVCacheResourcePtr collective(root, [held = std::move(held)](KVCacheResource*) mutable { held.clear(); });
        auto result = ranks_[0]->connector->decode_->scheduler_->asyncRead(collective, meta, {0, 2});
        check(result.ok() && result.context, "Decode kickoff failed");
        context_ = result.context;
    }
    void blockNonzeroCopy() {
        // execNoBlockCopy performs cudaDeviceSynchronize in debug logging mode,
        // which would also stall rank 0 behind this intentionally blocked stream.
        check(!Logger::getEngineLogger().isDebugMode(), "H2D gate requires LOG_LEVEL=INFO (no device-wide debug sync)");
        gate_ = std::make_unique<CopyGate>();
        gate_->init();
        ranks_[1]->gateCopy(*gate_);
    }
    void expectOnlyNonzeroRankPending() {
        ASSERT_TRUE(until([&] { return ranks_[1]->inCopy() && ranks_[0]->service.reads_done.load() == 1; }));
        ASSERT_FALSE(context_->done()) << context_->errorInfo().ToString();
        ranks_[0]->verify(*ranks_[0]->resource, id_, 0);
    }
    void expectSuccess() {
        ASSERT_TRUE(until([&] { return context_->done(); }));
        ASSERT_TRUE(context_->success()) << context_->errorInfo().ToString();
        ASSERT_NE(context_->sideChannelPayload(), nullptr);
        EXPECT_EQ(context_->sideChannelPayload()->first_token_id, id_);
        for (int r = 0; r < kRanks; ++r) ranks_[r]->verify(*ranks_[r]->resource, id_, r);
        const auto state = command('S');
        EXPECT_GT(state.handles[0], 0);
        EXPECT_GT(state.handles[1], 0);
    }
    void expectLoadTimeout() {
        ASSERT_FALSE(context_->success());
        // The first observed error can originate in D's checker/READ RPC or P's
        // StartLoad/send RPC. These share the load deadline but use distinct codes.
        const auto code = context_->errorInfo().code();
        EXPECT_TRUE(code == ErrorCode::GENERATE_TIMEOUT || code == ErrorCode::P2P_CONNECTOR_WORKER_READ_TIMEOUT
            || code == ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE
            || code == ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT
            || code == ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_TIMEOUT)
            << context_->errorInfo().ToString();
    }
    void recover() {
        if (context_) {
            ASSERT_TRUE(until([&] { return context_->done() && !context_->resourceHoldPending(); }, 30000));
        }
        context_.reset();
        for (const auto& rank : ranks_) rank->resource.reset();
        ASSERT_TRUE(until([&] {
            return ranks_[0]->allocator->freeBlocksNum() == ranks_[0]->baseline
                && ranks_[1]->allocator->freeBlocksNum() == ranks_[1]->baseline;
        }));
        ASSERT_TRUE(until([&] { const auto s = command('S'); return s.load_entered == s.load_done; }));
        ranks_[1]->service.join();
        ranks_[1]->service.fail_read = false;
        ++id_;
        allocateRequest();
        start();
        ASSERT_NO_FATAL_FAILURE(expectSuccess());
    }
    void expectRetainedThenRelease() {
        ASSERT_TRUE(until([&] { return context_->done(); }));
        ASSERT_FALSE(context_->success());
        ASSERT_TRUE(until([&] { return ranks_[1]->service.lease_queries.load() > 0; }));
        ASSERT_TRUE(until([&] {
            return ranks_[0]->service.cancels.load() > 0 && ranks_[1]->service.cancels.load() > 0;
        }));
        ASSERT_TRUE(context_->resourceHoldPending());
        const auto old = ranks_[1]->resource->blocks(0);
        for (const auto& rank : ranks_) rank->resource.reset();
        // Simulate the caller abandoning a failed request. A test-owned context
        // would itself retain all blocks and hide a broken checker lease hold.
        retired_context_ = context_;
        context_.reset();
        const int queries = ranks_[1]->service.lease_queries.load();
        ASSERT_TRUE(until([&] { return ranks_[1]->service.lease_queries.load() > queries; }));
        {
            const auto held = retired_context_.lock();
            ASSERT_NE(held, nullptr);
            ASSERT_TRUE(held->resourceHoldPending());
        }
        // Exhaust the remaining pool: neither old target can be allocated to B.
        auto b = ranks_[1]->allocate(ranks_[1]->baseline - old.size());
        for (auto block : old) EXPECT_EQ(std::count(b->blocks(0).begin(), b->blocks(0).end(), block), 0);
        EXPECT_EQ(ranks_[1]->allocator->freeBlocksNum(), 0);
        ranks_[1]->fill(*b, 900, 1);
        gate_->release();
        ASSERT_TRUE(until([&] { return retired_context_.expired(); }, 30000));
        ranks_[1]->verify(*b, 900, 1);
        // With B still occupying every other block, checker retirement must make
        // exactly A's former target blocks available for reuse.
        ASSERT_TRUE(until([&] { return ranks_[1]->allocator->freeBlocksNum() == old.size(); }));
        auto reused = ranks_[1]->allocate(old.size());
        for (auto block : old) EXPECT_EQ(std::count(reused->blocks(0).begin(), reused->blocks(0).end(), block), 1);
        ranks_[1]->fill(*reused, 901, 1);
        ranks_[1]->verify(*reused, 901, 1);
        reused.reset();
        b.reset();
        ASSERT_NO_FATAL_FAILURE(recover());
    }
    void prepareConcurrent(int load_ms = 5000, int kickoff_capacity = 0) {
        ranks_ = makeRanks(RoleType::DECODE, host_, load_ms);
        if (kickoff_capacity > 0) {
            auto& scheduler = ranks_[0]->connector->decode_->scheduler_;
            scheduler->checker_->stop();
            scheduler->async_read_pool_->stop();
            scheduler->async_read_pool_->join();
            auto pool = std::make_shared<autil::LockFreeThreadPool>(1, kickoff_capacity, nullptr, "BatchKickoff");
            ASSERT_TRUE(pool->start());
            scheduler->async_read_pool_ = pool;
            scheduler->checker_ = std::make_shared<P2PConnectorAsyncReadContextChecker>();
            ASSERT_TRUE(scheduler->checker_->init(nullptr, scheduler->tp_broadcast_client_, pool));
        }
    }
    // All requests in a wave coexist. Rank instances, pools, RPC channels and
    // request stores remain alive across waves, including failures and recovery.
    void runBatch(int count, bool mixed, bool overload, uint32_t seed) {
        ASSERT_TRUE(batch_.empty());
        ASSERT_GE(count, 1);
        ASSERT_LE(count, 64);
        const auto before = command('S');
        for (int i = 0; i < count; ++i) {
            BatchRequest request;
            request.id = ++batch_id_;
            request.blocks = i % 2 == 0 ? 1 : 4;
            request.mode = mixed ? i % 5 : 0; // 0..2 success, 3 cancel, 4 timeout.
            command('a', request.id, request.blocks);
            for (int rank = 0; rank < kRanks; ++rank) {
                auto resource = ranks_[rank]->allocate(request.blocks);
                ranks_[rank]->fill(*resource, request.id, rank, true);
                request.resources.push_back(std::move(resource));
            }
            ASSERT_EQ(request.resources[0]->blocks(0), request.resources[1]->blocks(0));
            batch_.push_back(std::move(request));
        }
        auto& scheduler = ranks_[0]->connector->decode_->scheduler_;
        if (overload) {
            queue_gate_ = std::make_shared<KickoffGate>();
            ASSERT_EQ(scheduler->async_read_pool_->pushTask([gate = queue_gate_] { gate->wait(); }, false),
                autil::ThreadPoolBase::ERROR_NONE);
            ASSERT_TRUE(queue_gate_->entered());
        }
        const auto wave_begin = std::chrono::steady_clock::now();
        size_t accepted = 0, rejected = 0;
        for (auto& request : batch_) {
            auto meta = std::make_shared<MockMeta>();
            meta->setUniqueKey(key(request.id));
            meta->setRequestId(request.id);
            meta->setDeadlineMs(currentTimeMs() + kRequestMs);
            meta->setPrefillAddr(peer_host_, report_.grpc_ports[0]);
            meta->setPrefillTpSize(kRanks);
            meta->setPrefillCpSize(1);
            auto held = request.resources;
            auto* root = held[0].get();
            KVCacheResourcePtr collective(root, [held = std::move(held)](KVCacheResource*) mutable { held.clear(); });
            request.started = std::chrono::steady_clock::now();
            auto result = scheduler->asyncRead(collective, meta, {0, request.blocks});
            if (!result.ok()) {
                ASSERT_TRUE(overload) << result.error_info.ToString();
                EXPECT_EQ(result.error_info.code(), ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED);
                ASSERT_EQ(result.context, nullptr);
                ++rejected;
            } else {
                ASSERT_NE(result.context, nullptr);
                request.context = result.context;
                ++accepted;
            }
        }
        if (overload) {
            ASSERT_GT(rejected, 0u) << "queue did not saturate; increase count or reduce test pool capacity";
            ASSERT_GT(accepted, 0u);
            EXPECT_EQ(command('S').load_entered, before.load_entered);
        }
        // Rejected kickoff requests never register resources in P. For accepted
        // requests use the same registration rendezvous as ordinary StartLoad.
        for (const auto& request : batch_) if (request.context) command('r', request.id);
        if (queue_gate_) queue_gate_->release();
        ASSERT_TRUE(until([&] {
            return command('S').load_entered == before.load_entered + static_cast<int>(accepted);
        }));
        std::vector<size_t> order(batch_.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 random(seed);
        std::shuffle(order.begin(), order.end(), random);
        for (auto index : order) {
            const auto& request = batch_[index];
            if (!request.context) continue;
            if (request.mode < 3) command('p', request.id);
            if (request.mode == 3) command('1', request.id);
        }
        for (const auto& request : batch_) {
            if (!request.context || request.mode != 3) continue;
            ASSERT_TRUE(until([&] {
                return ranks_[0]->hasCompletedTask(key(request.id)) && ranks_[1]->hasCompletedTask(key(request.id));
            }));
            ASSERT_FALSE(request.context->done()) << "cancel stage missed key=" << key(request.id);
            ranks_[0]->connector->cancelRead(request.context);
        }
        ASSERT_TRUE(until([&] {
            bool finished = true;
            for (auto& request : batch_) {
                if (!request.context) continue;
                const bool done = request.context->done();
                if (done && request.latency_ms < 0) {
                    request.latency_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - request.started).count();
                }
                if (!done || request.context->resourceHoldPending()) finished = false;
            }
            return finished;
        }, 30000));
        last_latencies_.clear();
        for (const auto& request : batch_) {
            SCOPED_TRACE(::testing::Message() << "seed=" << seed << " key=" << key(request.id) << " mode=" << request.mode);
            if (!request.context) continue;
            if (request.mode < 3) {
                ASSERT_TRUE(request.context->success()) << request.context->errorInfo().ToString();
                ASSERT_NE(request.context->sideChannelPayload(), nullptr);
                EXPECT_EQ(request.context->sideChannelPayload()->first_token_id, request.id);
                for (int rank = 0; rank < kRanks; ++rank) ranks_[rank]->verify(*request.resources[rank], request.id, rank);
                last_latencies_.push_back(request.latency_ms);
            } else {
                EXPECT_FALSE(request.context->success());
                if (request.mode == 4) {
                    context_ = request.context;
                    ASSERT_NO_FATAL_FAILURE(expectLoadTimeout());
                    context_.reset();
                } else {
                    EXPECT_TRUE(request.context->cancelRequested());
                }
            }
        }
        ASSERT_TRUE(until([&] {
            auto state = command('S');
            return state.load_done == state.load_entered && ranks_[0]->readsDrained() && ranks_[1]->readsDrained()
                && state.handles[0] == state.handles_done[0] && state.handles[1] == state.handles_done[1];
        }));
        command('f');
        for (const auto& request : batch_) command('d', request.id);
        batch_.clear();
        ASSERT_TRUE(until([&] {
            return scheduler->checker_->inflightContextCount() == 0
                && ranks_[0]->allocator->freeBlocksNum() == ranks_[0]->baseline
                && ranks_[1]->allocator->freeBlocksNum() == ranks_[1]->baseline;
        }));
        ASSERT_TRUE(until([&] {
            const auto state = command('S');
            return state.resources == 0 && state.computed_buffers == 0
                && state.free_blocks[0] == kBlocks - 1 && state.free_blocks[1] == kBlocks - 1;
        }));
        const auto state = command('S');
        EXPECT_EQ(state.source_requests, 0);
        EXPECT_EQ(state.resources, 0);
        EXPECT_EQ(state.computed_buffers, 0);
        EXPECT_EQ(state.registrations, state.deadline_entries);
        for (int rank = 0; rank < kRanks; ++rank) EXPECT_EQ(state.free_blocks[rank], kBlocks - 1);
        std::cerr << "[P2P-BATCH] seed=" << seed << " accepted=" << accepted << " rejected=" << rejected
                  << " mixed=" << mixed << " P_registrations=" << state.registrations << std::endl;
        last_wave_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - wave_begin).count();
    }
    struct BatchRequest {
        int64_t id = 0;
        int blocks = 0, mode = 0;
        std::vector<KVCacheResourcePtr> resources;
        std::shared_ptr<P2PConnectorAsyncReadContext> context;
        std::chrono::steady_clock::time_point started;
        double latency_ms = -1;
    };
    std::vector<BatchRequest> batch_;
    int64_t batch_id_ = 10000;
    std::shared_ptr<KickoffGate> queue_gate_;
    std::vector<double> last_latencies_;
    double last_wave_seconds_ = 0;
    Fd control_;
    std::string host_, peer_host_;
    Report report_;
    int64_t id_ = 1;
    std::vector<std::unique_ptr<Rank>> ranks_;
    std::unique_ptr<CopyGate> gate_;
    std::shared_ptr<P2PConnectorAsyncReadContext> context_;
    std::weak_ptr<P2PConnectorAsyncReadContext> retired_context_;
};

TEST_F(P2PMultiRankTransferTest, BothRanksTransferRealGpuBytes) {
    prepare();
    start();
    ASSERT_NO_FATAL_FAILURE(expectSuccess());
    ASSERT_NO_FATAL_FAILURE(recover());
}

TEST_F(P2PMultiRankTransferTest, CancelRetainsTargetsWhileNonzeroRankH2dIsBlocked) {
    prepare();
    blockNonzeroCopy();
    start();
    ASSERT_NO_FATAL_FAILURE(expectOnlyNonzeroRankPending());
    ranks_[0]->connector->cancelRead(context_);
    ASSERT_NO_FATAL_FAILURE(expectRetainedThenRelease());
}

TEST_F(P2PMultiRankTransferTest, LoadTimeoutRetainsTargetsWhileNonzeroRankH2dIsBlocked) {
    prepare(3000);
    blockNonzeroCopy();
    const auto begin = std::chrono::steady_clock::now();
    start();
    ASSERT_NO_FATAL_FAILURE(expectOnlyNonzeroRankPending());
    ASSERT_TRUE(until([&] { return context_->done(); }, 5000));
    ASSERT_NO_FATAL_FAILURE(expectLoadTimeout());
    EXPECT_GE(std::chrono::steady_clock::now() - begin, std::chrono::milliseconds(2500));
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(8));
    ASSERT_NO_FATAL_FAILURE(expectRetainedThenRelease());
}

TEST_F(P2PMultiRankTransferTest, NonzeroRankRpcFailureCancelsAlreadyStartedWrites) {
    prepare();
    ranks_[1]->service.fail_read = true;
    blockNonzeroCopy();
    start();
    ASSERT_NO_FATAL_FAILURE(expectOnlyNonzeroRankPending());
    ranks_[1]->service.releaseFailure();
    ASSERT_TRUE(until([&] { return context_->done(); }));
    EXPECT_NE(context_->errorInfo().ToString().find("injected nonzero-rank READ failure"), std::string::npos);
    ASSERT_NO_FATAL_FAILURE(expectRetainedThenRelease());
}

TEST_F(P2PMultiRankTransferTest, StartLoadWaitsForMatchingRegistration) {
    prepare();
    start(false);
    ASSERT_TRUE(until([&] { return command('S').load_entered == 1; }));
    EXPECT_FALSE(context_->done());
    const auto waiting = command('S');
    EXPECT_EQ(waiting.handles[0] + waiting.handles[1], 0);
    command('R');
    command('P');
    ASSERT_NO_FATAL_FAILURE(expectSuccess());
    ASSERT_NO_FATAL_FAILURE(recover());
}

TEST_F(P2PMultiRankTransferTest, MissingRegistrationExpiresWithinLoadBudget) {
    prepare(1000);
    const auto begin = std::chrono::steady_clock::now();
    start(false);
    ASSERT_TRUE(until([&] { return context_->done(); }, 5000));
    ASSERT_NO_FATAL_FAILURE(expectLoadTimeout());
    EXPECT_GE(std::chrono::steady_clock::now() - begin, std::chrono::milliseconds(800));
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(5));
    const auto state = command('S');
    EXPECT_EQ(state.load_entered, 1);
    EXPECT_EQ(state.handles[0] + state.handles[1], 0);
    ASSERT_NO_FATAL_FAILURE(recover());
}

TEST_F(P2PMultiRankTransferTest, CancelInterruptsRegistrationWaitBeforeLoadDeadline) {
    prepare();
    start(false);
    ASSERT_TRUE(until([&] { return command('S').load_entered == 1; }));
    ranks_[0]->connector->cancelRead(context_);
    ASSERT_TRUE(until([&] { return context_->done() && command('S').load_done == 1; }));
    EXPECT_FALSE(context_->success());
    EXPECT_EQ(command('S').handles[0] + command('S').handles[1], 0);
    ASSERT_NO_FATAL_FAILURE(recover());
}

TEST_F(P2PMultiRankTransferTest, ConcurrentRequestsKeepEveryRankPayloadIsolated) {
    ASSERT_NO_FATAL_FAILURE(prepareConcurrent(60000));
    for (int concurrency : {8, 16, 32}) {
        SCOPED_TRACE(concurrency);
        ASSERT_NO_FATAL_FAILURE(runBatch(concurrency, false, false, 20260917 + concurrency));
    }
}

TEST_F(P2PMultiRankTransferTest, ConcurrentSuccessCancelAndTimeoutRecoverTogether) {
    ASSERT_NO_FATAL_FAILURE(prepareConcurrent());
    for (uint32_t seed : {20260917u, 20260918u, 20260919u}) {
        ASSERT_NO_FATAL_FAILURE(runBatch(10, true, false, seed));
        ASSERT_NO_FATAL_FAILURE(runBatch(10, false, false, seed + 100));
    }
}

TEST_F(P2PMultiRankTransferTest, RepeatedOverloadDrainsAndRecoversWithoutRestart) {
    ASSERT_NO_FATAL_FAILURE(prepareConcurrent(60000, 8));
    for (uint32_t round = 0; round < 20; ++round) {
        SCOPED_TRACE(round);
        ASSERT_NO_FATAL_FAILURE(runBatch(8, false, false, 3000 + round));
        ASSERT_NO_FATAL_FAILURE(runBatch(16, false, true, 4000 + round));
        ASSERT_NO_FATAL_FAILURE(runBatch(8, false, false, 5000 + round));
    }
}

// Opt in separately: this keeps both hosts' connectors/stores alive for hours.
// Shortening the duration is useful for debugging, not release soak evidence.
TEST_F(P2PMultiRankTransferTest, DISABLED_SustainedMixedLoadHasBoundedResources) {
    const int duration = setting("P2P_SOAK_SECONDS", 7200, 60, 86400);
    const int concurrency = setting("P2P_SOAK_CONCURRENCY", 8, 5, 16);
    const int pause_ms = setting("P2P_SOAK_PAUSE_MS", 100, 0, 10000);
    const int64_t rss_budget = setting("P2P_SOAK_RSS_GROWTH_MB", 256, 1, 65536) * int64_t{1024};
    const bool check_perf = setting("P2P_SOAK_CHECK_PERF", 0, 0, 1) != 0;
    uint32_t seed = setting("P2P_SOAK_SEED", 20260917, 1, 2147483647);
    ASSERT_NO_FATAL_FAILURE(prepareConcurrent(5000, concurrency));
    // Warm all paths and the maximum simultaneous allocations before baselines.
    ASSERT_NO_FATAL_FAILURE(runBatch(concurrency * 2, false, true, seed++));
    ASSERT_NO_FATAL_FAILURE(runBatch(concurrency, true, false, seed++));
    for (int i = 0; i < 10; ++i) ASSERT_NO_FATAL_FAILURE(runBatch(concurrency, false, false, seed++));
    const auto d_baseline = sampleProcess();
    const auto p_baseline = command('S').process;
    const auto begin = std::chrono::steady_clock::now();
    auto sample_at = begin + std::chrono::seconds(60);
    auto fault_at = begin + std::chrono::minutes(10);
    std::vector<double> latencies;
    double work_seconds = 0, baseline_p99 = 0, baseline_rate = 0;
    int degraded_windows = 0;
    while (std::chrono::steady_clock::now() - begin < std::chrono::seconds(duration)) {
        if (std::chrono::steady_clock::now() >= fault_at) {
            ASSERT_NO_FATAL_FAILURE(runBatch(concurrency, true, false, seed++));
            ASSERT_NO_FATAL_FAILURE(runBatch(concurrency * 2, false, true, seed++));
            fault_at = std::chrono::steady_clock::now() + std::chrono::minutes(10);
        }
        ASSERT_NO_FATAL_FAILURE(runBatch(concurrency, false, false, seed++));
        latencies.insert(latencies.end(), last_latencies_.begin(), last_latencies_.end());
        work_seconds += last_wave_seconds_;
        if (std::chrono::steady_clock::now() >= sample_at) {
            // Logical terminal records are allowed until their own horizon.
            // Only overdue records are leaks; active resources were checked at
            // every wave boundary, without restarting either side.
            ASSERT_TRUE(until([&] {
                return command('S').expired_records == 0
                    && ranks_[0]->pendingCancels(true) == 0 && ranks_[1]->pendingCancels(true) == 0;
            }));
            const auto p = command('S');
            const auto d = sampleProcess();
            EXPECT_LE(d.rss_kib, d_baseline.rss_kib + rss_budget);
            EXPECT_LE(p.process.rss_kib, p_baseline.rss_kib + rss_budget);
            EXPECT_LE(d.fds, d_baseline.fds + 16);
            EXPECT_LE(p.process.fds, p_baseline.fds + 16);
            EXPECT_LE(d.threads, d_baseline.threads + 16);
            EXPECT_LE(p.process.threads, p_baseline.threads + 16);
            std::sort(latencies.begin(), latencies.end());
            ASSERT_FALSE(latencies.empty());
            const double p99 = latencies[(latencies.size() - 1) * 99 / 100];
            const double rate = latencies.size() / work_seconds;
            if (baseline_p99 == 0) { baseline_p99 = p99; baseline_rate = rate; }
            degraded_windows = p99 > baseline_p99 * 1.20 || rate < baseline_rate * .90 ? degraded_windows + 1 : 0;
            if (check_perf) EXPECT_LT(degraded_windows, 3);
            size_t gpu_free = 0, gpu_total = 0;
            checkCuda(cudaMemGetInfo(&gpu_free, &gpu_total));
            std::cerr << "[P2P-SOAK] seconds="
                      << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - begin).count()
                      << " seed=" << seed << " p99_ms=" << p99 << " connector_req_s=" << rate
                      << " degraded_windows=" << degraded_windows << " D_rss_kib=" << d.rss_kib
                      << " P_rss_kib=" << p.process.rss_kib << " D_cpu_us=" << d.cpu_us
                      << " P_cpu_us=" << p.process.cpu_us << " D_fd=" << d.fds << " P_fd=" << p.process.fds
                      << " D_threads=" << d.threads << " P_threads=" << p.process.threads
                      << " gpu_used_bytes=" << gpu_total - gpu_free << " P_records=" << p.registrations
                      << " P_source_markers=" << p.source_markers << " D_cancel_markers="
                      << ranks_[0]->pendingCancels() + ranks_[1]->pendingCancels() << std::endl;
            latencies.clear();
            work_seconds = 0;
            sample_at = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(pause_ms));
    }
    ASSERT_NO_FATAL_FAILURE(runBatch(concurrency, false, false, seed));
}

}  // namespace
}  // namespace rtp_llm
