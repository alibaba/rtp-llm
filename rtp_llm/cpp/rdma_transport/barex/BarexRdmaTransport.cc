#include "rtp_llm/cpp/rdma_transport/barex/BarexRdmaTransport.h"

#include <algorithm>
#include <type_traits>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/cuda.h>

#include "aios/network/accl-barex/include/accl/barex/highlevel/xclient_transport.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection_callback.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xserver_transport.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xtransport_config.h"
#include "aios/network/accl-barex/include/accl/barex/xchannel.h"
#include "aios/network/accl-barex/include/accl/barex/xsimple_mempool.h"
#include "aios/network/accl-barex/metric/KMonitorAcclMetricReporter.h"
#include "aios/network/rdma/accl/RdmaMempool.h"
#include "aios/network/rdma/accl/RdmaMempoolFactory.h"
#include "autil/NetUtil.h"

#include "rtp_llm/cpp/rdma_transport/RdmaTransport.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace rdma_transport {

namespace {

constexpr uint32_t kIoThreadCount = 2;
constexpr uint32_t kWorkerThreadCount = 2;
constexpr uint32_t kReadAttempts = 2;
constexpr uint64_t kMinGpuAllocBytes = 64UL * 1024;
constexpr auto kConnectionRetryBackoff = std::chrono::seconds(1);

using RdmaAllocFn = std::function<::accl::barex::BarexResult()>;
using RdmaShrinkFn = std::function<::accl::barex::BarexResult(float)>;

::accl::barex::BarexResult allocWithOomShrinkRetry(const RdmaAllocFn& alloc, const RdmaShrinkFn& shrink) {
    auto result = alloc();
    if (result != ::accl::barex::BAREX_ERR_OOM) {
        return result;
    }
    shrink(0.5F);
    result = alloc();
    if (result != ::accl::barex::BAREX_ERR_OOM) {
        return result;
    }
    shrink(1.0F);
    return alloc();
}

uint64_t alignUp(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

::accl::barex::BarexResult allocGpuBufferWithOomRecovery(const std::shared_ptr<::accl::barex::XSimpleMempool>& pool,
                                                         ::accl::barex::memp_t& memory,
                                                         uint64_t size,
                                                         int device_id) {
    auto alloc = [&]() { return pool->AllocLimitBuffer(memory, size, ::accl::barex::GPU, /*nic_id=*/-1, device_id); };
    auto shrink = [&](float ratio) {
        const auto before = pool->Stats(::accl::barex::GPU);
        RTP_LLM_LOG_WARNING("tensor rdma GPU allocation hit pool OOM; shrinking %.0f%% of free buckets "
                            "(total=%lu busy=%lu free=%lu)",
                            ratio * 100.0F,
                            before.total_bytes,
                            before.busy_bytes,
                            before.free_bytes);
        const auto result = pool->Shrink(::accl::barex::GPU, ratio, 0.0F);
        const auto after = pool->Stats(::accl::barex::GPU);
        RTP_LLM_LOG_WARNING("tensor rdma GPU pool shrink finished, result=%d (total=%lu busy=%lu free=%lu)",
                            result,
                            after.total_bytes,
                            after.busy_bytes,
                            after.free_bytes);
        return result;
    };
    return allocWithOomShrinkRetry(alloc, shrink);
}

TensorDataType toRdmaDtype(torch::ScalarType type) {
    switch (type) {
        case torch::kFloat32:
            return TensorDataType::FLOAT32;
        case torch::kInt32:
            return TensorDataType::INT32;
        case torch::kFloat16:
            return TensorDataType::FLOAT16;
        case torch::kBFloat16:
            return TensorDataType::BFLOAT16;
        default:
            throw std::runtime_error("unsupported tensor dtype");
    }
}

torch::ScalarType fromRdmaDtype(TensorDataType type) {
    switch (type) {
        case TensorDataType::FLOAT32:
            return torch::kFloat32;
        case TensorDataType::INT32:
            return torch::kInt32;
        case TensorDataType::FLOAT16:
            return torch::kFloat16;
        case TensorDataType::BFLOAT16:
            return torch::kBFloat16;
        default:
            throw std::runtime_error("unsupported protobuf dtype");
    }
}

int currentCudaDevice() {
    try {
        return static_cast<int>(c10::cuda::current_device());
    } catch (...) {
        return 0;
    }
}

class ServerConnectionCallback: public ::accl::barex::XConnectionCallback {
public:
    void OnRecvCall(const std::shared_ptr<::accl::barex::XConnection>&, const ::accl::barex::XMessage&) override {}

    void OnConnectSuccess(const std::shared_ptr<::accl::barex::XConnection>& connection) override {
        auto peer = connection->GetPeerAddrAndPort();
        RTP_LLM_LOG_DEBUG("tensor rdma accepted connection from %s:%d", peer.first.c_str(), peer.second);
    }

    void OnConnectFailed(const std::shared_ptr<::accl::barex::XConnection>& connection) override {
        auto peer = connection->GetPeerAddrAndPort();
        RTP_LLM_LOG_WARNING("tensor rdma server connection failed for %s:%d", peer.first.c_str(), peer.second);
    }
};

class ReadCleanupHandoff {
public:
    ReadCleanupHandoff(RdmaDeferredReadQueue::Ticket ticket, bool* release_deferred) noexcept:
        ticket_(std::move(ticket)), release_deferred_(release_deferred) {
        if (release_deferred_) {
            *release_deferred_ = true;
        }
    }
    ~ReadCleanupHandoff() {
        if (ticket_) {
            ticket_->store(RdmaDeferredReadQueue::Ownership::DEFERRED, std::memory_order_release);
        }
    }
    void returnToCaller() noexcept {
        ticket_->store(RdmaDeferredReadQueue::Ownership::CALLER, std::memory_order_release);
        ticket_.reset();
        if (release_deferred_) {
            *release_deferred_ = false;
        }
    }

private:
    RdmaDeferredReadQueue::Ticket ticket_;
    bool* release_deferred_;
};

struct ReadState {
    struct ChunkResult {
        bool done{false};
        bool ok{false};
    };

    ReadState(std::shared_ptr<::accl::barex::XSimpleMempool> pool, int device_id):
        pool(std::move(pool)), device_id(device_id) {}

    ~ReadState() {
        if (memory.buf != nullptr) {
            at::cuda::CUDAGuard device_guard(device_id);
            pool->ReleaseBuffer(memory.buf, ::accl::barex::GPU);
        }
    }

    std::shared_ptr<::accl::barex::XSimpleMempool> pool;
    int device_id;
    ::accl::barex::memp_t memory{};
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<ChunkResult> chunks;
    size_t completed{0};
    std::shared_ptr<std::atomic<bool>> completion_confirmed{std::make_shared<std::atomic<bool>>(false)};
};

}  // namespace

struct BarexRdmaConnectionState {
    enum class Status {
        CONNECTING,
        CONNECTED,
        FAILED
    };

    explicit BarexRdmaConnectionState(size_t expected_count): target_count(expected_count) {}

    std::mutex mutex;
    std::condition_variable cv;
    Status status{Status::CONNECTING};
    std::vector<std::shared_ptr<::accl::barex::XConnection>> connections;
    const size_t target_count;
    size_t pending{0};
    size_t next{0};
    bool retired{false};
    std::chrono::steady_clock::time_point retry_after;
};

namespace {

class ClientConnectionCallback: public ::accl::barex::XConnectionCallback {
public:
    explicit ClientConnectionCallback(std::weak_ptr<BarexRdmaConnectionState> state): state_(std::move(state)) {}

    void OnRecvCall(const std::shared_ptr<::accl::barex::XConnection>&, const ::accl::barex::XMessage&) override {}

    void OnConnectSuccess(const std::shared_ptr<::accl::barex::XConnection>& connection) override {
        bool close_connection = false;
        if (auto state = state_.lock()) {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->pending > 0) {
                --state->pending;
            }
            if (state->retired) {
                close_connection = true;
            } else {
                state->connections.push_back(connection);
                state->status = BarexRdmaConnectionState::Status::CONNECTED;
            }
            state->cv.notify_all();
        } else {
            close_connection = true;
        }
        if (close_connection && connection) {
            connection->Close();
        }
    }

    void OnConnectFailed(const std::shared_ptr<::accl::barex::XConnection>& connection) override {
        if (auto state = state_.lock()) {
            std::lock_guard<std::mutex> lock(state->mutex);
            auto it = std::find(state->connections.begin(), state->connections.end(), connection);
            if (it != state->connections.end()) {
                state->connections.erase(it);
            } else if (state->pending > 0) {
                --state->pending;
            }
            if (!state->connections.empty()) {
                state->status = BarexRdmaConnectionState::Status::CONNECTED;
                state->next %= state->connections.size();
            } else if (state->pending == 0) {
                state->status = BarexRdmaConnectionState::Status::FAILED;
            } else {
                state->status = BarexRdmaConnectionState::Status::CONNECTING;
            }
            state->retry_after = std::chrono::steady_clock::now() + kConnectionRetryBackoff;
            state->cv.notify_all();
        }
    }

private:
    std::weak_ptr<BarexRdmaConnectionState> state_;
};

}  // namespace

BarexRdmaRuntime::BarexRdmaRuntime(const RdmaConfig& config, Mode mode, int device_id):
    config_(config), mode_(mode), device_id_(device_id >= 0 ? device_id : currentCudaDevice()) {}

BarexRdmaRuntime::~BarexRdmaRuntime() {
    at::cuda::CUDAGuard device_guard(device_id_);
    running_.store(false);
    if (gc_thread_.joinable()) {
        gc_thread_.join();
    }

    std::vector<Slot> slots;
    {
        std::lock_guard<std::mutex> lock(slots_mutex_);
        for (const auto& item : slots_) {
            slots.push_back(item.second);
        }
        slots_.clear();
    }
    for (const auto& slot : slots) {
        releaseSlot(slot);
    }

    std::vector<std::shared_ptr<::accl::barex::XConnection>> connections;
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        for (const auto& item : connections_) {
            std::lock_guard<std::mutex> state_lock(item.second->mutex);
            item.second->retired = true;
            connections.insert(connections.end(), item.second->connections.begin(), item.second->connections.end());
            item.second->connections.clear();
        }
        connections_.clear();
    }
    for (const auto& connection : connections) {
        connection->Close();
    }
    client_.reset();
    server_.reset();
    deferred_reads_.reclaimCompleted();
    deferred_reads_.clearAfterTransportShutdown();
    x_mempool_.reset();
    mempool_.reset();
    if (factory_acquired_) {
        ::arpc::RdmaMempoolFactory::Shutdown();
    }
}

bool BarexRdmaRuntime::init() {
    if (config_.qp_count == 0) {
        RTP_LLM_LOG_WARNING("tensor rdma init failed: qp count must be positive");
        return false;
    }
    at::cuda::CUDAGuard device_guard(device_id_);
    if (!::arpc::RdmaMempoolFactory::Init()) {
        RTP_LLM_LOG_WARNING("tensor rdma init failed: RdmaMempoolFactory::Init failed");
        return false;
    }
    factory_acquired_ = true;
    mempool_ = ::arpc::RdmaMempoolFactory::GetMempool();
    x_mempool_ = mempool_ ? mempool_->GetXMempool() : nullptr;
    if (!x_mempool_) {
        RTP_LLM_LOG_WARNING("tensor rdma init failed: Barex mempool is unavailable");
        return false;
    }

    auto metrics = std::make_shared<::accl::barex::KMonitorAcclMetricReporter>(true, kmonitor::FATAL);
    if (!metrics->init()) {
        RTP_LLM_LOG_WARNING("tensor rdma init failed: metric reporter init failed");
        return false;
    }

    if (mode_ == Mode::EXPORT) {
        ::accl::barex::XServerTransportConfig transport_config(kIoThreadCount, kWorkerThreadCount, x_mempool_, metrics);
        server_ = ::accl::barex::XServerTransport::NewInstance(transport_config);
        if (!server_) {
            RTP_LLM_LOG_WARNING("tensor rdma export init failed: cannot create Barex server");
            return false;
        }

        rdma_port_ = config_.port > 0 ? static_cast<uint32_t>(config_.port) : autil::NetUtil::randomPort();
        auto callback = std::make_shared<ServerConnectionCallback>();
        if (server_->Listen(::accl::barex::TCP_V4, rdma_port_, callback) != ::accl::barex::BAREX_SUCCESS) {
            RTP_LLM_LOG_WARNING("tensor rdma export init failed: listen on port %u failed", rdma_port_);
            return false;
        }

        rdma_ip_ = config_.bind_ip.empty() ? autil::NetUtil::getBindIp() : config_.bind_ip;
        if (rdma_ip_.empty()) {
            RTP_LLM_LOG_WARNING("tensor rdma export init failed: no routable bind IP");
            return false;
        }
        running_.store(true);

        RTP_LLM_LOG_INFO("tensor rdma export listening on %s:%u", rdma_ip_.c_str(), rdma_port_);
        return true;
    }

    ::accl::barex::XClientTransportConfig transport_config(kIoThreadCount, kWorkerThreadCount, x_mempool_, metrics);
    client_ = ::accl::barex::XClientTransport::NewInstance(transport_config);
    if (!client_) {
        RTP_LLM_LOG_WARNING("tensor rdma read init failed: cannot create Barex client");
        return false;
    }
    RTP_LLM_LOG_INFO("tensor rdma reader initialized");
    running_.store(true);
    gc_thread_ = std::thread(&BarexRdmaRuntime::gcLoop, this);
    return true;
}

std::string BarexRdmaRuntime::genLeaseId() {
    return rdma_ip_ + ":" + std::to_string(rdma_port_) + ":" + std::to_string(lease_seq_.fetch_add(1)) + ":"
           + std::to_string(currentTimeUs());
}

void BarexRdmaRuntime::releaseSlot(const Slot& slot) {
    if (slot.addr && x_mempool_) {
        at::cuda::CUDAGuard device_guard(device_id_);
        x_mempool_->ReleaseBuffer(slot.addr, ::accl::barex::GPU);
    }
}

bool BarexRdmaRuntime::create(const std::vector<torch::Tensor>& tensors, RdmaDescriptor* descriptor) {
    if (mode_ != Mode::EXPORT || descriptor == nullptr || tensors.empty()) {
        RTP_LLM_LOG_WARNING("tensor rdma export: invalid mode, descriptor, or tensor list");
        return false;
    }
    at::cuda::CUDAGuard device_guard(device_id_);

    struct Segment {
        uint64_t offset;
        uint64_t nbytes;
        TensorDataType dtype;
    };
    std::vector<Segment> segments;
    segments.reserve(tensors.size());
    uint64_t total = 0;
    try {
        for (const auto& tensor : tensors) {
            if (!tensor.defined() || !tensor.is_cuda()) {
                RTP_LLM_LOG_WARNING("tensor rdma export: every tensor must be a defined CUDA tensor");
                return false;
            }
            if (tensor.device().index() != device_id_) {
                RTP_LLM_LOG_WARNING("tensor rdma export: tensor device %d differs from configured device %d",
                                    tensor.device().index(),
                                    device_id_);
                return false;
            }
            const uint64_t nbytes = static_cast<uint64_t>(tensor.numel()) * tensor.element_size();
            if (nbytes == 0) {
                RTP_LLM_LOG_WARNING("tensor rdma export: empty tensor is not supported");
                return false;
            }
            const uint64_t offset = alignUp(total, kRdmaSlotAlign);
            if (offset > std::numeric_limits<uint64_t>::max() - nbytes) {
                return false;
            }
            segments.push_back({offset, nbytes, toRdmaDtype(tensor.scalar_type())});
            total = offset + nbytes;
        }
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("tensor rdma export validation failed: %s", e.what());
        return false;
    }

    if (total == 0 || (config_.max_slot_bytes > 0 && total > static_cast<uint64_t>(config_.max_slot_bytes))) {
        RTP_LLM_LOG_WARNING(
            "tensor rdma export: packed tensors %lu exceed slot limit %ld", total, config_.max_slot_bytes);
        return false;
    }

    ::accl::barex::memp_t memory;
    auto result = allocGpuBufferWithOomRecovery(x_mempool_, memory, total, device_id_);
    if (result != ::accl::barex::BAREX_SUCCESS || memory.buf == nullptr) {
        RTP_LLM_LOG_WARNING(
            "tensor rdma export: registered GPU allocation failed, result=%d, nbytes=%lu", result, total);
        return false;
    }

    const uint64_t actual_bytes = memory.buf_len;

    try {
        for (size_t i = 0; i < tensors.size(); ++i) {
            torch::Tensor source = tensors[i].contiguous();
            auto options = torch::TensorOptions().dtype(source.scalar_type()).device(source.device());
            torch::Tensor target = torch::from_blob(memory.buf + segments[i].offset, source.sizes(), options);
            target.copy_(source);
        }
        // Do not publish the descriptor until packing into registered memory is complete.
        // This fulfills RdmaExport::create's NIC-visibility contract for the Barex provider.
        at::cuda::getCurrentCUDAStream(device_id_).synchronize();
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("tensor rdma export: packing failed: %s", e.what());
        x_mempool_->ReleaseBuffer(memory.buf, ::accl::barex::GPU);
        return false;
    }

    const std::string lease_id = genLeaseId();
    descriptor->lease_id = lease_id;
    descriptor->host = rdma_ip_;
    descriptor->port = rdma_port_;
    descriptor->remote_addr = reinterpret_cast<uint64_t>(memory.buf);
    descriptor->payload_bytes = total;
    descriptor->nic_keys.clear();
    descriptor->tensors.clear();
    for (const auto& item : memory.mrs) {
        RdmaNicKey key;
        key.nic_id = item.first;
        key.rkey = item.second->rkey;
        descriptor->nic_keys.push_back(std::move(key));
    }
    if (descriptor->nic_keys.empty()) {
        RTP_LLM_LOG_WARNING("tensor rdma export: allocation has no registered NIC");
        x_mempool_->ReleaseBuffer(memory.buf, ::accl::barex::GPU);
        return false;
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        TensorMeta meta;
        meta.shape.assign(tensors[i].sizes().begin(), tensors[i].sizes().end());
        meta.dtype = segments[i].dtype;
        meta.offset = segments[i].offset;
        meta.nbytes = segments[i].nbytes;
        descriptor->tensors.push_back(std::move(meta));
    }

    {
        std::lock_guard<std::mutex> lock(slots_mutex_);
        // Do not expire slots by wall time: a timed-out peer may still have an
        // unconfirmed NIC READ. Bound retained leases instead of reusing memory.
        if (actual_bytes > static_cast<uint64_t>(config_.max_receipt_bytes)
            || inflight_bytes_ > static_cast<uint64_t>(config_.max_receipt_bytes) - actual_bytes) {
            x_mempool_->ReleaseBuffer(memory.buf, ::accl::barex::GPU);
            descriptor->lease_id.clear();
            return false;
        }
        slots_[lease_id] = Slot{memory.buf, actual_bytes};
        inflight_bytes_ += actual_bytes;
    }
    RTP_LLM_LOG_DEBUG("tensor rdma export: lease_id=%s addr=%p logical=%lu allocated=%lu tensors=%zu",
                      lease_id.c_str(),
                      memory.buf,
                      total,
                      actual_bytes,
                      descriptor->tensors.size());
    return true;
}

void BarexRdmaRuntime::release(const std::vector<std::string>& lease_ids) {
    at::cuda::CUDAGuard device_guard(device_id_);
    std::vector<Slot> released;
    {
        std::lock_guard<std::mutex> lock(slots_mutex_);
        for (const auto& lease_id : lease_ids) {
            auto it = slots_.find(lease_id);
            if (it == slots_.end()) {
                RTP_LLM_LOG_DEBUG("tensor rdma release: unknown or expired lease %s", lease_id.c_str());
                continue;
            }
            inflight_bytes_ -= it->second.allocated_bytes;
            released.push_back(it->second);
            slots_.erase(it);
        }
    }
    for (const auto& slot : released) {
        releaseSlot(slot);
    }
}

void BarexRdmaRuntime::gcLoop() {
    while (running_.load()) {
        deferred_reads_.reclaimCompleted();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

std::vector<std::shared_ptr<::accl::barex::XConnection>>
BarexRdmaRuntime::getConnections(const std::string& ip, uint32_t port, int64_t timeout_ms) {
    const std::string endpoint = ip + ":" + std::to_string(port);
    std::shared_ptr<BarexRdmaConnectionState> state;
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto it = connections_.find(endpoint);
        if (it == connections_.end()) {
            state = std::make_shared<BarexRdmaConnectionState>(config_.qp_count);
            connections_[endpoint] = state;
        } else {
            state = it->second;
        }
    }

    size_t connect_count = 0;
    size_t active_count = 0;
    size_t pending_count = 0;
    bool wait_for_connections = false;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        active_count = state->connections.size();
        wait_for_connections = active_count == 0;
        const size_t provisioned = active_count + state->pending;
        if (!state->retired && provisioned < state->target_count
            && std::chrono::steady_clock::now() >= state->retry_after) {
            connect_count = state->target_count - provisioned;
            state->pending += connect_count;
            state->status = BarexRdmaConnectionState::Status::CONNECTING;
        }
        pending_count = state->pending;
    }
    if (connect_count > 0) {
        RTP_LLM_LOG_INFO("tensor rdma connecting %zu QP(s) to %s: active=%zu pending=%zu target=%zu",
                         connect_count,
                         endpoint.c_str(),
                         active_count,
                         pending_count,
                         state->target_count);
        auto callback = std::make_shared<ClientConnectionCallback>(state);
        for (size_t i = 0; i < connect_count; ++i) {
            client_->Connect(::accl::barex::TCP_V4, ip, port, callback);
        }
    }

    std::unique_lock<std::mutex> lock(state->mutex);
    if (wait_for_connections && state->pending > 0 && state->connections.size() < state->target_count) {
        const int64_t connect_timeout_ms = std::min<int64_t>(config_.connect_timeout_ms, timeout_ms);
        state->cv.wait_for(lock, std::chrono::milliseconds(connect_timeout_ms), [&]() {
            return state->pending == 0 || state->connections.size() >= state->target_count;
        });
    }
    if (!state->connections.empty()) {
        std::vector<std::shared_ptr<::accl::barex::XConnection>> result;
        result.reserve(state->connections.size());
        const size_t start = state->next++ % state->connections.size();
        for (size_t i = 0; i < state->connections.size(); ++i) {
            result.push_back(state->connections[(start + i) % state->connections.size()]);
        }
        return result;
    }
    lock.unlock();
    std::vector<std::shared_ptr<::accl::barex::XConnection>> stale_connections;
    {
        std::lock_guard<std::mutex> connections_lock(connections_mutex_);
        auto it = connections_.find(endpoint);
        if (it != connections_.end() && it->second == state) {
            std::lock_guard<std::mutex> state_lock(state->mutex);
            state->retired = true;
            stale_connections.swap(state->connections);
            connections_.erase(it);
        }
    }
    for (const auto& connection : stale_connections) {
        connection->Close();
    }
    RTP_LLM_LOG_WARNING("tensor rdma connect to %s timed out or failed", endpoint.c_str());
    return {};
}

void BarexRdmaRuntime::invalidateConnection(const std::string& ip,
                                            uint32_t port,
                                            const std::shared_ptr<::accl::barex::XConnection>& connection) {
    const std::string endpoint = ip + ":" + std::to_string(port);
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto it = connections_.find(endpoint);
        if (it != connections_.end()) {
            const auto state = it->second;
            std::lock_guard<std::mutex> state_lock(state->mutex);
            auto& state_connections = state->connections;
            state_connections.erase(std::remove(state_connections.begin(), state_connections.end(), connection),
                                    state_connections.end());
            state->retry_after = std::chrono::steady_clock::now() + kConnectionRetryBackoff;
            if (state_connections.empty() && state->pending == 0) {
                state->status = BarexRdmaConnectionState::Status::FAILED;
                state->retired = true;
                connections_.erase(it);
            } else if (state_connections.empty()) {
                state->status = BarexRdmaConnectionState::Status::CONNECTING;
            } else {
                state->status = BarexRdmaConnectionState::Status::CONNECTED;
                state->next %= state_connections.size();
            }
        }
    }
    if (connection) {
        connection->Close();
    }
}

bool BarexRdmaRuntime::readOne(const RdmaDescriptor& descriptor,
                               std::vector<torch::Tensor>* tensors,
                               int64_t timeout_ms,
                               std::function<void()> release_after_completion,
                               bool* release_deferred) {
    if (mode_ != Mode::READ || tensors == nullptr || descriptor.tensors.empty() || descriptor.payload_bytes == 0
        || descriptor.nic_keys.empty()) {
        RTP_LLM_LOG_WARNING("tensor rdma read: invalid mode, output, or descriptor");
        return false;
    }
    if (config_.max_slot_bytes > 0 && descriptor.payload_bytes > static_cast<uint64_t>(config_.max_slot_bytes)) {
        RTP_LLM_LOG_WARNING("tensor rdma read: descriptor %lu exceeds slot limit %ld",
                            descriptor.payload_bytes,
                            config_.max_slot_bytes);
        return false;
    }
    const std::string& rdma_ip = descriptor.host;
    const uint32_t rdma_port = descriptor.port;
    if (rdma_ip.empty() || rdma_port == 0) {
        RTP_LLM_LOG_WARNING("tensor rdma read: descriptor has no endpoint");
        return false;
    }

    const int64_t budget_ms =
        timeout_ms > 0 ? std::min<int64_t>(timeout_ms, config_.read_timeout_ms) : config_.read_timeout_ms;
    const auto operation_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(budget_ms);
    std::shared_ptr<ReadState> state;
    double read_us = 0;
    std::vector<int> selected_local_nics;
    std::vector<int> selected_remote_nics;
    for (uint32_t attempt = 0; attempt < kReadAttempts; ++attempt) {
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(operation_deadline - std::chrono::steady_clock::now())
                .count();
        if (remaining <= 0) {
            break;
        }
        auto connections = getConnections(rdma_ip, rdma_port, remaining);
        if (connections.empty()) {
            RTP_LLM_LOG_WARNING("tensor rdma read: connection attempt %u/%u failed", attempt + 1, kReadAttempts);
            continue;
        }

        auto candidate = std::make_shared<ReadState>(x_mempool_, device_id_);
        auto result =
            allocGpuBufferWithOomRecovery(x_mempool_, candidate->memory, descriptor.payload_bytes, device_id_);
        if (result != ::accl::barex::BAREX_SUCCESS || candidate->memory.buf == nullptr) {
            RTP_LLM_LOG_WARNING(
                "tensor rdma read: receive allocation failed, result=%d, nbytes=%lu", result, descriptor.payload_bytes);
            return false;
        }

        struct ReadPath {
            std::shared_ptr<::accl::barex::XConnection> connection;
            ::accl::barex::XChannel* channel;
            int local_nic;
            int remote_nic;
            uint32_t remote_rkey;
        };
        std::vector<ReadPath> paths;
        for (const auto& connection : connections) {
            auto* channel = connection ? connection->GetChannel() : nullptr;
            if (channel == nullptr) {
                invalidateConnection(rdma_ip, rdma_port, connection);
                continue;
            }
            const int local_nic = channel->GetLocalNicId();
            if (candidate->memory.mrs.find(local_nic) == candidate->memory.mrs.end()) {
                RTP_LLM_LOG_WARNING("tensor rdma read: no local MR for NIC %d", local_nic);
                invalidateConnection(rdma_ip, rdma_port, connection);
                continue;
            }
            const int remote_nic = channel->GetPeerNicId();
            auto remote_rkey = std::find_if(descriptor.nic_keys.begin(),
                                            descriptor.nic_keys.end(),
                                            [remote_nic](const auto& item) { return item.nic_id == remote_nic; });
            if (remote_rkey == descriptor.nic_keys.end()) {
                RTP_LLM_LOG_WARNING("tensor rdma read: no remote rkey for NIC %d", remote_nic);
                invalidateConnection(rdma_ip, rdma_port, connection);
                continue;
            }
            paths.push_back({connection, channel, local_nic, remote_nic, remote_rkey->rkey});
        }
        if (paths.empty()) {
            continue;
        }
        if (paths.size() > descriptor.payload_bytes) {
            paths.resize(descriptor.payload_bytes);
        }

        candidate->chunks.resize(paths.size());
        auto ticket = deferred_reads_.prepare(candidate->completion_confirmed, candidate, release_after_completion);
        // Allocate ownership tracking before any NIC submission. On exceptional
        // or timeout paths the reclaimer owns both local storage and remote lease.
        ReadCleanupHandoff handoff(std::move(ticket), release_deferred);
        const auto read_start = std::chrono::steady_clock::now();
        uint64_t offset = 0;
        for (size_t i = 0; i < paths.size(); ++i) {
            const uint64_t chunks_left = paths.size() - i;
            const uint64_t chunk_bytes = (descriptor.payload_bytes - offset + chunks_left - 1) / chunks_left;

            ::accl::barex::rw_memp_t read_memory;
            read_memory.data = candidate->memory;
            read_memory.data.buf = candidate->memory.buf + offset;
            read_memory.data.buf_len = chunk_bytes;
            read_memory.data.mr = candidate->memory.mrs.at(paths[i].local_nic);
            read_memory.r_addr = descriptor.remote_addr + offset;
            read_memory.r_key = paths[i].remote_rkey;
            read_memory.r_ttl_ms = currentTimeMs() + remaining;
            std::vector<::accl::barex::rw_memp_t> reads{read_memory};

            result = paths[i].channel->ReadBatch(
                reads,
                [candidate, i](::accl::barex::Status status) {
                    std::lock_guard<std::mutex> lock(candidate->mutex);
                    if (candidate->chunks[i].done) {
                        return;
                    }
                    candidate->chunks[i].done = true;
                    candidate->chunks[i].ok = status.IsOk();
                    ++candidate->completed;
                    if (candidate->completed == candidate->chunks.size()) {
                        candidate->completion_confirmed->store(true, std::memory_order_release);
                    }
                    candidate->cv.notify_one();
                },
                false);
            if (result != ::accl::barex::BAREX_SUCCESS) {
                RTP_LLM_LOG_WARNING("tensor rdma read: ReadBatch submission failed, qp=%zu result=%d", i, result);
                // A failed submit is not a completion receipt. A provider may
                // have submitted part of the work; only its callback can retire it.
                invalidateConnection(rdma_ip, rdma_port, paths[i].connection);
            }
            offset += chunk_bytes;
        }

        bool read_ok = false;
        std::vector<bool> path_ok(paths.size(), false);
        {
            std::unique_lock<std::mutex> lock(candidate->mutex);
            const auto remaining_wait = std::chrono::duration_cast<std::chrono::milliseconds>(
                                            operation_deadline - std::chrono::steady_clock::now())
                                            .count();
            if (remaining_wait <= 0 || !candidate->cv.wait_for(lock, std::chrono::milliseconds(remaining_wait), [&]() {
                    return candidate->completed == candidate->chunks.size();
                })) {
                RTP_LLM_LOG_WARNING("tensor rdma read: timed out after %ldms", budget_ms);
            } else {
                read_ok = std::all_of(
                    candidate->chunks.begin(), candidate->chunks.end(), [](const auto& chunk) { return chunk.ok; });
                if (!read_ok) {
                    RTP_LLM_LOG_WARNING("tensor rdma read: parallel completion failed");
                }
            }
            for (size_t i = 0; i < candidate->chunks.size(); ++i) {
                path_ok[i] = candidate->chunks[i].done && candidate->chunks[i].ok;
            }
        }
        const bool completed = candidate->completion_confirmed->load(std::memory_order_acquire);
        if (completed) {
            handoff.returnToCaller();
        }
        if (!completed) {
            for (const auto& path : paths) {
                invalidateConnection(rdma_ip, rdma_port, path.connection);
            }
            return false;
        }
        if (!read_ok) {
            for (size_t i = 0; i < paths.size(); ++i) {
                if (!path_ok[i]) {
                    invalidateConnection(rdma_ip, rdma_port, paths[i].connection);
                }
            }
            continue;
        }

        read_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - read_start).count();
        selected_local_nics.clear();
        selected_remote_nics.clear();
        for (const auto& path : paths) {
            selected_local_nics.push_back(path.local_nic);
            selected_remote_nics.push_back(path.remote_nic);
        }
        state = std::move(candidate);
        break;
    }
    if (!state) {
        RTP_LLM_LOG_WARNING("tensor rdma read: all %u attempts failed", kReadAttempts);
        return false;
    }

    const double bw_gbytes_per_sec =
        read_us > 0 ? static_cast<double>(descriptor.payload_bytes) / read_us / 1000.0 : 0.0;
    std::string local_nic_ids;
    std::string remote_nic_ids;
    for (size_t i = 0; i < selected_local_nics.size(); ++i) {
        if (i > 0) {
            local_nic_ids += ",";
            remote_nic_ids += ",";
        }
        local_nic_ids += std::to_string(selected_local_nics[i]);
        remote_nic_ids += std::to_string(selected_remote_nics[i]);
    }
    RTP_LLM_LOG_INFO("[RDMA-BW] read nbytes=%lu time_us=%.1f bw_GBps=%.2f cuda_device=%d qp_count=%zu "
                     "local_nic_ids=[%s] remote_nic_ids=[%s]",
                     descriptor.payload_bytes,
                     read_us,
                     bw_gbytes_per_sec,
                     device_id_,
                     selected_local_nics.size(),
                     local_nic_ids.c_str(),
                     remote_nic_ids.c_str());

    try {
        tensors->clear();
        tensors->reserve(descriptor.tensors.size());
        for (const auto& tensor : descriptor.tensors) {
            if (tensor.offset > descriptor.payload_bytes || tensor.nbytes > descriptor.payload_bytes - tensor.offset) {
                RTP_LLM_LOG_WARNING("tensor rdma read: tensor range [%lu,+%lu) exceeds slot %lu",
                                    tensor.offset,
                                    tensor.nbytes,
                                    descriptor.payload_bytes);
                return false;
            }
            const auto& shape = tensor.shape;
            uint64_t numel = 1;
            for (int64_t dim : shape) {
                if (dim < 0 || (dim > 0 && numel > std::numeric_limits<uint64_t>::max() / dim)) {
                    RTP_LLM_LOG_WARNING("tensor rdma read: invalid or overflowing tensor shape");
                    return false;
                }
                numel *= static_cast<uint64_t>(dim);
            }
            const auto scalar_type = fromRdmaDtype(tensor.dtype);
            auto options = torch::TensorOptions().dtype(scalar_type).device(torch::Device(torch::kCUDA, device_id_));
            const uint64_t element_bytes = torch::elementSize(scalar_type);
            if (element_bytes > 0 && numel > std::numeric_limits<uint64_t>::max() / element_bytes) {
                RTP_LLM_LOG_WARNING("tensor rdma read: tensor byte size overflows");
                return false;
            }
            const uint64_t expected_bytes = numel * element_bytes;
            if (expected_bytes != tensor.nbytes) {
                RTP_LLM_LOG_WARNING("tensor rdma read: manifest shape/dtype size %lu differs from nbytes %lu",
                                    expected_bytes,
                                    tensor.nbytes);
                return false;
            }
            torch::Tensor view = torch::from_blob(state->memory.buf + tensor.offset, shape, options);
            tensors->emplace_back(view.clone());
        }
        at::cuda::getCurrentCUDAStream(device_id_).synchronize();
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("tensor rdma read: materializing tensors failed: %s", e.what());
        tensors->clear();
        return false;
    }
    return true;
}

RdmaReadResult BarexRdmaRuntime::read(const std::vector<RdmaDescriptor>& descriptors,
                                      int64_t timeout_ms,
                                      std::function<void()> release_after_completion,
                                      bool* release_deferred) {
    RdmaReadResult result;
    if (release_deferred) {
        *release_deferred = false;
    }
    if (mode_ != Mode::READ || descriptors.empty()) {
        result.status = ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid mode or empty descriptor list");
        return result;
    }
    at::cuda::CUDAGuard device_guard(device_id_);
    std::unordered_set<std::string> lease_ids;
    for (const auto& descriptor : descriptors) {
        const auto validation = validateRdmaDescriptor(descriptor, config_.max_slot_bytes);
        if (!validation.ok()) {
            result.status = validation;
            return result;
        }
        if (!lease_ids.insert(descriptor.lease_id).second) {
            result.status = ErrorInfo(ErrorCode::INVALID_PARAMS, "duplicate RDMA descriptor lease id");
            return result;
        }
    }

    const int64_t budget_ms = timeout_ms > 0 ? timeout_ms : config_.read_timeout_ms;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(budget_ms);
    for (const auto& descriptor : descriptors) {
        const auto remaining_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (remaining_ms <= 0) {
            result.tensors.clear();
            result.status = ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "RDMA read deadline exceeded");
            return result;
        }
        std::vector<torch::Tensor> tensors;
        if (!readOne(descriptor, &tensors, remaining_ms, release_after_completion, release_deferred)) {
            result.tensors.clear();
            result.status = ErrorInfo(ErrorCode::REMOTE_ALLOCATE_RESOURCE_READ_FAILED, "RDMA read failed");
            return result;
        }
        result.tensors.insert(
            result.tensors.end(), std::make_move_iterator(tensors.begin()), std::make_move_iterator(tensors.end()));
    }
    return result;
}

bool hasRdmaImplementation() {
    return true;
}

namespace {

class BarexRdmaExport final: public RdmaExport {
public:
    explicit BarexRdmaExport(std::shared_ptr<BarexRdmaRuntime> runtime): runtime_(std::move(runtime)) {}

    RdmaDescriptor create(const std::vector<torch::Tensor>& tensors) override {
        RdmaDescriptor descriptor;
        if (!runtime_->create(tensors, &descriptor)) {
            descriptor = RdmaDescriptor{};
        }
        return descriptor;
    }

    void release(const std::vector<std::string>& lease_ids) override {
        runtime_->release(lease_ids);
    }

private:
    std::shared_ptr<BarexRdmaRuntime> runtime_;
};

class BarexRdmaRead final: public RdmaRead {
public:
    explicit BarexRdmaRead(std::shared_ptr<BarexRdmaRuntime> runtime): runtime_(std::move(runtime)) {}

    RdmaReadResult readWithCleanup(const std::vector<RdmaDescriptor>& descriptors,
                                   int64_t timeout_ms,
                                   std::function<void()> release_after_completion,
                                   bool* release_deferred) override {
        return runtime_->read(descriptors, timeout_ms, std::move(release_after_completion), release_deferred);
    }
    RdmaReadResult read(const std::vector<RdmaDescriptor>& descriptors, int64_t timeout_ms) override {
        return runtime_->read(descriptors, timeout_ms);
    }

private:
    std::shared_ptr<BarexRdmaRuntime> runtime_;
};

}  // namespace

std::shared_ptr<RdmaExport> createRdmaExport(const RdmaConfig& config, int device_id) {
    auto runtime = std::make_shared<BarexRdmaRuntime>(config, BarexRdmaRuntime::Mode::EXPORT, device_id);
    if (!runtime->init()) {
        return nullptr;
    }
    return std::make_shared<BarexRdmaExport>(std::move(runtime));
}

std::shared_ptr<RdmaRead> createRdmaRead(const RdmaConfig& config, int device_id) {
    auto runtime = std::make_shared<BarexRdmaRuntime>(config, BarexRdmaRuntime::Mode::READ, device_id);
    if (!runtime->init()) {
        return nullptr;
    }
    return std::make_shared<BarexRdmaRead>(std::move(runtime));
}

}  // namespace rdma_transport
}  // namespace rtp_llm
