#include <algorithm>
#include <cerrno>
#include <csignal>
#include <poll.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew2.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServerNew2.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

extern char** environ;

namespace rtp_llm {
namespace {

constexpr int kLayers         = 3;
constexpr int kBlocks         = 32;
constexpr int kTokensPerBlock = 4;
constexpr int kNewTokens      = 3;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void cudaCheck(cudaError_t status) {
    require(status == cudaSuccess, cudaGetErrorString(status));
}

std::string environment(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value ? value : fallback;
}

int64_t steadyTimeMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct PayloadTimeouts {
    int request_ms;
    int load_ms;
};

PayloadTimeouts payloadTimeouts() {
    const auto read = [](const char* name, int fallback) {
        const auto value = environment(name, "");
        if (value.empty())
            return fallback;
        size_t    parsed  = 0;
        const int timeout = std::stoi(value, &parsed);
        require(parsed == value.size() && timeout > 0 && timeout <= 600000,
                std::string(name) + " must be between 1 and 600000 ms");
        return timeout;
    };
    // Cross-host runs use the same short budgets; wire timeouts do not depend on clock offsets.
    PayloadTimeouts result{read("P2P_SMOKE_REQUEST_TIMEOUT_MS", 10000), read("P2P_SMOKE_LOAD_TIMEOUT_MS", 3000)};
    require(result.load_ms < result.request_ms, "payload load timeout must be less than request timeout");
    return result;
}

bool waitFor(const std::function<bool()>& condition) {
    // Cover the production 20s lease hold after a failed physical transfer.
    const auto deadline = steadyTimeMs() + 30000;
    do {
        if (condition()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (steadyTimeMs() < deadline);
    return condition();
}

// The expected bytes depend on logical coordinates, never physical block IDs.
// The range excludes the destination poison (0xff), including for scale buffers.
std::vector<uint8_t> payload(int64_t request_id, int layer, size_t block, size_t buffer, size_t bytes) {
    std::vector<uint8_t> result(bytes);
    for (size_t offset = 0; offset < bytes; ++offset) {
        result[offset] = (request_id * 17 + layer * 43 + block * 71 + buffer * 97 + offset * 13) % 251;
    }
    return result;
}

void writeBytes(const BlockInfo& info, const std::vector<uint8_t>& bytes) {
    require(info.is_cuda && info.addr && info.size_bytes == bytes.size(), "expected a real GPU cache buffer");
    cudaCheck(cudaMemcpy(info.addr, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
}

std::vector<uint8_t> readBytes(const BlockInfo& info) {
    require(info.is_cuda && info.addr && info.size_bytes > 0, "expected a nonempty GPU cache buffer");
    std::vector<uint8_t> bytes(info.size_bytes);
    cudaCheck(cudaMemcpy(bytes.data(), info.addr, bytes.size(), cudaMemcpyDeviceToHost));
    return bytes;
}

enum class Fault {
    None,
    CorruptByte,
    OmitLastLayer
};

enum class CancelStage : int32_t {
    None,
    BeforePublish,
    AfterFirstLayer,
    AfterFirstDecodeOutput
};

struct CancelControl {
    int32_t stage         = static_cast<int32_t>(CancelStage::None);
    int32_t step_delay_ms = 0;
};

// Only model execution/sampling is substituted. NormalGenerateStream, the FIFO
// state machine, allocation, connector registration and side-channel handling
// are production code. P and D own separate GPU cache pools and TCP/RDMA backends.
class PayloadEngine: public EngineBase {
public:
    PayloadEngine(const CacheConfig& cache_config, const RuntimeConfig& runtime, const PDSepConfig& pd):
        EngineBase(EngineInitParams{}), runtime_(runtime), pd_(pd) {
        model_.max_seq_len                  = 64;
        model_.vocab_size                   = 256;
        model_.attn_config.tokens_per_block = kTokensPerBlock;
        resource_context_.role_type         = pd.role_type;
        resource_context_.decode_entrance   = true;
        resource_context_.reuse_cache       = false;
        resource_context_.cache_manager     = std::make_shared<KVCacheManager>(cache_config,
                                                                           false,
                                                                           nullptr,
                                                                           KVCacheConfig{},
                                                                           ParallelismConfig{},
                                                                           runtime,
                                                                           SpeculativeExecutionConfig{},
                                                                           pd,
                                                                           CacheStoreConfig{});
        require(getCacheManager()->init(), "real P2P cache manager init failed (check backend and ports)");
        require(getCacheManager()->hasP2PConnector(), "P2P connector was not initialized");
        baseline_free = getCacheManager()->freeBlocksNum();
        // Poison every allocatable D block before the first request. Subsequent
        // requests use a different pattern to detect stale data after pool reuse.
        if (pd.role_type == RoleType::DECODE) {
            for (int layer = 0; layer < kLayers; ++layer) {
                for (int block = 1; block < kBlocks; ++block) {
                    for (const auto& info : getCacheManager()->convertIndexToBufferByTag(block, layer, "default")) {
                        writeBytes(info, std::vector<uint8_t>(info.size_bytes, 0xff));
                    }
                }
            }
        }
        scheduler_ = std::make_unique<FIFOScheduler>(
            runtime, model_, pd, ParallelismConfig{}, ModelSpecificConfig{}, getCacheManager());
    }

    ~PayloadEngine() override {
        (void)stop();
    }

    void start() {
        thread_ = std::thread([this] { run(); });
    }

    std::shared_ptr<GenerateStream> makeStream(const std::shared_ptr<GenerateInput>& input) override {
        return std::make_shared<NormalGenerateStream>(input, model_, runtime_, resource_context_, nullptr);
    }

    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>& input) override {
        auto stream = makeStream(input);
        enqueue(stream);
        return stream;
    }

    void enqueue(std::shared_ptr<GenerateStream>& stream) override {
        const auto status = scheduler_->enqueue(stream);
        if (!status.ok()) {
            stream->reportError(ErrorCode::UNKNOWN_ERROR, status.ToString());
        }
    }

    absl::Status stop() override {
        stopping_ = true;
        {
            // Stop must not free a cache pool while the stub is inspecting it.
            std::lock_guard<std::mutex> lock(execution_mutex_);
            if (scheduler_) {
                (void)scheduler_->stop();
            }
        }
        if (thread_.joinable()) {
            thread_.join();
        }
        return absl::OkStatus();
    }

    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("model warmup is outside the payload smoke test");
    }

    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }

    std::string failure() const {
        std::lock_guard<std::mutex> lock(error_mutex_);
        return error_;
    }

    void configureCancel(CancelControl control) {
        require(control.stage >= 0 && control.stage <= static_cast<int32_t>(CancelStage::AfterFirstDecodeOutput)
                    && control.step_delay_ms >= 0 && control.step_delay_ms <= 1000,
                "invalid cancel test control");
        std::lock_guard<std::mutex> lock(execution_mutex_);
        cancel_control_ = control;
        layer_progress_.clear();
        decode_steps_.clear();
        next_step_ms_.clear();
        stage_reached = false;
    }

    uint64_t blockRefs() const {
        // The target already enables -fno-access-control for test observations.
        const auto pool = getCacheManager()->allocator_->getDeviceBlockPool();
        uint64_t   refs = 0;
        for (int block = 1; block < kBlocks; ++block)
            refs += pool->refCount(block);
        return refs;
    }

    std::atomic<bool> stage_reached{false};

    bool idle() const {
        return scheduler_->empty();
    }

    Fault               fault         = Fault::None;  // Set before starting the executor thread.
    size_t              baseline_free = 0;
    std::atomic<size_t> published_bytes{0};
    std::atomic<size_t> checked_bytes{0};
    std::atomic<int>    checked_requests{0};
    std::atomic<int>    published_layers{0};

private:
    bool cacheStep(const GenerateStreamPtr& stream, bool prefill) {
        const auto&  resource      = stream->kvCache().cacheResource(0);
        const size_t prompt_blocks = (stream->inputLength() + kTokensPerBlock - 1) / kTokensPerBlock;
        require(prompt_blocks > 0, "empty prompt block coverage");
        auto& progress = layer_progress_[stream->streamId()];
        for (int layer = prefill ? progress : 0; layer < kLayers; ++layer) {
            const auto stage = static_cast<CancelStage>(cancel_control_.stage);
            if (prefill
                && ((stage == CancelStage::BeforePublish && layer == 0)
                    || (stage == CancelStage::AfterFirstLayer && layer == 1))) {
                stage_reached = true;
                return false;  // Keep scheduling so cancellation can reclaim this stream.
            }
            if (prefill && fault == Fault::OmitLastLayer && layer == kLayers - 1) {
                continue;
            }
            const auto& ids = resource.blocksForLayer(layer, "default");
            require(ids.size() >= prompt_blocks && (!prefill || resource.cacheKeys().size() >= prompt_blocks),
                    "missing allocated prompt blocks/cache keys");
            for (size_t block = 0; block < prompt_blocks; ++block) {
                const auto buffers = getCacheManager()->convertIndexToBufferByTag(ids[block], layer, "default");
                require(!buffers.empty(), "empty cache block buffer list");
                for (size_t buffer = 0; buffer < buffers.size(); ++buffer) {
                    auto expected = payload(stream->streamId(), layer, block, buffer, buffers[buffer].size_bytes);
                    require(!expected.empty(), "empty cache payload");
                    if (prefill) {
                        if (fault == Fault::CorruptByte && layer == 1 && block == 0 && buffer == 0) {
                            expected[expected.size() / 2] ^= 1;
                        }
                        writeBytes(buffers[buffer], expected);
                        published_bytes += expected.size();
                    } else {
                        const auto actual   = readBytes(buffers[buffer]);
                        auto       mismatch = std::mismatch(expected.begin(), expected.end(), actual.begin());
                        if (mismatch.first != expected.end()) {
                            throw std::runtime_error("payload mismatch request=" + std::to_string(stream->streamId())
                                                     + " layer=" + std::to_string(layer) + " tag=default block="
                                                     + std::to_string(block) + " buffer=" + std::to_string(buffer)
                                                     + " byte=" + std::to_string(mismatch.first - expected.begin()));
                        }
                        checked_bytes += actual.size();
                    }
                }
            }
            if (prefill) {
                // Synchronous H2D copies above complete before the production
                // layer-ready seam; no fabricated transfer completion or event.
                const CacheKeysType    keys(resource.cacheKeys().begin(), resource.cacheKeys().begin() + prompt_blocks);
                const BlockIndicesType blocks(ids.begin(), ids.begin() + prompt_blocks);
                require(getCacheManager()->writeP2PLayer(0,
                                                         layer,
                                                         "default",
                                                         keys,
                                                         blocks,
                                                         stream->streamId(),
                                                         nullptr,
                                                         stream->generateInput()->request_deadline_ms),
                        "writeP2PLayer rejected real allocated buffers");
                ++published_layers;
                progress = layer + 1;
                if (cancel_control_.step_delay_ms > 0 && progress < kLayers) {
                    next_step_ms_[stream->streamId()] = steadyTimeMs() + cancel_control_.step_delay_ms;
                    return false;
                }
            }
        }
        if (!prefill) {
            ++checked_requests;
        }
        return true;
    }

    void run() {
        // The test uses one CUDA device; P/D communication still goes through
        // their real transport endpoints, never a direct pool-to-pool copy.
        const auto device_status = cudaSetDevice(0);
        while (!stopping_) {
            auto                        scheduled = scheduler_->schedule();
            std::lock_guard<std::mutex> execution_lock(execution_mutex_);
            if (stopping_) {
                break;
            }
            if (!scheduled.ok()) {
                continue;
            }
            for (const auto& stream : scheduled.value()) {
                try {
                    cudaCheck(device_status);
                    const bool prefill = pd_.role_type == RoleType::PREFILL;
                    if (stream->hasError())
                        continue;
                    if (!prefill && cancel_control_.stage == static_cast<int32_t>(CancelStage::AfterFirstDecodeOutput)
                        && decode_steps_[stream->streamId()] > 0) {
                        stage_reached = true;
                        continue;
                    }
                    if (steadyTimeMs() < next_step_ms_[stream->streamId()])
                        continue;
                    if ((prefill && stream->queryPdSep())
                        || (!prefill && stream->seqLength() == stream->inputLength() + 1)) {
                        if (!cacheStep(stream, prefill))
                            continue;
                    }
                    // D must have applied the real first-token side channel
                    // before becoming runnable. The remaining tokens are stubbed.
                    if (!prefill) {
                        require(!stream->isContextStream(), "D became runnable before P2P side channel");
                        require(stream->seqLength() > stream->inputLength(), "missing prefill token");
                    }
                    const int  token  = 100 + stream->seqLength() - stream->inputLength();
                    const auto width  = static_cast<int64_t>(stream->nextBatchSize());
                    auto       tokens = torch::full({width, 1}, token, torch::kInt32);
                    if (stream->hasNumBeams()) {
                        tokens = torch::empty({width, static_cast<int64_t>(stream->seqLength() + 1)}, torch::kInt32);
                        for (int64_t row = 0; row < width; ++row) {
                            const int  source = std::min<int64_t>(row, stream->currentBatchSize() - 1);
                            const auto prefix = stream->completeTokenIdsVec(source);
                            auto*      data   = tokens.data_ptr<int32_t>() + row * tokens.size(1);
                            std::copy(prefix.begin(), prefix.end(), data);
                            data[stream->seqLength()] = token;
                        }
                    }
                    stream->step();
                    StreamUpdateInfo update{tokens,
                                            1,
                                            {},
                                            {},
                                            {},
                                            torch::zeros({width}, torch::kFloat32),
                                            {},
                                            {},
                                            {},
                                            {},
                                            true,
                                            false,
                                            std::nullopt};
                    stream->update(update);
                    if (!prefill)
                        ++decode_steps_[stream->streamId()];
                    next_step_ms_[stream->streamId()] = steadyTimeMs() + cancel_control_.step_delay_ms;
                } catch (const std::exception& error) {
                    {
                        std::lock_guard<std::mutex> lock(error_mutex_);
                        error_ = error.what();
                    }
                    stream->reportError(ErrorCode::EXECUTION_EXCEPTION, error.what());
                }
            }
        }
    }

    ModelConfig                          model_;
    RuntimeConfig                        runtime_;
    PDSepConfig                          pd_;
    std::atomic<bool>                    stopping_{false};
    std::thread                          thread_;
    std::mutex                           execution_mutex_;
    mutable std::mutex                   error_mutex_;
    std::string                          error_;
    CancelControl                        cancel_control_;
    std::unordered_map<int64_t, int>     layer_progress_, decode_steps_;
    std::unordered_map<int64_t, int64_t> next_step_ms_;
};

// Count wire RPCs while delegating all request handling to production servers.
class PayloadRpcService: public RpcService::Service {
public:
    LocalRpcServer*       target  = nullptr;
    PrefillRpcServerNew2* prefill = nullptr;
    std::atomic<int>      generate_calls{0}, peer_calls{0}, load_calls{0}, read_calls{0}, handle_read_calls{0};
    std::atomic<int>      generate_done{0}, generate_cancelled{0}, load_done{0}, load_cancelled{0};
    std::atomic<int>      read_done{0}, handle_read_done{0}, cancel_read_calls{0};

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        ++generate_calls;
        const auto status = target->GenerateStreamCall(context, request, writer);
        if (context->IsCancelled())
            ++generate_cancelled;
        ++generate_done;
        return status;
    }
    grpc::Status GetPeerInfo(grpc::ServerContext*        context,
                             const GetPeerInfoRequestPB* request,
                             GetPeerInfoResponsePB*      response) override {
        ++peer_calls;
        return prefill ? prefill->GetPeerInfo(context, request, response) :
                         grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "D");
    }
    grpc::Status StartLoad(grpc::ServerContext*                  context,
                           const P2PConnectorStartLoadRequestPB* request,
                           P2PConnectorStartLoadResponsePB*      response) override {
        ++load_calls;
        const auto status = prefill ? prefill->StartLoad(context, request, response) :
                                      grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "D");
        if (context->IsCancelled())
            ++load_cancelled;
        ++load_done;
        return status;
    }
    grpc::Status ExecuteFunction(grpc::ServerContext*     context,
                                 const FunctionRequestPB* request,
                                 FunctionResponsePB*      response) override {
        if (request->has_p2p_request()) {
            if (request->p2p_request().type() == P2PConnectorBroadcastType::READ)
                ++read_calls;
            if (request->p2p_request().type() == P2PConnectorBroadcastType::HANDLE_READ)
                ++handle_read_calls;
            if (request->p2p_request().type() == P2PConnectorBroadcastType::CANCEL_READ)
                ++cancel_read_calls;
        }
        const auto status = target->ExecuteFunction(context, request, response);
        if (request->has_p2p_request()) {
            if (request->p2p_request().type() == P2PConnectorBroadcastType::READ)
                ++read_done;
            if (request->p2p_request().type() == P2PConnectorBroadcastType::HANDLE_READ)
                ++handle_read_done;
        }
        return status;
    }
};

class PayloadEndpoint {
public:
    PayloadEndpoint(LocalRpcServer& target, PrefillRpcServerNew2* prefill, const std::string& host): host(host) {
        service.target  = &target;
        service.prefill = prefill;
        grpc::ServerBuilder builder;
        builder.AddListeningPort(host + ":0", grpc::InsecureServerCredentials(), &grpc_port);
        builder.RegisterService(&service);
        server = builder.BuildAndStart();
        require(server != nullptr && grpc_port > 0, "gRPC endpoint bind failed");
    }
    ~PayloadEndpoint() {
        server->Shutdown(std::chrono::system_clock::now());
        server->Wait();
    }
    std::string address() const {
        return host + ":" + std::to_string(grpc_port);
    }
    PayloadRpcService             service;
    std::string                   host;
    int                           grpc_port = 0;
    std::unique_ptr<grpc::Server> server;
};

std::shared_ptr<PayloadEngine> makePayloadEngine(
    const CacheConfig& config, RoleType role, const PayloadEndpoint& endpoint, bool rdma, int load_timeout_ms) {
    PDSepConfig pd;
    pd.role_type             = role;
    pd.decode_entrance       = true;
    pd.cache_store_rdma_mode = rdma;
    // Same port convention as serving: transfer backend listens at base+1.
    const auto transfer_port = autil::NetUtil::randomPort();
    require(transfer_port > 1 && transfer_port <= 65535, "invalid transfer port");
    pd.cache_store_listen_port = transfer_port - 1;
    pd.load_cache_timeout_ms   = load_timeout_ms;
    RuntimeConfig runtime;
    runtime.max_generate_batch_size                     = 4;
    runtime.fifo_scheduler_config.max_batch_tokens_size = 256;
    runtime.worker_grpc_addrs                           = {endpoint.address()};
    runtime.worker_addrs                                = {endpoint.host + ":" + std::to_string(transfer_port) + ":"
                                                           + std::to_string(endpoint.grpc_port)};
    runtime.p2p_worker_addrs                            = runtime.worker_addrs;
    return std::make_shared<PayloadEngine>(config, runtime, pd);
}

// Test control protocol carries configuration, counters and diagnostics only.
// Cache bytes and the first token exclusively use the production RPC/transport.
struct WorkerHello {
    uint32_t magic              = 0x50325033;
    int32_t  dtype              = 0;
    int32_t  fault              = 0;
    int32_t  rdma               = 0;
    int32_t  request_timeout_ms = 0;
    int32_t  load_timeout_ms    = 0;
};

struct WorkerReport {
    uint32_t magic          = 0x50325033;
    int32_t  pid            = 0;
    int32_t  grpc_port      = 0;
    int32_t  generate_calls = 0, peer_calls = 0, load_calls = 0, handle_read_calls = 0;
    int32_t  published_layers = 0;
    uint64_t published_bytes = 0, free_blocks = 0, baseline_free = 0;
    uint64_t block_refs    = 0;
    int32_t  generate_done = 0, generate_cancelled = 0, load_done = 0, load_cancelled = 0;
    int32_t  handle_read_done = 0, stage_reached = 0, scheduler_idle = 0;
    char     host[64]      = {};
    char     failure[1024] = {};
};

class ControlFd {
public:
    explicit ControlFd(int fd = -1): fd(fd) {}
    ~ControlFd() {
        if (fd >= 0)
            ::close(fd);
    }
    ControlFd(const ControlFd&)            = delete;
    ControlFd& operator=(const ControlFd&) = delete;
    int        fd;
};

void awaitControl(int fd, short events, int64_t deadline) {
    while (steadyTimeMs() < deadline) {
        pollfd    poll_fd{fd, events, 0};
        const int result = ::poll(&poll_fd, 1, static_cast<int>(std::max<int64_t>(0, deadline - steadyTimeMs())));
        if (result < 0 && errno == EINTR)
            continue;
        require(result > 0, "P2P worker control timed out or poll failed");
        // POLLHUP may accompany the last buffered response; recv reports EOF.
        require(!(poll_fd.revents & POLLNVAL), "invalid P2P worker control descriptor");
        return;
    }
    throw std::runtime_error("P2P worker control deadline exceeded");
}

void controlIO(int fd, void* data, size_t size, bool sending, int timeout_ms = 10000) {
    const auto deadline = steadyTimeMs() + timeout_ms;
    auto*      bytes    = static_cast<char*>(data);
    while (size > 0) {
        awaitControl(fd, sending ? POLLOUT : POLLIN, deadline);
        const auto count =
            sending ? ::send(fd, bytes, size, MSG_NOSIGNAL | MSG_DONTWAIT) : ::recv(fd, bytes, size, MSG_DONTWAIT);
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
            continue;
        require(count > 0, "P2P worker control closed or IO failed; inspect worker log");
        bytes += count;
        size -= static_cast<size_t>(count);
    }
}

sockaddr_in controlAddress(const std::string& address) {
    const auto colon = address.rfind(':');
    require(colon != std::string::npos, "control address must be IPv4:port");
    size_t      parsed    = 0;
    const auto  port_text = address.substr(colon + 1);
    const int   port      = std::stoi(port_text, &parsed);
    sockaddr_in result{};
    result.sin_family = AF_INET;
    require(parsed == port_text.size() && port > 0 && port <= 65535, "invalid control port");
    result.sin_port = htons(port);
    require(::inet_pton(AF_INET, address.substr(0, colon).c_str(), &result.sin_addr) == 1,
            "control host must be an IPv4 address");
    return result;
}

// D owns this process only for automatic local execution. In cross-host mode it
// connects to an explicitly launched P worker and never signals a remote PID.
class PrefillProcess {
public:
    PrefillProcess(DataType dtype, Fault fault, bool rdma, PayloadTimeouts timeouts) {
        try {
            const auto remote = environment("P2P_SMOKE_CONTROL_ADDR", "");
            if (remote.empty()) {
                launch();
            } else {
                const auto address = controlAddress(remote);
                control_.fd        = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
                require(control_.fd >= 0, "control socket creation failed");
                const int result = ::connect(control_.fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address));
                require(result == 0 || errno == EINPROGRESS, "cannot connect to remote P control endpoint");
                awaitControl(control_.fd, POLLOUT, steadyTimeMs() + 10000);
                int       error = 0;
                socklen_t size  = sizeof(error);
                require(::getsockopt(control_.fd, SOL_SOCKET, SO_ERROR, &error, &size) == 0 && error == 0,
                        "remote P control connection failed");
            }
            WorkerHello hello;
            hello.dtype              = dtype == DataType::TYPE_INT8 ? 1 : 0;
            hello.fault              = static_cast<int32_t>(fault);
            hello.rdma               = rdma;
            hello.request_timeout_ms = timeouts.request_ms;
            hello.load_timeout_ms    = timeouts.load_ms;
            controlIO(control_.fd, &hello, sizeof(hello), true);
            ready = receive(120000);
            require(ready.grpc_port > 0 && ready.host[0] != '\0', "P did not advertise a ready RPC endpoint");
            if (pid_ > 0) {
                require(ready.pid == pid_ && ready.pid != ::getpid(), "P must run in a distinct exec process");
            } else {
                require(std::string(ready.host) != "127.0.0.1" && std::string(ready.host) != "0.0.0.0",
                        "remote P must advertise its reachable P2P_SMOKE_HOST");
            }
        } catch (...) {
            terminate();
            throw;
        }
    }

    ~PrefillProcess() {
        terminate();
    }

    WorkerReport snapshot() {
        char command = 'S';
        controlIO(control_.fd, &command, 1, true);
        return receive(10000);
    }

    void configureCancel(CancelControl control) {
        char command = 'C';
        controlIO(control_.fd, &command, 1, true);
        controlIO(control_.fd, &control, sizeof(control), true);
        (void)receive(10000);
    }

    bool finish() {
        char command = 'Q';
        controlIO(control_.fd, &command, 1, true);
        // ACK is sent after P destroys its engine, RPC endpoint and backend.
        char ack = 0;
        controlIO(control_.fd, &ack, 1, false, 30000);
        require(ack == 'Q', "invalid P shutdown acknowledgement");
        ::close(control_.fd);
        control_.fd = -1;
        if (pid_ <= 0)
            return true;
        int        status = 0;
        const bool exited = waitFor([&] {
            const auto result = ::waitpid(pid_, &status, WNOHANG);
            if (result == pid_) {
                pid_ = -1;
                return true;
            }
            return false;
        });
        return exited && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }

    WorkerReport ready;

private:
    WorkerReport receive(int timeout_ms) {
        WorkerReport report;
        controlIO(control_.fd, &report, sizeof(report), false, timeout_ms);
        require(report.magic == WorkerHello{}.magic && report.host[sizeof(report.host) - 1] == '\0'
                    && report.failure[sizeof(report.failure) - 1] == '\0',
                "invalid worker report");
        return report;
    }

    void launch() {
        int sockets[2];
        require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) == 0, "control socketpair failed");
        control_.fd = sockets[0];
        ControlFd                  worker_socket(sockets[1]);
        posix_spawn_file_actions_t actions;
        require(::posix_spawn_file_actions_init(&actions) == 0, "spawn actions init failed");
        const int dup_result = ::posix_spawn_file_actions_adddup2(&actions, worker_socket.fd, STDIN_FILENO);
        if (dup_result != 0) {
            ::posix_spawn_file_actions_destroy(&actions);
            throw std::runtime_error("spawn control descriptor setup failed");
        }
        std::vector<std::string> env;
        for (char** item = environ; *item; ++item) {
            const std::string value(*item);
            // The worker must not overwrite D's Bazel XML or be filtered out by sharding.
            if (value.rfind("XML_OUTPUT_FILE=", 0) == 0 || value.rfind("GTEST_", 0) == 0
                || value.rfind("TEST_SHARD", 0) == 0 || value.rfind("TEST_TOTAL_SHARDS=", 0) == 0
                || value.rfind("P2P_SMOKE_CONTROL_", 0) == 0)
                continue;
            env.push_back(value);
        }
        std::vector<char*> envp;
        for (auto& value : env)
            envp.push_back(value.data());
        envp.push_back(nullptr);
        std::vector<std::string> args{"/proc/self/exe",
                                      "--gtest_filter=P2PPayloadWorker.DISABLED_PrefillProcess",
                                      "--gtest_also_run_disabled_tests",
                                      "--gtest_repeat=1",
                                      "--gtest_output="};
        std::vector<char*>       argv;
        for (auto& value : args)
            argv.push_back(value.data());
        argv.push_back(nullptr);
        const int result = ::posix_spawn(&pid_, "/proc/self/exe", &actions, nullptr, argv.data(), envp.data());
        ::posix_spawn_file_actions_destroy(&actions);
        require(result == 0, "cannot exec P worker: " + std::string(std::strerror(result)));
    }

    void terminate() noexcept {
        // Closing control also asks an externally started worker to clean up.
        if (control_.fd >= 0) {
            ::close(control_.fd);
            control_.fd = -1;
        }
        if (pid_ > 0) {
            int status = 0;
            if (::waitpid(pid_, &status, WNOHANG) == 0)
                ::kill(pid_, SIGKILL);
            while (::waitpid(pid_, &status, 0) < 0 && errno == EINTR) {}
            pid_ = -1;
        }
    }

    ControlFd control_;
    pid_t     pid_ = -1;
};

class P2PPayloadWorker: public DeviceTestBase {};

// Internal exec entry; cross-host runs explicitly select this disabled test.
TEST_F(P2PPayloadWorker, DISABLED_PrefillProcess) {
    ControlFd  listener;
    ControlFd  control;
    const auto listen_address = environment("P2P_SMOKE_CONTROL_LISTEN", "");
    if (listen_address.empty()) {
        control.fd = ::dup(STDIN_FILENO);
        ASSERT_GE(control.fd, 0);
    } else {
        const auto address = controlAddress(listen_address);
        listener.fd        = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        ASSERT_GE(listener.fd, 0);
        const int reuse = 1;
        ASSERT_EQ(::setsockopt(listener.fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)), 0);
        ASSERT_EQ(::bind(listener.fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)), 0);
        ASSERT_EQ(::listen(listener.fd, 1), 0);
        std::cerr << "P2P control listening at " << listen_address << "; waiting for D (120s)" << std::endl;
        awaitControl(listener.fd, POLLIN, steadyTimeMs() + 120000);
        control.fd = ::accept4(listener.fd, nullptr, nullptr, SOCK_CLOEXEC);
        ASSERT_GE(control.fd, 0);
    }
    WorkerHello hello;
    controlIO(control.fd, &hello, sizeof(hello), false);
    ASSERT_EQ(hello.magic, WorkerHello{}.magic);
    ASSERT_GT(hello.load_timeout_ms, 0);
    ASSERT_LT(hello.load_timeout_ms, hello.request_timeout_ms);
    ASSERT_LE(hello.request_timeout_ms, 600000);
    ASSERT_TRUE(hello.dtype == 0 || hello.dtype == 1);
    ASSERT_GE(hello.fault, 0);
    ASSERT_LE(hello.fault, static_cast<int>(Fault::OmitLastLayer));
    const auto transport = environment("P2P_SMOKE_TRANSPORT", "tcp");
    ASSERT_TRUE(transport == "tcp" || transport == "rdma");
    ASSERT_EQ(hello.rdma, transport == "rdma");
    const auto host = environment("P2P_SMOKE_HOST", "127.0.0.1");
    ASSERT_LT(host.size(), sizeof(WorkerReport{}.host));
    ASSERT_TRUE(transport != "rdma" || host != "127.0.0.1");
    {
        PrefillRpcServerNew2 prefill;
        prefill.meta_ = std::make_shared<RpcServerRuntimeMeta>();
        PayloadEndpoint endpoint(prefill, &prefill, host);
        const auto      config = test::makeSimpleMhaCacheConfig(
            kLayers, kBlocks, kTokensPerBlock, hello.dtype == 1 ? DataType::TYPE_INT8 : DataType::TYPE_FP16, 2, 16);
        auto engine     = makePayloadEngine(config, RoleType::PREFILL, endpoint, hello.rdma, hello.load_timeout_ms);
        prefill.engine_ = engine;
        prefill.dp_grpc_addrs_ = {endpoint.address()};
        engine->fault          = static_cast<Fault>(hello.fault);
        engine->start();
        const auto report = [&] {
            WorkerReport state;
            state.pid                = ::getpid();
            state.grpc_port          = endpoint.grpc_port;
            state.generate_calls     = endpoint.service.generate_calls.load();
            state.peer_calls         = endpoint.service.peer_calls.load();
            state.load_calls         = endpoint.service.load_calls.load();
            state.handle_read_calls  = endpoint.service.handle_read_calls.load();
            state.published_layers   = engine->published_layers.load();
            state.published_bytes    = engine->published_bytes.load();
            state.free_blocks        = engine->getCacheManager()->freeBlocksNum();
            state.baseline_free      = engine->baseline_free;
            state.block_refs         = engine->blockRefs();
            state.stage_reached      = engine->stage_reached.load();
            state.scheduler_idle     = engine->idle();
            state.generate_done      = endpoint.service.generate_done.load();
            state.generate_cancelled = endpoint.service.generate_cancelled.load();
            state.load_done          = endpoint.service.load_done.load();
            state.load_cancelled     = endpoint.service.load_cancelled.load();
            state.handle_read_done   = endpoint.service.handle_read_done.load();
            std::copy(host.begin(), host.end(), state.host);
            const auto failure = engine->failure();
            std::copy_n(failure.data(), std::min(failure.size(), sizeof(state.failure) - 1), state.failure);
            controlIO(control.fd, &state, sizeof(state), true);
        };
        try {
            report();
            std::cerr << "P2P prefill ready: pid=" << ::getpid() << " grpc=" << endpoint.address()
                      << " transport=" << transport << std::endl;
            while (true) {
                char command = 0;
                controlIO(control.fd, &command, 1, false, hello.request_timeout_ms + 60000);
                if (command == 'Q')
                    break;
                if (command == 'C') {
                    CancelControl config;
                    controlIO(control.fd, &config, sizeof(config), false);
                    engine->configureCancel(config);
                } else {
                    require(command == 'S', "unknown P worker control command");
                }
                report();
            }
        } catch (...) {
            (void)engine->stop();
            throw;
        }
        (void)engine->stop();
    }
    char ack = 'Q';
    controlIO(control.fd, &ack, 1, true);
}

class P2PGenerateStreamSmokeTest: public DeviceTestBase {
protected:
    void initialize(DataType dtype, Fault fault = Fault::None, bool cancellation_test = false) {
        timeouts_ = payloadTimeouts();
        if (cancellation_test) {
            // The 30s drain assertion must expire before production deadlines;
            // waiting for the regular timeout cannot make a cancel test pass.
            timeouts_.load_ms    = std::max(timeouts_.load_ms, 90000);
            timeouts_.request_ms = std::max(timeouts_.request_ms, std::min(600000, timeouts_.load_ms + 30000));
        }
        const auto transport = environment("P2P_SMOKE_TRANSPORT", "tcp");
        require(transport == "tcp" || transport == "rdma", "P2P_SMOKE_TRANSPORT must be tcp or rdma");
        host_ = environment("P2P_SMOKE_HOST", "127.0.0.1");
        require(transport != "rdma" || host_ != "127.0.0.1", "RDMA requires P2P_SMOKE_HOST on its network interface");
        // P execs a fresh image (or is launched on another host) before D starts its backend.
        prefill_process_ = std::make_unique<PrefillProcess>(dtype, fault, transport == "rdma", timeouts_);
        std::cerr << "P2P decode pid=" << ::getpid() << ", prefill pid=" << prefill_process_->ready.pid
                  << " grpc=" << prefill_process_->ready.host << ':' << prefill_process_->ready.grpc_port
                  << " request_timeout_ms=" << timeouts_.request_ms << " load_timeout_ms=" << timeouts_.load_ms
                  << std::endl;
        decode_.meta_           = std::make_shared<RpcServerRuntimeMeta>();
        decode_rpc_             = std::make_unique<PayloadEndpoint>(decode_, nullptr, host_);
        const auto cache_config = test::makeSimpleMhaCacheConfig(kLayers, kBlocks, kTokensPerBlock, dtype, 2, 16);
        decode_engine_ =
            makePayloadEngine(cache_config, RoleType::DECODE, *decode_rpc_, transport == "rdma", timeouts_.load_ms);
        decode_.engine_                = decode_engine_;
        decode_.prefill_server_caller_ = std::make_shared<PrefillServerCaller>("p2p-payload-smoke");
        decode_engine_->start();
    }

    GenerateInputPB request(int64_t id, int prompt_length) const {
        GenerateInputPB request;
        request.set_request_id(id);
        for (int token = 1; token <= prompt_length; ++token) {
            request.add_token_ids(token);
        }
        auto* config = request.mutable_generate_config();
        config->set_can_use_pd_separation(true);
        config->set_max_new_tokens(kNewTokens);
        config->set_num_beams(1);
        config->set_num_return_sequences(1);
        config->set_timeout_ms(timeouts_.request_ms);
        config->set_is_streaming(true);
        auto* role = config->add_role_addrs();
        role->set_role(RoleAddrPB::PREFILL);
        role->set_ip(prefill_process_->ready.host);
        role->set_grpc_port(prefill_process_->ready.grpc_port);
        return request;
    }

    grpc::Status generate(const GenerateInputPB& request, std::vector<GenerateOutputsPB>& outputs) {
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(timeouts_.request_ms + 2000));
        return generate(request, outputs, context);
    }

    grpc::Status generate(const GenerateInputPB&          request,
                          std::vector<GenerateOutputsPB>& outputs,
                          grpc::ClientContext&            context,
                          std::atomic<int>*               output_count = nullptr) {
        auto stub =
            RpcService::NewStub(grpc::CreateChannel(decode_rpc_->address(), grpc::InsecureChannelCredentials()));
        auto              reader = stub->GenerateStreamCall(&context, request);
        GenerateOutputsPB output;
        while (reader->Read(&output)) {
            outputs.push_back(output);
            if (output_count)
                ++*output_count;
        }
        return reader->Finish();
    }

    void expectReleased() {
        ASSERT_TRUE(waitFor([this] {
            const auto prefill = prefill_process_->snapshot();
            return prefill.free_blocks == prefill.baseline_free
                   && decode_engine_->getCacheManager()->freeBlocksNum() == decode_engine_->baseline_free;
        })) << "P/D cache or connector references were not released";
    }

    void expectCancelDrained() {
        require(waitFor([this] {
                    const auto  p = prefill_process_->snapshot();
                    const auto& d = decode_rpc_->service;
                    return p.generate_done == p.generate_calls && p.load_done == p.load_calls
                           && p.handle_read_done == p.handle_read_calls && d.generate_done == d.generate_calls
                           && d.read_done == d.read_calls && p.free_blocks == p.baseline_free
                           && decode_engine_->getCacheManager()->freeBlocksNum() == decode_engine_->baseline_free
                           && p.block_refs == 0 && decode_engine_->blockRefs() == 0 && p.scheduler_idle
                           && decode_engine_->idle();
                }),
                "cancel did not drain P/D RPCs and KV block references before cleanup timeout");
        const auto p = prefill_process_->snapshot();
        require(std::string(p.failure).empty() && decode_engine_->failure().empty(),
                "payload engine failure P=" + std::string(p.failure) + " D=" + decode_engine_->failure());
    }

    void configureCancel(CancelStage stage, int step_delay_ms) {
        const CancelControl control{static_cast<int32_t>(stage), step_delay_ms};
        prefill_process_->configureCancel(control);
        decode_engine_->configureCancel(control);
    }

    void healthyAfterCancel(int64_t id) {
        configureCancel(CancelStage::None, 0);
        const auto                     checked = decode_engine_->checked_requests.load();
        std::vector<GenerateOutputsPB> outputs;
        const auto                     status = generate(request(id, 11), outputs);
        require(status.ok(), "post-cancel request failed: " + status.error_message());
        require(!outputs.empty() && outputs.back().flatten_output().finished_size() == 1
                    && outputs.back().flatten_output().finished(0),
                "post-cancel request did not finish");
        require(decode_engine_->checked_requests == checked + 1, "post-cancel cache bytes were not checked");
        expectCancelDrained();
    }

    bool cancelRound(int64_t id, CancelStage stage, int delay_ms) {
        const auto before             = prefill_process_->snapshot();
        const auto d_cancel_before    = decode_rpc_->service.generate_cancelled.load();
        const auto read_before        = decode_rpc_->service.read_calls.load();
        const auto cancel_read_before = decode_rpc_->service.cancel_read_calls.load();
        configureCancel(stage, 10);
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(timeouts_.request_ms + 2000));
        std::vector<GenerateOutputsPB> outputs;
        std::atomic<int>               output_count{0};
        auto                           call =
            std::async(std::launch::async, [&] { return generate(request(id, 11), outputs, context, &output_count); });
        try {
            if (stage != CancelStage::None) {
                require(waitFor([&] {
                            const auto p = prefill_process_->snapshot();
                            if (stage == CancelStage::AfterFirstDecodeOutput)
                                return decode_engine_->stage_reached.load() && output_count.load() > 0;
                            return p.stage_reached && p.load_calls > before.load_calls
                                   && decode_rpc_->service.read_calls > read_before;
                        }),
                        "cancel stage was not reached");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            context.TryCancel();
            const auto status    = call.get();
            const bool cancelled = status.error_code() == grpc::StatusCode::CANCELLED;
            require(cancelled || (stage == CancelStage::None && status.ok()),
                    "unexpected cancel result: " + std::to_string(status.error_code()) + " " + status.error_message());
            if (status.ok()) {
                require(!outputs.empty() && outputs.back().flatten_output().finished_size() == 1
                            && outputs.back().flatten_output().finished(0),
                        "successful racing request is incomplete");
            }
            expectCancelDrained();
            const auto after = prefill_process_->snapshot();
            if (stage != CancelStage::None) {
                require(decode_rpc_->service.generate_cancelled > d_cancel_before,
                        "D handler did not observe outer RPC cancellation");
            }
            if (stage == CancelStage::BeforePublish || stage == CancelStage::AfterFirstLayer) {
                require(after.generate_cancelled > before.generate_cancelled,
                        "cancellation did not reach P GenerateStreamCall");
                require(after.load_cancelled > before.load_cancelled, "cancellation did not reach P StartLoad");
                require(decode_rpc_->service.cancel_read_calls > cancel_read_before,
                        "cancellation did not reach D READ workers");
            }
            std::cerr << "cancel request=" << id << " stage=" << static_cast<int>(stage) << " delay_ms=" << delay_ms
                      << " result=" << status.error_code()
                      << " P_cancel=" << after.generate_cancelled - before.generate_cancelled
                      << " D_cancel=" << decode_rpc_->service.generate_cancelled - d_cancel_before << std::endl;
            healthyAfterCancel(id + 1);
            return cancelled;
        } catch (...) {
            context.TryCancel();
            if (call.valid())
                call.wait();
            throw;
        }
    }

    void TearDown() override {
        if (decode_engine_)
            (void)decode_engine_->stop();
        decode_rpc_.reset();
        decode_.engine_.reset();
        decode_engine_.reset();
        if (prefill_process_) {
            try {
                EXPECT_TRUE(prefill_process_->finish()) << "P process did not exit successfully";
            } catch (const std::exception& error) {
                ADD_FAILURE() << "P shutdown failed: " << error.what();
            }
            prefill_process_.reset();
        }
        DeviceTestBase::TearDown();
    }

    void roundTrips(DataType dtype) {
        initialize(dtype);
        int         requests       = 0;
        size_t      expected_bytes = 0;
        const auto& cache_config   = decode_engine_->getCacheManager()->cacheConfig();
        if (dtype == DataType::TYPE_INT8) {
            ASSERT_GT(cache_config.kv_scale_size_bytes, 0u);
        }
        for (int prompt_length : {8, 11}) {
            SCOPED_TRACE(prompt_length);
            std::vector<GenerateOutputsPB> outputs;
            const auto                     status  = generate(request(701 + requests, prompt_length), outputs);
            const auto                     prefill = prefill_process_->snapshot();
            ASSERT_TRUE(status.ok()) << status.error_message() << " P=" << std::string(prefill.failure)
                                     << " D=" << decode_engine_->failure();
            ASSERT_FALSE(outputs.empty());
            const auto& final = outputs.back().flatten_output();
            ASSERT_EQ(final.finished_size(), 1);
            EXPECT_TRUE(final.finished(0));
            std::vector<int32_t> ids;
            for (const auto& output : outputs) {
                EXPECT_EQ(output.request_id(), 701 + requests);
                const auto& tensor = output.flatten_output().output_ids();
                ASSERT_EQ(tensor.data_type(), TensorPB::INT32);
                ASSERT_EQ(tensor.int32_data().size() % sizeof(int32_t), 0u);
                const size_t old_size = ids.size();
                ids.resize(old_size + tensor.int32_data().size() / sizeof(int32_t));
                std::memcpy(ids.data() + old_size, tensor.int32_data().data(), tensor.int32_data().size());
            }
            EXPECT_EQ(ids, (std::vector<int32_t>{100, 101, 102}));
            ++requests;
            EXPECT_EQ(prefill.generate_calls, requests);
            EXPECT_EQ(decode_rpc_->service.generate_calls.load(), requests);
            EXPECT_GE(prefill.peer_calls, 1);
            EXPECT_EQ(prefill.load_calls, requests);
            EXPECT_EQ(prefill.handle_read_calls, requests);
            EXPECT_EQ(decode_rpc_->service.read_calls.load(), requests);
            EXPECT_EQ(prefill.published_layers, requests * kLayers);
            EXPECT_EQ(decode_engine_->checked_requests.load(), requests);
            EXPECT_GT(decode_engine_->checked_bytes.load(), 0u);
            expected_bytes += ((prompt_length + kTokensPerBlock - 1) / kTokensPerBlock) * cache_config.block_size_bytes;
            EXPECT_EQ(decode_engine_->checked_bytes.load(), expected_bytes);
            EXPECT_EQ(prefill.published_bytes, decode_engine_->checked_bytes.load());
            EXPECT_TRUE(std::string(prefill.failure).empty()) << std::string(prefill.failure);
            EXPECT_TRUE(decode_engine_->failure().empty()) << decode_engine_->failure();
            ASSERT_NO_FATAL_FAILURE(expectReleased());
        }
    }

    PayloadTimeouts                  timeouts_{10000, 3000};
    std::string                      host_;
    DecodeRpcServerNew2              decode_;
    std::shared_ptr<PayloadEngine>   decode_engine_;
    std::unique_ptr<PrefillProcess>  prefill_process_;
    std::unique_ptr<PayloadEndpoint> decode_rpc_;
};

TEST_F(P2PGenerateStreamSmokeTest, GenerateStreamTransfersEveryFp16CacheByte) {
    roundTrips(DataType::TYPE_FP16);
}

TEST_F(P2PGenerateStreamSmokeTest, GenerateStreamTransfersInt8CacheAndScaleBytes) {
    roundTrips(DataType::TYPE_INT8);
}

TEST_F(P2PGenerateStreamSmokeTest, CancelAtEachPDStageDrainsBothSides) {
    initialize(DataType::TYPE_FP16, Fault::None, true);
    int64_t id = 2001;
    for (const auto stage :
         {CancelStage::BeforePublish, CancelStage::AfterFirstLayer, CancelStage::AfterFirstDecodeOutput}) {
        SCOPED_TRACE(static_cast<int>(stage));
        EXPECT_TRUE(cancelRound(id, stage, 0));
        id += 2;
    }
}

TEST_F(P2PGenerateStreamSmokeTest, RandomCancelRepeatedlyDrainsBothSides) {
    const auto parse = [](const char* name, const char* fallback, unsigned long maximum) {
        const auto text   = environment(name, fallback);
        size_t     parsed = 0;
        const auto value  = std::stoul(text, &parsed);
        require(parsed == text.size() && value <= maximum, std::string("invalid ") + name);
        return value;
    };
    const auto seed   = static_cast<uint32_t>(parse("P2P_CANCEL_SEED", "20260915", UINT32_MAX));
    const auto rounds = parse("P2P_CANCEL_ROUNDS", "32", 1000);
    require(rounds > 0, "P2P_CANCEL_ROUNDS must be positive");
    initialize(DataType::TYPE_INT8, Fault::None, true);
    std::mt19937                       random(seed);
    std::uniform_int_distribution<int> delay(0, 80);
    size_t                             cancelled = 0;
    for (size_t round = 0; round < rounds; ++round) {
        const auto delay_ms = delay(random);
        SCOPED_TRACE(::testing::Message() << "seed=" << seed << " round=" << round << " delay_ms=" << delay_ms);
        std::cerr << "random cancel seed=" << seed << " round=" << round << '/' << rounds << " delay_ms=" << delay_ms
                  << std::endl;
        cancelled += cancelRound(3001 + 2 * round, CancelStage::None, delay_ms);
    }
    EXPECT_GT(cancelled, 0u) << "all requests completed before cancellation; increase rounds or reduce delays";
    std::cerr << "random cancel summary seed=" << seed << " rounds=" << rounds << " cancelled=" << cancelled
              << " completed_before_cancel=" << rounds - cancelled << std::endl;
}

TEST_F(P2PGenerateStreamSmokeTest, NonPDQueriesFinishOnPrefillWithoutP2PTransfer) {
    initialize(DataType::TYPE_FP16);
    int requests = 0;
    for (int mode = 0; mode < 5; ++mode) {
        for (const std::string key : {std::string{}, std::string{"business-key"}}) {
            SCOPED_TRACE(::testing::Message() << "mode=" << mode << " key=" << key);
            auto  input  = request(801 + requests, 8);
            auto* config = input.mutable_generate_config();
            config->set_unique_key(key);
            if (mode == 0)
                config->set_max_new_tokens(1);
            if (mode == 1)
                config->set_num_beams(2);
            if (mode == 2) {
                config->add_variable_num_beams(2);
                config->add_variable_num_beams(4);
                config->add_variable_num_beams(2);
            }
            if (mode == 3)
                config->set_num_return_sequences(2);
            if (mode == 4)
                config->set_can_use_pd_separation(false);
            std::vector<GenerateOutputsPB> outputs;
            const auto                     status  = generate(input, outputs);
            const auto                     prefill = prefill_process_->snapshot();
            ASSERT_TRUE(status.ok()) << status.error_message() << " P=" << prefill.failure;
            ASSERT_FALSE(outputs.empty());
            ASSERT_GT(outputs.back().flatten_output().finished_size(), 0);
            for (bool finished : outputs.back().flatten_output().finished())
                EXPECT_TRUE(finished);
            ++requests;
            EXPECT_EQ(prefill.generate_calls, requests);
            EXPECT_EQ(prefill.peer_calls, 0);
            EXPECT_EQ(prefill.load_calls, 0);
            EXPECT_EQ(prefill.published_bytes, 0u);
            EXPECT_EQ(decode_engine_->checked_requests.load(), 0);
            EXPECT_EQ(decode_rpc_->service.read_calls.load(), 0);
            EXPECT_TRUE(std::string(prefill.failure).empty()) << prefill.failure;
            ASSERT_NO_FATAL_FAILURE(expectReleased());
        }
    }
}

TEST_F(P2PGenerateStreamSmokeTest, CorruptedTransferredByteFailsBeforeDecodeOutput) {
    initialize(DataType::TYPE_FP16, Fault::CorruptByte);
    std::vector<GenerateOutputsPB> outputs;
    const auto                     status  = generate(request(711, 11), outputs);
    const auto                     prefill = prefill_process_->snapshot();
    EXPECT_FALSE(status.ok());
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details())) << status.error_message();
    EXPECT_EQ(details.error_code(), static_cast<int>(ErrorCode::EXECUTION_EXCEPTION));
    for (const auto& output : outputs) {
        for (bool finished : output.flatten_output().finished())
            EXPECT_FALSE(finished);
    }
    EXPECT_NE(decode_engine_->failure().find("payload mismatch request=711 layer=1 tag=default block=0 buffer=0 byte="),
              std::string::npos)
        << decode_engine_->failure();
    EXPECT_EQ(prefill.load_calls, 1);
    EXPECT_EQ(prefill.published_layers, kLayers);
    EXPECT_EQ(decode_engine_->checked_requests.load(), 0);
    expectReleased();
}

TEST_F(P2PGenerateStreamSmokeTest, MissingLayerFailsLoadWithoutRunningDecode) {
    initialize(DataType::TYPE_FP16, Fault::OmitLastLayer);
    std::vector<GenerateOutputsPB> outputs;
    const auto                     status  = generate(request(721, 11), outputs);
    const auto                     prefill = prefill_process_->snapshot();
    EXPECT_FALSE(status.ok());
    // A server-reported load error is required; expiry of the outer client
    // watchdog alone must not make this negative case pass.
    ErrorDetailsPB details;
    ASSERT_FALSE(status.error_details().empty()) << status.error_message();
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_NE(details.error_code(), 0);
    for (const auto& output : outputs) {
        for (bool finished : output.flatten_output().finished())
            EXPECT_FALSE(finished);
    }
    EXPECT_EQ(prefill.load_calls, 1);
    EXPECT_EQ(prefill.published_layers, kLayers - 1);
    EXPECT_TRUE(std::string(prefill.failure).empty()) << std::string(prefill.failure);
    EXPECT_TRUE(decode_engine_->failure().empty()) << decode_engine_->failure();
    EXPECT_EQ(decode_engine_->checked_requests.load(), 0);
    EXPECT_EQ(decode_engine_->checked_bytes.load(), 0u);
    expectReleased();
}

}  // namespace
}  // namespace rtp_llm
