#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientFactory.h"

#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DirectSubscriber.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/PackedSsdTransferClient.h"
#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"
#include "autil/legacy/json.h"
#include <map>
#include <chrono>
#include <grpcpp/create_channel.h>
#include <grpcpp/generic/generic_stub.h>
#include <grpcpp/security/credentials.h>
#include "rtp_llm/cpp/cache/events/kvcm_discovery.pb.h"
#if USING_CUDA
#include <cuda_runtime_api.h>
#endif
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/VIPServerSubscriber.h"

namespace rtp_llm {
namespace kvcm {
namespace {
// The feature gate belongs at the call site: when disabled, do not parse or
// constrain the upstream SDK configuration or allocate our GPU packing arena.
// Once enabled, select packing from the backend, never a second flag.
bool isPaceSsd(const std::string& config) {
    using namespace autil::legacy;
    const auto fields = AnyCast<json::JsonMap>(json::ParseJson(config));
    auto       sdk_it = fields.find("sdk_config");
    if (sdk_it == fields.end())
        return false;  // Let the SDK validate other configuration formats.
    const auto sdk     = AnyCast<json::JsonMap>(sdk_it->second);
    auto       configs = sdk.find("sdk_backend_configs");
    if (configs == sdk.end())
        return false;
    const auto backends = AnyCast<json::JsonArray>(configs->second);
    for (const auto& backend : backends) {
        const auto entry = AnyCast<json::JsonMap>(backend);
        if (AnyCast<std::string>(entry.at("type")) == "pace_ssd") {
            if (backends.size() != 1)
                throw std::invalid_argument("GPU SSD requires exactly one pace_ssd backend");
            return true;
        }
    }
    return false;
}
std::unique_ptr<kv_cache_manager::TransferClient>
createNativeSsdGpu(const std::string&                                 config,
                   const kv_cache_manager::InitParams&                params,
                   const kv_cache_manager::ClientMemoryRegistrations& registrations) {
#if USING_CUDA
    // PACE SSD treats each IOV as a complete object write. A model page has
    // several layer IOVs, so gather into one registered GPU span per URI.
    // This is D2D packing, not the legacy CPU snapshot/packed wrapper.
    using namespace autil::legacy;
    const auto fields   = AnyCast<json::JsonMap>(json::ParseJson(config));
    const auto sdk      = AnyCast<json::JsonMap>(fields.at("sdk_config"));
    const auto backends = AnyCast<json::JsonArray>(sdk.at("sdk_backend_configs"));
    if (backends.size() != 1 || AnyCast<std::string>(AnyCast<json::JsonMap>(backends.front()).at("type")) != "pace_ssd")
        throw std::invalid_argument("native SSD GDR requires exactly one pace_ssd backend");
    // PACE seals GPU registration at the first Init. Reuse one pre-registered
    // span across this process/device's tag clients; lease disjoint ranges per operation.
    using Context = PackedSsdTransferClient::GpuPackingContext;
    static std::mutex                            registry_mutex;
    static std::map<int, std::weak_ptr<Context>> contexts;
    static std::atomic<size_t>                   resident_bytes{0};
    constexpr size_t                             bytes  = PackedSsdTransferClient::max_operation_bytes;
    int                                          device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        return {};
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    auto                        context = contexts[device].lock();
    if (registrations.gpu.empty()) {
        // ClientWrapper registers every pool at its first Init and clears the
        // later descriptors. Those clients may reuse only a proven subspan of
        // that successful registration, never an arbitrary empty descriptor.
        if (!context || !params.regist_span || !context->coversRegisteredGpuSpan(*params.regist_span, device))
            throw std::invalid_argument("PACE SSD requires dedicated GPU memory registrations");
    } else if (context && context->registered) {
        // PACE seals registrations at its first Init. A second engine cannot
        // quietly introduce a new GPU allocation into an old process session.
        for (const auto& gpu : registrations.gpu) {
            if (gpu.device_id != device || !context->coversRegisteredGpuSpan({gpu.base, gpu.size}, device))
                throw std::invalid_argument("PACE SSD cannot register new GPU spans after its first Init");
        }
    }
    if (!context) {
        size_t previous = resident_bytes.load();
        do {
            if (previous > PackedSsdTransferClient::max_process_bytes - bytes)
                throw std::runtime_error("native SSD registered GPU packing capacity exhausted");
        } while (!resident_bytes.compare_exchange_weak(previous, previous + bytes));
        void* base = nullptr;
        if (cudaMalloc(&base, bytes) != cudaSuccess) {
            resident_bytes.fetch_sub(bytes);
            return {};
        }
        auto arena       = std::shared_ptr<void>(base, [device](void* p) {
            int previous_device = device;
            cudaGetDevice(&previous_device);
            cudaSetDevice(device);
            cudaFree(p);
            cudaSetDevice(previous_device);
            resident_bytes.fetch_sub(bytes);
        });
        context          = std::make_shared<Context>(std::move(arena));
        contexts[device] = context;
    }
    auto memory = registrations;
    if (!context->registered)
        memory.gpu.push_back({context->memory.get(), bytes, device});
    auto registered_spans = context->registered ? std::vector<kv_cache_manager::GpuMemorySpan>{} : memory.gpu;
    auto delegate         = kv_cache_manager::TransferClient::Create(config, params, memory);
    if (!delegate)
        return {};
    if (!context->registered)
        context->registered_gpu_spans.swap(registered_spans);
    context->registered = true;
    return std::make_unique<PackedSsdTransferClient>(
        std::move(delegate),
        [](void* dst, kv_cache_manager::MemoryType dt, const void* src, kv_cache_manager::MemoryType st, size_t size) {
            using M = kv_cache_manager::MemoryType;
            // D2D cudaMemcpy need not wait on the host. External RDMA is
            // not ordered by the CUDA default stream: publish/scatter must
            // finish before handing off or reusing the registered range.
            return dt == M::GPU && st == M::GPU && cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice) == cudaSuccess
                   && cudaStreamSynchronize(nullptr) == cudaSuccess;
        },
        std::move(context),
        [device] { return cudaSetDevice(device) == cudaSuccess; });
#else
    throw std::invalid_argument("native SSD GDR requires CUDA");
#endif
}
}  // namespace

std::unique_ptr<kv_cache_manager::MetaClient>
ClientFactory::createMetaClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const {
    return kv_cache_manager::MetaClient::Create(config, init_params);
}

std::unique_ptr<kv_cache_manager::TransferClient>
ClientFactory::createTransferClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const {
    if (remoteCacheDecodePublicationEnabled() && isPaceSsd(config))
        throw std::invalid_argument("PACE SSD requires GPU memory registrations; CPU packing is unsupported");
    return kv_cache_manager::TransferClient::Create(config, init_params);
}

std::unique_ptr<kv_cache_manager::TransferClient>
ClientFactory::createTransferClientWithMemory(
    const std::string&                                config,
    const kv_cache_manager::InitParams&                init_params,
    const kv_cache_manager::ClientMemoryRegistrations& memory_registrations) const {
    if (remoteCacheDecodePublicationEnabled() && isPaceSsd(config))
        return createNativeSsdGpu(config, init_params, memory_registrations);
    return kv_cache_manager::TransferClient::Create(config, init_params, memory_registrations);
}

std::unique_ptr<Subscriber> ClientFactory::createSubscriber(bool enable_vipserver) const {
    if (enable_vipserver) {
        return std::make_unique<VIPServerSubscriber>();
    }
    return std::make_unique<DirectSubscriber>();
}

std::string ClientFactory::getLeaderAddress(const std::string& address,
                                           const std::string& instance_id,
                                           uint32_t timeout_ms) const {
    discovery::GetClusterInfoRequest request;
    request.set_trace_id("rtp-kvcm-leader-discovery");
    request.set_instance_id(instance_id);
    grpc::Slice request_slice(request.SerializeAsString());
    grpc::ByteBuffer request_buffer(&request_slice, 1), response_buffer;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms));
    grpc::GenericStub stub(grpc::CreateChannel(address, grpc::InsecureChannelCredentials()));
    grpc::CompletionQueue queue;
    auto call = stub.PrepareUnaryCall(
        &context, "/kv_cache_manager.proto.meta.MetaService/GetClusterInfo", request_buffer, &queue);
    grpc::Status status;
    if (call) {
        call->StartCall();
        call->Finish(&response_buffer, &status, &status);
    }
    void* tag = nullptr;
    bool ok = false;
    const bool succeeded = call && queue.Next(&tag, &ok) && ok && status.ok();
    queue.Shutdown();
    while (queue.Next(&tag, &ok)) {}
    if (!succeeded) {
        return {};
    }
    std::vector<grpc::Slice> slices;
    if (!response_buffer.Dump(&slices).ok()) {
        return {};
    }
    std::string response_data;
    for (const auto& slice : slices) {
        response_data.append(reinterpret_cast<const char*>(slice.begin()), slice.size());
    }
    discovery::GetClusterInfoResponse response;
    if (!response.ParseFromString(response_data) || response.header().status().code() != 1
        || !response.has_leader_endpoint()) {
        return {};
    }
    const auto& leader = response.leader_endpoint();
    if (response.leader_node_id().empty() || leader.node_id() != response.leader_node_id()
        || leader.host().empty() || leader.meta_rpc_port() <= 0 || leader.meta_rpc_port() > 65535) {
        return {};
    }
    const auto host = leader.host().find(':') == std::string::npos ? leader.host() : "[" + leader.host() + "]";
    return host + ":" + std::to_string(leader.meta_rpc_port());
}

}  // namespace kvcm
}  // namespace rtp_llm
