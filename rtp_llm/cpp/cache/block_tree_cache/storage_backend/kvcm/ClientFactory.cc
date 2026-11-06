#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientFactory.h"

#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DirectSubscriber.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/VIPServerSubscriber.h"
#include <chrono>
#include <grpcpp/create_channel.h>
#include <grpcpp/generic/generic_stub.h>
#include <grpcpp/security/credentials.h>
#include "rtp_llm/cpp/cache/events/kvcm_discovery.pb.h"

namespace rtp_llm {
namespace kvcm {

std::unique_ptr<kv_cache_manager::MetaClient>
ClientFactory::createMetaClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const {
    return kv_cache_manager::MetaClient::Create(config, init_params);
}

std::unique_ptr<kv_cache_manager::TransferClient>
ClientFactory::createTransferClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const {
    return kv_cache_manager::TransferClient::Create(config, init_params);
}

std::unique_ptr<kv_cache_manager::TransferClient>
ClientFactory::createTransferClientWithMemory(
    const std::string&                                config,
    const kv_cache_manager::InitParams&                init_params,
    const kv_cache_manager::ClientMemoryRegistrations& memory_registrations) const {
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
