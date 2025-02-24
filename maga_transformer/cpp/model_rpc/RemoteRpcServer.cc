#include "autil/NetUtil.h"
#include "maga_transformer/cpp/utils/NetUtil.h"
#include "maga_transformer/cpp/model_rpc/RemoteRpcServer.h"

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

grpc::Status RemoteRpcServer::init(const EngineInitParams&                                maga_init_params,
                                   py::object                                             mm_process_engine,
                                   std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    auto ret = LocalRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params));
    if (!ret.ok()) {
        return ret;
    }
    initLocalHostInfo();
    initLocalPeerInfo();
    initCacheStore(maga_init_params.gpt_init_parameter);
    return grpc::Status::OK;
}

void RemoteRpcServer::initLocalHostInfo() {
    string local_id, local_ip, hostname;
    if (!autil::NetUtil::GetDefaultIp(local_ip) || local_ip.empty()) {
        FT_LOG_WARNING("failed to get local ip, use hostname instead");
        FT_CHECK_WITH_INFO(autil::NetUtil::GetHostName(hostname), "get hostname failed");
        local_id = "hostname_" + hostname;
    } else {
        local_id = "ip_" + local_ip;
    }
    auto    pid        = getpid();
    auto    start_time = currentTimeUs();
    process_id_ = local_id + "_pid_" + std::to_string(pid)
                        + "_timestamp_" + std::to_string(start_time);
    FT_LOG_INFO("local process id is %s", process_id_.c_str());
}

void RemoteRpcServer::initLocalPeerInfo() {
    // not init when tp rank != 0
    if (maga_init_params_.gpt_init_parameter.tp_rank_ > 0) {
        return;
    }
    // worker 0 is master (rank 0)
    for(auto& worker_addr : maga_init_params_.gpt_init_parameter.worker_addrs_) {
        FT_LOG_INFO("In gpt init params: worker address is %s", worker_addr.c_str());
        resource_.workers.push_back(worker_addr);
    }
    string worker_info = "worker address is ";
    for (auto& worker : resource_.workers) {
        worker_info += worker + ", ";
    }
    FT_LOG_INFO(worker_info);
}

void RemoteRpcServer::initCacheStore(const GptInitParameter& init_params) {
    FT_LOG_INFO("init_params.use_cache_store = %d, init_params.pd_separation = %d",
                init_params.use_cache_store_, init_params.pd_separation_);

    if (!init_params.use_cache_store_) {
        FT_FAIL("cache store not used in RemoteRpcServer is unexpected");
    }
    const_cast<ResourceContext*>(&engine_->resourceContext())->use_cache_store = true;
    auto device = engine_->getDevice();
    auto cache_manager = engine_->resourceContext().cache_manager;

    CacheStoreInitParams params;
    params.listen_port = init_params.cache_store_listen_port_;
    params.connect_port = init_params.cache_store_connect_port_;
    params.rdma_listen_port = init_params.cache_store_rdma_listen_port_;
    params.rdma_connect_port = init_params.cache_store_rdma_connect_port_;
    params.rdma_mode = init_params.cache_store_rdma_mode_;
    params.thread_count = 4;
    params.queue_size = 500;
    params.device = device;
    FT_LOG_INFO("cache store listen port is [%ld], connect port is [%d], rdma_mode is [%d]",
        params.listen_port, params.connect_port, params.rdma_mode);
    cache_store_ = NormalCacheStore::createNormalCacheStore(params);
    FT_CHECK_WITH_INFO(cache_store_ != nullptr, "cache store init failed");
    FT_LOG_INFO("cache store init success");

    device->setCacheStore(cache_store_);
    cache_manager->regUserMr();

    resource_.cache_store = std::dynamic_pointer_cast<NormalCacheStore>(cache_store_);
}

}  // namespace rtp_llm
