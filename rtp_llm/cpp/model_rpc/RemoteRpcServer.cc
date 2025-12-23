#include "autil/NetUtil.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"

using namespace std;

namespace rtp_llm {

grpc::Status RemoteRpcServer::init(const EngineInitParams&                                maga_init_params,
                                   py::object                                             mm_process_engine,
                                   std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    rtp_llm::ProposeModelEngineInitParams* propose_params_ptr = propose_params ? propose_params.get() : nullptr;
    auto ret = LocalRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params));
    if (!ret.ok()) {
        return ret;
    }
    initLocalHostInfo();
    initLocalPeerInfo();
    initCacheStore(maga_init_params, propose_params_ptr);
    return grpc::Status::OK;
}

void RemoteRpcServer::initLocalHostInfo() {
    string local_id, local_ip, hostname;
    if (!autil::NetUtil::GetDefaultIp(local_ip) || local_ip.empty()) {
        RTP_LLM_LOG_WARNING("failed to get local ip, use hostname instead");
        RTP_LLM_CHECK_WITH_INFO(autil::NetUtil::GetHostName(hostname), "get hostname failed");
        local_id = "hostname_" + hostname;
    } else {
        local_id = "ip_" + local_ip;
    }
    auto pid        = getpid();
    auto start_time = currentTimeUs();
    process_id_     = local_id + "_pid_" + std::to_string(pid) + "_timestamp_" + std::to_string(start_time);
    RTP_LLM_LOG_INFO("local process id is %s", process_id_.c_str());
}

void RemoteRpcServer::initLocalPeerInfo() {
    // not init when tp rank != 0
    if (maga_init_params_.parallelism_config.tp_rank > 0) {
        return;
    }
    // worker 0 is master (rank 0)
    resource_.workers      = maga_init_params_.runtime_config.worker_addrs;
    resource_.grpc_workers = maga_init_params_.runtime_config.worker_grpc_addrs;

    string worker_info = "worker address is ";
    for (auto& worker : resource_.workers) {
        worker_info += worker + ", ";
    }
    RTP_LLM_LOG_INFO("%s", worker_info.c_str());

    string worker_grpc_info = "worker grpc address is ";
    for (auto& worker : resource_.grpc_workers) {
        worker_grpc_info += worker + ", ";
    }
    RTP_LLM_LOG_INFO("%s", worker_grpc_info.c_str());
}

void RemoteRpcServer::initCacheStore(const EngineInitParams&                init_params,
                                     rtp_llm::ProposeModelEngineInitParams* propose_params) {
    RTP_LLM_LOG_INFO("init_params.role_type : %d", init_params.pd_sep_config.role_type);

    if (init_params.pd_sep_config.role_type != RoleType::PREFILL
        && init_params.pd_sep_config.role_type != RoleType::DECODE) {
        RTP_LLM_FAIL("role_type must be prefill or decode, but it is %d", init_params.pd_sep_config.role_type);
    }
    auto device        = engine_->getDevice();
    auto cache_manager = engine_->resourceContext().cache_manager;

    CacheStoreInitParams params;
    params.listen_port                  = init_params.pd_sep_config.cache_store_listen_port;
    params.rdma_listen_port             = init_params.pd_sep_config.cache_store_rdma_listen_port;
    params.rdma_mode                    = init_params.pd_sep_config.cache_store_rdma_mode;
    params.thread_count                 = init_params.cache_store_config.thread_count;
    params.queue_size                   = 500;
    params.rdma_connect_timeout_ms      = init_params.cache_store_config.rdma_connect_timeout_ms;
    params.rdma_qp_count_per_connection = init_params.cache_store_config.rdma_qp_count_per_connection;
    params.rdma_io_thread_count         = init_params.cache_store_config.rdma_io_thread_count;
    params.rdma_worker_thread_count     = init_params.cache_store_config.rdma_worker_thread_count;
    params.messager_io_thread_count     = init_params.cache_store_config.messager_io_thread_count;
    params.messager_worker_thread_count = init_params.cache_store_config.messager_worker_thread_count;
    params.device                       = device;
    params.metrics_reporter             = metrics_reporter_;
    RTP_LLM_LOG_INFO("cache store listen port is [%ld], rdma listen port is [%ld] rdma_mode is [%d]",
                     params.listen_port,
                     params.rdma_listen_port,
                     params.rdma_mode);
    cache_store_ = NormalCacheStore::createNormalCacheStore(params);
    RTP_LLM_CHECK_WITH_INFO(cache_store_ != nullptr, "cache store init failed");
    RTP_LLM_LOG_INFO("cache store init success");

    device->setCacheStore(cache_store_);
    cache_manager->regUserMr(maga_init_params_.model_id);

    resource_.cache_store = std::dynamic_pointer_cast<NormalCacheStore>(cache_store_);
}

}  // namespace rtp_llm
