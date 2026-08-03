#include "autil/NetUtil.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"

#include <algorithm>
#include <limits>

using namespace std;

namespace rtp_llm {

grpc::Status RemoteRpcServer::validateBeforeCacheStoreInit() const {
    const auto& parallelism = maga_init_params_.parallelism_config;
    size_t      cp_size     = 1;
    if (parallelism.prefill_cp_config.kv_cache_sharded) {
        if (maga_init_params_.pd_sep_config.role_type == RoleType::PREFILL && parallelism.tp_size > 1) {
            cp_size = static_cast<size_t>(parallelism.tp_size);
        } else if (maga_init_params_.pd_sep_config.role_type == RoleType::DECODE
                   && parallelism.prefill_cp_config.prefill_cp_size > 1) {
            cp_size = static_cast<size_t>(parallelism.prefill_cp_config.prefill_cp_size);
        }
    }
    return validateNormalCacheStoreTopologies(engine_->resourceContext().cache_manager->cacheConfig(), cp_size);
}

grpc::Status RemoteRpcServer::validateNormalCacheStoreTopologies(const CacheConfig& config, size_t cp_size) {
    cp_size          = std::max<size_t>(cp_size, 1);
    size_t main_span = 0;
    for (const auto& group : config.topology().groups()) {
        if (!group.policy.enable_prefix_reuse || group.layer_ids.empty()) {
            continue;
        }
        if (group.policy.cp_slice == CpBlockSliceMode::NONE) {
            main_span = group.seq_size_per_block;
            break;
        }
        if (main_span == 0 && cp_size > 0 && group.seq_size_per_block % cp_size == 0) {
            main_span = group.seq_size_per_block / cp_size;
        }
    }
    if (main_span == 0) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "NormalCacheStore requires at least one reusable cache group");
    }
    auto status = validateNormalCacheStoreWireSpan(config, main_span, cp_size, "main");
    if (!status.ok()) {
        return status;
    }
    for (size_t module_index = 0; module_index < config.mtp_sub_configs.size(); ++module_index) {
        const auto& module_config = config.mtp_sub_configs[module_index];
        if (!module_config) {
            continue;
        }
        status = validateNormalCacheStoreWireSpan(
            *module_config, main_span, cp_size, "MTP module " + std::to_string(module_index));
        if (!status.ok()) {
            return status;
        }
    }
    return grpc::Status::OK;
}

grpc::Status RemoteRpcServer::validateNormalCacheStoreWireSpan(const CacheConfig& config,
                                                               size_t             expected_span,
                                                               size_t             cp_size,
                                                               const std::string& topology_name) {
    for (const auto& group : config.topology().groups()) {
        if (group.layer_ids.empty()) {
            continue;
        }
        const bool   direct_match     = group.seq_size_per_block == expected_span;
        const bool   cp_product_valid = expected_span <= std::numeric_limits<size_t>::max() / cp_size;
        const size_t cp_span          = cp_product_valid ? expected_span * cp_size : 0;
        const bool   cp_slice_match = cp_size > 1 && group.policy.cp_slice != CpBlockSliceMode::NONE && cp_product_valid
                                    && group.seq_size_per_block == cp_span;
        if (!direct_match && !cp_slice_match) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                topology_name + " cache group " + group.tag + " has physical span "
                                    + std::to_string(group.seq_size_per_block) + ", expected "
                                    + std::to_string(expected_span) + " (or CP-sliced " + std::to_string(cp_span)
                                    + ") for the NormalCacheStore key vector");
        }
    }
    return grpc::Status::OK;
}

grpc::Status RemoteRpcServer::init(const EngineInitParams&                                maga_init_params,
                                   std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                                   py::object                                             mm_process_engine) {
    rtp_llm::ProposeModelEngineInitParams* propose_params_ptr = propose_params ? propose_params.get() : nullptr;
    auto ret = LocalRpcServer::init(maga_init_params, std::move(propose_params), mm_process_engine);
    if (!ret.ok()) {
        return ret;
    }
    ret = validateBeforeCacheStoreInit();
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
    params.metrics_reporter             = metrics_reporter_;
    params.device_id                    = static_cast<int>(init_params.parallelism_config.local_rank);
    RTP_LLM_LOG_INFO("cache store listen port is [%ld], rdma listen port is [%ld] rdma_mode is [%d]",
                     params.listen_port,
                     params.rdma_listen_port,
                     params.rdma_mode);
    cache_store_ = NormalCacheStore::createNormalCacheStore(params);
    RTP_LLM_CHECK_WITH_INFO(cache_store_ != nullptr, "cache store init failed");
    RTP_LLM_LOG_INFO("cache store init success");

    cache_manager->setCacheStore(cache_store_);
    cache_manager->regUserMr(maga_init_params_.model_id, cache_store_);

    resource_.cache_store = std::dynamic_pointer_cast<NormalCacheStore>(cache_store_);
}

}  // namespace rtp_llm
