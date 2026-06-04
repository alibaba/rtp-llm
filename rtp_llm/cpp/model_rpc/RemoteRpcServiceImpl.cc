#include <memory>
#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew2.h"
#include "autil/NetUtil.h"

namespace rtp_llm {

grpc::Status RemoteRpcServiceImpl::init(const EngineInitParams&                                maga_init_params,
                                        py::object                                             mm_process_engine,
                                        std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    decode_entrance_ = maga_init_params.pd_sep_config.decode_entrance;
    RTP_LLM_LOG_INFO("remote rpc service init, decode_entrance is %d", decode_entrance_);

    if (decode_entrance_) {
        if (maga_init_params.pd_sep_config.role_type == RoleType::PREFILL) {
            prefill_server_new2_ = std::make_shared<PrefillRpcServerNew2>();
            local_server_        = prefill_server_new2_;
            return prefill_server_new2_->init(maga_init_params, mm_process_engine, std::move(propose_params));
        }

        decode_server_new2_ = std::make_shared<DecodeRpcServerNew2>();
        local_server_       = decode_server_new2_;
        return decode_server_new2_->init(maga_init_params, mm_process_engine, std::move(propose_params));
    }

    if (maga_init_params.pd_sep_config.role_type == RoleType::PREFILL) {
        prefill_server_ = std::make_shared<PrefillRpcServer>();
        local_server_   = prefill_server_;
        auto ret        = prefill_server_->init(maga_init_params, mm_process_engine, std::move(propose_params));
        if (!ret.ok()) {
            return ret;
        }
        initPrefillDpForwarder(maga_init_params);
        return grpc::Status::OK;
    } else {
        decode_server_ = std::make_shared<DecodeRpcServer>();
        local_server_  = decode_server_;
        return decode_server_->init(maga_init_params, mm_process_engine, std::move(propose_params));
    }
}

void RemoteRpcServiceImpl::initPrefillDpForwarder(const EngineInitParams& maga_init_params) {
    const auto& pc  = maga_init_params.parallelism_config;
    const auto& pdc = maga_init_params.pd_sep_config;
    if (pc.dp_size <= 1) {
        return;
    }

    int64_t local_rpc_port = 0;
    {
        pybind11::gil_scoped_acquire acquire;
        local_rpc_port = maga_init_params.server_config.attr("rpc_server_port").cast<int64_t>();
    }

    int64_t     stride   = pc.tp_size * pdc.worker_port_offset;
    int64_t     dp0_port = local_rpc_port - pc.dp_rank * stride;
    std::string bind_ip  = autil::NetUtil::getBindIp();

    for (int64_t i = 0; i < pc.dp_size; ++i) {
        prefill_dp_addrs_.push_back(bind_ip + ":" + std::to_string(dp0_port + i * stride));
    }
    local_dp_index_ = static_cast<size_t>(pc.dp_rank);

    std::string local_ip;
    std::string local_id;
    if (!autil::NetUtil::GetDefaultIp(local_ip) || local_ip.empty()) {
        std::string hostname;
        autil::NetUtil::GetHostName(hostname);
        local_id = "hostname_" + hostname;
    } else {
        local_id = "ip_" + local_ip;
    }
    std::string process_id =
        local_id + "_pid_" + std::to_string(getpid()) + "_timestamp_" + std::to_string(currentTimeUs());

    prefill_dp_forwarder_ = std::make_shared<PrefillServerCaller>(process_id);

    std::string addrs_str;
    for (size_t i = 0; i < prefill_dp_addrs_.size(); ++i) {
        if (i > 0)
            addrs_str += ", ";
        addrs_str += prefill_dp_addrs_[i];
        if (i == local_dp_index_)
            addrs_str += "(local)";
    }
    RTP_LLM_LOG_INFO(
        "prefill DP forwarder: dp_size=%ld, local_dp=%zu, addrs=[%s]", pc.dp_size, local_dp_index_, addrs_str.c_str());
}

}  // namespace rtp_llm
