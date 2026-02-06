#include <memory>
#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"

namespace rtp_llm {

grpc::Status RemoteRpcServiceImpl::init(const EngineInitParams&                                maga_init_params,
                                        std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                                        py::object                                             mm_process_engine) {
    decode_entrance_ = maga_init_params.pd_sep_config.decode_entrance;
    RTP_LLM_LOG_INFO("remote rpc service init, decode_entrance is %d", decode_entrance_);
    if (decode_entrance_) {
        if (maga_init_params.pd_sep_config.role_type == RoleType::PREFILL) {
            prefill_server_new_ = std::make_shared<PrefillRpcServerNew>();
            local_server_       = prefill_server_new_;
            return prefill_server_new_->init(maga_init_params, std::move(propose_params), mm_process_engine);
        } else {
            decode_server_new_ = std::make_shared<DecodeRpcServerNew>();
            local_server_      = decode_server_new_;
            return decode_server_new_->init(maga_init_params, std::move(propose_params), mm_process_engine);
        }
    } else {
        if (maga_init_params.pd_sep_config.role_type == RoleType::PREFILL) {
            prefill_server_ = std::make_shared<PrefillRpcServer>();
            local_server_   = prefill_server_;
            return prefill_server_->init(maga_init_params, std::move(propose_params), mm_process_engine);
        } else {
            decode_server_ = std::make_shared<DecodeRpcServer>();
            local_server_  = decode_server_;
            return decode_server_->init(maga_init_params, std::move(propose_params), mm_process_engine);
        }
    }
}

}  // namespace rtp_llm
