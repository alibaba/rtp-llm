#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"
#include "rtp_llm/cpp/telemetry/TelemetryRuntime.h"
#include <functional>

namespace th = torch;

namespace rtp_llm {

class RtpLLMOp: public th::jit::CustomClassHolder {
public:
    RtpLLMOp();
    ~RtpLLMOp();

    void init(py::object model,
              py::object engine_config,
              py::object vit_config,
              py::object propose_model,
              py::object mm_process_engine,
              py::object trace_config = py::none());
    void stop(const std::function<void(int64_t)>& before_rpc_shutdown = {});
    void setServing(bool serving);

private:
    EngineInitParams initModel(py::object model, py::object engine_config, py::object vit_config);
    std::unique_ptr<ProposeModelEngineInitParams> initProposeModel(py::object              propose_model,
                                                                   const EngineInitParams& base_params);
    void                                          initRPCServer(const EngineInitParams                        maga_init_params,
                                                                std::unique_ptr<ProposeModelEngineInitParams> propose_params,
                                                                py::object                                    mm_process_engine,
                                                                telemetry::TelemetryConfig                    trace_config);

private:
    std::unique_ptr<RpcServiceImpl> model_rpc_service_;
    std::unique_ptr<grpc::Server>   grpc_server_;
    std::thread                     grpc_server_thread_;
    std::atomic<bool>               is_server_ready_{false};
    std::atomic<bool>               is_server_shutdown_{false};
    size_t                          model_id_ = 0;
};

void registerRtpLLMOp(const py::module& m);

}  // namespace rtp_llm
