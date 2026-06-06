#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillServerCaller.h"

namespace rtp_llm {

bool decodeEntranceRequiresPrefill(const GenerateInputPB& request);

struct DecodeEntranceKeys {
    std::string business_unique_key;
    std::string handoff_unique_key;
};

DecodeEntranceKeys buildDecodeEntranceKeys(const GenerateInputPB& request,
                                           const std::string&     bind_ip,
                                           int64_t                unique_key_id,
                                           int64_t                current_time_us);

GenerateInputPB makeDecodeEntranceHandoffRequest(const GenerateInputPB& request, const std::string& handoff_unique_key);

void updateDecodeAuxInfo(GenerateOutputsPB&                                 outputs_pb,
                         std::shared_ptr<GenerateStream>&                   stream,
                         const std::shared_ptr<PrefillServerCallerContext>& prefill_ctx);

class DecodeRpcServerNew2: public RemoteRpcServer {
public:
    DecodeRpcServerNew2()  = default;
    ~DecodeRpcServerNew2() = default;

public:
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      py::object                                             mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) override;

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   server_context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* response_writer);

private:
    grpc::Status pollStreamOutputWithPrefill(grpc::ServerContext*                               context,
                                             const std::string&                                 request_key,
                                             WriterInterface*                                   writer,
                                             std::shared_ptr<GenerateStream>&                   stream,
                                             const std::shared_ptr<PrefillServerCallerContext>& prefill_ctx);

private:
    std::atomic<int64_t>                 unique_key_id_{0};
    std::shared_ptr<PrefillServerCaller> prefill_server_caller_;
};

}  // namespace rtp_llm
