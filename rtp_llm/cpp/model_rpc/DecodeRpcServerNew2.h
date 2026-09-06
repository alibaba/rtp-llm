#pragma once

#include <cstddef>

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

size_t selectDecodeEntranceDpIndex(size_t dp_count, int64_t handoff_id);

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
    static grpc::Status parsePrefillDpAddr(const std::string& addr, std::string* ip, uint32_t* port);
    void updateAuxInfo(GenerateOutputsPB& outputs_pb, std::shared_ptr<GenerateStream>& stream) override;

private:
    std::atomic<int64_t>                 unique_key_id_{0};
    std::shared_ptr<PrefillServerCaller> prefill_server_caller_;
};

}  // namespace rtp_llm
