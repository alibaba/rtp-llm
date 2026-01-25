#pragma once

#include <string>
#include <vector>
#include <optional>
#include <tuple>
#include <torch/torch.h>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

class IGenerateStream {
public:
    virtual ~IGenerateStream() = default;

public:
    virtual void             appendTokenId(int batch_id, int token_id) = 0;
    virtual std::vector<int> currentExecuteTokens(int batch_id)        = 0;

    virtual void appendSPInfo(const std::vector<int>& propose_tokens,
                              const TensorPB&         propose_probs,
                              const TensorPB&         propose_hidden)                             = 0;
    virtual std::optional<std::tuple<std::vector<int>, TensorPB, TensorPB>> getSPInfoPB() = 0;

    virtual int                       reuseBlockNum()  = 0;
    virtual std::tuple<int, int, int> getReuseLength() = 0;  // reuse_length, local_reuse_length, remote_reuse_length
    virtual void setPrefillReuseLength(int reuse_length, int local_reuse_length, int remote_reuse_length) = 0;

    virtual std::pair<std::string, uint32_t> getPrefillAddr() = 0;  // prefill_ip, prefill_port

    virtual std::vector<int32_t> getContextPositionIdsPB()                              = 0;
    virtual void                 setContextPositionIds(const std::vector<int32_t>& ids) = 0;

    // Wait for first token to be generated and set via updateOutput (used in PD separation)
    virtual bool waitForRemoteGenerate() = 0;

    // Get original request (GenerateInputPB) for calling prefill server
    virtual const GenerateInputPB* getOriginalRequest() const = 0;

    // Check if need to call prefill server
    virtual bool needCallPrefill() const = 0;
};
using IGenerateStreamPtr = std::shared_ptr<IGenerateStream>;

}  // namespace rtp_llm