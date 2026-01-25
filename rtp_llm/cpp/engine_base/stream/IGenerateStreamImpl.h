#pragma once

#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"

namespace rtp_llm {

class IGenerateStreamImpl: public IGenerateStream {
public:
    IGenerateStreamImpl(const std::shared_ptr<GenerateStream>& stream, rtp_llm::DeviceBase* device);
    ~IGenerateStreamImpl() = default;

public:
    // IGenerateStream interface implementation
    void             appendTokenId(int batch_id, int token_id) override;
    std::vector<int> currentExecuteTokens(int batch_id) override;

    void appendSPInfo(const std::vector<int>& propose_tokens,
                      const TensorPB&         propose_probs,
                      const TensorPB&         propose_hidden) override;

    std::optional<std::tuple<std::vector<int>, TensorPB, TensorPB>> getSPInfoPB() override;

    int                       reuseBlockNum() override;
    std::tuple<int, int, int> getReuseLength() override;
    void setPrefillReuseLength(int reuse_length, int local_reuse_length, int remote_reuse_length) override;

    std::pair<std::string, uint32_t> getPrefillAddr() override;

    std::vector<int32_t> getContextPositionIdsPB() override;
    void                 setContextPositionIds(const std::vector<int32_t>& ids) override;

    // Wait for first token to be generated and set via updateOutput (used in PD separation)
    bool waitForRemoteGenerate() override;

    // Get original request (GenerateInputPB) for calling prefill server
    const GenerateInputPB* getOriginalRequest() const override;

    // Check if need to call prefill server
    bool needCallPrefill() const override;

    // Set stream to stop with error code and message
    void setStop(ErrorCode error_code, const std::string& error_msg) override;

    // Additional helper methods
    std::shared_ptr<GenerateStream> getStream() const {
        return stream_;
    }

    rtp_llm::BufferPtr getContextPositionIds() const {
        return stream_->getContextPositionIds();
    }

    void setContextPositionIds(rtp_llm::BufferPtr context_position_ids) {
        stream_->setContextPositionIds(context_position_ids);
    }

private:
    std::shared_ptr<GenerateStream> stream_;
    rtp_llm::DeviceBase*            device_;
};

}  // namespace rtp_llm
