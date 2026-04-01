#pragma once

#include "rtp_llm/cpp/cache/connector/IGenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

class IGenerateStreamImpl: public IGenerateStream {
public:
    explicit IGenerateStreamImpl(const std::shared_ptr<GenerateStream>& stream);
    ~IGenerateStreamImpl() override = default;

public:
    int64_t     deadlineMs() const override;
    std::string uniqueKey() const override;
    int64_t     requestId() const override;

    void             appendTokenId(int batch_id, int token_id) override;
    std::vector<int> currentExecuteTokens(int batch_id) override;

    void appendSPInfo(const std::vector<int>& propose_tokens,
                      const TensorPB&         propose_probs,
                      const TensorPB&         propose_hidden) override;

    std::optional<std::tuple<std::vector<int>, TensorPB, TensorPB>> getSPInfoPB() override;

    int                            reuseBlockNum() override;
    std::tuple<int, int, int, int> getReuseLength() override;
    void                           setPrefillReuseLength(int reuse_length,
                                                         int local_reuse_length,
                                                         int remote_reuse_length,
                                                         int memory_reuse_length) override;

    std::pair<std::string, uint32_t> getPrefillAddr() override;

    std::vector<int32_t> getContextPositionIdsPB() override;
    void                 setContextPositionIds(const std::vector<int32_t>& ids) override;

    bool waitForRemoteGenerate() override;

    int getPrefillTpSize() const override;

    void setStop(ErrorCode error_code, const std::string& error_msg) override;

    std::shared_ptr<GenerateStream> getStream() const {
        return stream_;
    }

    torch::Tensor getContextPositionIds() const {
        return stream_->getContextPositionIds();
    }

    void setContextPositionIds(torch::Tensor context_position_ids) {
        stream_->setContextPositionIds(std::move(context_position_ids));
    }

private:
    std::shared_ptr<GenerateStream> stream_;
};

}  // namespace rtp_llm
