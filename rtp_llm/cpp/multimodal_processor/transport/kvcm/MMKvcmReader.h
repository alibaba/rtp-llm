#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <torch/python.h>

#include "rtp_llm/cpp/config/MMKvcmConfig.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/multimodal_processor/transport/MMRemoteOutputTransport.h"
#include "rtp_llm/cpp/multimodal_processor/transport/kvcm/MMKvcmClient.h"

namespace rtp_llm {

bool assembleMMKvcmOutput(const std::vector<torch::Tensor>& tensors,
                          const MultimodalOutputPB&         receipt,
                          MultimodalOutput*                 output);

class MMKvcmReader: public MMReceiptReader {
public:
    explicit MMKvcmReader(std::shared_ptr<MMKvcmClient> client): client_(std::move(client)) {}
    MMKvcmReader(std::shared_ptr<MMKvcmClient> client, const MMKvcmConfig& config, int device_id):
        client_(std::move(client)), config_(config), device_id_(device_id), validate_manifest_(true) {}

    const char* name() const override {
        return "kvcm";
    }

    bool          advertise(const std::string& endpoint, MultimodalInputsPB& request_pb) override;
    bool          matches(const MultimodalOutputPB& receipt) const override;
    ConsumeResult consume(const MultimodalOutputPB& receipt, DeliveryContext& context) override;
    void          discard(const MultimodalOutputPB& receipt, DeliveryContext& context) override;

private:
    static std::vector<std::string> handlesOf(const MultimodalOutputPB& receipt);

    bool validateAndAllocate(const MultimodalOutputPB&   receipt,
                             std::vector<torch::Tensor>* tensors,
                             std::vector<MMKvcmBuffer>*  objects,
                             std::string*                error) const;

    std::shared_ptr<MMKvcmClient> client_;
    MMKvcmConfig                  config_;
    int                           device_id_         = -1;
    bool                          validate_manifest_ = false;
};

std::unique_ptr<MMReceiptReader> createMMKvcmReader(std::shared_ptr<MMKvcmClient> client);

}  // namespace rtp_llm
