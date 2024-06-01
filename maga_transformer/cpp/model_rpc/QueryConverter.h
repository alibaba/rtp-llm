#pragma once

#include <cstring>
#include <memory>
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/proto/model_rpc_service.pb.h"
#include "src/fastertransformer/core/Buffer.h"

namespace rtp_llm {
class QueryConverter {
public:
    static std::shared_ptr<GenerateInput> transQuery(const GenerateInputPB* input);

    static void transResponse(GenerateOutputsPB* outputs, const GenerateOutputs* response);

private:
    static std::shared_ptr<GenerateConfig> transGenerateConfig(const GenerateConfigPB* config_proto);

    static void transTensor(TensorPB* t, const fastertransformer::Buffer* buffer);
};

}  // namespace rtp_llm
