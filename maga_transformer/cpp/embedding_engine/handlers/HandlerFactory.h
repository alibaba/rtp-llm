#pragma once

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/fastertransformer/core/Buffer.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/embedding_engine/handlers/HandlerBase.h"
#include "maga_transformer/cpp/embedding_engine/handlers/MainseHandler.h"
#include <memory>

namespace rtp_llm {
    static std::unique_ptr<HandlerBase> create_handler_from_factory(const GptInitParameter& params) {
        return std::make_unique<MainseHandler>(params);
    }
}  // namespace rtp_llm
