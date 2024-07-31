#pragma once

#include <vector>
#include <google/protobuf/service.h>
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingEngine.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/EmbeddingHandlerOp.h"

namespace rtp_llm {

std::unique_ptr<::google::protobuf::Service>
createEmbeddingArpcService(const ft::GptInitParameter&               gpt_init_params,
                           py::object                                py_render,
                           std::shared_ptr<rtp_llm::EmbeddingEngine> engine,
                           kmonitor::MetricsReporterPtr              reporter);

} // namespace rtp_llm
