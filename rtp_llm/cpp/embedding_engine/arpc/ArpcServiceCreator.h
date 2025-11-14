#pragma once

#include <vector>
#include <google/protobuf/service.h>
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingEngine.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

namespace rtp_llm {

std::unique_ptr<::google::protobuf::Service>
createEmbeddingArpcService(int64_t model_rpc_port,
                           int64_t arpc_thread_num,
                           int64_t arpc_queue_num,
                           int64_t arpc_io_thread_num,
                           py::object                                    py_render,
                           py::object                                    py_tokenizer,
                           std::shared_ptr<rtp_llm::MultimodalProcessor> mm_processor,
                           std::shared_ptr<rtp_llm::EmbeddingEngine>     engine,
                           kmonitor::MetricsReporterPtr                  reporter);

}  // namespace rtp_llm
