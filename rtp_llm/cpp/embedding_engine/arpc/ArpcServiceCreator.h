#pragma once

#include <vector>
#include <google/protobuf/service.h>
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingEngine.h"
#include "rtp_llm/cpp/embedding_engine/arpc/ArpcServerWrapper.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

namespace rtp_llm {

std::unique_ptr<::google::protobuf::Service>
createEmbeddingArpcService(int64_t                                       model_rpc_port,
                           int64_t                                       arpc_thread_num,
                           int64_t                                       arpc_queue_num,
                           int64_t                                       arpc_io_thread_num,
                           py::object                                    py_render,
                           py::object                                    py_tokenizer,
                           std::shared_ptr<rtp_llm::MultimodalProcessor> mm_processor,
                           std::shared_ptr<rtp_llm::EmbeddingEngine>     engine,
                           kmonitor::MetricsReporterPtr                  reporter,
                           bool                                          arpc_rdma_mode = false);

// Factory: open-source stub throws for RDMA; internal_source provides real RDMA impl.
std::unique_ptr<ArpcServerWrapper> createArpcServerWrapper(bool                                         arpc_rdma_mode,
                                                           std::unique_ptr<::google::protobuf::Service> service,
                                                           int                                          threadNum,
                                                           int                                          queueNum,
                                                           int                                          ioThreadNum,
                                                           int                                          port);

}  // namespace rtp_llm
