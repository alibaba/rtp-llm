#include <vector>
#include <google/protobuf/service.h>
#include "rtp_llm/cpp/th_op/GptInitParameter.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingEngine.h"
#include "rtp_llm/cpp/embedding_engine/arpc/ArpcServiceCreator.h"

namespace rtp_llm {

std::unique_ptr<::google::protobuf::Service>
createEmbeddingArpcService(const rtp_llm::GptInitParameter&               gpt_init_params,
                           py::object                                py_render,
                           std::shared_ptr<rtp_llm::MultimodalProcessor> mm_processor,
                           std::shared_ptr<rtp_llm::EmbeddingEngine> engine,
                           kmonitor::MetricsReporterPtr              reporter) {    
    return nullptr;
}

}  // namespace rtp_llm
