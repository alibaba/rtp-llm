
#include "rtp_llm/cpp/normal_engine/arpc/ArpcServiceCreator.h"

namespace rtp_llm {

std::unique_ptr<::google::protobuf::Service> createNormalArpcService(const EngineInitParams& maga_init_params,
                                                                     std::shared_ptr<rtp_llm::EngineBase> engine,
                                                                     py::object                           py_tokenizer,
                                                                     kmonitor::MetricsReporterPtr         reporter) {
    return nullptr;
}

}  // namespace rtp_llm
