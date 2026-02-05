#pragma once

#include <google/protobuf/service.h>
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"

namespace rtp_llm {

std::unique_ptr<::google::protobuf::Service> createNormalArpcService(const EngineInitParams& maga_init_params,
                                                                     std::shared_ptr<rtp_llm::EngineBase> engine,
                                                                     py::object                           py_tokenizer,
                                                                     kmonitor::MetricsReporterPtr         reporter);

}  // namespace rtp_llm
