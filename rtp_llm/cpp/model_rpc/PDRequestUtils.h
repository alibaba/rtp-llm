#pragma once

#include <cstdint>
#include <functional>
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <string>
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/trace/span.h"

#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

namespace rtp_llm {

struct PDSupportDecision {
    bool        supported;
    const char* reason;
};

PDSupportDecision checkPDSupport(const GenerateInputPB& request);
ErrorInfo         validatePDHandoff(const GenerateInputPB& request);
std::string       masterEnqueuedHandoffUniqueKey(int64_t request_id);

ErrorInfo updateMultimodalFeaturesWithTrace(
    MultimodalProcessor& processor,
    std::shared_ptr<GenerateInput>& input,
    const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& span,
    grpc::ServerContext* server_context);

// Decode consumes FE-expanded IDs; Prefill performs multimodal embedding preparation.
ErrorInfo preprocessForPD(std::shared_ptr<GenerateInput>& input,
                          MultimodalProcessor*            processor,
                          bool                            is_mtp_eagle,
                          RoleType                        role,
                          grpc::ServerContext*            server_context,
                          const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& span);

// Rebuild a failed Prefill MM attempt from immutable wire input, preserving the
// original local request deadline and start time. Decode never calls this path.
ErrorInfo preprocessPrefillWithRetry(const GenerateInputPB&                                              request,
                                     std::shared_ptr<GenerateInput>&                                     input,
                                     MultimodalProcessor*                                                processor,
                                     bool                                                                is_mtp_eagle,
                                     const PDSepConfig&                                                  config,
                                     grpc::ServerContext*                                                server_context,
                                     const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& span,
                                     const std::function<ErrorInfo()>&                                   cancellation);

}  // namespace rtp_llm
