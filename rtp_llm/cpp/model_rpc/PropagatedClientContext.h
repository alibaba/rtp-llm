#pragma once

#include <chrono>
#include <memory>

#include "grpc++/grpc++.h"

namespace rtp_llm {

inline grpc::PropagationOptions remoteLoadPropagationOptions() {
    grpc::PropagationOptions options;
    options.enable_deadline_propagation();
    options.enable_cancellation_propagation();
    return options;
}

inline std::shared_ptr<grpc::ClientContext>
makePropagatedClientContext(grpc::ServerContext*                       parent_context,
                            std::chrono::system_clock::time_point business_deadline) {
    std::unique_ptr<grpc::ClientContext> context;
    if (parent_context == nullptr) {
        context = std::make_unique<grpc::ClientContext>();
    } else {
        context = grpc::ClientContext::FromServerContext(*parent_context, remoteLoadPropagationOptions());
    }
    context->set_deadline(business_deadline);
    return std::shared_ptr<grpc::ClientContext>(std::move(context));
}

}  // namespace rtp_llm
