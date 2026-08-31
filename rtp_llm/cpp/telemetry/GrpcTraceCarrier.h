#pragma once

#include <string>

#include "grpc++/grpc++.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/context/propagation/global_propagator.h"
#include "opentelemetry/context/propagation/text_map_propagator.h"
#include "opentelemetry/nostd/string_view.h"

namespace rtp_llm {
namespace telemetry {

// Read-only carrier over gRPC server metadata for W3C context extraction.
// grpc keys are lower-cased by the transport, matching "traceparent"/"tracestate".
class GrpcServerMetadataCarrier: public opentelemetry::context::propagation::TextMapCarrier {
public:
    explicit GrpcServerMetadataCarrier(const grpc::ServerContext* server_context): server_context_(server_context) {}

    opentelemetry::nostd::string_view Get(opentelemetry::nostd::string_view key) const noexcept override {
        if (server_context_ == nullptr) {
            return "";
        }
        const auto& metadata = server_context_->client_metadata();
        auto        it       = metadata.find(grpc::string_ref(key.data(), key.size()));
        if (it == metadata.end()) {
            return "";
        }
        return opentelemetry::nostd::string_view(it->second.data(), it->second.size());
    }

    void Set(opentelemetry::nostd::string_view, opentelemetry::nostd::string_view) noexcept override {
        // extraction-only carrier
    }

private:
    const grpc::ServerContext* server_context_;
};

// Write-only carrier over gRPC client metadata for W3C context injection.
class GrpcClientMetadataCarrier: public opentelemetry::context::propagation::TextMapCarrier {
public:
    explicit GrpcClientMetadataCarrier(grpc::ClientContext* client_context): client_context_(client_context) {}

    opentelemetry::nostd::string_view Get(opentelemetry::nostd::string_view) const noexcept override {
        return "";
    }

    void Set(opentelemetry::nostd::string_view key, opentelemetry::nostd::string_view value) noexcept override {
        if (client_context_ == nullptr) {
            return;
        }
        try {
            client_context_->AddMetadata(std::string(key.data(), key.size()), std::string(value.data(), value.size()));
        } catch (...) {
            // fail-open: dropping the carrier must never break the RPC
        }
    }

private:
    grpc::ClientContext* client_context_;
};

// Extracts remote OTel context from gRPC server metadata via the global W3C
// propagator. Invalid/missing headers yield an empty context (safe fallback);
// never throws.
inline opentelemetry::context::Context extractContextFromServerMetadata(const grpc::ServerContext* server_context) {
    opentelemetry::context::Context empty_context{};
    try {
        GrpcServerMetadataCarrier carrier(server_context);
        auto propagator = opentelemetry::context::propagation::GlobalTextMapPropagator::GetGlobalPropagator();
        return propagator->Extract(carrier, empty_context);
    } catch (...) {
        return empty_context;
    }
}

// Injects the given OTel context into gRPC client metadata via the global W3C
// propagator. Must be called after ClientContext creation and before the RPC
// is initiated (covers every retry re-creation); never throws.
inline void injectContextToClientMetadata(grpc::ClientContext*                   client_context,
                                          const opentelemetry::context::Context& context) {
    try {
        GrpcClientMetadataCarrier carrier(client_context);
        auto propagator = opentelemetry::context::propagation::GlobalTextMapPropagator::GetGlobalPropagator();
        propagator->Inject(carrier, context);
    } catch (...) {
        // fail-open
    }
}

}  // namespace telemetry
}  // namespace rtp_llm
