#pragma once

#include "rtp_llm/cpp/model_rpc/DecodeAdmissionController.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

namespace rtp_llm {

// AcquireResult -> grpc::Status. Kept in its own header so the mapping and the controller can
// be unit tested without linking the RPC server (and therefore without a GPU): the mapping
// carries two cross-module contracts PrefillRpcServer depends on, so it is asserted directly
// instead of through the gRPC handler.
inline grpc::Status admissionResultToStatus(DecodeAdmissionController::AcquireResult result) {
    switch (result) {
        case DecodeAdmissionController::AcquireResult::ACQUIRED:
            return grpc::Status::OK;
        case DecodeAdmissionController::AcquireResult::CANCELLED:
            return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled while waiting for decode admission");
        case DecodeAdmissionController::AcquireResult::TIMED_OUT:
            // Deliberately not RESOURCE_EXHAUSTED: PrefillRpcServer maps that code to
            // DECODE_MALLOC_FAILED(8211) (PrefillRpcServer.cc:138), which would report a
            // queueing timeout as a decode KV allocation failure and send operators looking
            // at memory water marks instead of the admission limit. The message also
            // deliberately avoids the "Deadline Exceeded" / "Connection timed out"
            // substrings PrefillRpcServer greps for (:126, :129): those branches close the
            // gRPC connection, and a decode role that is merely saturated is healthy --
            // forcing a reconnect would turn back-pressure into a transport fault. The cost
            // is a less specific error code upstream.
            return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED,
                                "request timed out while waiting for decode admission");
        case DecodeAdmissionController::AcquireResult::OVERSIZED:
            return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                                "request batch exceeds the decode admission limit");
    }
    return grpc::Status(grpc::StatusCode::INTERNAL, "unknown decode admission result");
}

}  // namespace rtp_llm
