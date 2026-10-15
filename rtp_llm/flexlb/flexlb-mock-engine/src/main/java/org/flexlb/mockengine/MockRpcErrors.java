package org.flexlb.mockengine;

import io.grpc.Metadata;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import org.flexlb.engine.grpc.EngineRpcService;

/** PrefillRpcServer::parseDownstreamError: domain details precede transport fallback. */
final class MockRpcErrors {
    private static final Metadata.Key<byte[]> DETAILS = Metadata.Key.of("grpc-status-details-bin", Metadata.BINARY_BYTE_MARSHALLER);
    private MockRpcErrors() { }
    static StatusRuntimeException error(Status status, long code, String message) {
        Metadata trailers = new Metadata();
        trailers.put(DETAILS, EngineRpcService.ErrorDetailsPB.newBuilder().setErrorCode(code).setErrorMessage(message).build().toByteArray());
        return status.withDescription(message).asRuntimeException(trailers);
    }
    static Status status(EngineRpcService.ErrorDetailsPB error) {
        return switch ((int) error.getErrorCode()) {
            case 602, 8106, 8211, 8429 -> Status.RESOURCE_EXHAUSTED;
            case 603, 8203, 8204, 8213, 8214 -> Status.DEADLINE_EXCEEDED;
            case 8100 -> Status.CANCELLED;
            case 8207 -> Status.UNAVAILABLE;
            default -> Status.INTERNAL;
        };
    }
    static EngineRpcService.ErrorDetailsPB downstream(Throwable error) { return downstream(error, 8207); }
    static EngineRpcService.ErrorDetailsPB downstream(Throwable error, long fallback) {
        Metadata trailers = error == null ? null : Status.trailersFromThrowable(error);
        if (trailers != null && trailers.get(DETAILS) != null) {
            try {
                var details = EngineRpcService.ErrorDetailsPB.parseFrom(trailers.get(DETAILS));
                if (details.getErrorCode() != 0) return details;
            } catch (com.google.protobuf.InvalidProtocolBufferException ignored) { /* transport fallback */ }
        }
        Status status = error == null ? Status.ALREADY_EXISTS : Status.fromThrowable(error);
        return EngineRpcService.ErrorDetailsPB.newBuilder()
                .setErrorCode(status.getCode() == Status.Code.RESOURCE_EXHAUSTED ? 8211 : fallback)
                .setErrorMessage("Remote ALLOCATE failed: " + status).build();
    }
}
