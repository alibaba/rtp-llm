package org.flexlb.mockengine;

import io.grpc.Metadata;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;

/** vit_rpc_server._abort_ft_runtime: domain details and cross-layer reporting ownership. */
final class MockVitErrors {
    private static final Metadata.Key<String> REPORTED = Metadata.Key.of("x-rtp-vit-error-reported", Metadata.ASCII_STRING_MARSHALLER);
    private static final Metadata.Key<byte[]> DETAILS = Metadata.Key.of("grpc-status-details-bin", Metadata.BINARY_BYTE_MARSHALLER);
    static StatusRuntimeException cancelled() { return MockRpcErrors.error(Status.CANCELLED, 8100, "ViT request cancelled"); }
    static boolean reported(Throwable error) {
        var trailers = Status.trailersFromThrowable(error);
        return trailers != null && "1".equals(trailers.get(REPORTED));
    }
    static StatusRuntimeException normalize(Throwable error) {
        var trailers = Status.trailersFromThrowable(error);
        if (error instanceof StatusRuntimeException status && trailers != null && trailers.containsKey(DETAILS)) return status;
        var status = Status.fromThrowable(error);
        long code = switch (status.getCode()) {
            case CANCELLED -> 8100;
            case DEADLINE_EXCEEDED -> 603;
            case INVALID_ARGUMENT -> 902;
            case RESOURCE_EXHAUSTED -> 409;
            default -> error instanceof java.util.concurrent.TimeoutException ? 603
                    : error instanceof java.util.concurrent.CancellationException || error instanceof InterruptedException ? 8100
                    : error instanceof IllegalArgumentException ? 902 : 903;
        };
        if (status.getCode() == Status.Code.UNKNOWN) status = switch ((int)code) {
            case 603 -> Status.DEADLINE_EXCEEDED;
            case 8100 -> Status.CANCELLED;
            case 902 -> Status.INVALID_ARGUMENT;
            default -> Status.INTERNAL;
        };
        return MockRpcErrors.error(status, code, error.getMessage() == null ? error.getClass().getSimpleName() : error.getMessage());
    }
    static StatusRuntimeException markReported(StatusRuntimeException error) {
        var trailers = new Metadata();
        if (error.getTrailers() != null) trailers.merge(error.getTrailers());
        trailers.put(REPORTED, "1");
        return error.getStatus().asRuntimeException(trailers);
    }
}
