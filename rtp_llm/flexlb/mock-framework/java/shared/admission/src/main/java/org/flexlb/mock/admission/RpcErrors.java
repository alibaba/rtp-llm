package org.flexlb.mock.admission;

import io.grpc.*;
import org.flexlb.mock.engine.RequestFailure;
import static org.flexlb.mock.engine.RequestFailure.Reason.*;
import org.flexlb.engine.grpc.EngineRpcService.ErrorDetailsPB;

/** Native C++ grpc::Status.error_details wire encoding, shared by P and D. */
public final class RpcErrors {
    private RpcErrors() {}
    private static final Metadata.Key<byte[]> DETAILS = Metadata.Key.of("grpc-status-details-bin", Metadata.BINARY_BYTE_MARSHALLER);
    static StatusRuntimeException failure(Status status, long code) {
        if (code == 0) return status.asRuntimeException();
        var trailers = new Metadata();
        trailers.put(DETAILS, ErrorDetailsPB.newBuilder().setErrorCode(code).setErrorMessage(status.getDescription()).build().toByteArray());
        return status.asRuntimeException(trailers);
    }
    static RequestFailure scheduling(Status status, long code) {
        if (status.isOk()) return null;
        var reason = switch (status.getCode()) {
            case CANCELLED -> CANCELLED;
            case DEADLINE_EXCEEDED -> TIMEOUT;
            case RESOURCE_EXHAUSTED -> code == 8429 ? PREEMPTED : CAPACITY;
            case UNAVAILABLE -> CLOSED;
            case ALREADY_EXISTS -> DUPLICATE;
            case INVALID_ARGUMENT, FAILED_PRECONDITION, UNIMPLEMENTED -> INVALID;
            default -> INTERNAL;
        };
        return new RequestFailure(reason, status.getDescription(), failure(status, code));
    }
    public static Status status(RequestFailure failure) {
        if (failure == null) return Status.OK;
        if (failure.getCause() instanceof StatusRuntimeException nativeError) return nativeError.getStatus();
        return (switch (failure.reason()) {
            case CANCELLED -> Status.CANCELLED;
            case TIMEOUT -> Status.DEADLINE_EXCEEDED;
            case CAPACITY, OVERLOADED, PREEMPTED -> Status.RESOURCE_EXHAUSTED;
            case CLOSED -> Status.UNAVAILABLE;
            case DUPLICATE -> Status.ALREADY_EXISTS;
            case INVALID -> Status.FAILED_PRECONDITION;
            case INTERNAL -> Status.INTERNAL;
        }).withDescription(failure.getMessage());
    }
    public static long code(RequestFailure failure) {
        if (failure == null) return 0;
        if (failure.getCause() instanceof StatusRuntimeException nativeError) {
            var trailers = nativeError.getTrailers();
            var bytes = trailers == null ? null : trailers.get(DETAILS);
            if (bytes == null) return 0;
            try { return ErrorDetailsPB.parseFrom(bytes).getErrorCode(); }
            catch (com.google.protobuf.InvalidProtocolBufferException error) { throw new IllegalStateException("Invalid local RTP error details", error); }
        }
        return switch (failure.reason()) {
            case TIMEOUT -> 603;
            case CAPACITY -> 602;
            case PREEMPTED -> 8429;
            case CANCELLED -> 8100;
            default -> 0;
        };
    }
    static StatusRuntimeException failure(RequestFailure failure) { return failure(status(failure), code(failure)); }
}
