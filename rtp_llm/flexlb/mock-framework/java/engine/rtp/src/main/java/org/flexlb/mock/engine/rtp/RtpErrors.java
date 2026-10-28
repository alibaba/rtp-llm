package org.flexlb.mock.engine.rtp;

import io.grpc.*;
import org.flexlb.engine.grpc.EngineRpcService.ErrorDetailsPB;

/** Native C++ grpc::Status.error_details wire encoding, shared by P and D. */
final class RtpErrors {
    private static final Metadata.Key<byte[]> DETAILS = Metadata.Key.of("grpc-status-details-bin", Metadata.BINARY_BYTE_MARSHALLER);
    static StatusRuntimeException failure(Status status, long code) {
        if (code == 0) return status.asRuntimeException();
        var trailers = new Metadata();
        trailers.put(DETAILS, ErrorDetailsPB.newBuilder().setErrorCode(code).setErrorMessage(status.getDescription()).build().toByteArray());
        return status.asRuntimeException(trailers);
    }
}
