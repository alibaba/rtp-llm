package org.flexlb.sync.util;

import io.grpc.Status;

public final class GrpcStatusUtils {

    private GrpcStatusUtils() {
    }

    public static boolean isDeadlineExceeded(Throwable throwable) {
        return Status.fromThrowable(throwable).getCode() == Status.Code.DEADLINE_EXCEEDED;
    }
}
