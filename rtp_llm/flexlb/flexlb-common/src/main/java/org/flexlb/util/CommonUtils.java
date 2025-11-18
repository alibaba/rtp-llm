package org.flexlb.util;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class CommonUtils {

    /**
     * Convert HTTP port to gRPC port
     *
     * @param httpPort http port
     * @return gRPC port
     */
    public static int toGrpcPort(int httpPort) {
        return httpPort + 1;
    }
}
