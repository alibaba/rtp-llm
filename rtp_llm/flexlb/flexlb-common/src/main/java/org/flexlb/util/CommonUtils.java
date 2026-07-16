package org.flexlb.util;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.flexlb.constant.CommonConstants;

@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class CommonUtils {

    /**
     * Convert HTTP port to gRPC port
     *
     * @param httpPort http port
     * @return gRPC port
     */
    public static int toGrpcPort(int httpPort) {
        return httpPort + CommonConstants.RPC_PORT_OFFSET;
    }

    /**
     * Convert gRPC port back to HTTP port
     *
     * @param grpcPort gRPC port
     * @return HTTP port
     */
    public static int toHttpPort(int grpcPort) {
        return grpcPort - CommonConstants.RPC_PORT_OFFSET;
    }

    /**
     * Convert HTTP port to ARPC port (embedding/BERT MainseBertRpcService)
     *
     * @param httpPort http port
     * @return ARPC port
     */
    public static int toArpcPort(int httpPort) {
        return httpPort + CommonConstants.RPC_PORT_OFFSET;
    }
}
