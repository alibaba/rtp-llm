package org.flexlb.util;

public class CommonUtils {

    private CommonUtils() {

    }

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
