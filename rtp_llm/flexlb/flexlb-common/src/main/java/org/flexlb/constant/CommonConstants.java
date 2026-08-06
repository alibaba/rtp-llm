package org.flexlb.constant;

import java.util.Set;

public class CommonConstants {

    public static final String FUNCTION = "aigc.text-generation.generation";

    public static final String CODEC = "codec";

    public static final String TIMEOUT_HANDLER = "timeoutHandler";

    /**
     * gRPC timeout message
     */
    public static final String DEADLINE_EXCEEDED_MESSAGE = "DEADLINE_EXCEEDED";

    /**
     * Port offset between HTTP port and gRPC port = HTTP port + GRPC_PORT_OFFSET
     */
    public static final int GRPC_PORT_OFFSET = 1;

    /**
     * Default Auto-TPM request QoS priority. Applied when the client omits
     * the priority (proto3 int32 default 0) or sends an invalid value.
     */
    public static final int DEFAULT_REQUEST_PRIORITY = 50;

    /**
     * Valid Auto-TPM request QoS priority levels. Any other value is
     * normalized to {@link #DEFAULT_REQUEST_PRIORITY} on the Java side.
     */
    public static final Set<Integer> VALID_REQUEST_PRIORITIES = Set.of(30, 40, 50, 60, 70);

}
