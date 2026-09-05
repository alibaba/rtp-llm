package org.flexlb.constant;

public class CommonConstants {

    public static final String FUNCTION = "aigc.text-generation.generation";

    public static final String CODEC = "codec";

    public static final String TIMEOUT_HANDLER = "timeoutHandler";

    /** Separator between a physical {@code ip:port} address and its logical engine index. */
    public static final String LOGICAL_WORKER_ENGINE_INDEX_SEPARATOR = "@";

    /**
     * gRPC timeout message
     */
    public static final String DEADLINE_EXCEEDED_MESSAGE = "DEADLINE_EXCEEDED";

    /**
     * Port offset between HTTP port and gRPC port = HTTP port + GRPC_PORT_OFFSET
     */
    public static final int GRPC_PORT_OFFSET = 1;

}
