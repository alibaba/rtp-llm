package org.flexlb.constant;

public class CommonConstants {

    public static final String FUNCTION = "aigc.text-generation.generation";

    public static final String START = "start";

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
     * Port offset between HTTP port and ARPC port = HTTP port + ARPC_PORT_OFFSET.
     * Embedding/BERT workers expose MainseBertRpcService over ARPC on rpc_server_port (= base+1),
     * while http_port = base+5 in the rtp_llm ServerConfig port layout, so ARPC = http_port - 4.
     * Kept as a named constant (not a magic number) since the layout has shifted before.
     */
    public static final int ARPC_PORT_OFFSET = -4;

}