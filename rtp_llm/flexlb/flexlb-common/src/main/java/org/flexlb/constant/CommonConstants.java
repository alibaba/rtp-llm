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
     * Both engines bind their RPC service to rpc_server_port = registered port + 1
     * (RTP-LLM deployments register base+0, the frontend HTTP port, into service discovery;
     * port layout: {@code rtp_llm/config/py_config_modules.py}). The LLM engine serves gRPC
     * ({@code LocalRpcServer}) there, the embedding engine serves ARPC
     * ({@code MainseArpcServiceImpl}) — same offset, protocol differs by engine type, and
     * each caller reads the field matching the protocol it speaks. One offset for both, since
     * the port layout is protocol-independent.
     */
    public static final int RPC_PORT_OFFSET = 1;

}
