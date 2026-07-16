package org.flexlb.enums;

/**
 * Engine type of the workers behind this flexlb instance. One declared fact that drives
 * both worker liveness and which RPC port slot the scheduler returns — mirroring the
 * engine-side reality that the two types bind different protocols to rpc_server_port
 * (= registered port + 1, see {@code rtp_llm/config/py_config_modules.py}).
 *
 * <p>{@link #LLM} ({@code RtpLLMOp}): serves gRPC ({@code LocalRpcServer}) on base+1 with
 * {@code GetWorkerStatus} — liveness is probed per worker via gRPC, schedule targets carry
 * {@code grpc_port}.
 *
 * <p>{@link #EMBEDDING} ({@code RtpEmbeddingOp}): serves ARPC ({@code MainseArpcServiceImpl})
 * on base+1 and has no {@code GetWorkerStatus} — liveness degrades to trusting the
 * service-discovery host list (no probing, no load metrics, so only load-unaware strategies
 * are permitted), schedule targets carry {@code arpc_port}.
 */
public enum EngineType {

    LLM,

    EMBEDDING
}
