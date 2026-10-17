package org.flexlb.testfixture;

import lombok.Data;
import org.flexlb.dao.route.RoleType;

import java.util.Map;

/** Mutable input used to assemble worker observations in tests. */
@Data
public class WorkerStatusResponse {

    private RoleType role;

    /**
     * Compatibility-only field. RTP-LLM's gRPC GetWorkerStatus currently does
     * not populate available_concurrency, so the observed value is the protobuf
     * default 0. Do not use it for routing, admission control, or batch sizing.
     */
    private long availableConcurrency;

    private long runningQueryLen;

    private long waitingQueryLen;

    private Map<String, TaskInfo> runningTaskInfo;

    private Map<String, TaskInfo> finishedTaskInfo;

    private Long statusVersion = 0L;

    private Long latestFinishedVersion = 0L;

    private double stepLatencyMs;

    private long iterateCount;

    private long dpSize;

    private long tpSize;

    private long dpRank;

    private boolean alive;

    private long availableKvCacheTokens;

    private long totalKvCacheTokens;

    /** Model-level maximum sequence length reported by the Engine. */
    private long maxSeqLen;

    /**
     * FIFO scheduler's strict aggregate context-token limit for one admitted
     * batch/group. A group whose total context length is greater than or equal
     * to this value cannot be admitted by the Engine.
     */
    private long maxBatchTokensSize;
}
