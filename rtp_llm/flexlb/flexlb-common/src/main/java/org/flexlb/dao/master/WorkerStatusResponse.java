package org.flexlb.dao.master;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.flexlb.dao.route.RoleType;

import java.util.Map;

/**
 * @author zjw
 * description:
 * date: 2025/3/10
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class WorkerStatusResponse {

    @JsonProperty("role")
    private RoleType role;

    /**
     * Compatibility-only field. RTP-LLM's gRPC GetWorkerStatus currently does
     * not populate available_concurrency, so the observed value is the protobuf
     * default 0. Do not use it for routing, admission control, or batch sizing.
     */
    @JsonProperty("available_concurrency")
    private long availableConcurrency;

    @JsonProperty("running_query_len")
    private long runningQueryLen;

    @JsonProperty("waiting_query_len")
    private long waitingQueryLen;

    @JsonProperty("running_task_info")
    private Map<String, TaskInfo> runningTaskInfo;

    @JsonProperty("finished_task_info")
    private Map<String, TaskInfo> finishedTaskInfo;

    @JsonProperty("status_version")
    private Long statusVersion = 0L;

    @JsonProperty("latest_finished_version")
    private Long latestFinishedVersion = 0L;

    @JsonProperty("cache_status")
    private CacheStatus cacheStatus;

    @JsonProperty("step_latency_ms")
    private double stepLatencyMs;

    @JsonProperty("iterate_count")
    private long iterateCount;

    @JsonProperty("dpSize")
    private long dpSize;

    @JsonProperty("tpSize")
    private long tpSize;

    @JsonProperty("dpRank")
    private long dpRank;

    @JsonProperty("alive")
    private boolean alive;

    @JsonProperty("available_kv_cache")
    private long availableKvCacheTokens;

    @JsonProperty("total_kv_cache")
    private long totalKvCacheTokens;

    /** Model-level maximum sequence length reported by the Engine. */
    @JsonProperty("max_seq_len")
    private long maxSeqLen;

    /**
     * FIFO scheduler's strict aggregate context-token limit for one admitted
     * batch/group. A group whose total context length is greater than or equal
     * to this value cannot be admitted by the Engine.
     */
    @JsonProperty("max_batch_tokens_size")
    private long maxBatchTokensSize;

    @JsonProperty("version")
    private long version;

    @JsonProperty("message")
    private String message;

}
