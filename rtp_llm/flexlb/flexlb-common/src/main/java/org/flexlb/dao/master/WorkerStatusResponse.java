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

    @JsonProperty("version")
    private long version;

    @JsonProperty("message")
    private String message;

}
