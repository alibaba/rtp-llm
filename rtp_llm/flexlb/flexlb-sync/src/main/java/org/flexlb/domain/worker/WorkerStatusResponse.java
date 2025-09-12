package org.flexlb.domain.worker;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.ProfileMeta;
import org.flexlb.dao.master.TaskInfo;

/**
 * @author zjw
 * description:
 * date: 2025/3/10
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class WorkerStatusResponse {

    @JsonProperty("role")
    private String role;

    @JsonProperty("available_concurrency")
    private long availableConcurrency;

    @JsonProperty("running_query_len")
    private long runningQueryLen;

    @JsonProperty("waiting_query_len")
    private long waitingQueryLen;

    @JsonProperty("running_task_info")
    private List<TaskInfo> runningTaskInfo;

    @JsonProperty("finished_task_list")
    private List<TaskInfo> finishedTaskList;

    @JsonProperty("status_version")
    private Long statusVersion = 0L;

    @JsonProperty("cache_status")
    private CacheStatus cacheStatus;

    @JsonProperty("profile_meta")
    private ProfileMeta profileMeta;

    @JsonProperty("step_latency_ms")
    private double stepLatencyMs;

    @JsonProperty("iterate_count")
    private long iterateCount;

    @JsonProperty("dpSize")
    private long dpSize;

    @JsonProperty("tpSize")
    private long tpSize;

    @JsonProperty("alive")
    private boolean alive;

    @JsonProperty("version")
    private long version;

    @JsonProperty("message")
    private String message;

}