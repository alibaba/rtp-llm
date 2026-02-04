package org.flexlb.dao.master;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.flexlb.enums.TaskStateEnum;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class TaskInfo {
    @JsonProperty("inter_request_id")
    private String interRequestId;
    @JsonProperty("prefix_length")
    private long prefixLength;    // cache hit len
    @JsonProperty("prefill_time")
    private long prefillTime;
    @JsonProperty("input_length")
    private long inputLength;
    @JsonProperty("waiting_time")
    private long waitingTime;
    @JsonProperty("iterate_count")
    private long iterateCount;
    @JsonProperty("end_time_ms")
    private long endTimeMs;
    @JsonProperty("dp_rank")
    private long dpRank;

    // 任务状态相关字段
    private TaskStateEnum taskState = TaskStateEnum.CREATED;
    private long lastActiveTimeUs = System.nanoTime() / 1000;

    public long estimatePrefillTime() {
        return estimatePrefillTimeMs(inputLength, prefixLength);
    }

    public static long estimatePrefillTimeMs(long tokens, long hitCacheTokens) {
        return (long) (tokens * 1.0 - hitCacheTokens * 0.7);
    }
    
    /**
     * 更新任务状态
     */
    public void updateTaskState(TaskStateEnum newState) {
        if (this.taskState != newState) {
            this.taskState = newState;
            this.lastActiveTimeUs = System.nanoTime() / 1000;
        }
    }
    
    /**
     * 判断任务是否丢失
     */
    public boolean isLost() {
        return taskState == TaskStateEnum.LOST;
    }
    
    /**
     * 判断任务是否超时
     */
    public boolean isTimeout(long currentTimeUs, long timeoutUs) {
        return (currentTimeUs - lastActiveTimeUs) > timeoutUs;
    }
}
