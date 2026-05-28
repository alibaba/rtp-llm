package org.flexlb.dao.master;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.flexlb.enums.TaskStateEnum;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class TaskInfo {

    private static volatile double coeff0 = 190.0;
    private static volatile double coeff1 = 0.0076;
    private static volatile double coeff2 = 0.000000009;
    private static volatile boolean profiled = false;

    public static void updateCoefficients(double c0, double c1, double c2) {
        coeff0 = c0;
        coeff1 = c1;
        coeff2 = c2;
        profiled = true;
    }

    public static boolean isProfiled() {
        return profiled;
    }
    @JsonProperty("request_id")
    private long requestId;
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

    // Task state related fields
    private TaskStateEnum taskState = TaskStateEnum.CREATED;
    private long lastActiveTimeUs = System.nanoTime() / 1000;

    public long estimatePrefillTime() {
        return estimatePrefillTimeMs(inputLength, prefixLength);
    }

    public static long estimatePrefillTimeMs(long tokens, long hitCacheTokens) {
        long compute = tokens - hitCacheTokens;
        return (long) (coeff0 + coeff1 * compute + coeff2 * compute * compute);
    }

    /**
     * Update task state
     */
    public void updateTaskState(TaskStateEnum newState) {
        if (this.taskState != newState) {
            this.taskState = newState;
            this.lastActiveTimeUs = System.nanoTime() / 1000;
        }
    }

    /**
     * Check if task is lost
     */
    public boolean isLost() {
        return taskState == TaskStateEnum.LOST;
    }

    /**
     * Check if task is timed out
     */
    public boolean isTimeout(long currentTimeUs, long timeoutUs) {
        return (currentTimeUs - lastActiveTimeUs) > timeoutUs;
    }
}
