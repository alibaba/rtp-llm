package org.flexlb.dao.master;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.flexlb.enums.TaskStateEnum;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class TaskInfo {

    @JsonProperty("request_id")
    private long requestId;
    @JsonProperty("prefix_length")
    private long prefixLength;    // cache hit len
    @JsonProperty("prefix_length_valid")
    private boolean prefixLengthValid;
    @JsonProperty("predicted_prefix_length")
    private long predictedPrefixLength;
    @JsonProperty("cache_match_source")
    private String cacheMatchSource;
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
    @JsonProperty("error_code")
    private long errorCode;
    @JsonProperty("error_message")
    private String errorMessage;
    @JsonProperty("batch_id")
    private long batchId = -1;
    @JsonProperty("phase")
    private TaskPhase phase;
    @JsonProperty("execution_time_ms")
    private long executionTimeMs = -1;

    @JsonProperty("priority_preemption_progress")
    private PriorityPreemptionProgress priorityPreemptionProgress =
            PriorityPreemptionProgress.NONE;

    @JsonProperty("waiting_entered_time_ms")
    private long waitingEnteredTimeMs;
    @JsonProperty("running_entered_time_ms")
    private long runningEnteredTimeMs;
    @JsonProperty("request_received_time_ms")
    private long requestReceivedTimeMs;
    @JsonProperty("input_queue_enqueue_time_ms")
    private long inputQueueEnqueueTimeMs;
    @JsonProperty("input_queue_drain_time_ms")
    private long inputQueueDrainTimeMs;
    @JsonProperty("remote_kv_wait_ms")
    private long remoteKvWaitMs;
    @JsonProperty("first_token_time_ms")
    private long firstTokenTimeMs;
    @JsonProperty("hbm_local_match_tokens")
    private long hbmLocalMatchTokens;
    @JsonProperty("remote_kv_added_match_tokens")
    private long remoteKvAddedMatchTokens;
    @JsonProperty("first_prefill_step_id")
    private long firstPrefillStepId;
    @JsonProperty("last_prefill_step_id")
    private long lastPrefillStepId;
    @JsonProperty("prefill_step_count")
    private long prefillStepCount;
    @JsonProperty("prefill_nonfinal_chunk_tokens_min")
    private long prefillNonfinalChunkTokensMin;
    @JsonProperty("prefill_nonfinal_chunk_tokens_max")
    private long prefillNonfinalChunkTokensMax;
    @JsonProperty("completed_prefill_tokens")
    private long completedPrefillTokens;
    @JsonProperty("remaining_prefill_tokens")
    // -1 means the engine omitted this optional field; 0 means no work remains.
    private long remainingPrefillTokens = -1;
    @JsonProperty("last_completed_prefill_step_id")
    private long lastCompletedPrefillStepId;

    @JsonIgnore
    private boolean kvcmMatchAvailable;
    @JsonIgnore
    private long kvcmLocalMatchTokens;
    @JsonIgnore
    private long kvcmP2pFetchTokens;
    @JsonIgnore
    private long kvcmP2pTotalMatchTokens;

    // Task state related fields
    private TaskStateEnum taskState = TaskStateEnum.CREATED;
    private long lastActiveTimeUs = System.nanoTime() / 1000;
    private long waitingConfirmTimeUs = -1;

    public long estimatePrefillTime() {
        return estimatePrefillTimeMs(inputLength, prefixLength);
    }

    public static long estimatePrefillTimeMs(long tokens, long hitCacheTokens) {
        return tokens - hitCacheTokens;
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
