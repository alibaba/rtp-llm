package org.flexlb.dao.master;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.flexlb.enums.TaskStateEnum;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class TaskInfo {
    public static final double DEFAULT_CACHE_HIT_DISCOUNT = 0.7;

    @JsonProperty("request_id")
    private String requestId;
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

    @JsonIgnore
    private double cacheHitDiscount = DEFAULT_CACHE_HIT_DISCOUNT;

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
        return estimatePrefillTimeMs(inputLength, prefixLength, cacheHitDiscount);
    }

    public static long estimatePrefillTimeMs(long tokens, long hitCacheTokens) {
        return estimatePrefillTimeMs(tokens, hitCacheTokens, DEFAULT_CACHE_HIT_DISCOUNT);
    }

    public static long estimatePrefillTimeMs(long tokens, long hitCacheTokens, double cacheHitDiscount) {
        return (long) (tokens - hitCacheTokens * cacheHitDiscount);
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
