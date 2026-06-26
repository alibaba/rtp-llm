package org.flexlb.dao.master;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.TaskPhase;
import org.flexlb.enums.TaskStateEnum;

import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
@Slf4j
public class TaskInfo {
    static final String PREFILL_TIME_ESTIMATE_FORMULA_ENV = "PREFILL_TIME_ESTIMATE_FORMULA";
    static final String DEFAULT_PREFILL_TIME_ESTIMATE_FORMULA = "tokens * 1.0 - hitCacheTokens * 0.7";
    private static final PrefillTimeFormula PREFILL_TIME_ESTIMATE_FORMULA = readPrefillTimeFormula(System.getenv());

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
    @JsonProperty("error_code")
    private long errorCode;
    @JsonProperty("error_message")
    private String errorMessage;
    @JsonProperty("batch_id")
    private long batchId = -1;
    @JsonProperty("phase")
    private TaskPhase phase;

    private long predictedMs;

    // Task state related fields
    private TaskStateEnum taskState = TaskStateEnum.CREATED;
    private long lastActiveTimeUs = System.nanoTime() / 1000;

    public long estimatePrefillTime() {
        return estimatePrefillTimeMs(inputLength, prefixLength);
    }

    public static long estimatePrefillTimeMs(long tokens, long hitCacheTokens) {
        return PREFILL_TIME_ESTIMATE_FORMULA.estimate(tokens, hitCacheTokens);
    }

    static PrefillTimeFormula readPrefillTimeFormula(Map<String, String> environment) {
        String formula = environment.get(PREFILL_TIME_ESTIMATE_FORMULA_ENV);
        if (formula == null || formula.trim().isEmpty()) {
            formula = DEFAULT_PREFILL_TIME_ESTIMATE_FORMULA;
        }
        try {
            return PrefillTimeFormula.parse(formula);
        } catch (IllegalArgumentException e) {
            log.warn(
                    "Invalid {}={}: {}, use default formula {}",
                    PREFILL_TIME_ESTIMATE_FORMULA_ENV,
                    formula,
                    e.getMessage(),
                    DEFAULT_PREFILL_TIME_ESTIMATE_FORMULA);
            return PrefillTimeFormula.parse(DEFAULT_PREFILL_TIME_ESTIMATE_FORMULA);
        }
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
