package org.flexlb.dao.master;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.flexlb.enums.TaskPhase;
import org.flexlb.enums.PriorityPreemptionProgress;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class TaskInfo {

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
    @JsonProperty("execution_time_ms")
    private long executionTimeMs = -1;

    @JsonProperty("priority_preemption_progress")
    private PriorityPreemptionProgress priorityPreemptionProgress =
            PriorityPreemptionProgress.NONE;

    /**
     * Whether this record is only the control-plane cancel overlay and does
     * not prove that the request is actively executing on the worker.
     */
    public boolean priorityCancelOverlayOnly() {
        return phase == TaskPhase.PENDING
                && (priorityPreemptionProgress == PriorityPreemptionProgress.CANCELING
                    || priorityPreemptionProgress == PriorityPreemptionProgress.CANCELED);
    }

}
