package org.flexlb.dao.master;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.flexlb.enums.TaskPhase;

import java.util.Map;

/**
 * Task information DTO transferred from engine gRPC status report.
 * <p>Fields mirror the engine's {@code TaskInfoPB} protobuf message.
 * Only the subset actually consumed by FlexLB is retained here;
 * unused legacy fields ({@code prefillTime}, {@code predictedMs},
 * {@code taskState}) were removed during ShortestTTFT → CostBased refactoring.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class TaskInfo {

    @JsonProperty("request_id")
    private long requestId;
    @JsonProperty("prefix_length")
    private long prefixLength;
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
}
