package org.flexlb.testfixture;

import lombok.Data;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;

@Data
public class TaskInfo {

    private long requestId;

    private long prefixLength;    // cache hit len

    private long prefillTime;

    private long inputLength;

    private long waitingTime;

    private long iterateCount;

    private long endTimeMs;

    private long dpRank;

    private long errorCode;

    private String errorMessage;

    private long batchId = -1;

    private TaskPhase phase;

    private long executionTimeMs = -1;

    private PriorityPreemptionProgress priorityPreemptionProgress =
            PriorityPreemptionProgress.NONE;
}
