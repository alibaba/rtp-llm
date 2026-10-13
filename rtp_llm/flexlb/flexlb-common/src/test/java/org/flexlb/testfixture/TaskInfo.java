package org.flexlb.testfixture;

import lombok.Data;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;

@Data
public class TaskInfo {

    private long requestId;


    private long prefixLength;    // cache hit len
    private boolean prefixLengthValid;
    private long predictedPrefixLength;
    private String cacheMatchSource;
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

    private long waitingEnteredTimeMs;
    private long runningEnteredTimeMs;
    private long requestReceivedTimeMs;
    private long inputQueueEnqueueTimeMs;
    private long inputQueueDrainTimeMs;
    private long remoteKvWaitMs;
    private long firstTokenTimeMs;
    private long hbmLocalMatchTokens;
    private long remoteKvAddedMatchTokens;
    private long firstPrefillStepId;
    private long lastPrefillStepId;
    private long prefillStepCount;
    private long prefillNonfinalChunkTokensMin;
    private long prefillNonfinalChunkTokensMax;
    private long completedPrefillTokens;
    // -1 means the engine omitted this optional field; 0 means no work remains.
    private long remainingPrefillTokens = -1;
    private long lastCompletedPrefillStepId;

}
