package org.flexlb.dao.loadbalance;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class QueueSnapshot {

    private long sequenceId;
    private String requestId;
    private long enqueueTime;
    private long waitTimeMs;
    private int retryCount;
    private String queueType;
}