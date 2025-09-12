package org.flexlb.domain.monitor;

import lombok.Getter;
import lombok.Setter;

/**
 * @author zjw
 * description:
 * date: 2025/3/14
 */
@Setter
@Getter
public class BatchMonitorContext {

    private long startTime;

    private boolean success = true;
    private String batchFailType;
    private long totalCost;

    private long tokenizeStartTime;
    private long tokenizeEndTime;

    private long batchStartTime;
    private long batchEndTime;

    private int reqInBatch;

    public void markBatchSuccess(int reqInBatch) {
        this.success = true;
        this.reqInBatch = reqInBatch;
    }

    public void markBatchFail(String batchFailType) {
        this.success = false;
        this.batchFailType = batchFailType;
    }

}
