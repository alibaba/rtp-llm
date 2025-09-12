package org.flexlb.domain.batch;

import lombok.Getter;
import lombok.Setter;

/**
 * @author zjw
 * description:
 * date: 2025/3/11
 */
@Getter
@Setter
public class RequestBatchMeta {
    private int reqNum;
    private long startTime;
    private long estimatedCostTime;

    public RequestBatchMeta(int reqNum, long startTime, long estimatedCostTime) {
        this.reqNum = reqNum;
        this.startTime = startTime;
        this.estimatedCostTime = estimatedCostTime;
    }

}
