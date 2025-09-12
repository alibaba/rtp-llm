package org.flexlb.domain.worker;

import lombok.Getter;
import org.flexlb.dao.master.WorkerStatus;

/**
 * @author zjw
 * description:
 * date: 2025/3/12
 */
@Getter
public class WorkerTTFT {

    private final WorkerStatus workerStatus;
    private final long ttft;
    private final long prefillTime;
    private final long hitCacheTokens;

    public WorkerTTFT(WorkerStatus workerStatus, long ttft, long prefillTime, long hitCacheTokens) {
        this.workerStatus = workerStatus;
        this.ttft = ttft;
        this.prefillTime = prefillTime;
        this.hitCacheTokens = hitCacheTokens;
    }
}
