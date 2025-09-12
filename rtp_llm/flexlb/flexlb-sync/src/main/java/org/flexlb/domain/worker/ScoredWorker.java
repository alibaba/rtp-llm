package org.flexlb.domain.worker;

import lombok.Data;
import org.flexlb.dao.master.WorkerStatus;

@Data
public class ScoredWorker {
    private final WorkerStatus worker;
    private final WorkerTTFT workerTTFT;

    public ScoredWorker(WorkerStatus worker, WorkerTTFT workerTTFT) {
        this.worker = worker;
        this.workerTTFT = workerTTFT;
    }
    public WorkerStatus worker() {
        return worker;
    }

    public long ttft() {
        return workerTTFT.getTtft();
    }
}
