package org.flexlb.balance.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
@Component
public class ExpirationWorkerCleaner {

    private final EngineWorkerStatus engineWorkerStatus;

    public ExpirationWorkerCleaner(EngineWorkerStatus engineWorkerStatus) {
        this.engineWorkerStatus = engineWorkerStatus;
    }

    @Scheduled(fixedRate = 2000)
    public void cleanExpiredWorkers() {
        ModelWorkerStatus modelWorkerStatus = engineWorkerStatus.getModelRoleWorkerStatusMap().get("engine_service");
        if (modelWorkerStatus == null) {
            log.error("modelWorkerStatus is null, modelName: engine_service");
            return;
        }
        doClean(modelWorkerStatus.getPrefillStatusMap());
        doClean(modelWorkerStatus.getDecodeStatusMap());
        doClean(modelWorkerStatus.getPdFusionStatusMap());
        doClean(modelWorkerStatus.getVitStatusMap());
    }

    public static void doClean(ConcurrentHashMap<String, WorkerStatus> workerStatusMap) {
        if (workerStatusMap == null) {
            return;
        }
        long curTimeMillis = System.currentTimeMillis();
        for (Iterator<Map.Entry<String, WorkerStatus>> it = workerStatusMap.entrySet().iterator(); it.hasNext(); ) {
            Map.Entry<String, WorkerStatus> item = it.next();
            WorkerStatus workerStatus = item.getValue();
            long expirationTime = workerStatus.getExpirationTime().get();
            if (curTimeMillis > expirationTime) {
                it.remove();
            }
        }
    }
}