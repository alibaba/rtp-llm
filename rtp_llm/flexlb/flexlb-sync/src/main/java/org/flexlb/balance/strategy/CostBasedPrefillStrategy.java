package org.flexlb.balance.strategy;

import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;

/** Cost-based Prefill placement that decides each request independently. */
public final class CostBasedPrefillStrategy extends PrefillStrategy {

    public CostBasedPrefillStrategy(
            WorkerDirectory workerDirectory,
            CacheAwareService cacheAwareService,
            EngineHealthReporter engineHealthReporter) {
        super(workerDirectory, cacheAwareService, engineHealthReporter);
    }

}
