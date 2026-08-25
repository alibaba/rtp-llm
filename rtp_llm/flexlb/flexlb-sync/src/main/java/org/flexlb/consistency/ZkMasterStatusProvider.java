package org.flexlb.consistency;

import org.flexlb.metric.MasterStatusProvider;
import org.springframework.context.annotation.Lazy;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Component;

/**
 * Provides master status based on Zookeeper master election.
 * Used by KMonitorAdapter to tag all metrics with isMaster=true/false.
 *
 * <p>When flexlb-sync is on the classpath, this bean takes precedence over
 * {@link org.flexlb.metric.NoOpMasterStatusProvider} (which is guarded by
 * {@code @ConditionalOnMissingBean}).</p>
 *
 * <p>{@link LBStatusConsistencyService#isMaster()} already checks
 * {@code isNeedConsistency()} internally and returns {@code false} for roles
 * that don't participate in ZK election (e.g. frontend).</p>
 *
 * @author saichen.sm
 */
@Component
@Primary
public class ZkMasterStatusProvider implements MasterStatusProvider {

    private final LBStatusConsistencyService lbStatusConsistencyService;

    public ZkMasterStatusProvider(@Lazy LBStatusConsistencyService lbStatusConsistencyService) {
        this.lbStatusConsistencyService = lbStatusConsistencyService;
    }

    @Override
    public boolean isMaster() {
        return lbStatusConsistencyService.isMaster();
    }
}
