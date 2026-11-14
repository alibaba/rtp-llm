package org.flexlb.consistency;

import org.flexlb.metric.MasterStatusProvider;
import org.springframework.context.annotation.Lazy;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Component;

/**
 * Exposes ZooKeeper election status for metric tags when the synchronization module is present.
 * Roles that do not participate in election report non-master status.
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
