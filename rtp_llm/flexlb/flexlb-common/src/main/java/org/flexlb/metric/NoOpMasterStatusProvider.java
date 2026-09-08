package org.flexlb.metric;

import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.stereotype.Component;

/**
 * Default implementation for roles that don't participate in master election (e.g., frontend).
 * Always returns false.
 *
 * @author saichen.sm
 */
@Component
@ConditionalOnMissingBean(MasterStatusProvider.class)
public class NoOpMasterStatusProvider implements MasterStatusProvider {

    @Override
    public boolean isMaster() {
        return false;
    }
}
