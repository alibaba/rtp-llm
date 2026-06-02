package org.flexlb.service.grace;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.service.grace.strategy.LbConsistencyHooker;
import org.flexlb.service.grace.strategy.QueryWarmerHooker;
import org.springframework.context.EnvironmentAware;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

import java.util.Arrays;

/**
 * Graceful online orchestrator.
 * <p>
 * Executes online hooks in a fixed order:
 * 1. LB consistency — register with service discovery (ZK)
 * 2. Query warmer — warm up service before accepting traffic
 */
@Slf4j
@Component
public class GracefulOnlineService implements EnvironmentAware {

    private final LbConsistencyHooker lbConsistencyHooker;
    private final QueryWarmerHooker queryWarmerHooker;
    private Environment environment;

    public GracefulOnlineService(LbConsistencyHooker lbConsistencyHooker,
                                 QueryWarmerHooker queryWarmerHooker) {
        this.lbConsistencyHooker = lbConsistencyHooker;
        this.queryWarmerHooker = queryWarmerHooker;
    }

    @Override
    public void setEnvironment(Environment environment) {
        this.environment = environment;
    }

    public void online() {
        boolean isTestEnv = Arrays.stream(environment.getActiveProfiles())
                .anyMatch(e -> "test".equals(e));
        if (isTestEnv) {
            log.info("test env, skip online service");
            return;
        }

        log.info("Graceful online: step 1 — register with service discovery");
        lbConsistencyHooker.afterStartUp();

        log.info("Graceful online: step 2 — warm up service");
        queryWarmerHooker.afterStartUp();
    }
}
