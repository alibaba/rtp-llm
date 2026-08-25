package org.flexlb.it.scenario.worker;

import org.flexlb.dao.route.RoleType;
import org.flexlb.it.configuration.AbstractIntegrationTestContextInitializer;
import org.flexlb.it.fixture.engine.WorkerTopology;
import org.springframework.context.ApplicationContextInitializer;
import org.springframework.context.ConfigurableApplicationContext;

import java.util.Map;

/**
 * Prepares a mixed-role topology to verify role-specific worker discovery and status maps.
 */
public final class MultiRoleContextInitializer
        extends AbstractIntegrationTestContextInitializer
        implements ApplicationContextInitializer<ConfigurableApplicationContext> {

    private static final WorkerTopology TOPOLOGY = new WorkerTopology(Map.of(
            RoleType.PREFILL, 2,
            RoleType.DECODE, 3,
            RoleType.PDFUSION, 1));

    @Override
    public void initialize(ConfigurableApplicationContext applicationContext) {
        initializeShortestTtftContext(TOPOLOGY);
    }
}
