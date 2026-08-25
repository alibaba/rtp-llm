package org.flexlb.it.scenario.stability;

import org.flexlb.dao.route.RoleType;
import org.flexlb.it.configuration.AbstractIntegrationTestContextInitializer;
import org.flexlb.it.fixture.engine.WorkerTopology;
import org.springframework.context.ApplicationContextInitializer;
import org.springframework.context.ConfigurableApplicationContext;

/** Prepares the 200-worker SHORTEST_TTFT context used only by the stress integration profile. */
public final class WorkerScaleContextInitializer
        extends AbstractIntegrationTestContextInitializer
        implements ApplicationContextInitializer<ConfigurableApplicationContext> {

    private static final WorkerTopology TWO_HUNDRED_PDFUSION_WORKERS =
            WorkerTopology.of(RoleType.PDFUSION, 200);

    @Override
    public void initialize(ConfigurableApplicationContext applicationContext) {
        initializeShortestTtftContext(TWO_HUNDRED_PDFUSION_WORKERS, 8);
    }
}
