package org.flexlb.it.scenario.queue;

import org.flexlb.it.configuration.AbstractIntegrationTestContextInitializer;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.springframework.context.ApplicationContextInitializer;
import org.springframework.context.ConfigurableApplicationContext;

/** Prepares the queue-enabled, one-scheduler-worker short-bucket test context. */
public final class QueueingShortestTtftContextInitializer
        extends AbstractIntegrationTestContextInitializer
        implements ApplicationContextInitializer<ConfigurableApplicationContext> {

    @Override
    public void initialize(ConfigurableApplicationContext applicationContext) {
        initializeQueueingShortestTtftContext(IntegrationTestFixtures.PDFUSION_TWO_WORKERS);
    }
}
