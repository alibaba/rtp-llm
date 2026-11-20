package org.flexlb.balance.schedule;

import org.flexlb.config.ConfigService;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Independent implementations exercise the public extension points and executor authority.
 */
class CustomDecodePreemptionContractTest {
    @Test
    void controlledWaitKeepsItsWorkUntilTheResultIsReady() {
        var service = RequestPreemptionTestSupport.controlled();
        var request = mock(RequestContext.class);
        var selection = new RequestContext.SelectionAttempt(request, 1L);
        var endpoint = SchedulerTestSupport.mockDecodeEndpoint();
        var result = new CompletableFuture<RequestPreemptionTestSupport.AcquiredResource>();
        var deliveries = new AtomicInteger();
        when(service.acquisition.acquire(request, endpoint)).thenReturn(result);
        assertTrue(service.start(selection, endpoint, ignored -> deliveries.incrementAndGet()));
        var work = service.pendingWork().getFirst();
        try {
            assertTimeoutPreemptively(Duration.ofSeconds(1), work.action()::run);
            assertEquals(0, deliveries.get());
            assertSame(work, service.pendingWork().getFirst());
        } finally {
            result.complete(null);
        }
        work.action().run();
        assertEquals(1, deliveries.get());
        assertTrue(service.isIdle());
    }

    @Test
    void customServiceReplacesDefaultWithoutConstructingExecutorDependencies() {
        var config = SchedulingTestConfig.newConfig();
        RequestPreemption service = RequestPreemptionTestSupport.mock();
        new ApplicationContextRunner()
                .withUserConfiguration(SchedulerConfiguration.class)
                .withBean(
                        "environmentConfigSource",
                        org.flexlb.config.loading.source.ConfigSource.class,
                        () -> mock(org.flexlb.config.loading.source.ConfigSource.class))
                .withBean(
                        "nacosConfigSource",
                        org.flexlb.config.loading.source.ConfigSource.class,
                        () -> mock(org.flexlb.config.loading.source.ConfigSource.class))
                .withBean(
                        "uniConfigConfigSource",
                        org.flexlb.config.loading.source.ConfigSource.class,
                        () -> mock(org.flexlb.config.loading.source.ConfigSource.class))
                .withBean(ConfigService.class, () -> SchedulingTestConfig.configService(config))
                .withBean(RequestScheduler.class, () -> mock(RequestScheduler.class))
                .withBean(RequestPreemption.class, () -> service)
                .run(
                        context -> {
                            org.assertj.core.api.Assertions.assertThat(context)
                                    .hasSingleBean(RequestPreemption.class)
                                    .doesNotHaveBean(DefaultRequestPreemption.class);
                            assertSame(service, context.getBean(RequestPreemption.class));
                        });
    }
}
