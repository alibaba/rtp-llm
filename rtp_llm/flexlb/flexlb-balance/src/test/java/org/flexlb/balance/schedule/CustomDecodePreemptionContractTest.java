package org.flexlb.balance.schedule;

import org.flexlb.config.ConfigService;
import org.flexlb.worker.endpoint.DecodeResources.Capacity;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;

/**
 * Independent implementations exercise the public extension points and executor authority.
 */
class CustomDecodePreemptionContractTest {
    @Test
    void customServiceAndPolicyReplaceDefaultsWithoutConstructingExecutorDependencies() {
        var config = SchedulingTestConfig.newConfig();
        DecodePreemption service =
                new DecodePreemption() {
                    @Override
                    public CompletableFuture<RequestResult> tryPreempt(
                            RequestContext context, WorkerEndpoint endpoint) {
                        return null;
                    }

                    @Override
                    public boolean awaitIdle(long deadlineNanos) {
                        return true;
                    }

                    @Override
                    public void close() {
                        stopAdmission();
                    }

                    @Override
                    public void stopAdmission() {}
                };
        DecodePreemptionPolicy policy =
                (request, priority, capacity, snapshot, settings, reasons) ->
                        new DecodePreemptionPolicy.DecodePreemptionDecision(
                                capacity.evaluate(
                                        snapshot.selection().usage(),
                                        request.getHardKvTokens(),
                                        request.getExpectedKvTokens(),
                                        Capacity.NONE),
                                null);
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
                .withBean(DecodePreemption.class, () -> service)
                .withBean(DecodePreemptionPolicy.class, () -> policy)
                .run(
                        context -> {
                            org.assertj.core.api.Assertions.assertThat(context)
                                    .hasSingleBean(DecodePreemption.class)
                                    .hasSingleBean(DecodePreemptionPolicy.class)
                                    .doesNotHaveBean(DefaultDecodePreemption.class);
                            assertSame(service, context.getBean(DecodePreemption.class));
                            assertSame(policy, context.getBean(DecodePreemptionPolicy.class));
                        });
    }
}
