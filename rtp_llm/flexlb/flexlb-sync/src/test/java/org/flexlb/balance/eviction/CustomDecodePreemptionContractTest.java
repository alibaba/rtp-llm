package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeResources.CapacityRelease;
import org.flexlb.balance.endpoint.DecodeResources.CapacityUsage;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRoutingView;
import org.flexlb.balance.endpoint.DecodeResources.ReservationHandle;
import org.flexlb.balance.endpoint.DecodeResources.ResourceSnapshot;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodePreemptionPolicy.DecodeEvictionProposal;
import org.flexlb.balance.scheduler.DeliveryDispatcher;
import org.flexlb.balance.scheduler.PlacementConfiguration;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestLifecycle;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.balance.scheduler.ResponseCompletionExecutor.RequestFuture;
import org.flexlb.balance.scheduler.SchedulerTestSupport;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/** Independent implementations exercise the public extension points and executor authority. */
class CustomDecodePreemptionContractTest {
    @Test
    void customServiceAndPolicyReplaceDefaultsWithoutConstructingExecutorDependencies() {
        var config = SchedulingTestConfig.newConfig();
        DecodePreemption service =
                new DecodePreemption() {
                    @Override
                    public CompletableFuture<PreemptionResult> tryReclaim(
                            RequestContext context, WorkerEndpoint endpoint) {
                        return null;
                    }

                    @Override
                    public void stopAdmission() {}

                    @Override
                    public boolean awaitIdle(long deadlineNanos) {
                        return true;
                    }

                    @Override
                    public void close() {
                        stopAdmission();
                    }
                };
        DecodePreemptionPolicy policy =
                (request, priority, capacity, snapshot, settings, reasons) ->
                        new DecodePreemptionPolicy.DecodeEvictionDecision(
                                capacity.evaluate(
                                        snapshot.routing().placementUsage(),
                                        request.getHardKvTokens(),
                                        request.getExpectedKvTokens(),
                                        CapacityRelease.NONE),
                                null);
        new ApplicationContextRunner()
                .withUserConfiguration(PlacementConfiguration.class)
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

    @ParameterizedTest
    @ValueSource(
            strings = {
                "selected",
                "declined",
                "endpoint",
                "stale",
                "priority",
                "duplicate",
                "stage",
                "unknown"
            })
    void customVictimChoiceIsPreservedButCannotExceedItsAuthority(String mode) {
        var config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        var settings = new PreemptionConfig();
        settings.setAllowedVictimStages(
                Set.of(
                        mode.equals("stage")
                                ? VictimStage.DECODE_RESERVED
                                : VictimStage.DECODE_ENGINE_OWNED));
        config.priorityOrdering().setPreemption(settings);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(2L);
        var first =
                new DecodeRequestView(
                        901L,
                        10,
                        128L,
                        128L,
                        DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        true,
                        11L,
                        false);
        var chosen =
                new DecodeRequestView(
                        902L,
                        mode.equals("priority") ? 90 : mode.equals("unknown") ? 0 : 30,
                        128L,
                        128L,
                        DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        true,
                        12L,
                        false);
        var endpoint = SchedulerTestSupport.mockDecodeEndpoint();
        var routing = mock(DecodeRoutingView.class);
        var snapshot = mock(ResourceSnapshot.class);
        when(endpoint.resourceSnapshot()).thenReturn(snapshot);
        when(endpoint.ipPort()).thenReturn("worker:8080");
        when(snapshot.routing()).thenReturn(routing);
        when(routing.address()).thenReturn("worker:8080");
        when(routing.placementUsage())
                .thenReturn(new CapacityUsage(2L, 20_000L, 10_000L, 0L, 10_000L));
        when(snapshot.requests())
                .thenReturn(Map.of(first.requestId(), first, chosen.requestId(), chosen));
        var owner = mock(RequestLifecycle.class);
        var scheduler = mock(RequestScheduler.class);
        org.springframework.test.util.ReflectionTestUtils.setField(owner, "owner", scheduler);
        var dispatcher = mock(DeliveryDispatcher.class);
        var context = new RequestContext(config);
        context.data()
                .setRequest(Request.builder().requestId(903L).seqLen(128L).rawPriority(70).build());
        context.data()
                .setSchedulingMetadata(
                        SchedulingMetadata.explicit(70, System.currentTimeMillis() + 60_000L));
        var future = new RequestFuture((kind, response, failure) -> false);
        ReflectionTestUtils.invokeMethod(context.data(), "register", owner, future);
        SchedulerTestSupport.bindOwner(context, owner);
        when(owner.isAdmissionOpen(903L, future)).thenReturn(true);
        DecodePreemptionPolicy policy =
                (request, priority, capacity, view, allowed, reasons) -> {
                    assertSame(snapshot, view);
                    var victim =
                            mode.equals("stale")
                                    ? new DecodeRequestView(
                                            chosen.requestId(),
                                            chosen.priority(),
                                            128L,
                                            128L,
                                            chosen.phase(),
                                            true,
                                            999L,
                                            false)
                                    : chosen;
                    var proposal =
                            new DecodeEvictionProposal(
                                    mode.equals("endpoint")
                                            ? "another:8080"
                                            : view.routing().address(),
                                    mode.equals("duplicate")
                                            ? List.of(victim, victim)
                                            : List.of(victim),
                                    DecodeEvictionProposal.CASE_SLOT,
                                    128L,
                                    0L);
                    return new DecodePreemptionPolicy.DecodeEvictionDecision(
                            capacity.evaluate(
                                    view.routing().placementUsage(),
                                    request.getHardKvTokens(),
                                    request.getExpectedKvTokens(),
                                    CapacityRelease.NONE),
                            mode.equals("declined") ? null : proposal);
                };
        var executor =
                spy(
                        new DefaultDecodePreemption(
                                dispatcher,
                                SchedulerTestSupport.repository(owner),
                                mock(org.flexlb.balance.scheduler.RequestTaskExecutor.class),
                                mock(org.flexlb.balance.scheduler.TaskTimer.class),
                                mock(RequestSchedulerReporter.class),
                                SchedulingTestConfig.configService(config),
                                SchedulingTestConfig.capacity(config),
                                policy));
        var exact = new ReservationHandle(7L, 903L, 31L);
        doAnswer(
                        invocation -> {
                            DefaultDecodePreemption.PreemptionCommand command =
                                    invocation.getArgument(1);
                            assertEquals(List.of(chosen), command.victims());
                            return CompletableFuture.completedFuture(
                                    new DecodePreemption.PreemptionResult(
                                            exact, false, "committed"));
                        })
                .when(executor)
                .preempt(any(), any());
        if (mode.equals("selected")) {
            assertSame(exact, executor.tryReclaim(context, endpoint).join().reservation());
            executor.stopAdmission();
            assertNull(executor.tryReclaim(context, endpoint));
            verify(executor, times(1)).preempt(any(), any());
        } else {
            if (mode.equals("declined")) assertNull(executor.tryReclaim(context, endpoint));
            else
                assertThrows(
                        IllegalArgumentException.class,
                        () -> executor.tryReclaim(context, endpoint));
            verify(executor, never()).preempt(any(), any());
            verifyNoInteractions(dispatcher);
            verifyNoInteractions(scheduler);
        }
    }
}
