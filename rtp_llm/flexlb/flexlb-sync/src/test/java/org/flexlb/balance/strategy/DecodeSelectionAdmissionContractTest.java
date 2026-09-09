package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.ScheduledRequest.DecodeBinding;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Selection and authoritative admission must interpret the same ownership scope. */
class DecodeSelectionAdmissionContractTest {
    private static final String QUEUED_IP = "127.0.0.1";
    private static final String FREE_IP = "127.0.0.2";
    private static final long PROMPT_TOKENS = 100L;
    private static final long EXPECTED_TOKENS = 300L;

    @ParameterizedTest
    @MethodSource("preemptionCases")
    void preemptiveQueueSelectsAnEndpointWhosePlacementCanActuallySucceed(Policy policy, CapacityDimension dimension) {
        try (Fixture fixture = new Fixture(policy, dimension)) {
            fixture.assertPlacementAndDispatchDisagree();
            var queuedUsage = fixture.queued.routingView().placementUsage();
            var freeUsage = fixture.free.routingView().placementUsage();

            for (long requestId = 100L; requestId < 106L; requestId++) {
                DecodeBinding request = fixture.request(requestId);
                PlacementResult<SelectedRole, RoleType> result = fixture.strategy.select(request, null);
                assertEquals(PlacementResult.Status.SUCCESS, result.status());
                try (SelectedRole selected = result.value()) {
                    assertEquals(FREE_IP, selected.serverStatus().getServerIp(),
                            "free placement capacity must win over a queued endpoint with only dispatch capacity");
                    fixture.assertSelectionHasNoReservation(requestId, queuedUsage, freeUsage);
                    try (WorkerEndpoint.GenerationPin pin = selected.takeGenerationPin()) {
                        DecodeEndpoint endpoint = (DecodeEndpoint) pin.endpoint();
                        DecodeEndpoint.ReservationHandle reservation = endpoint.tryReservePlacementPinned(pin, request.requestId(), request.hardKvTokens(),
                                request.expectedKvTokens(), request.priority(), request.capacity());
                        assertNotNull(reservation, "the selected endpoint must pass the same placement gate");
                        try {
                            assertEquals(requestId, reservation.requestId());
                            assertTrue(endpoint.layeredAdmissionView().isQueued(requestId));
                            var reserved = endpoint.layeredAdmissionView().reserved().get(requestId);
                            assertEquals(70, reserved.priority());
                            assertEquals(PROMPT_TOKENS, reserved.kvTokens());
                            assertEquals(EXPECTED_TOKENS, reserved.expectedKvTokens());
                        } finally {
                            endpoint.releaseReservationExact(reservation);
                        }
                    }
                }
                assertEquals(queuedUsage, fixture.queued.routingView().placementUsage());
                assertEquals(freeUsage, fixture.free.routingView().placementUsage());
            }
        }
    }

    @ParameterizedTest(name = "{0}: {1}")
    @MethodSource("dispatchAdmissionCases")
    void dispatchAdmissionKeepsQueuedEndpointsEligibleWithoutTakingCapacity(
            Policy policy, CapacityDimension dimension) {
        try (Fixture fixture = new Fixture(policy, dimension)) {
            fixture.assertPlacementAndDispatchDisagree();
            var queuedUsage = fixture.queued.routingView().placementUsage();
            var freeUsage = fixture.free.routingView().placementUsage();
            Map<String, Integer> selectedCounts = new HashMap<>();

            for (long requestId = 200L; requestId < 206L; requestId++) {
                PlacementResult<SelectedRole, RoleType> result = fixture.strategy.select(fixture.request(requestId), null);
                assertEquals(PlacementResult.Status.SUCCESS, result.status());
                try (SelectedRole selected = result.value()) {
                    selectedCounts.merge(selected.serverStatus().getServerIp(), 1, Integer::sum);
                    fixture.assertSelectionHasNoReservation(requestId, queuedUsage, freeUsage);
                }
            }

            assertEquals(Map.of(QUEUED_IP, 3, FREE_IP, 3), selectedCounts,
                    "DIRECT and queues without preemption must evaluate dispatch ownership, excluding queued work");
        }
    }

    private static Stream<Arguments> dispatchAdmissionCases() {
        return Stream.of(Policy.DIRECT, Policy.FIFO, Policy.PRIORITY_ONLY)
                .flatMap(policy -> Stream.of(CapacityDimension.values())
                        .map(dimension -> Arguments.of(policy, dimension)));
    }

    private enum CapacityDimension { EXPECTED_KV, REQUESTS }
    private static Stream<Arguments> preemptionCases() {
        return Stream.of(Policy.PRIORITY_DEFAULT, Policy.PREEMPTIVE)
                .flatMap(policy -> Stream.of(CapacityDimension.values()).map(dimension -> Arguments.of(policy, dimension)));
    }

    private enum Policy { DIRECT, FIFO, PRIORITY_ONLY, PRIORITY_DEFAULT, PREEMPTIVE }

    private static final class Fixture implements AutoCloseable {
        private final FlexlbConfig config = StrategyTestSupport.config();
        private final EndpointRegistry endpoints;
        private final DecodeSelector strategy;
        private final DecodeEndpoint queued;
        private final DecodeEndpoint free;
        private final DecodeEndpoint.AdmissionCapacity limits;

        private Fixture(Policy policy, CapacityDimension dimension) {
            config.getRouter().getRoles().getDecode().getCostEstimator().setExpression("0");
            if (policy == Policy.DIRECT) {
                config.setScheduler(SchedulerConfig.direct());
            } else if (policy != Policy.FIFO) {
                QueueOrderingConfig ordering = QueueOrderingConfig.priority();
                if (policy == Policy.PRIORITY_ONLY) { ordering.setPreemption(null); }
                if (policy == Policy.PREEMPTIVE) {
                    PreemptionConfig preemption = new PreemptionConfig();
                    preemption.setAllowedVictimStages(EnumSet.of(VictimStage.DECODE_RESERVED));
                    ordering.setPreemption(preemption);
                }
                config.queueScheduler().setOrdering(ordering);
            }
            long maxRequests = dimension == CapacityDimension.REQUESTS ? 2L : 0L;
            var availability = config.getRouter().getRoles().getDecode().getAvailability();
            availability.setMaxEngineRequests(maxRequests == 0L ? null : maxRequests);
            availability.setMaxKvUsagePercent(90L);
            limits = new DecodeEndpoint.AdmissionCapacity(maxRequests, 90L);
            ConfigService configs = mock(ConfigService.class);
            when(configs.loadBalanceConfig()).thenReturn(config);
            endpoints = StrategyTestSupport.endpointRegistry(configs);
            queued = publish(QUEUED_IP);
            free = publish(FREE_IP);
            strategy = new DecodeSelector(new WorkerDirectory(endpoints));

            long expectedKv = dimension == CapacityDimension.EXPECTED_KV ? 400L : 0L;
            try (WorkerEndpoint.GenerationPin pin = queued.tryPinGeneration()) {
                assertNotNull(pin);
                assertNotNull(queued.tryReservePlacementPinned(pin, 1L, 0L, expectedKv, 50));
                assertNotNull(queued.tryReservePlacementPinned(pin, 2L, 0L, expectedKv, 50));
            }
        }

        private DecodeEndpoint publish(String ip) {
            return (DecodeEndpoint) StrategyTestSupport.publishEndpoint(endpoints, RoleType.DECODE,
                    ip + ":8080", StrategyTestSupport.workerStatus(
                            RoleType.DECODE, null, ip, 8080, 9090, true, 1_000L, 1_000L));
        }

        private DecodeBinding request(long requestId) {
            Request request = new Request();
            request.setRequestId(requestId);
            request.setSeqLen(100L);
            request.setMaxNewTokens(200);
            request.setPriority(70);
            BalanceContext context = new BalanceContext();
            context.setRequest(request);
            context.setConfig(config);
            return DecodeBinding.capture(context);
        }

        private void assertPlacementAndDispatchDisagree() {
            assertTrue(limits.evaluate(queued.routingView().dispatchUsage(), PROMPT_TOKENS, EXPECTED_TOKENS).fits());
            assertFalse(limits.evaluate(queued.routingView().placementUsage(), PROMPT_TOKENS, EXPECTED_TOKENS).fits());
            assertTrue(limits.evaluate(free.routingView().dispatchUsage(), PROMPT_TOKENS, EXPECTED_TOKENS).fits());
            assertTrue(limits.evaluate(free.routingView().placementUsage(), PROMPT_TOKENS, EXPECTED_TOKENS).fits());
        }

        private void assertSelectionHasNoReservation(long requestId,
                DecodeEndpoint.CapacityUsage queuedUsage, DecodeEndpoint.CapacityUsage freeUsage) {
            assertEquals(queuedUsage, queued.routingView().placementUsage());
            assertEquals(freeUsage, free.routingView().placementUsage());
            assertFalse(queued.layeredAdmissionView().isQueued(requestId));
            assertFalse(free.layeredAdmissionView().isQueued(requestId));
        }

        @Override
        public void close() {
            endpoints.close();
        }
    }
}
