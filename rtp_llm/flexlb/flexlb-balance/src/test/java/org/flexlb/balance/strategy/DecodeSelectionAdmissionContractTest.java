package org.flexlb.balance.strategy;

import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.SchedulingTestConfig;
import org.flexlb.balance.strategy.StrategyTestSupport.Selection;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.scheduler.PreemptionConfig;
import org.flexlb.config.scheduler.PreemptionStage;
import org.flexlb.config.scheduler.QueueOrderingConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.DecodeResources.Capacity;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.EndpointTestSupport;
import org.flexlb.worker.endpoint.Request;
import org.flexlb.worker.endpoint.WorkerEndpoint;
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

/**
 * Selection and authoritative admission must interpret the same ownership scope.
 */
class DecodeSelectionAdmissionContractTest {
    private static final String QUEUED_IP = "127.0.0.1";
    private static final String FREE_IP = "127.0.0.2";

    private static final long PROMPT_TOKENS = 100L;
    private static final long EXPECTED_TOKENS = 300L;

    @ParameterizedTest
    @MethodSource("preemptionCases")
    void preemptiveQueueSelectsAnEndpointWhosePlacementCanActuallySucceed(
            Policy policy, CapacityDimension dimension) {
        try (Fixture fixture = new Fixture(policy, dimension)) {
            fixture.assertOnlyFreeEndpointHasCapacity();
            var queuedUsage = fixture.queued.selectionSnapshot().usage();
            var freeUsage = fixture.free.selectionSnapshot().usage();

            for (long requestId = 100L; requestId < 106L; requestId++) {
                Request request = fixture.request(requestId);
                RoutingResult<Selection, RoleType> result =
                        StrategyTestSupport.select(
                                fixture.strategy,
                                request,
                                org.flexlb.balance.schedule.SchedulingMetadata.explicit(
                                        70, Long.MAX_VALUE),
                                null);
                assertEquals(RoutingResult.Status.SUCCESS, result.status());
                try (Selection selected = result.value()) {
                    assertEquals(
                            FREE_IP,
                            selected.decode().getServerIp(),
                            "the selected endpoint must have capacity for the request");
                    WorkerEndpoint selectedEndpoint = StrategyTestSupport.decodeEndpoint(selected);
                    DecodeEndpoint endpoint = (DecodeEndpoint) selectedEndpoint;
                    var reservation = EndpointTestSupport.decodeRegistration(endpoint, requestId);
                    assertNotNull(
                            reservation, "the selected endpoint must pass the same placement gate");
                    try {
                        assertEquals(String.valueOf(requestId), reservation.requestId());
                        assertEquals(
                                org.flexlb.worker.status.DecodeTaskPhase.LOCAL_RESERVED,
                                endpoint.resourceSnapshot()
                                        .requests()
                                        .get(String.valueOf(requestId))
                                        .phase());
                        var reserved =
                                endpoint.resourceSnapshot()
                                        .requests()
                                        .get(String.valueOf(requestId));
                        assertEquals(70, reserved.priority());
                        assertEquals(PROMPT_TOKENS, reserved.kvTokens());
                        assertEquals(EXPECTED_TOKENS, reserved.expectedKvTokens());
                    } finally {
                        endpoint.releaseRequest(reservation);
                    }
                }
                assertEquals(queuedUsage, fixture.queued.selectionSnapshot().usage());
                assertEquals(freeUsage, fixture.free.selectionSnapshot().usage());
            }
        }
    }

    @ParameterizedTest(name = "{0}: {1}")
    @MethodSource("dispatchAdmissionCases")
    void selectionRegistersUnsentCapacityInEverySchedulerMode(
            Policy policy, CapacityDimension dimension) {
        try (Fixture fixture = new Fixture(policy, dimension)) {
            fixture.assertOnlyFreeEndpointHasCapacity();
            var queuedUsage = fixture.queued.selectionSnapshot().usage();
            var freeUsage = fixture.free.selectionSnapshot().usage();
            Map<String, Integer> selectedCounts = new HashMap<>();

            for (long requestId = 200L; requestId < 206L; requestId++) {
                RoutingResult<Selection, RoleType> result =
                        StrategyTestSupport.select(
                                fixture.strategy,
                                fixture.request(requestId),
                                org.flexlb.balance.schedule.SchedulingMetadata.explicit(
                                        70, Long.MAX_VALUE),
                                null);
                assertEquals(RoutingResult.Status.SUCCESS, result.status());
                try (Selection selected = result.value()) {
                    selectedCounts.merge(selected.decode().getServerIp(), 1, Integer::sum);
                    assertEquals(
                            org.flexlb.worker.status.DecodeTaskPhase.LOCAL_RESERVED,
                            fixture.free
                                    .resourceSnapshot()
                                    .requests()
                                    .get(String.valueOf(requestId))
                                    .phase());
                    assertEquals(
                            freeUsage.occupiedRequests() + 1,
                            fixture.free.selectionSnapshot().usage().occupiedRequests());
                }
            }

            assertEquals(queuedUsage, fixture.queued.selectionSnapshot().usage());
            assertEquals(freeUsage, fixture.free.selectionSnapshot().usage());
            assertEquals(
                    Map.of(FREE_IP, 6),
                    selectedCounts,
                    "every scheduler counts selections which have not been sent");
        }
    }

    private static Stream<Arguments> dispatchAdmissionCases() {
        return Stream.of(Policy.DIRECT, Policy.FIFO, Policy.PRIORITY_ONLY)
                .flatMap(
                        policy ->
                                Stream.of(CapacityDimension.values())
                                        .map(dimension -> Arguments.of(policy, dimension)));
    }

    private static Stream<Arguments> preemptionCases() {
        return Stream.of(Policy.PRIORITY_DEFAULT, Policy.PREEMPTIVE)
                .flatMap(
                        policy ->
                                Stream.of(CapacityDimension.values())
                                        .map(dimension -> Arguments.of(policy, dimension)));
    }

    private enum CapacityDimension {
        EXPECTED_KV,
        REQUESTS
    }

    private enum Policy {
        DIRECT,
        FIFO,
        PRIORITY_ONLY,
        PRIORITY_DEFAULT,
        PREEMPTIVE
    }

    /**
     * Builds a full and a free Decode endpoint to compare placement capacity with dispatch capacity under each scheduling policy.
     */
    private static final class Fixture implements AutoCloseable {
        private final FlexlbConfig config = StrategyTestSupport.config();
        private final EndpointRegistry endpoints;
        private final StrategyTestSupport.DecodeRoutingFixture strategy;
        private final DecodeEndpoint queued;
        private final DecodeEndpoint free;
        private final DecodeResources.AdmissionCapacity limits;

        private Fixture(Policy policy, CapacityDimension dimension) {
            config.getRouter().getRoles().getDecode().getCostEstimator().setExpression("0");
            if (policy == Policy.DIRECT) {
                config.setScheduler(ConfigTestSupport.direct());
            } else if (policy != Policy.FIFO) {
                QueueOrderingConfig ordering = ConfigTestSupport.priority();
                if (policy == Policy.PRIORITY_ONLY) {
                    ordering.setPreemption(null);
                }
                if (policy == Policy.PREEMPTIVE) {
                    PreemptionConfig preemption = new PreemptionConfig();
                    preemption.setAllowedPreemptionStages(
                            EnumSet.of(PreemptionStage.DECODE_ENGINE_OWNED));
                    ordering.setPreemption(preemption);
                }
                config.queueScheduler().setOrdering(ordering);
            }
            long maxRequests = dimension == CapacityDimension.REQUESTS ? 2L : 0L;
            var availability = config.getRouter().getRoles().getDecode().getAvailability();
            availability.setMaxEngineRequests(maxRequests == 0L ? null : maxRequests);
            availability.setMaxKvUsagePercent(90L);
            limits = new DecodeResources.AdmissionCapacity(maxRequests, 90L);
            ConfigService configs = mock(ConfigService.class);
            when(configs.loadBalanceConfig()).thenReturn(config);
            endpoints = StrategyTestSupport.endpointRegistry(configs);
            queued = publish(QUEUED_IP);
            free = publish(FREE_IP);
            strategy =
                    new StrategyTestSupport.DecodeRoutingFixture(
                            endpoints,
                            SchedulingTestConfig.configService(config),
                            SchedulingTestConfig.capacity(
                                    SchedulingTestConfig.configService(config)
                                            .loadBalanceConfig()));

            long expectedKv = dimension == CapacityDimension.EXPECTED_KV ? 400L : 0L;
            {
                WorkerEndpoint selectedEndpoint = queued;

                assertNotNull(selectedEndpoint);
                assertNotNull(
                        org.flexlb.balance.schedule.SelectionTestSupport.registerQueuedDecode(
                                        queued, String.valueOf(1L), 0L, expectedKv, 50)
                                .value());
                assertNotNull(
                        org.flexlb.balance.schedule.SelectionTestSupport.registerQueuedDecode(
                                        queued, String.valueOf(2L), 0L, expectedKv, 50)
                                .value());
            }
        }

        @Override
        public void close() {
            endpoints.close();
        }

        private DecodeEndpoint publish(String ip) {
            return (DecodeEndpoint)
                    StrategyTestSupport.publishEndpoint(
                            endpoints,
                            StrategyTestSupport.workerStatus(
                                    RoleType.DECODE, null, ip, 8080, 9090, true, 1_000L, 1_000L));
        }

        private Request request(long requestId) {
            Request request =
                    Request.builder()
                            .requestId(String.valueOf(requestId))
                            .seqLen(100L)
                            .maxNewTokens(200)
                            .rawPriority(70)
                            .build();
            RequestContext context = new RequestContext(config);
            context.setRequest(request);
            return request;
        }

        private void assertOnlyFreeEndpointHasCapacity() {
            assertFalse(
                    limits.evaluate(
                                    queued.selectionSnapshot().usage(),
                                    PROMPT_TOKENS,
                                    EXPECTED_TOKENS,
                                    Capacity.NONE)
                            .isZero());
            assertTrue(
                    limits.evaluate(
                                    free.selectionSnapshot().usage(),
                                    PROMPT_TOKENS,
                                    EXPECTED_TOKENS,
                                    Capacity.NONE)
                            .isZero());
            assertTrue(
                    limits.evaluate(
                                    free.selectionSnapshot().usage(),
                                    PROMPT_TOKENS,
                                    EXPECTED_TOKENS,
                                    Capacity.NONE)
                            .isZero());
        }
    }
}
