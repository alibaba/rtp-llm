package org.flexlb.balance.strategy;

import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.EncoderEndpoint;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.Request;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.WorkerStatus;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

/**
 * Checks Encoder load selection across health, cache estimates and scheduling modes.
 */
class EncoderStrategyTest {
    private final EncoderStrategy strategy =
            new EncoderStrategy(
                    configuration(10),
                    org.mockito.Mockito.mock(org.flexlb.worker.endpoint.EndpointRegistry.class));

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void configuredLimitIsRecheckedAtRegistrationWithoutAScheduler(boolean capacityTaken) {
        var status =
                WorkerStatus.createDiscovered(
                        RoleType.ENCODER, "group", "127.0.0.1", 8001, 8002, "test");
        status.lock.lock();
        try {
            status.recordSuccessfulPoll(true);
        } finally {
            status.lock.unlock();
        }
        var endpoint = new EncoderEndpoint(status, mock(CapacityNotifier.class));
        var endpoints = mock(EndpointRegistry.class);
        when(endpoints.<WorkerEndpoint>endpoints(RoleType.ENCODER, "group"))
                .thenReturn(List.of(endpoint));
        when(endpoints.getEndpoint(RoleType.ENCODER, endpoint.ipPort())).thenReturn(endpoint);
        var selection = spy(new EncoderStrategy(configuration(1), endpoints));
        var context = new RequestContext(new org.flexlb.config.FlexlbConfig());
        context.setRequest(Request.builder().requestId("incoming").seqLen(100L).build());
        doAnswer(
                        call -> {
                            var result = call.callRealMethod();
                            if (capacityTaken) {
                                assertEquals(
                                        WorkerEndpoint.RegistrationStatus.REGISTERED,
                                        endpoint.tryRegisterSelection("competitor", 1, 50L)
                                                .status());
                            }
                            return result;
                        })
                .when(selection)
                .select(anyList(), eq("group"), anyBoolean(), anyInt());
        try {
            var result = selection.select(context, "group");
            assertEquals(
                    capacityTaken ? RoutingResult.Status.BLOCKED : RoutingResult.Status.SUCCESS,
                    result.status());
            assertEquals(1L, endpoint.selectionSnapshot().requestCount());
            assertEquals(capacityTaken ? 50L : 100L, endpoint.selectionSnapshot().uncachedTokens());
        } finally {
            endpoint.evictExpiredRequests(-1L, ignored -> false);
        }
    }

    @Test
    void selectsLeastConcurrentWorkerWithoutUsingAvailableKvCache() {
        var busy = candidate("busy", true, 4, 0);
        var winner = candidate("zero-kv", true, 2, 100);
        var sameCount = candidate("small-kv", true, 2, 0);
        assertSame(
                winner,
                strategy.select(List.of(busy, winner, sameCount), "group", false, 10)
                        .orElseThrow());
    }

    @Test
    void ignoresDeadWorkersAndIncludesPendingLocalRequests() {
        var dead = candidate("dead", false, 0, 0);
        var pending = candidate("pending", true, 3, 0);
        var winner = candidate("winner", true, 1, 0);
        assertSame(
                winner,
                strategy.select(List.of(dead, pending, winner), "group", false, 10).orElseThrow());
    }

    @Test
    void returnsExactBlockerWhenNoLiveWorkerExists() {
        var result = strategy.select(List.of(candidate("dead", false, 0, 0)), "group", false, 10);
        org.junit.jupiter.api.Assertions.assertTrue(result.isEmpty());
    }

    @Test
    void selectsZeroKvWorkerWhenItHasLessConcurrentWork() {
        var busy = candidate("positive", true, 1, 0);
        var winner = candidate("zero", true, 0, 0);
        assertSame(
                winner, strategy.select(List.of(busy, winner), "group", false, 10).orElseThrow());
    }

    @Test
    void knownCacheEstimateSelectsLessUncachedWorkEvenWhenItHasMoreRequests() {
        var longJob = candidate("long-job", true, 1, 1000);
        var shortJobs = candidate("short-jobs", true, 4, 40);
        assertSame(
                shortJobs,
                strategy.select(List.of(longJob, shortJobs), "group", true, 10).orElseThrow());
    }

    @Test
    void queueModeSkipsWorkersAtEncoderInflightLimit() {
        var atCap = candidate("at-cap", true, 2, 0);
        var winner = candidate("winner", true, 1, 0);
        assertSame(
                winner, strategy.select(List.of(atCap, winner), "group", false, 2).orElseThrow());
    }

    @Test
    void queueModeBlocksWhenEveryWorkerIsAtEncoderInflightLimit() {
        var result =
                strategy.select(
                        List.of(candidate("running", true, 1, 0), candidate("pending", true, 1, 0)),
                        "group",
                        false,
                        1);
        org.junit.jupiter.api.Assertions.assertTrue(result.isEmpty());
    }

    @Test
    void doesNotSelectAnotherGroup() {
        var endpoint = org.mockito.Mockito.mock(EncoderEndpoint.class);
        var other = new EncoderEndpoint.SelectionSnapshot(endpoint, true, "another-group", 0, 0);
        org.junit.jupiter.api.Assertions.assertTrue(
                strategy.select(List.of(other), "group", false, 10).isEmpty());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void registrationRechecksTheRequestedGroupAfterRanking(boolean movesToAnotherGroup) {
        var status =
                WorkerStatus.createDiscovered(
                        RoleType.ENCODER, "g1", "127.0.0.1", 8001, 8002, "test");
        status.lock.lock();
        try {
            status.recordSuccessfulPoll(true);
        } finally {
            status.lock.unlock();
        }
        var endpoint = new EncoderEndpoint(status, mock(CapacityNotifier.class));
        assertTrue(endpoint.selectionSnapshot().available());
        var endpoints = mock(EndpointRegistry.class);
        when(endpoints.<WorkerEndpoint>endpoints(RoleType.ENCODER, "g1"))
                .thenReturn(List.of(endpoint));
        when(endpoints.getEndpoint(RoleType.ENCODER, endpoint.ipPort())).thenReturn(endpoint);
        var context = mock(RequestContext.class);
        when(context.getRequest())
                .thenReturn(Request.builder().requestId("encoder-group-race").seqLen(100L).build());
        var selection = spy(new EncoderStrategy(configuration(10), endpoints));
        doAnswer(
                        call -> {
                            var chosen = call.callRealMethod();
                            if (movesToAnotherGroup) {
                                status.lock.lock();
                                try {
                                    status.updateDiscoveryLabels("test", "g2");
                                } finally {
                                    status.lock.unlock();
                                }
                            }
                            return chosen;
                        })
                .when(selection)
                .select(anyList(), eq("g1"), anyBoolean(), anyInt());

        try {
            var result = selection.select(context, "g1");
            if (movesToAnotherGroup) {
                assertAll(
                        () ->
                                assertNotEquals(
                                        RoutingResult.Status.SUCCESS,
                                        result.status(),
                                        "a worker that moved to g2 cannot satisfy a g1 request"),
                        () ->
                                assertEquals(
                                        0L,
                                        endpoint.selectionSnapshot().requestCount(),
                                        "a group mismatch must leave no Encoder registration"));
            } else {
                assertEquals(RoutingResult.Status.SUCCESS, result.status());
                assertEquals("g1", result.value().group());
                assertEquals(1L, endpoint.selectionSnapshot().requestCount());
            }
        } finally {
            endpoint.evictExpiredRequests(-1L, ignored -> false);
        }
    }

    private static EncoderEndpoint.SelectionSnapshot candidate(
            String address, boolean available, long requests, long tokens) {
        var endpoint = org.mockito.Mockito.mock(EncoderEndpoint.class);
        org.mockito.Mockito.when(endpoint.ipPort()).thenReturn(address);
        return new EncoderEndpoint.SelectionSnapshot(
                endpoint, available, "group", requests, tokens);
    }

    private static org.flexlb.config.ConfigService configuration(int maxRequests) {
        var config = new org.flexlb.config.FlexlbConfig();
        config.getRouter().getRoles().getEncoder().setMaxRequests(maxRequests);
        var service = mock(org.flexlb.config.ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        return service;
    }
}
