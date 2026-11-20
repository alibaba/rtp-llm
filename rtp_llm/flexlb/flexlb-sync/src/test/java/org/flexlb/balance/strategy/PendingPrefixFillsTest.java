package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class PendingPrefixFillsTest {
    private static final RoleType ROLE = RoleType.PREFILL;
    private static final Set<String> LEADERS = Set.of("p:80", "other:80");
    private static final List<Long> BLOCKS = List.of(10L, 11L, 12L, 13L);
    private final CompletableFuture<Response> response = new CompletableFuture<>();
    private final RequestContext owner = new RequestContext(StrategyTestSupport.config());
    private final PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
    private final WorkerStatus status = mock(WorkerStatus.class, RETURNS_DEEP_STUBS);
    private final CountDownLatch reconsidered = new CountDownLatch(1);
    private final PendingPrefixFills fills = new PendingPrefixFills(ep -> reconsidered.countDown());
    private final Request extension = request(2, List.of(10L, 11L, 12L, 13L, 14L));

    private static Request request(long id, List<Long> keys) {
        return Request.builder().requestId(id).seqLen(keys.size() * 512L)
                .cacheKeyBlockSize(512L).blockCacheKeys(keys).build();
    }

    @BeforeEach
    void setUp() {
        var producer = request(1, BLOCKS);
        owner.data().setRequest(producer);
        owner.data().setFuture(response);
        when(endpoint.ipPort()).thenReturn("p:80");
        when(endpoint.getStatus()).thenReturn(status);
        when(status.topologySnapshot().ip()).thenReturn("p");
        when(status.topologySnapshot().port()).thenReturn(80);
        when(status.committedEngineObservation().runningTaskList()).thenReturn(Map.of());
        fills.remember(PendingPrefixFills.suffix(producer, ROLE, "g", 512), owner, endpoint, 40, 100, 30_100);
    }

    private PrefillEndpoint waitFor(Request request, int priority, long latestWake, long now) {
        return fills.waitingFor(PendingPrefixFills.suffix(request, ROLE, "g", 512),
                request, priority, LEADERS, latestWake, now);
    }

    private void confirm(String ip) {
        var worker = new ServerStatus();
        worker.setRole(ROLE);
        worker.setServerIp(ip);
        worker.setHttpPort(80);
        worker.setRequestId(1L);
        worker.setSuccess(true);
        var reply = new Response();
        reply.setSuccess(true);
        reply.setServerStatus(List.of(worker));
        response.complete(reply);
    }

    @Test
    void onlyConfirmedOverlappingWorkDefersARequest() {
        assertNull(waitFor(extension, 40, 10_000, 200));
        confirm("p");
        assertSame(endpoint, waitFor(extension, 40, 10_000, 200));
        assertNull(waitFor(request(1, BLOCKS), 40, 10_000, 200));
        assertNull(waitFor(request(3, List.of(10L, 11L, 99L, 13L)), 40, 10_000, 200));
        assertNull(waitFor(request(4, List.of(10L, 11L, 12L, 99L)), 40, 10_000, 200));
        assertNull(PendingPrefixFills.suffix(extension, ROLE, "g", 4 * 512));
    }

    @Test
    void aPartialIndexPublicationStillFindsTheInProgressTerminalBlock() {
        confirm("p");
        var advanced = PendingPrefixFills.suffix(extension, ROLE, "g", 1024);
        assertSame(endpoint, fills.waitingFor(advanced, extension, 40, LEADERS, 10_000, 200));
        assertNull(fills.waitingFor(PendingPrefixFills.suffix(extension, ROLE, "other", 1024),
                extension, 40, LEADERS, 10_000, 200));
    }

    @Test
    void aLongerProducerCanServeAnEarlierShorterTurn() {
        confirm("p");
        assertSame(endpoint, waitFor(request(3, BLOCKS.subList(0, 3)), 40, 10_000, 200));
    }

    @Test
    void anExceptionalFutureCannotParkPeers() {
        response.completeExceptionally(new IllegalStateException("dispatch failed"));
        assertNull(waitFor(extension, 40, 10_000, 200));
    }

    @Test
    void matchingIntervalEndsDoNotHideAnInteriorMismatch() {
        var producer = request(1, List.of(1L, 2L, 3L, 4L, 5L, 6L));
        owner.data().setRequest(producer);
        fills.remember(PendingPrefixFills.suffix(producer, ROLE, "g", 512), owner, endpoint, 40, 100, 30_100);
        confirm("p");
        var different = request(5, List.of(1L, 2L, 3L, 99L, 5L, 6L, 7L));
        assertNull(waitFor(different, 40, 10_000, 200));
    }

    @Test
    void anAssignmentElsewhereIsNotAProducer() {
        confirm("other");
        assertNull(waitFor(extension, 40, 10_000, 200));
    }

    @Test
    void cancelledOrFailedAssignmentCannotBlock() {
        response.cancel(false);
        assertNull(waitFor(extension, 40, 10_000, 200));
    }

    @Test
    void anUnsuccessfulRoutingReplyCannotBlock() {
        response.complete(new Response());
        assertNull(waitFor(extension, 40, 10_000, 200));
    }

    @Test
    void retiredGenerationAndNearDeadlineDoNotBlock() {
        confirm("p");
        assertNull(waitFor(extension, 40, 2000, 200));
        when(endpoint.isGenerationRetiringOrRetired()).thenReturn(true);
        assertNull(waitFor(extension, 40, 10_000, 200));
    }

    @Test
    void higherPriorityPassesUntilTheProducerIsActuallyRunning() {
        confirm("p");
        assertNull(waitFor(extension, 50, 10_000, 200));
        assertSame(endpoint, fills.waitingFor(PendingPrefixFills.suffix(extension, ROLE, "g", 512),
                extension, 50, Set.of("p:80"), 10_000, 200));
        var task = mock(WorkerStatus.TaskObservation.class);
        when(task.phase()).thenReturn(TaskPhase.RUNNING);
        when(task.priorityPreemptionProgress()).thenReturn(PriorityPreemptionProgress.NONE);
        when(status.committedEngineObservation().runningTaskList()).thenReturn(Map.of("1", task));
        assertSame(endpoint, waitFor(extension, 50, 10_000, 200));
        when(task.priorityPreemptionProgress()).thenReturn(PriorityPreemptionProgress.CANCELING);
        assertNull(waitFor(extension, 50, 10_000, 200));
    }

    @Test
    void expiryReconsidersParkedWorkWithoutAnyCacheOrStatusUpdate() throws Exception {
        confirm("p");
        assertSame(endpoint, waitFor(extension, 40, 10_000, 3090));
        assertTrue(reconsidered.await(2, TimeUnit.SECONDS));
        assertNull(waitFor(extension, 40, 10_000, 3100));
    }
}
