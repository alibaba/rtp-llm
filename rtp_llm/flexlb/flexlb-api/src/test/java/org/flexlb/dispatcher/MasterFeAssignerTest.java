package org.flexlb.dispatcher;

import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Direct unit coverage for {@link MasterFeAssigner}, the single FE-stamp point. The two production
 * entry points ({@link BatchScheduleClient} and {@link org.flexlb.httpserver.HttpLoadBalanceServer})
 * exercise the happy paths, but three guard/exception branches are only reachable — and only
 * assertable — here: the null/empty-targets early return, the absent-{@link FePool}-bean no-op
 * (the documented deployment precondition failure), and the two "swallow and leave fe_url null"
 * exception paths. Each test doubles as a mutation guard: deleting the guard/catch it targets turns
 * it red.
 */
class MasterFeAssignerTest {

    private static BatchScheduleTarget target(String ip) {
        return new BatchScheduleTarget(ip, 23840, 23841);
    }

    private static BatchScheduleRequest request(boolean assignFe) {
        BatchScheduleRequest request = new BatchScheduleRequest();
        request.setAssignFe(assignFe);
        return request;
    }

    private static BatchScheduleResponse response(
            List<BatchScheduleTarget> targets, boolean resolvedLocally) {
        BatchScheduleResponse response = BatchScheduleResponse.success(targets);
        response.setResolvedLocally(resolvedLocally);
        return response;
    }

    @Test
    void assignFeFalseDoesNotAdvanceMasterPool() {
        FePool pool = mock(FePool.class);
        MasterFeAssigner assigner = DispatcherTestSupport.masterFeAssigner(pool);
        BatchScheduleRequest request = request(false);
        BatchScheduleTarget target = target("10.0.0.1");

        assigner.assign(request, response(List.of(target), true));

        assertNull(target.getFeUrl());
        verifyNoInteractions(pool);
    }

    @Test
    void nullTargetsIsNoOpAndNeverTouchesPool() {
        // Early return before any consistency/pool interaction. Mutation guard: drop the null check
        // and assign(null) NPEs on targets.isEmpty()/the for-loop.
        FePool pool = mock(FePool.class);
        MasterFeAssigner assigner = DispatcherTestSupport.masterFeAssigner(pool);
        assertDoesNotThrow(() -> assigner.assign(request(true), null));
        verifyNoInteractions(pool);
    }

    @Test
    void emptyTargetsIsNoOpAndNeverTouchesPool() {
        FePool pool = mock(FePool.class);
        MasterFeAssigner assigner = DispatcherTestSupport.masterFeAssigner(pool);
        assertDoesNotThrow(() -> assigner.assign(request(true), response(List.of(), true)));
        verifyNoInteractions(pool);
    }

    @Test
    void absentFePoolBeanLeavesFeUrlNullWithoutThrowing() {
        // The documented precondition failure: an elected master that does NOT run the dispatcher
        // has no FePool bean. resolvedLocally is true (consistency off by the mock default), so we
        // reach the pool lookup — getIfAvailable() returns null → no stamp, no throw, fe_url stays
        // null and the chunk fails downstream with CHUNK_NO_FE. Mutation guard: delete the
        // `pool == null` check and this NPEs on pool.next().
        MasterFeAssigner assigner = DispatcherTestSupport.noopFeAssigner();
        BatchScheduleTarget t = target("10.0.0.1");
        assertDoesNotThrow(() -> assigner.assign(
                request(true), response(List.of(t), true)));
        assertNull(t.getFeUrl(), "no FePool bean → fe_url must stay null (chunk fails CHUNK_NO_FE)");
    }

    @Test
    void slaveDoesNotStampAndNeverTouchesPool() {
        // resolvedLocally=false (consistency on, not master): a slave already holds the master's
        // stamp; re-stamping with a local cursor would reintroduce the collision the feature removes.
        // Mutation guard: drop the resolvedLocally guard and the pool gets touched + the master's
        // fe_url is overwritten.
        FePool pool = mock(FePool.class);
        MasterFeAssigner assigner = DispatcherTestSupport.masterFeAssigner(pool);
        BatchScheduleTarget t = target("10.0.0.1");
        t.setFeUrl("http://master-fe");
        assigner.assign(request(true), response(List.of(t), false));
        assertEquals("http://master-fe", t.getFeUrl(), "a slave must preserve the master's fe_url");
        verifyNoInteractions(pool);
    }

    @Test
    void masterStampsEachTargetFromTheCursorInOrder() {
        // One nextBatch(count) call feeds the whole batch: url[i] must land on target[i]. Mutation
        // guard: zip the urls in reverse (or off by one) and the fe_url assignments swap.
        FePool pool = mock(FePool.class);
        when(pool.nextBatch(2)).thenReturn(List.of("http://fe-1", "http://fe-2"));
        MasterFeAssigner assigner = DispatcherTestSupport.masterFeAssigner(pool);
        BatchScheduleTarget a = target("10.0.0.1");
        BatchScheduleTarget b = target("10.0.0.2");
        assigner.assign(request(true), response(List.of(a, b), true));
        assertEquals("http://fe-1", a.getFeUrl());
        assertEquals("http://fe-2", b.getFeUrl());
    }

    @Test
    void emptyPoolSnapshotIsSwallowedLeavingFeUrlNull() {
        // FePool.nextBatch() throws IllegalStateException when its snapshot is empty (FE outage).
        // This is the expected operational failure: swallow (throttled WARN), leave fe_url null, do
        // NOT abort the schedule — the chunk fails downstream with CHUNK_NO_FE. Mutation guard:
        // remove the IllegalStateException catch and the exception aborts the whole schedule.
        FePool pool = mock(FePool.class);
        when(pool.nextBatch(1)).thenThrow(new IllegalStateException("no FE endpoints available"));
        MasterFeAssigner assigner = DispatcherTestSupport.masterFeAssigner(pool);
        BatchScheduleTarget t = target("10.0.0.1");
        assertDoesNotThrow(() -> assigner.assign(
                request(true), response(List.of(t), true)));
        assertNull(t.getFeUrl());
    }

    @Test
    void emptyPoolLeavesEveryTargetNullWithNoPartialStamp() {
        // nextBatch resolves the FE snapshot once and throws before any pick, so an empty pool
        // assigns nothing rather than stamping a prefix: assignment is all-or-nothing per batch.
        // Mutation guard: stamp inside a per-target loop that calls a throwing pick and the first
        // target would come back partially stamped.
        FePool pool = mock(FePool.class);
        when(pool.nextBatch(2)).thenThrow(new IllegalStateException("no FE endpoints available"));
        MasterFeAssigner assigner = DispatcherTestSupport.masterFeAssigner(pool);
        BatchScheduleTarget a = target("10.0.0.1");
        BatchScheduleTarget b = target("10.0.0.2");
        assertDoesNotThrow(() -> assigner.assign(
                request(true), response(List.of(a, b), true)));
        assertNull(a.getFeUrl(), "empty pool must stamp nothing (chunk fails CHUNK_NO_FE)");
        assertNull(b.getFeUrl(), "empty pool must stamp nothing (chunk fails CHUNK_NO_FE)");
    }

    @Test
    void unexpectedRuntimeExceptionIsSwallowedLeavingFeUrlNull() {
        // A non-empty-pool RuntimeException is an unexpected bug: it is logged loud (ERROR) rather
        // than folded into the throttled "pool empty" WARN, but still swallowed so the already-
        // computed BE assignment is not lost. fe_url stays null → chunk fails CHUNK_NO_FE. Mutation
        // guard: narrow the second catch away and this propagates out of assign().
        FePool pool = mock(FePool.class);
        when(pool.nextBatch(1)).thenThrow(new IllegalArgumentException("unexpected"));
        MasterFeAssigner assigner = DispatcherTestSupport.masterFeAssigner(pool);
        BatchScheduleTarget t = target("10.0.0.1");
        assertDoesNotThrow(() -> assigner.assign(
                request(true), response(List.of(t), true)));
        assertNull(t.getFeUrl());
    }
}
