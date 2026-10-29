package org.flexlb.balance.strategy;

import org.flexlb.balance.projection.*;
import org.flexlb.balance.planner.*;
import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class PrefillSloPolicyTest {
    @Test
    void duplicatePlanTargetsReleaseExactlyOnceAndAdvanceToTheNextTarget() {
        var ledger = new PrefillTokenWorkload();
        var first = ledger.plan(100, 10, 2000);
        var second = ledger.plan(200, 20, 2000);
        var third = ledger.plan(300, 30, 3000);
        assertEquals(2000, ledger.snapshot().earliestPlannedTargetMs());
        assertTrue(first.release());
        assertFalse(first.release());
        assertEquals(2000, ledger.snapshot().earliestPlannedTargetMs());
        assertEquals(50, ledger.snapshot().plannedExecutionMs());
        second.close();
        assertEquals(3000, ledger.snapshot().earliestPlannedTargetMs());
        third.close();
        assertEquals(Long.MAX_VALUE, ledger.snapshot().earliestPlannedTargetMs());
        assertEquals(0, ledger.snapshot().plannedTokens());
        assertEquals(0, ledger.snapshot().plannedExecutionMs());
    }

    @Test
    void failedTimeReservationDoesNotPartiallyChangeTheLedger() {
        var ledger = new PrefillTokenWorkload();
        var first = ledger.plan(10, Long.MAX_VALUE, 3000);
        var before = ledger.snapshot();
        assertThrows(ArithmeticException.class, () -> ledger.plan(20, 1, 2000));
        assertEquals(before, ledger.snapshot());
        first.close();
        assertEquals(0, ledger.snapshot().plannedTokens());
        assertEquals(0, ledger.snapshot().plannedExecutionMs());
    }

    RouteProjection.Inputs inputs(List<GroupPlanner.Item> items, boolean unknown) {
        return new RouteProjection.Inputs(new QueueSnapshot(10000,true,GroupingPolicy.FIXED_WINDOW,
                Comparator.comparingLong(GroupPlanner.Item::enqueueSeq),new GroupPlanner.Constraints(1024,100000,100000,2700,20),items,null),
                new WorkSnapshot(10000,List.of(),List.of(),unknown?1:0),0);
    }
    @Test void additionalBatchMemberMustRespectAlreadyQueuedRequestTarget() {
        // Existing request arrived at 8000, expires at 14000: original SLO target is 11000.
        var old=new GroupPlanner.Item(1,50,1,9000,14000,1000,0);
        assertTrue(PrefillSloPolicy.admits(3000,10000,10000,600,0,inputs(List.of(old),false),false));
        assertFalse(PrefillSloPolicy.admits(3000,10000,10000,800,0,inputs(List.of(old),false),false));
        assertTrue(PrefillSloPolicy.admits(3000,10000,10000,800,0,inputs(List.of(),false),false));
    }
    @Test void subTenMillisecondServiceStillCollectsButNoWaitWhenTargetIsSpent() {
        assertEquals(1, PrefillSloPolicy.collectionWindowMs(20, 100, 9));
        assertEquals(20, PrefillSloPolicy.collectionWindowMs(20, 1000, 300));
        assertEquals(0, PrefillSloPolicy.collectionWindowMs(20, 9, 9));
        assertEquals(1, PrefillSloPolicy.collectionWindowMs(20, 10, 9));
    }
    @Test void unknownEngineWorkAndMissingPredictionNeverMeanIdle() {
        assertFalse(PrefillSloPolicy.admits(3000,10000,10000,1,0,inputs(List.of(),true),true));
        assertFalse(PrefillSloPolicy.admits(3000,10000,10000,-1,0,inputs(List.of(),false),true));
    }
    @Test void plusMinusTenPercentServiceChangesMoveAdmissionBoundary() {
        var inputs=inputs(List.of(),false);
        assertTrue(PrefillSloPolicy.admits(3000,10000,10000,2500,0,inputs,false));
        assertTrue(PrefillSloPolicy.admits(3000,10000,10000,2250,0,inputs,false));
        assertFalse(PrefillSloPolicy.admits(3000,10000,10000,2750,0,inputs,false));
    }
    @Test void saturatedPredictionCannotWrapIntoAdmission() {
        assertFalse(PrefillSloPolicy.admits(3000,10000,10000,Long.MAX_VALUE,10,inputs(List.of(),false),false));
    }
    @Test void randomWorkloadReservationLifecycleConservesPredictedTime() {
        var ledger=new PrefillTokenWorkload();
        var random=new Random(19);
        var plans=new ArrayList<PrefillTokenWorkload.Reservation>();
        var durations=new ArrayList<Long>();
        for(int n=0;n<10000;n++) {
            if(plans.isEmpty()||random.nextBoolean()) {
                long ms=random.nextInt(1000); plans.add(ledger.plan(random.nextInt(100000),ms)); durations.add(ms);
            } else {
                int i=random.nextInt(plans.size()); var plan=plans.remove(i); plan.close();plan.close();durations.remove(i);
            }
            assertEquals(durations.stream().mapToLong(Long::longValue).sum(),ledger.snapshot().plannedExecutionMs());
        }
        plans.forEach(PrefillTokenWorkload.Reservation::close);
        assertEquals(0,ledger.snapshot().plannedExecutionMs());
    }
    @Test void cacheProtectionRequiresQueuedCompetitionAndKeepsAnIdleWorkerUsable() {
        assertTrue(PrefillSloPolicy.admitsCacheWork(30000,2000,65536,0,.875,100,0,false));
        assertFalse(PrefillSloPolicy.admitsCacheWork(30000,2000,65536,0,.875,3000,0,false));
        assertTrue(PrefillSloPolicy.admitsCacheWork(30000,2000,65536,57344,.875,3000,0,false));
        assertTrue(PrefillSloPolicy.admitsCacheWork(30000,2000,65536,0,0,20000,0,false));
        assertTrue(PrefillSloPolicy.admitsCacheWork(30000,2000,65536,0,.875,3000,0,true));
    }
    @Test void agingExpandsColdFeedWithoutAllowingItToConsumeTheEntireSloBudget() {
        assertTrue(PrefillSloPolicy.admitsCacheWork(30000,2000,65536,0,.875,375,30000,false));
        assertFalse(PrefillSloPolicy.admitsCacheWork(30000,2000,65536,0,.875,20000,60000,false));
        assertFalse(PrefillSloPolicy.admitsCacheWork(30000,2000,65536,0,.875,375,-1,false));
    }
}
