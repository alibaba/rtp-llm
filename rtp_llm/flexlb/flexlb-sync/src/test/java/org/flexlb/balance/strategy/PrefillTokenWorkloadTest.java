package org.flexlb.balance.strategy;

import org.junit.jupiter.api.Test;
import java.util.Random;
import java.util.ArrayList;
import static org.junit.jupiter.api.Assertions.*;

class PrefillTokenWorkloadTest {
    @Test void abandonedPlansReleaseExactlyOnceWithoutConsumingHardCapacity() {
        var work = new PrefillTokenWorkload();
        var plan = work.plan(1000);
        assertEquals(1000, work.snapshot().score());
        assertEquals(0, work.snapshot().pendingTokens());
        plan.close(); plan.close();
        assertEquals(0, work.snapshot().score());
    }

    @Test void completionOrCancellationReleasesOnlyOwnedWork() {
        var work = new PrefillTokenWorkload();
        work.accept(1000);
        work.accept(250);
        work.release(1000);
        assertEquals(250, work.snapshot().score());
        work.release(250);
        assertEquals(0, work.snapshot().score());
        assertThrows(IllegalArgumentException.class, () -> work.release(1));
    }

    @Test void randomReservationsMatchOutstandingWorkWithoutHistoricalDebt() {
        var work = new PrefillTokenWorkload();
        var pending = new ArrayList<Long>();
        var plannedTokens = new ArrayList<Long>();
        var plans = new ArrayList<PrefillTokenWorkload.Reservation>();
        var random = new Random(7);
        for (int i = 0; i < 10000; i++) {
            switch (random.nextInt(4)) {
                case 0 -> {
                    long tokens = random.nextInt(100000);
                    pending.add(tokens);
                    work.accept(tokens);
                }
                case 1 -> {
                    if (!pending.isEmpty()) { work.release(pending.remove(random.nextInt(pending.size()))); }
                }
                case 2 -> {
                    long tokens = random.nextInt(100000);
                    plannedTokens.add(tokens);
                    plans.add(work.plan(tokens));
                }
                case 3 -> {
                    if (!plans.isEmpty()) {
                        int at = random.nextInt(plans.size());
                        var plan = plans.remove(at);
                        plannedTokens.remove(at);
                        assertTrue(plan.release());
                        assertFalse(plan.release());
                    }
                }
            }
            assertEquals(pending.stream().mapToLong(Long::longValue).sum(), work.snapshot().pendingTokens());
            assertEquals(plannedTokens.stream().mapToLong(Long::longValue).sum(), work.snapshot().plannedTokens());
        }
        pending.forEach(work::release);
        plans.forEach(PrefillTokenWorkload.Reservation::close);
        assertEquals(0, work.snapshot().score());
    }
    @Test void strictThresholdAndWeakCacheNeverSendToZeroHitWorker() {
        assertFalse(PrefillTokenBalance.cacheFirst(500, 1000, 50));
        assertTrue(PrefillTokenBalance.cacheFirst(501, 1000, 50));
        assertFalse(PrefillTokenBalance.cacheFirst(0, 0, 0));
        assertFalse(PrefillTokenBalance.eligible(400, 900, true));
        assertTrue(PrefillTokenBalance.eligible(900, 900, true));
        assertTrue(PrefillTokenBalance.eligible(200, 400, false));
        assertFalse(PrefillTokenBalance.eligible(0, 400, false));
        assertTrue(PrefillTokenBalance.eligible(0, 0, false));
    }
    @Test void cacheCompetitionRespectsPriorityAndPlanTransferDoesNotLeaveHistory() {
        var work = new PrefillTokenWorkload();
        var hot = work.plan(8192,117,30000,65536,50);
        assertEquals(.875,work.cacheWorkAtOrAbove(50).hitFraction());
        assertEquals(0,work.cacheWorkAtOrAbove(51).inputTokens());
        hot.close();
        work.accept(8192,65536,50);
        assertEquals(65536,work.cacheWorkAtOrAbove(50).inputTokens());
        work.accept(65536,65536,90);
        assertEquals(0,work.cacheWorkAtOrAbove(90).hitFraction());
        assertEquals(.4375,work.cacheWorkAtOrAbove(50).hitFraction());
        work.release(8192,65536,50);
        work.release(65536,65536,90);
        hot.close();
        assertEquals(0,work.cacheWorkAtOrAbove(0).inputTokens());
        assertEquals(0,work.snapshot().score());
    }
    @Test void invalidCacheReservationDoesNotConsumePlannedCapacity() {
        var work = new PrefillTokenWorkload();
        assertThrows(IllegalArgumentException.class,()->work.plan(100,1,30000,99,50));
        assertEquals(0,work.snapshot().score());
        assertEquals(0,work.snapshot().plannedExecutionMs());
        assertEquals(0,work.cacheWorkAtOrAbove(0).inputTokens());
        assertThrows(IllegalArgumentException.class,()->work.accept(100,99,50));
        assertEquals(0,work.snapshot().score());
    }
    @Test void randomCachedPlansTransfersAndReleasesConserveEveryPriorityTier() {
        record Job(long input, long compute, int priority, PrefillTokenWorkload.Reservation plan) { }
        var work = new PrefillTokenWorkload();
        var jobs = new ArrayList<Job>();
        var random = new Random(71);
        for (int i=0;i<2000;i++) {
            int action = random.nextInt(4);
            if (jobs.isEmpty() || action<2) {
                long input = random.nextInt(100000), compute = random.nextInt((int)input+1);
                int priority = new int[]{1,50,90}[random.nextInt(3)];
                var plan = action==0 ? work.plan(compute,compute/70+1,30000,input,priority) : null;
                if (plan==null) work.accept(compute,input,priority);
                jobs.add(new Job(input,compute,priority,plan));
            } else {
                int index = random.nextInt(jobs.size());
                var job = jobs.remove(index);
                if (job.plan()!=null) {
                    job.plan().close();job.plan().close();
                    if (action==3) {
                        work.accept(job.compute(),job.input(),job.priority());
                        jobs.add(new Job(job.input(),job.compute(),job.priority(),null));
                    }
                } else work.release(job.compute(),job.input(),job.priority());
            }
            for (int minimum : new int[]{0,1,50,90,100}) {
                long input = jobs.stream().filter(j->j.priority()>=minimum).mapToLong(Job::input).sum();
                long compute = jobs.stream().filter(j->j.priority()>=minimum).mapToLong(Job::compute).sum();
                assertEquals(new PrefillTokenWorkload.CacheWork(input,compute),work.cacheWorkAtOrAbove(minimum));
            }
            assertEquals(jobs.stream().mapToLong(Job::compute).sum(),work.snapshot().score());
        }
        for (var job : jobs) {
            if (job.plan()!=null) job.plan().close();
            else work.release(job.compute(),job.input(),job.priority());
        }
        assertEquals(0,work.cacheWorkAtOrAbove(0).inputTokens());
        assertEquals(0,work.snapshot().score());
    }
}
