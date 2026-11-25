package org.flexlb.balance.delivery;

import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;

import static org.flexlb.balance.delivery.RouteDispatchTiming.Stage.*;
import static org.mockito.Mockito.*;

class RouteDispatchTimingTest {
    @Test
    void reportsRepeatedAttemptsAndReopenLatencyOnceWithTheTotalWaitDenominator() {
        var timer = new RouteDispatchTiming();
        timer.enter(DISPATCH_ATTEMPT, 10_000_000);
        timer.enter(QUEUE_OR_RESELECTION, 11_000_000);
        timer.enter(PREFILL_INFLIGHT, 11_000_000);
        timer.enter(PREFILL_INFLIGHT, 12_000_000);
        timer.enter(REOPEN_TO_RESUME, 14_000_000);
        timer.enter(REOPEN_TO_RESUME, 15_000_000);
        timer.enter(QUEUE_OR_RESELECTION, 16_000_000);
        timer.enter(DISPATCH_ATTEMPT, 19_000_000);

        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        timer.report(reporter, "worker", 50, 20, 20_000_000);

        verify(reporter).reportRouteDispatchStage("worker", 50, "QUEUE_OR_RESELECTION", 13);
        verify(reporter).reportRouteDispatchStage("worker", 50, "DISPATCH_ATTEMPT", 2);
        verify(reporter).reportRouteDispatchStage("worker", 50, "PREFILL_INFLIGHT", 3);
        verify(reporter).reportRouteDispatchStage("worker", 50, "REOPEN_TO_RESUME", 2);
        for (var unused : new RouteDispatchTiming.Stage[]{PREFILL_KV, PREFILL_ENGINE_WAITING,
                PREFILL_RUNNING, PREFILL_OTHER, DECODE_CAPACITY}) {
            verify(reporter).reportRouteDispatchStage("worker", 50, unused.name(), 0);
        }
        verifyNoMoreInteractions(reporter);
    }
}
