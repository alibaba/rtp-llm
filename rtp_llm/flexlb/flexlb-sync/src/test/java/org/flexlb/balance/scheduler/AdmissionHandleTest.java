package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.nullable;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class AdmissionHandleTest {
    @Test
    void competingCompletionsSettleOnceOutsideTheHandleMonitor() throws Exception {
        var context = new RequestContext(SchedulingTestConfig.newConfig());
        var lifecycle = mock(RequestLifecycle.class);
        SchedulerTestSupport.bindOwner(context, lifecycle);
        var handle = new AdmissionHandle(context, null);
        doAnswer(
                        call -> {
                            assertFalse(Thread.holdsLock(handle));
                            return null;
                        })
                .when(lifecycle)
                .completePlacement(same(handle), any(), nullable(Response.class));
        var start = new CountDownLatch(1);
        try (var workers = Executors.newFixedThreadPool(4)) {
            var pending = new ArrayList<Future<?>>();
            for (int i = 0; i < 4; i++) {
                boolean success = i % 2 == 0;
                pending.add(
                        workers.submit(
                                () -> {
                                    assertTrue(start.await(5, TimeUnit.SECONDS));
                                    if (success) handle.finish();
                                    else
                                        handle.terminate(
                                                Response.error(StrategyErrorType.DISPATCH_FAILED));
                                    return null;
                                }));
            }
            start.countDown();
            for (var task : pending) task.get(5, TimeUnit.SECONDS);
        }
        verify(lifecycle).completePlacement(same(handle), any(), nullable(Response.class));
        assertDoesNotThrow(handle::close);
    }
}
