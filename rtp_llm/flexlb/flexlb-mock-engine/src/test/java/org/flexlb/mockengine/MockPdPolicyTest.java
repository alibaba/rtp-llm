package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

class MockPdPolicyTest {
    private MockPdPolicy policy(int retries, long budget) {
        return new MockPdPolicy(retries, budget, 1000, 1, 0, 50, 128, 2000);
    }
    @Test void retriesOnlyPrepareUntilSuccess() throws Exception {
        var attempts = new AtomicInteger();
        assertTrue(policy(3, 1000).prepare(() -> attempts.incrementAndGet() == 3, () -> false, () -> 5000));
        assertEquals(3, attempts.get());
    }
    @Test void retryBudgetDoesNotInterruptFirstLiveAllocation() throws Exception {
        var attempts = new AtomicInteger();
        assertTrue(policy(5, 1).prepare(() -> { attempts.incrementAndGet(); Thread.sleep(10); return true; }, () -> false, () -> 5000));
        assertEquals(1, attempts.get());
    }
    @Test void spentRetryBudgetStopsNewAttemptsButZeroDisablesBudget() throws Exception {
        var attempts = new AtomicInteger();
        assertFalse(policy(5, 1).prepare(() -> { attempts.incrementAndGet(); Thread.sleep(10); return false; }, () -> false, () -> 5000));
        assertEquals(1, attempts.get());
        attempts.set(0);
        assertFalse(policy(2, 0).prepare(() -> { attempts.incrementAndGet(); return false; }, () -> false, () -> 5000));
        assertEquals(3, attempts.get());
    }
    @Test void cancellationAndRequestDeadlineAreCheckedAfterFailedAttempt() {
        var cancelled = new java.util.concurrent.atomic.AtomicBoolean();
        var cancel = assertThrows(RuntimeException.class, () -> policy(0, 1).prepare(() -> {
            cancelled.set(true); return false;
        }, cancelled::get, () -> 5000));
        assertEquals(8100, MockRpcErrors.downstream(cancel).getErrorCode());
        var remaining = new java.util.concurrent.atomic.AtomicLong(1000);
        var timeout = assertThrows(RuntimeException.class, () -> policy(0, 1).prepare(() -> {
            remaining.set(0); return false;
        }, () -> false, remaining::get));
        assertEquals(603, MockRpcErrors.downstream(timeout).getErrorCode());
    }
    @Test void stageBudgetsCapUnlimitedRequestAndLoadHasRealFallback() {
        assertEquals(50, policy(0, 1).rpcMs(Long.MAX_VALUE));
        assertEquals(50, policy(0, 1).loadMs(Long.MAX_VALUE));
        assertEquals(12, policy(0, 1).loadMs(12));
        assertEquals(5000, new MockPdPolicy(0, 0, 0, 1, 0, 0, 128, 0).loadMs(Long.MAX_VALUE));
    }
    @Test void channelReadinessUsesTheTightestNativeDeadline() throws Exception {
        var unlimited = new MockPdPolicy(0, 0, 0, 1, 0, 0, 128, 0);
        assertEquals(15000, unlimited.readyMs(Long.MAX_VALUE));
        assertEquals(100, unlimited.readyMs(100));
        assertEquals(50, policy(0, 1000).readyMs(Long.MAX_VALUE));
        assertTrue(policy(0, 20).prepare(() -> {
            long ready = policy(0, 20).readyMs(Long.MAX_VALUE);
            assertTrue(ready > 0 && ready <= 20, "retry budget still bounds channel readiness");
            return true;
        }, () -> false, () -> Long.MAX_VALUE));
    }

    @Test void importsNativeInt64TimeoutsWithoutNarrowing() {
        var config = (com.fasterxml.jackson.databind.node.ObjectNode) MockEngineConfigFixture.load().snapshot().path("roles").path("prefill").path("pd_sep");
        config.put("prefill_retry_timeout_ms", 3000000000L).put("decode_retry_times", Long.MAX_VALUE);
        assertEquals(3000000000L, MockPdPolicy.parse(config).prefillRetryTimeoutMs());
    }

}
