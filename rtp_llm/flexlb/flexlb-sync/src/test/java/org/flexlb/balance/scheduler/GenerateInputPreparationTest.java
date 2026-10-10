package org.flexlb.balance.scheduler;

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.config.ConfigService;
import org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@Timeout(10)
class GenerateInputPreparationTest {
    @Test
    void failedPreparationIsDeferredAndPayloadReplacementRemainsAuthoritative() throws Exception {
        var context = new RequestContext(SchedulingTestConfig.batchConfig());
        context.setGenerateInputPb(ByteString.copyFrom(new byte[] {(byte) 0xff}));
        assertDoesNotThrow(context::prepareGenerateInput);
        var failure = assertThrows(InvalidProtocolBufferException.class, context::getGenerateInput);
        assertSame(failure, assertThrows(InvalidProtocolBufferException.class, context::getGenerateInput));
        var valid = GenerateInputPB.newBuilder().setRequestId(71).addTokenIds(42).build();
        context.setGenerateInputPb(valid.toByteString());
        context.prepareGenerateInput();
        var parsed = context.getGenerateInput();
        assertEquals(valid, parsed);
        assertSame(parsed, context.getGenerateInput());
        assertEquals(71, context.getGenerateInput().getRequestId());
        parsed.toBuilder().setRequestId(72).build();
        assertEquals(71, context.getGenerateInput().getRequestId());
        context.setGenerateInputPb(ByteString.EMPTY);
        assertFalse(context.hasGenerateInput());
        assertNull(context.getGenerateInput());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cancellationOrDeadlineDuringPreparationCannotReachSelection(boolean deadline) throws Exception {
        var config = SchedulingTestConfig.batchConfig();
        var source = RequestProtocolTestSupport.context(config, 17001L);
        var context = spy(source);
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var clockAdvanced = new AtomicBoolean();
        Thread submittingThread = Thread.currentThread();
        doAnswer(call -> {
            assertFalse(Thread.holdsLock(context));
            assertNotEquals(Thread.currentThread(), submittingThread);
            entered.countDown();
            assertTrue(release.await(5, TimeUnit.SECONDS));
            return call.callRealMethod();
        }).when(context).prepareGenerateInput();
        // Advance only the request clock, leaving the timer pending to exercise the absolute check.
        doAnswer(call -> clockAdvanced.get() || (boolean) call.callRealMethod())
                .when(context).requestExpired(anyLong());
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var reporter = mock(DeliveryMetricsReporter.class);
        var registry = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, reporter, mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        var router = mock(RequestWorkerSelector.class);
        var queue = org.flexlb.balance.scheduler.SchedulerTestSupport.configure(registry, config, router, reporter,
                mock(DecodeCapacityAcquirer.class), new PlacementAvailability());
        try {
            var future = queue.submit(context);
            assertTrue(entered.await(3, TimeUnit.SECONDS), "submit returns while planning is blocked");
            if (deadline) {
                clockAdvanced.set(true);
            } else {
                registry.cancel(17001L, 0L, CancelReason.CLIENT_CANCELLED);
                assertEquals(8504, future.get(2, TimeUnit.SECONDS).getCode(),
                        "input preparation owns no admission handle and cannot hold cancellation open");
            }
            release.countDown();
            assertEquals(deadline ? 8431 : 8504, future.get(3, TimeUnit.SECONDS).getCode());
            RequestProtocolTestSupport.close(queue);
            registry.awaitAdmissionMutations();
            verify(router, never()).select(any(), org.mockito.ArgumentMatchers.nullable(String.class));
            assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).liveRequestCount());
        } finally {
            release.countDown();
            RequestProtocolTestSupport.close(queue);
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry)) {
                registry.closeOutstandingAndTerminalize();
            }
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).timer().close();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).closeRequestExecutors();
        }
    }

}
