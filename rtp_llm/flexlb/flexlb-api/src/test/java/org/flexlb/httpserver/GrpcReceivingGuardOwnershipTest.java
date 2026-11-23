package org.flexlb.httpserver;

import io.grpc.Context;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.ServerCall;
import io.grpc.ServerStreamTracer;
import io.grpc.Status;
import io.grpc.internal.SerializingExecutor;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.PooledByteBufAllocator;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies that receiving permits outlive queued buffer cleanup.
 */
class GrpcReceivingGuardOwnershipTest {
    @Test
    void scheduleBodyBudgetRemainsChargedUntilTerminalCleanupAndCancelHasCapacity() {
        try (var guard = new GrpcReceivingGuard(1, 1, 60_000, Runnable::run, Runnable::run, 8, 4)) {
            Metadata headers = new Metadata();
            var schedule = guard.newServerStreamTracer("test/Schedule", headers);
            var rejected = guard.newServerStreamTracer("test/Schedule", headers);
            var cancel = guard.newServerStreamTracer("test/Cancel", headers);
            var rejectStatus = new CompletableFuture<Status>();
            rejected.filterContext(Context.ROOT)
                    .run(
                            () ->
                                    guard.getExecutor(
                                            new Call() {
                                                @Override
                                                public void close(
                                                        Status status, Metadata trailers) {
                                                    rejectStatus.complete(status);
                                                }
                                            },
                                            headers));
            assertEquals(
                    Status.Code.RESOURCE_EXHAUSTED, rejectStatus.getNow(Status.UNKNOWN).getCode());
            rejected.streamClosed(Status.RESOURCE_EXHAUSTED);
            assertEquals(1, guard.receiving());
            assertEquals(1, guard.controlReceiving());

            cancel.filterContext(Context.ROOT)
                    .run(
                            () -> {
                                Call call = new Call();
                                guard.getExecutor(call, headers);
                                var listener =
                                        guard.interceptCall(
                                                call,
                                                headers,
                                                (unused, metadata) ->
                                                        new ServerCall.Listener<>() {});
                                cancel.inboundWireSize(4);
                                assertEquals(4, guard.controlReceivingBytes());
                                listener.onHalfClose();
                                assertEquals(0, guard.controlReceivingBytes());
                                assertEquals(0, guard.controlReceiving());
                            });

            schedule.filterContext(Context.ROOT)
                    .run(
                            () -> {
                                var closeStatus = new CompletableFuture<Status>();
                                Call call =
                                        new Call() {
                                            @Override
                                            public void close(Status status, Metadata trailers) {
                                                closeStatus.complete(status);
                                                schedule.streamClosed(status);
                                            }
                                        };
                                guard.getExecutor(call, headers);
                                var listener =
                                        guard.interceptCall(
                                                call,
                                                headers,
                                                (unused, metadata) ->
                                                        new ServerCall.Listener<>() {});
                                schedule.inboundWireSize(8);
                                schedule.inboundWireSize(1);
                                assertEquals(
                                        Status.Code.RESOURCE_EXHAUSTED,
                                        closeStatus.getNow(Status.UNKNOWN).getCode());
                                assertEquals(8, guard.receivingBytes());
                                assertEquals(1, guard.receiving());
                                listener.onCancel();
                                listener.onComplete();
                                assertEquals(0, guard.receivingBytes());
                                assertEquals(0, guard.receiving());
                            });
        }
    }

    @Test
    void incompleteReceiveExpiresAndReturnsItsPermit() throws Exception {
        try (var guard = new GrpcReceivingGuard(1, 1, 20, Runnable::run, Runnable::run, 8, 4)) {
            Metadata headers = new Metadata();
            var schedule = guard.newServerStreamTracer("test/Schedule", headers);
            var closeStatus = new CompletableFuture<Status>();
            schedule.filterContext(Context.ROOT)
                    .run(
                            () ->
                                    guard.getExecutor(
                                            new Call() {
                                                @Override
                                                public void close(
                                                        Status status, Metadata trailers) {
                                                    schedule.streamClosed(status);
                                                    closeStatus.complete(status);
                                                }
                                            },
                                            headers));
            assertEquals(
                    Status.Code.DEADLINE_EXCEEDED, closeStatus.get(2, TimeUnit.SECONDS).getCode());
            assertEquals(1, guard.timedOut());
            assertEquals(0, guard.receiving());
        }
    }

    @Test
    void errorCannotReturnReservationBeforeQueuedActualByteBufCleanup() {
        queued(false);
        queued(true);
    }

    void queued(boolean startFails) {
        ArrayDeque<Runnable> work = new ArrayDeque<>();
        var guard = new GrpcReceivingGuard(1, 1, 60000, work::add, work::add, 1 << 20, 1 << 20);
        var allocator = new PooledByteBufAllocator(true);
        ByteBuf retained = allocator.directBuffer(65536, 65536);
        ServerStreamTracer tracer = guard.newServerStreamTracer("test/Schedule", new Metadata());
        Context c = tracer.filterContext(Context.ROOT), before = c.attach();
        try {
            Call call = new Call();
            Executor e = guard.getExecutor(call, new Metadata());
            ServerCall.Listener<Object> listener = null;
            try {
                listener =
                        guard.interceptCall(
                                call,
                                new Metadata(),
                                (x, h) -> {
                                    if (startFails) throw new AssertionError("start failed");
                                    return new ServerCall.Listener<>() {};
                                });
            } catch (AssertionError expected) {
                assertTrue(startFails);
            }
            SerializingExecutor serial = new SerializingExecutor(e);
            serial.execute(
                    () -> {
                        throw new AssertionError("callback Error");
                    });
            serial.execute(retained::release);
            if (!startFails) {
                final var terminal = listener;
                serial.execute(terminal::onCancel);
            }
            tracer.streamClosed(Status.CANCELLED);
            assertEquals(1, guard.receiving());
            assertEquals(65536, allocator.pinnedDirectMemory());
            assertThrows(AssertionError.class, () -> work.remove().run());
            // Error has escaped real grpc-core SerializingExecutor, but cleanup is still queued.
            assertEquals(1, guard.receiving());
            assertEquals(65536, allocator.pinnedDirectMemory());
            int n = 0;
            while (!work.isEmpty()) {
                work.remove().run();
                assertTrue(++n < 10);
            }
            assertEquals(0, guard.receiving());
            assertEquals(0, allocator.pinnedDirectMemory());
            assertEquals(0, guard.pendingTimers());
        } finally {
            if (retained.refCnt() > 0) retained.release();
            c.detach(before);
            guard.close();
        }
    }

    @Test
    void terminalBeforeSupplierDoesNotCreateOrphanWatchdog() {
        var guard =
                new GrpcReceivingGuard(1, 1, 60000, Runnable::run, Runnable::run, 1 << 20, 1 << 20);
        var tracer = guard.newServerStreamTracer("test/Schedule", new Metadata());
        Context c = tracer.filterContext(Context.ROOT), before = c.attach();
        try {
            tracer.streamClosed(Status.CANCELLED);
            guard.getExecutor(new Call(), new Metadata());
            assertEquals(0, guard.receiving());
            assertEquals(0, guard.pendingTimers());
        } finally {
            c.detach(before);
            guard.close();
        }
    }

    @Test
    void shutdownClosesReceivingCallsAndTerminatesWatchdog() throws Exception {
        var guard =
                new GrpcReceivingGuard(1, 1, 60000, Runnable::run, Runnable::run, 1 << 20, 1 << 20);
        var tracer = guard.newServerStreamTracer("test/Schedule", new Metadata());
        Context c = tracer.filterContext(Context.ROOT), before = c.attach();
        try {
            guard.getExecutor(
                    new Call() {
                        @Override
                        public void close(Status s, Metadata h) {
                            tracer.streamClosed(s);
                        }
                    },
                    new Metadata());
            assertEquals(1, guard.pendingTimers());
            guard.close();
            assertEquals(0, guard.receiving());
            assertEquals(0, guard.pendingTimers());
            var f = GrpcReceivingGuard.class.getDeclaredField("timer");
            f.setAccessible(true);
            assertTrue(((ScheduledThreadPoolExecutor) f.get(guard)).isTerminated());
        } finally {
            c.detach(before);
            guard.close();
        }
    }

    /**
     * Supplies the inbound call for cleanup tests without a network peer.
     */
    static class Call extends ServerCall<Object, Object> {
        public void request(int n) {}

        public void sendHeaders(Metadata h) {}

        public void sendMessage(Object o) {}

        public void close(Status s, Metadata h) {}

        public boolean isCancelled() {
            return false;
        }

        public MethodDescriptor<Object, Object> getMethodDescriptor() {
            return null;
        }
    }
}
